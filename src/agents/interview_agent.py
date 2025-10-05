"""
Fixed interview agent with proper counter management and single-invocation pattern.

Key fixes:
1. Counter incremented ONLY in generate_question_node (atomic operation)
2. Router only reads state, never modifies it
3. Single graph invocation processes response AND generates next question
4. Proper demo mode limits integrated into state
5. Eliminated two-phase invocation anti-pattern
"""
from typing import Dict, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import logging

from .state import InterviewState, ResponseAnalysis, FollowUpQuestion
from config.settings import settings

logger = logging.getLogger(__name__)


class InterviewAgent:
    """
    Manages interview conversations using proper single-invocation pattern.
    
    The graph flow is:
    START → process_response → check_completion → [generate_question OR finalize] → END
    
    Each user interaction requires exactly ONE graph.invoke() call.
    The checkpointer maintains state between invocations.
    """
    
    def __init__(self, llm, demo_mode: bool = False, checkpointer=None):
        self.llm = llm
        self.demo_mode = demo_mode
        self.limits = settings.get_question_limits(demo_mode)
        self.checkpointer = checkpointer if checkpointer else MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """
        Build graph with proper node sequencing for Q&A flow.
        
        Critical design decisions:
        - Counter incremented in generate_question_node only
        - Router functions only read state
        - Each invoke processes one complete turn
        """
        builder = StateGraph(InterviewState)
        
        # Add processing nodes
        builder.add_node("process_response", self._process_response_node)
        builder.add_node("check_completion", self._check_completion_node)
        builder.add_node("generate_question", self._generate_question_node)
        builder.add_node("finalize", self._finalize_node)
        
        # Define flow
        builder.add_edge(START, "process_response")
        builder.add_edge("process_response", "check_completion")
        
        # Conditional routing based on completion status
        builder.add_conditional_edges(
            "check_completion",
            self._route_after_check,
            {
                "generate": "generate_question",
                "finalize": "finalize"
            }
        )
        
        # Both paths end the invocation
        builder.add_edge("generate_question", END)
        builder.add_edge("finalize", END)
        
        return builder.compile(checkpointer=self.checkpointer)
    
    def _generate_question_node(self, state: InterviewState) -> Dict:
        """
        Generate next question and increment counter atomically.
        
        This is the ONLY place where questions_asked is incremented.
        This ensures the counter always matches actual questions generated.
        """
        current_count = state.get("questions_asked", 0)
        max_questions = state.get("max_questions", self.limits["max_questions"])
        review_type = state.get("review_type", "self_review")
        
        logger.info(
            f"Generating question {current_count + 1}/{max_questions} "
            f"for {review_type}"
        )
        
        # If this is the first question, use the template
        if current_count == 0:
            question_text = settings.INITIAL_QUESTIONS.get(
                review_type,
                "Tell me about your performance this review period."
            )
        else:
            # Generate contextual follow-up
            question_text = self._create_followup_question(state)
        
        # Return BOTH the question and incremented counter
        # These updates happen atomically when the node completes
        return {
            "messages": [AIMessage(content=question_text)],
            "questions_asked": current_count + 1  # Atomic increment
        }
    
    def _create_followup_question(self, state: InterviewState) -> str:
        """Generate contextual follow-up question using LLM."""
        qa_pairs = state.get("qa_pairs", [])
        questions_asked = state.get("questions_asked", 0)
        max_questions = state.get("max_questions", 10)
        
        # Get recent conversation context
        recent_qa = qa_pairs[-3:] if qa_pairs else []
        context = "\n\n".join([
            f"Q: {qa['question']}\nA: {qa['answer']}"
            for qa in recent_qa
        ])
        
        # Calculate remaining questions for urgency
        remaining = max_questions - questions_asked
        urgency_context = ""
        if remaining <= 2:
            urgency_context = (
                f"\n\nIMPORTANT: Only {remaining} question(s) remaining. "
                "Make this question count by exploring the most important uncovered areas."
            )
        elif remaining <= 4:
            urgency_context = f"\n\nNote: {remaining} questions remaining."
        
        prompt = f"""You are conducting a performance review interview.

                    CONVERSATION HISTORY:
                    {context if context else "No previous questions yet."}

                    METADATA:
                    - Review Type: {state.get('review_type', 'self_review')}
                    - Questions Asked: {questions_asked}
                    - Topics Covered: {', '.join(state.get('topics_covered', [])) if state.get('topics_covered') else 'None yet'}
                    - Competencies to Explore: {', '.join(settings.COMPETENCY_AREAS)}{urgency_context}

                    Generate the next question that either:
                    1. Deepens understanding of an already-discussed area (ask for specific examples, impact, or learnings)
                    2. Explores a new competency area that hasn't been covered yet
                    3. Clarifies a previous response if it was vague

                    The question should encourage detailed, specific responses with examples.
                    Return ONLY the question text, nothing else."""

        try:
            response = self.llm.with_structured_output(FollowUpQuestion).invoke(prompt)
            return response.question
        except Exception as e:
            logger.warning(f"Structured output failed, using fallback: {e}")
            response = self.llm.invoke(prompt)
            return response.content
    
    def _process_response_node(self, state: InterviewState) -> Dict:
        """
        Process user's response and create QA pair.
        
        This node does NOT generate the next question or increment counters.
        It only processes what the user said and extracts insights.
        """
        messages = state.get("messages", [])
        
        # Skip if no messages or less than 2 (need Q&A pair)
        if len(messages) < 2:
            logger.debug("Skipping process_response - insufficient messages")
            return {}
        
        # Find the last Q&A exchange
        last_human = None
        last_ai = None
        
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and last_human is None:
                last_human = msg
            elif isinstance(msg, AIMessage) and last_ai is None:
                last_ai = msg
            
            if last_human and last_ai:
                break
        
        if not last_human or not last_ai:
            logger.debug("Could not find Q&A pair")
            return {}
        
        # Analyze the response quality
        analysis = self._analyze_response_content(
            question=str(last_ai.content),
            answer=str(last_human.content)
        )
        
        # Create QA pair with analysis metadata
        qa_pair = {
            "question": last_ai.content,
            "answer": last_human.content,
            "depth_score": analysis.depth_score,
            "topics": analysis.topics,
            "question_number": state.get("questions_asked", 0)
        }
        
        logger.info(
            f"Processed response - depth: {analysis.depth_score:.2f}, "
            f"topics: {analysis.topics}"
        )
        
        # Update state with new QA pair and discovered topics
        existing_qa = state.get("qa_pairs", [])
        existing_topics = state.get("topics_covered", [])
        
        return {
            "qa_pairs": existing_qa + [qa_pair],
            "topics_covered": list(set(existing_topics + analysis.topics)),
            "depth_score": analysis.depth_score
        }
    
    def _analyze_response_content(
        self,
        question: str,
        answer: str
    ) -> ResponseAnalysis:
        """Analyze the depth and content of a user's response."""
        prompt = f"""Analyze this interview response for depth and content:

                    QUESTION: {question}

                    ANSWER: {answer}

                    Provide:
                    - depth_score: A decimal between 0.0 and 1.0 indicating response quality
                    * 0.0-0.3: Vague, brief, or surface-level
                    * 0.4-0.6: Some detail but could be deeper
                    * 0.7-0.9: Detailed with specific examples
                    * 1.0: Exceptional depth with metrics and learnings

                    - topics: List of competency areas or themes discussed (e.g., ["communication", "leadership"])
                    - key_insights: Important information revealed
                    - follow_up_areas: What could benefit from deeper exploration"""

        try:
            return self.llm.with_structured_output(ResponseAnalysis).invoke(prompt)
        except Exception as e:
            logger.warning(f"Analysis failed, using fallback: {e}")
            # Fallback heuristic analysis
            word_count = len(answer.split())
            return ResponseAnalysis(
                depth_score=min(word_count / 100, 1.0),
                topics=[],
                key_insights=[],
                follow_up_areas=[]
            )
    
    def _check_completion_node(self, state: InterviewState) -> Dict:
        """
        Determine if interview should complete.
        
        This node sets is_complete flag based on:
        1. Question limit reached
        2. User explicitly requested exit
        3. Sufficient depth and coverage achieved (full mode only)
        """
        questions_asked = state.get("questions_asked", 0)
        max_questions = state.get("max_questions", 10)
        topics_covered = len(state.get("topics_covered", []))
        qa_pairs = state.get("qa_pairs", [])
        
        # Calculate average depth
        if qa_pairs:
            avg_depth = sum(qa.get("depth_score", 0) for qa in qa_pairs) / len(qa_pairs)
        else:
            avg_depth = 0.0
        
        logger.info(
            f"Completion check - Q:{questions_asked}/{max_questions}, "
            f"Topics:{topics_covered}, AvgDepth:{avg_depth:.2f}"
        )
        
        # Hard limit: max questions reached
        if questions_asked >= max_questions:
            logger.info("Completing: max questions reached")
            return {"is_complete": True}
        
        # User explicitly requested exit - only check for very explicit phrases
        # to avoid false positives from normal responses mentioning "done" or "finished"
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            content = messages[-1].content
            # Handle case where content might be a list or other type
            if isinstance(content, str):
                last_content = content.lower().strip()
            else:
                last_content = str(content).lower().strip()
            
            # Only trigger on standalone commands or very explicit phrases
            # Not just any occurrence of these words in longer responses
            explicit_exit_phrases = [
                "exit", "quit", "i'm done", "i am done", 
                "stop the interview", "end interview",
                "that's all", "thats all", "no more questions"
            ]
            
            # Check if the entire message is one of these phrases (with some flexibility)
            words = last_content.split()
            if (len(words) <= 5 and  # Short message
                any(phrase in last_content for phrase in explicit_exit_phrases)):
                logger.info("Completing: user requested exit")
                return {"is_complete": True}
        
        # Natural completion (for full mode only, not demo mode)
        # In demo mode, we ONLY stop at max_questions, never early
        if not self.demo_mode:
            min_questions = self.limits["min_questions"]
            min_topics = self.limits["min_topics"]
            min_depth = self.limits["min_depth_score"]
            
            if (questions_asked >= min_questions and 
                topics_covered >= min_topics and 
                avg_depth >= min_depth):
                logger.info("Completing: natural criteria met (full mode)")
                return {"is_complete": True}
        
        # Continue interview
        logger.debug(f"Continuing interview - {max_questions - questions_asked} questions remaining")
        return {"is_complete": False}
    
    def _route_after_check(self, state: InterviewState) -> Literal["generate", "finalize"]:
        """
        Route based on completion status.
        
        CRITICAL: This function only READS state, never modifies it.
        All state changes happen in nodes, not in routing logic.
        """
        is_complete = state.get("is_complete", False)
        
        if is_complete:
            logger.info("Routing to finalize")
            return "finalize"
        else:
            logger.info("Routing to generate next question")
            return "generate"
    
    def _finalize_node(self, state: InterviewState) -> Dict:
        """
        Generate completion message when interview ends.
        
        This provides a summary and thanks the user for their responses.
        """
        questions_asked = state.get("questions_asked", 0)
        topics_covered = len(state.get("topics_covered", []))
        mode = "Demo" if self.demo_mode else "Full"
        
        completion_message = f"""✅ {mode} Interview Complete!

                                Thank you for your thoughtful responses. Here's what we covered:

                                📊 Questions Answered: {questions_asked}
                                🎯 Topics Discussed: {topics_covered}
                                📝 Total Responses: {len(state.get('qa_pairs', []))}

                                Your insights have been recorded and will contribute to a comprehensive performance review.

                                You can now save this session or start a new review type."""
        
        logger.info(f"Interview finalized - {questions_asked} questions, {topics_covered} topics")
        
        return {
            "messages": [AIMessage(content=completion_message)],
            "is_complete": True
        }
    
    def get_initial_message(self) -> str:
        """
        Get the initial question for a new interview.
        
        This is called by the UI when starting a fresh session.
        It doesn't increment the counter - that happens when
        generate_question_node runs.
        """
        # This is just a preview - the actual counter increment
        # happens when the graph runs
        return "Initializing interview..."