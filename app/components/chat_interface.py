"""
Fixed chat interface with proper single-invocation pattern.

Key fixes:
1. Single graph.invoke() per user interaction
2. Proper state initialization with max_questions
3. Accurate progress tracking from graph state
4. Eliminated two-step invocation anti-pattern
"""
import streamlit as st
import logging
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

from config.settings import settings

logger = logging.getLogger(__name__)


class ChatInterface:
    """
    Manages chat interface with proper single-invocation pattern.
    
    Each user message triggers ONE graph invocation that:
    1. Processes the response
    2. Checks completion
    3. Generates next question (or finalizes)
    
    The checkpointer maintains state between invocations.
    """
    
    def __init__(self, interview_agent, vector_store):
        self.interview_agent = interview_agent
        self.vector_store = vector_store
    
    def render(self):
        """Render the complete chat interface."""
        active_session = st.session_state.active_session
        session_display_name = settings.REVIEW_TYPES.get(active_session, "Interview")
        
        # Display header
        st.title(f"🎤 {session_display_name}")
        
        # Demo mode banner
        if st.session_state.demo_mode:
            limits = settings.get_question_limits(True)
            st.warning(
                f"**🎭 DEMO MODE ACTIVE** - Quick {limits['max_questions']}-question assessment. "
                "Switch to Full Mode in the sidebar for comprehensive interviews.",
                icon="⚠️"
            )
        
        # Show context
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"**Employee:** {st.session_state.employee_name}")
        with col2:
            st.caption(f"**Cycle:** {st.session_state.review_cycle}")
        with col3:
            if active_session in st.session_state.completed_sessions:
                st.caption("**Status:** ✅ Complete")
            else:
                st.caption("**Status:** 🔵 In Progress")
        
        # Get configuration for this session
        thread_id = st.session_state.thread_ids[active_session]
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initialize if needed
        if not self._is_initialized(active_session):
            self._initialize_interview(active_session, config)
        
        # Get current state from graph
        current_state = self._get_current_state(config)
        
        # Show accurate progress from graph state
        self._display_progress(current_state, active_session)
        
        st.divider()
        
        # Display all messages
        self._display_messages(current_state)
        
        # Handle user input
        if not current_state.get("is_complete", False):
            self._handle_user_input(active_session, config, current_state)
        else:
            st.success("✅ Interview complete! You can now save this session.")
        
        # Show tips
        self._display_tips()
    
    def _is_initialized(self, active_session: str) -> bool:
        """Check if this session has been initialized."""
        return f"{active_session}_initialized" in st.session_state
    
    def _get_current_state(self, config: dict) -> dict:
        """Get current state from graph checkpointer."""
        try:
            snapshot = self.interview_agent.graph.get_state(config)
            if snapshot and snapshot.values:
                return snapshot.values
        except Exception as e:
            logger.warning(f"Could not retrieve state: {e}")
        
        # Return empty state if nothing found
        return {
            "messages": [],
            "qa_pairs": [],
            "questions_asked": 0,
            "max_questions": settings.get_question_limits(st.session_state.demo_mode)["max_questions"],
            "is_complete": False
        }
    
    def _initialize_interview(self, active_session: str, config: dict):
        """
        Initialize a new interview session.
        
        This creates the initial state and generates the first question
        using a single graph invocation.
        """
        try:
            mode_text = "demo" if st.session_state.demo_mode else "full"
            logger.info(f"Initializing {active_session} ({mode_text} mode)")
            
            # Get limits for current mode
            limits = settings.get_question_limits(st.session_state.demo_mode)
            
            # Prepare initial state with proper max_questions
            reviewer_type = active_session.replace("_review", "")
            initial_state = {
                "messages": [],
                "qa_pairs": [],
                "topics_covered": [],
                "depth_score": 0.0,
                "questions_asked": 0,  # Counter starts at 0
                "max_questions": limits["max_questions"],  # Set limit based on mode
                "is_complete": False,
                "review_type": active_session,
                "employee_id": st.session_state.employee_id,
                "employee_name": st.session_state.employee_name,
                "reviewer_id": "current_user",
                "review_cycle": st.session_state.review_cycle
            }
            
            # Single invocation to initialize and generate first question
            with st.spinner("Generating first question..."):
                result = self.interview_agent.graph.invoke(initial_state, config)
            
            # Mark as initialized
            st.session_state[f"{active_session}_initialized"] = True
            
            logger.info(
                f"Initialized successfully - questions_asked: {result.get('questions_asked', 0)}, "
                f"max_questions: {result.get('max_questions', 0)}"
            )
            
        except Exception as e:
            logger.error(f"Error initializing interview: {e}", exc_info=True)
            st.error(f"Failed to start interview: {str(e)}")
    
    def _display_progress(self, state: dict, active_session: str):
        """Display accurate progress from graph state."""
        questions_asked = state.get("questions_asked", 0)
        max_questions = state.get("max_questions", 10)
        
        # Only show progress bar if we have questions
        if questions_asked > 0 or max_questions > 0:
            progress = min(questions_asked / max_questions, 1.0) if max_questions > 0 else 0
            
            # Show progress bar
            st.progress(
                progress,
                text=f"Question {questions_asked} of {max_questions}"
            )
            
            # Show detailed metrics in expander
            with st.expander("📊 Session Metrics", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Questions", questions_asked)
                
                with col2:
                    topics_count = len(state.get("topics_covered", []))
                    st.metric("Topics", topics_count)
                
                with col3:
                    qa_pairs = state.get("qa_pairs", [])
                    if qa_pairs:
                        avg_depth = sum(qa.get("depth_score", 0) for qa in qa_pairs) / len(qa_pairs)
                        st.metric("Avg Depth", f"{avg_depth:.2f}")
                    else:
                        st.metric("Avg Depth", "N/A")
    
    def _display_messages(self, state: dict):
        """Display conversation history from state."""
        messages = state.get("messages", [])
        
        if not messages:
            st.info("👋 Preparing your first question...")
            return
        
        # Display each message
        for msg in messages:
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            with st.chat_message(role):
                st.markdown(msg.content)
    
    def _handle_user_input(self, active_session: str, config: dict, current_state: dict):
        """
        Handle user input with single-invocation pattern.
        
        This is the critical fix: ONE graph.invoke() call that processes
        the answer AND generates the next question in a single execution.
        """
        # Get user input
        prompt = st.chat_input(
            f"Your response for {settings.REVIEW_TYPES[active_session]}...",
            key=f"chat_input_{active_session}"
        )
        
        if not prompt:
            return  # No input yet
        
        try:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Single graph invocation processes response AND generates next question
            with st.chat_message("assistant"):
                with st.spinner("Processing your response..."):
                    # CRITICAL: Single invoke() call
                    result = self.interview_agent.graph.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config
                    )
                
                # Display the result (either next question or completion message)
                if result.get("messages"):
                    last_message = result["messages"][-1]
                    st.markdown(last_message.content)
                
                # Log state after invocation
                logger.info(
                    f"After invoke - questions_asked: {result.get('questions_asked', 0)}, "
                    f"is_complete: {result.get('is_complete', False)}"
                )
                
                # Check if completed
                if result.get("is_complete"):
                    st.session_state.completed_sessions.add(active_session)
                    
                    # Show statistics
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Questions Answered", result.get("questions_asked", 0))
                    
                    with col2:
                        topics_count = len(result.get("topics_covered", []))
                        st.metric("Topics Covered", topics_count)
                    
                    with col3:
                        qa_pairs = result.get("qa_pairs", [])
                        if qa_pairs:
                            avg_depth = sum(qa.get("depth_score", 0) for qa in qa_pairs) / len(qa_pairs)
                            st.metric("Depth Score", f"{avg_depth:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            st.error(
                f"An error occurred: {str(e)}\n\n"
                "Please try again or start a new session."
            )
            return
        
        # Trigger rerun to update UI with new state
        st.rerun()
    
    def _display_tips(self):
        """Display helpful tips for users."""
        with st.expander("💡 Tips for a Great Interview"):
            if st.session_state.demo_mode:
                st.markdown("""
                **Demo Mode Tips:**
                - Provide concise but specific responses
                - Focus on one clear example per question
                - Each answer demonstrates the AI's intelligent follow-up capabilities
                - Perfect for quick presentations or initial assessments
                """)
            else:
                st.markdown("""
                **Full Interview Tips:**
                - Provide detailed responses with specific examples
                - Include quantifiable impact where possible (numbers, percentages, outcomes)
                - Be honest and reflective in your responses
                - Use STAR format: Situation, Task, Action, Result
                - The AI will ask follow-up questions to help you elaborate on important points
                - Take your time - quality responses lead to better insights
                """)
    
    def get_save_data(self, active_session: str, config: dict) -> dict:
        """
        Get data for saving the current session.
        
        This retrieves the complete state from the graph for storage.
        """
        try:
            state = self._get_current_state(config)
            
            return {
                "qa_pairs": state.get("qa_pairs", []),
                "questions_asked": state.get("questions_asked", 0),
                "max_questions": state.get("max_questions", 0),
                "topics_covered": state.get("topics_covered", []),
                "is_complete": state.get("is_complete", False),
                "demo_mode": st.session_state.demo_mode,
                "review_type": active_session
            }
        except Exception as e:
            logger.error(f"Error getting save data: {e}")
            return {}