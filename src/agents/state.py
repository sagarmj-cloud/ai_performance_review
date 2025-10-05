"""
State definitions for the interview agent using LangGraph.

Key fixes:
- Removed unnecessary reducers from simple counters
- Added max_questions to state for demo mode support
- Clarified which fields use reducers vs default override behavior
"""
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class InterviewState(TypedDict):
    """
    State for managing interview conversations.
    
    Fields with Annotated reducers accumulate values (like messages).
    Fields without reducers use default override behavior (like counters).
    This distinction is critical for maintaining accurate state.
    """
    # Message history with automatic merging via add_messages reducer
    # This accumulates messages across invocations
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Structured Q&A pairs extracted from conversation
    # Uses default list concatenation (override behavior)
    qa_pairs: List[Dict]
    
    # Topics identified and covered during interview
    # Uses default list behavior (override)
    topics_covered: List[str]
    
    # Depth score (0-1) indicating quality of responses
    # Simple float, uses override behavior
    depth_score: float
    
    # Number of questions asked in this session
    # CRITICAL: This counter uses override behavior (no reducer needed)
    # It should only be incremented in generate_question_node
    questions_asked: int
    
    # Maximum questions for this session (demo mode limit)
    # Set once at initialization, never changes
    max_questions: int
    
    # Whether the interview has gathered sufficient information
    # Boolean flag, uses override behavior
    is_complete: bool
    
    # Type of review being conducted
    review_type: str
    
    # Employee and review metadata
    employee_id: str
    reviewer_id: str
    review_cycle: str


# Pydantic models for structured outputs from LLM
class ResponseAnalysis(BaseModel):
    """Analysis of a single interview response."""
    depth_score: float = Field(
        description="Score from 0 to 1 indicating response depth and detail",
        ge=0.0,
        le=1.0
    )
    topics: List[str] = Field(
        description="List of topics or themes mentioned in the response"
    )
    key_insights: List[str] = Field(
        description="Important insights or information extracted from response"
    )
    follow_up_areas: List[str] = Field(
        description="Areas that could benefit from follow-up questions"
    )


class FollowUpQuestion(BaseModel):
    """Generated follow-up question with context."""
    question: str = Field(
        description="The follow-up question to ask"
    )
    focus_area: str = Field(
        description="The competency or topic this question addresses"
    )
    question_type: str = Field(
        description="Type: clarifying, deepening, exploring_new, or behavioral"
    )
    reasoning: str = Field(
        description="Why this question is appropriate at this point"
    )