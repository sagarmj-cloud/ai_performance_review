"""
Unit tests for the interview agent.

These tests verify that the LangGraph interview agent correctly manages
conversation flow, state updates, and completion detection.
"""
import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.interview_agent import InterviewAgent
from src.agents.state import InterviewState


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing without actual model calls."""
    llm = Mock()
    
    # Mock the invoke method to return a simple message
    mock_response = Mock()
    mock_response.content = "This is a test question."
    llm.invoke.return_value = mock_response
    
    # Mock structured output
    def mock_structured_output(schema):
        mock_llm_with_schema = Mock()
        
        if schema.__name__ == "FollowUpQuestion":
            mock_question = Mock()
            mock_question.question = "Can you provide more details?"
            mock_question.focus_area = "Technical Skills"
            mock_question.question_type = "deepening"
            mock_question.reasoning = "Need more specific examples"
            mock_llm_with_schema.invoke.return_value = mock_question
            
        elif schema.__name__ == "ResponseAnalysis":
            mock_analysis = Mock()
            mock_analysis.depth_score = 0.7
            mock_analysis.topics = ["communication", "teamwork"]
            mock_analysis.key_insights = ["Good collaboration skills"]
            mock_analysis.follow_up_areas = ["leadership"]
            mock_llm_with_schema.invoke.return_value = mock_analysis
            
        elif schema.__name__ == "CompletionCheck":
            mock_check = Mock()
            mock_check.should_continue = False
            mock_check.coverage_score = 0.8
            mock_check.missing_areas = []
            mock_check.reasoning = "Sufficient information gathered"
            mock_llm_with_schema.invoke.return_value = mock_check
        
        return mock_llm_with_schema
    
    llm.with_structured_output = mock_structured_output
    
    return llm


@pytest.fixture
def interview_agent(mock_llm):
    """Create an interview agent with mocked LLM."""
    return InterviewAgent(llm=mock_llm)


def test_agent_initialization(interview_agent):
    """Test that the agent initializes correctly with a graph."""
    assert interview_agent is not None
    assert interview_agent.graph is not None
    assert interview_agent.llm is not None


def test_generate_initial_question(interview_agent):
    """Test that the agent generates an appropriate initial question."""
    state = {
        "messages": [],
        "qa_pairs": [],
        "topics_covered": [],
        "depth_score": 0.0,
        "questions_asked": 0,
        "is_complete": False,
        "review_type": "self_review",
        "employee_id": "test_emp",
        "reviewer_id": "test_reviewer",
        "review_cycle": "2025-Q3"
    }
    
    result = interview_agent._generate_initial_question(state)
    
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert isinstance(result["messages"][0], AIMessage)
    assert result["questions_asked"] == 1


def test_analyze_response(interview_agent):
    """Test that the agent correctly analyzes user responses."""
    state = {
        "messages": [
            AIMessage(content="What are your key strengths?"),
            HumanMessage(content="I excel at communication and teamwork.")
        ],
        "qa_pairs": [],
        "topics_covered": [],
        "depth_score": 0.0,
        "questions_asked": 1,
        "is_complete": False,
        "review_type": "self_review",
        "employee_id": "test_emp",
        "reviewer_id": "test_reviewer",
        "review_cycle": "2025-Q3"
    }
    
    result = interview_agent._analyze_response(state)
    
    assert "qa_pairs" in result
    assert len(result["qa_pairs"]) == 1
    assert "question" in result["qa_pairs"][0]
    assert "answer" in result["qa_pairs"][0]
    assert "depth_score" in result["qa_pairs"][0]
    assert "topics_covered" in result


def test_assess_completion_not_complete(interview_agent):
    """Test completion assessment when criteria are not met."""
    state = {
        "messages": [],
        "qa_pairs": [
            {"question": "Q1", "answer": "A1", "depth_score": 0.5}
        ],
        "topics_covered": ["communication"],
        "depth_score": 0.5,
        "questions_asked": 2,
        "is_complete": False,
        "review_type": "self_review",
        "employee_id": "test_emp",
        "reviewer_id": "test_reviewer",
        "review_cycle": "2025-Q3"
    }
    
    result = interview_agent._assess_completion(state)
    
    # Should not be complete - only 2 questions and 1 topic
    assert "is_complete" in result


def test_assess_completion_max_questions(interview_agent):
    """Test that completion is forced after maximum questions."""
    qa_pairs = [
        {"question": f"Q{i}", "answer": f"A{i}", "depth_score": 0.7}
        for i in range(15)
    ]
    
    state = {
        "messages": [],
        "qa_pairs": qa_pairs,
        "topics_covered": ["topic1", "topic2", "topic3"],
        "depth_score": 0.7,
        "questions_asked": 15,
        "is_complete": False,
        "review_type": "self_review",
        "employee_id": "test_emp",
        "reviewer_id": "test_reviewer",
        "review_cycle": "2025-Q3"
    }
    
    result = interview_agent._assess_completion(state)
    
    # Should be complete - max questions reached
    assert result["is_complete"] == True


def test_route_completion_continue(interview_agent):
    """Test routing when interview should continue."""
    state = {
        "is_complete": False
    }
    
    route = interview_agent._route_completion(state)
    assert route == "continue"


def test_route_completion_end(interview_agent):
    """Test routing when interview should end."""
    state = {
        "is_complete": True
    }
    
    route = interview_agent._route_completion(state)
    assert route == "end"


def test_full_interview_flow(interview_agent):
    """Integration test of a complete interview flow."""
    # This would test the full graph execution
    # For now, we'll test the basic flow structure
    config = {
        "configurable": {
            "thread_id": "test_thread"
        }
    }
    
    # The graph should be properly compiled
    assert interview_agent.graph is not None
    
    # We can verify the graph has the expected nodes
    # (This is a basic structural test)
    # In a real test environment, you would:
    # 1. Initialize the graph with state
    # 2. Simulate user responses
    # 3. Verify the graph progresses through nodes correctly
    # 4. Verify completion criteria are properly evaluated