"""
Fixed main Streamlit application with proper state management.

Key fixes:
1. Proper checkpointer initialization (one per app instance)
2. Clear session state when switching demo/full mode
3. Correct graph state initialization with max_questions
4. Proper save functionality that reads from graph state
"""
import streamlit as st
import uuid
import logging
from datetime import datetime

from components.chat_interface import ChatInterface
from components.report_generator import ReportGenerator
from src.agents.interview_agent import InterviewAgent
from src.storage.vector_store import ReviewVectorStore
from src.llm.ollama_client import create_ollama_client
from src.aggregation.aggregator import PerformanceReviewAggregator
from config.settings import Settings as settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_llm_and_storage():
    """
    Initialize LLM and storage components with caching.
    These components are mode-independent so can be cached.
    """
    try:
        logger.info("Initializing LLM and storage components...")
        
        # Initialize LLM client
        logger.info(f"Connecting to Ollama at {settings.OLLAMA_BASE_URL}")
        llm_client = create_ollama_client()
        llm = llm_client.get_llm()
        
        # Initialize vector store
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
        vector_store = ReviewVectorStore(
            collection_name=settings.QDRANT_COLLECTION,
            qdrant_url=settings.QDRANT_URL,
            qdrant_api_key=settings.QDRANT_API_KEY
        )
        
        logger.info("Initialization complete!")
        return llm_client, llm, vector_store
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        st.error(
            f"Failed to initialize the system. Please check that Ollama and Qdrant are running.\n\n"
            f"Error: {str(e)}"
        )
        st.stop()


def get_or_create_agents(llm, vector_store, demo_mode, checkpointer):
    """
    Get existing agents or create new ones when demo_mode changes.
    
    CRITICAL: Agents must be recreated when demo_mode changes because
    they have different question limits and behavior.
    """
    recreate_needed = (
        'interview_agent' not in st.session_state or 
        'aggregator' not in st.session_state or
        st.session_state.get('agent_demo_mode') != demo_mode
    )
    
    if recreate_needed:
        logger.info(f"Creating agents with demo_mode={demo_mode}")
        st.session_state.interview_agent = InterviewAgent(
            llm=llm, 
            demo_mode=demo_mode, 
            checkpointer=checkpointer
        )
        st.session_state.aggregator = PerformanceReviewAggregator(
            vector_store=vector_store,
            llm=llm
        )
        st.session_state.agent_demo_mode = demo_mode
        logger.info("Agents created successfully")
    else:
        logger.info("Using cached agents from session state")
    
    return st.session_state.interview_agent, st.session_state.aggregator


def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    
    This sets up all the state needed for the application,
    including the checkpointer which is shared across all threads.
    """
    from langgraph.checkpoint.memory import MemorySaver
    
    # Demo mode - default to demo for presentations
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = True
    
    # Checkpointer - ONE instance shared by all threads
    # This is critical for proper state persistence
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()
        logger.info("Created new checkpointer in session state")
    
    # Application mode
    if "mode" not in st.session_state:
        st.session_state.mode = "interview"
    
    # Employee and review information
    if "employee_id" not in st.session_state:
        st.session_state.employee_id = "emp_001"
    
    if "employee_name" not in st.session_state:
        st.session_state.employee_name = "John Doe"
    
    if "review_cycle" not in st.session_state:
        current_year = datetime.now().year
        current_quarter = (datetime.now().month - 1) // 3 + 1
        st.session_state.review_cycle = f"{current_year}-Q{current_quarter}"
    
    # Current active session type
    if "active_session" not in st.session_state:
        st.session_state.active_session = "self_review"
    
    # Thread IDs for conversation persistence (one per review type)
    if "thread_ids" not in st.session_state:
        st.session_state.thread_ids = {
            session_type: str(uuid.uuid4())
            for session_type in settings.REVIEW_TYPES.keys()
        }
    
    # Interview completion status per session
    if "completed_sessions" not in st.session_state:
        st.session_state.completed_sessions = set()


def clear_all_sessions():
    """
    Clear all interview sessions when switching between demo and full mode.
    
    This is necessary because:
    1. Different modes have different max_questions limits
    2. Graph state needs to be reinitialized with new limits
    3. Old thread state would have incorrect counters
    """
    from langgraph.checkpoint.memory import MemorySaver
    
    logger.info("Clearing all sessions due to mode change...")
    
    # Create a FRESH checkpointer to clear all thread state
    st.session_state.checkpointer = MemorySaver()
    logger.info("Created fresh checkpointer")
    
    # Clear cached agents so they get recreated with new demo_mode
    for key in ['interview_agent', 'aggregator', 'agent_demo_mode']:
        if key in st.session_state:
            del st.session_state[key]
    logger.info("Cleared cached agents")
    
    # Reset all thread IDs
    st.session_state.thread_ids = {
        session_type: str(uuid.uuid4())
        for session_type in settings.REVIEW_TYPES.keys()
    }
    
    # Clear initialization flags
    for session_type in settings.REVIEW_TYPES.keys():
        init_key = f"{session_type}_initialized"
        if init_key in st.session_state:
            del st.session_state[init_key]
    
    # Clear completion status
    st.session_state.completed_sessions = set()
    
    logger.info("All sessions cleared successfully")


def render_sidebar(vector_store, chat_interface):
    """
    Render sidebar with configuration and session management.
    """
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Demo Mode Toggle - Prominent at the top
        st.subheader("🎭 Application Mode")
        
        previous_demo_mode = st.session_state.demo_mode
        
        demo_mode = st.radio(
            "Interview Length",
            options=[True, False],
            format_func=lambda x: f"🎯 Demo Mode ({settings.get_question_limits(True)['max_questions']} questions)" if x else f"📋 Full Mode (up to {settings.get_question_limits(False)['max_questions']} questions)",
            index=0 if st.session_state.demo_mode else 1,
            key="demo_mode_selector",
            help="Demo mode conducts shorter interviews perfect for presentations"
        )
        
        # Check if demo mode changed
        if demo_mode != previous_demo_mode:
            st.session_state.demo_mode = demo_mode
            clear_all_sessions()
            st.rerun()
        
        # Show current limits
        limits = settings.get_question_limits(st.session_state.demo_mode)
        if st.session_state.demo_mode:
            st.info(
                f"**🎭 Demo Mode Active**\n\n"
                f"• Questions: {limits['min_questions']}-{limits['max_questions']}\n"
                f"• Topics: {limits['min_topics']}+\n"
                f"• Perfect for quick presentations!"
            )
        else:
            st.success(
                f"**📋 Full Mode Active**\n\n"
                f"• Questions: {limits['min_questions']}-{limits['max_questions']}\n"
                f"• Topics: {limits['min_topics']}+\n"
                f"• Comprehensive assessment"
            )
        
        st.divider()
        
        # Mode selection: Interview or Report
        st.session_state.mode = st.radio(
            "Function",
            ["interview", "report"],
            format_func=lambda x: "🎤 Conduct Interview" if x == "interview" else "📊 Generate Report",
            key="mode_selector"
        )
        
        st.divider()
        
        # Employee information
        st.subheader("Employee Information")
        
        st.session_state.employee_id = st.text_input(
            "Employee ID",
            value=st.session_state.employee_id,
            help="Unique identifier for the employee"
        )
        
        st.session_state.employee_name = st.text_input(
            "Employee Name",
            value=st.session_state.employee_name
        )
        
        st.session_state.review_cycle = st.text_input(
            "Review Cycle",
            value=st.session_state.review_cycle,
            help="Format: YYYY-QN (e.g., 2025-Q3) or annual-YYYY"
        )
        
        # Interview mode specific controls
        if st.session_state.mode == "interview":
            st.divider()
            st.subheader("Interview Session")
            
            # Session type selector with visual indicators
            for session_key, session_name in settings.REVIEW_TYPES.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    button_type = "primary" if session_key == st.session_state.active_session else "secondary"
                    if st.button(
                        session_name,
                        key=f"btn_{session_key}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state.active_session = session_key
                        st.rerun()
                
                with col2:
                    # Show completion indicator
                    if session_key in st.session_state.completed_sessions:
                        st.caption("✅")
                
                with col3:
                    # Show question count from graph state
                    try:
                        thread_id = st.session_state.thread_ids[session_key]
                        config = {"configurable": {"thread_id": thread_id}}
                        snapshot = st.session_state.interview_agent.graph.get_state(config)
                        if snapshot and snapshot.values:
                            q_count = snapshot.values.get("questions_asked", 0)
                            st.caption(f"📝 {q_count}")
                        else:
                            st.caption("📝 0")
                    except:
                        st.caption("📝 0")
            
            st.divider()
            
            # Session management actions
            st.subheader("Actions")
            
            # Save current session
            if st.button("💾 Save Session", use_container_width=True):
                save_current_session(vector_store, chat_interface)
            
            # Clear current session
            if st.button("🗑️ Clear Current", use_container_width=True):
                clear_current_session()
            
            # Start new session
            if st.button("🔄 New Session", use_container_width=True):
                start_new_session()
        
        st.divider()
        
        # System status
        st.caption("🟢 System Online")
        mode_text = "Demo Mode" if st.session_state.demo_mode else "Full Mode"
        st.caption(f"Mode: {mode_text}")
        st.caption(f"Model: {settings.OLLAMA_MODEL}")


def save_current_session(vector_store, chat_interface):
    """
    Save the current interview session to vector store.
    
    This reads the complete state from the graph and saves it.
    """
    active_session = st.session_state.active_session
    thread_id = st.session_state.thread_ids[active_session]
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Get save data from chat interface (which reads from graph state)
        save_data = chat_interface.get_save_data(active_session, config)
        
        if not save_data or not save_data.get("qa_pairs"):
            st.sidebar.warning("No Q&A pairs to save in this session.")
            return
        
        qa_pairs = save_data["qa_pairs"]
        
        # Format as complete transcript
        content = "\n\n".join([
            f"Q{qa.get('question_number', i+1)}: {qa['question']}\n\nA: {qa['answer']}"
            for i, qa in enumerate(qa_pairs)
        ])
        
        # Store in vector database
        reviewer_type = active_session.replace("_review", "")
        doc_id = vector_store.add_review_session(
            content=content,
            employee_id=st.session_state.employee_id,
            employee_name=st.session_state.employee_name,
            reviewer_type=reviewer_type,
            review_cycle=st.session_state.review_cycle,
            session_date=datetime.now().isoformat(),
            questions_asked=save_data.get("questions_asked", 0),
            demo_mode=st.session_state.demo_mode
        )
        
        # Mark as complete
        st.session_state.completed_sessions.add(active_session)
        
        mode_text = "Demo" if st.session_state.demo_mode else "Full"
        st.sidebar.success(f"✅ {mode_text} session saved! ({len(qa_pairs)} Q&A pairs)")
        logger.info(
            f"Saved {reviewer_type} review session for {st.session_state.employee_id} "
            f"(demo={st.session_state.demo_mode})"
        )
        
    except Exception as e:
        st.sidebar.error(f"Failed to save session: {str(e)}")
        logger.error(f"Error saving session: {e}")


def clear_current_session():
    """Clear the current session by resetting its thread ID."""
    active_session = st.session_state.active_session
    
    # Generate new thread ID (creates fresh state in checkpointer)
    st.session_state.thread_ids[active_session] = str(uuid.uuid4())
    
    # Clear initialization flag
    init_key = f"{active_session}_initialized"
    if init_key in st.session_state:
        del st.session_state[init_key]
    
    # Remove from completed
    if active_session in st.session_state.completed_sessions:
        st.session_state.completed_sessions.remove(active_session)
    
    st.sidebar.info("🗑️ Session cleared")
    st.rerun()


def start_new_session():
    """Start a new session with fresh thread ID."""
    active_session = st.session_state.active_session
    
    # Generate new thread ID
    st.session_state.thread_ids[active_session] = str(uuid.uuid4())
    
    # Clear initialization flag
    init_key = f"{active_session}_initialized"
    if init_key in st.session_state:
        del st.session_state[init_key]
    
    # Remove from completed
    if active_session in st.session_state.completed_sessions:
        st.session_state.completed_sessions.remove(active_session)
    
    st.sidebar.info("🔄 New session started")
    st.rerun()


def main():
    """Main application entry point."""
    
    # Configure page
    st.set_page_config(
        page_title="AI Performance Review System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize LLM and storage (cached)
    llm_client, llm, vector_store = initialize_llm_and_storage()
    
    # Get or create agents (recreates when demo_mode changes)
    interview_agent, aggregator = get_or_create_agents(
        llm, 
        vector_store, 
        st.session_state.demo_mode,
        st.session_state.checkpointer
    )
    
    # Create chat interface
    chat_interface = ChatInterface(
        interview_agent=interview_agent,
        vector_store=vector_store
    )
    
    # Render sidebar
    render_sidebar(vector_store, chat_interface)
    
    # Main content area
    if st.session_state.mode == "interview":
        chat_interface.render()
    else:
        report_generator = ReportGenerator(
            aggregator=aggregator,
            vector_store=vector_store
        )
        report_generator.render()


if __name__ == "__main__":
    main()