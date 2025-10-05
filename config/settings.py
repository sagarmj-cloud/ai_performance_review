"""
Configuration settings for the AI Performance Review System
"""
import os
from typing import List, Dict, Union
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables and defaults"""
    
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = "performance_reviews"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_NUM_CTX: int = 4096
    OLLAMA_KEEP_ALIVE: str = "10m"
    
    # OpenAI Configuration (for embeddings)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Interview Configuration - Full Mode
    MIN_QUESTIONS: int = 5
    MAX_QUESTIONS: int = 15
    MIN_TOPICS: int = 4
    MIN_DEPTH_SCORE: float = 0.6
    
    # Interview Configuration - Demo Mode
    DEMO_MIN_QUESTIONS: int = 3
    DEMO_MAX_QUESTIONS: int = 5
    DEMO_MIN_TOPICS: int = 2
    DEMO_MIN_DEPTH_SCORE: float = 0.5
    
    # Review Types
    REVIEW_TYPES: Dict[str, str] = {
        "self_review": "Self Review 📝",
        "peer_review": "Peer Review 👥",
        "manager_review": "Manager Review 👔"
    }
    
    # Competency Areas
    COMPETENCY_AREAS: List[str] = [
        "Technical Skills",
        "Communication",
        "Collaboration",
        "Leadership",
        "Problem Solving",
        "Innovation",
        "Time Management",
        "Adaptability"
    ]
    
    # Question Templates by Review Type
    INITIAL_QUESTIONS: Dict[str, str] = {
        "self_review": """Let's begin your self-assessment. I'll ask you a series of questions to understand your performance over this review period.

First, tell me about your most significant achievements this period. What accomplishments are you most proud of, and what impact did they have?""",
        
        "peer_review": """Thank you for providing feedback on your colleague. Your honest insights help create a comprehensive review.

To start, what are this person's greatest strengths in your collaboration? Please provide specific examples of when these strengths made a difference.""",
        
        "manager_review": """Let's discuss your team member's performance. Your perspective is crucial for their development.

Begin by describing their overall performance this period. What stands out to you about their contributions and growth?"""
    }
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_question_limits(cls, demo_mode: bool = False) -> Dict[str, Union[int, float]]:
        """Get question limits based on current mode"""
        if demo_mode:
            return {
                "min_questions": cls.DEMO_MIN_QUESTIONS,
                "max_questions": cls.DEMO_MAX_QUESTIONS,
                "min_topics": cls.DEMO_MIN_TOPICS,
                "min_depth_score": cls.DEMO_MIN_DEPTH_SCORE
            }
        return {
            "min_questions": cls.MIN_QUESTIONS,
            "max_questions": cls.MAX_QUESTIONS,
            "min_topics": cls.MIN_TOPICS,
            "min_depth_score": cls.MIN_DEPTH_SCORE
        }
    

settings = Settings()