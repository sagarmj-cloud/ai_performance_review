"""
Unit tests for the vector store.

These tests verify that review sessions are correctly stored, retrieved,
and filtered using the Qdrant vector database.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.storage.vector_store import ReviewVectorStore


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = Mock()
    
    # Mock get_collections
    collections_response = Mock()
    collections_response.collections = []
    client.get_collections.return_value = collections_response
    
    # Mock create_collection
    client.create_collection.return_value = None
    
    # Mock create_payload_index
    client.create_payload_index.return_value = None
    
    return client


@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    embeddings = Mock()
    embeddings.embed_query.return_value = [0.1] * 3072
    embeddings.embed_documents.return_value = [[0.1] * 3072]
    return embeddings


@pytest.fixture
def mock_vector_store(mock_qdrant_client, mock_embeddings):
    """Create a vector store with mocked dependencies."""
    with patch('src.storage.vector_store.QdrantClient', return_value=mock_qdrant_client):
        with patch('src.storage.vector_store.OpenAIEmbeddings', return_value=mock_embeddings):
            with patch('src.storage.vector_store.FastEmbedSparse'):
                with patch('src.storage.vector_store.QdrantVectorStore'):
                    store = ReviewVectorStore(
                        collection_name="test_reviews",
                        qdrant_url="http://localhost:6333"
                    )
                    return store


def test_parse_review_cycle_quarterly():
    """Test parsing quarterly review cycles."""
    year, quarter = ReviewVectorStore._parse_review_cycle("2025-Q3")
    assert year == 2025
    assert quarter == 3


def test_parse_review_cycle_annual():
    """Test parsing annual review cycles."""
    year, quarter = ReviewVectorStore._parse_review_cycle("annual-2025")
    assert year == 2025
    assert quarter == 0


def test_parse_review_cycle_year_only():
    """Test parsing year-only review cycles."""
    year, quarter = ReviewVectorStore._parse_review_cycle("2025")
    assert year == 2025
    assert quarter == 0


def test_parse_review_cycle_invalid():
    """Test parsing invalid review cycles defaults to current year."""
    year, quarter = ReviewVectorStore._parse_review_cycle("invalid")
    assert year == datetime.now().year
    assert quarter == 0


def test_add_review_session(mock_vector_store):
    """Test adding a review session to the vector store."""
    # Mock the add_documents method
    mock_vector_store.vector_store.add_documents = Mock(return_value=["doc_id_123"])
    
    doc_id = mock_vector_store.add_review_session(
        content="This is a test review session.",
        employee_id="emp_001",
        employee_name="John Doe",
        reviewer_type="self",
        review_cycle="2025-Q3",
        department="Engineering"
    )
    
    assert doc_id == "doc_id_123"
    assert mock_vector_store.vector_store.add_documents.called


def test_add_qa_pairs(mock_vector_store):
    """Test adding Q&A pairs to the vector store."""
    mock_vector_store.vector_store.add_documents = Mock(
        return_value=["id1", "id2", "id3"]
    )
    
    qa_pairs = [
        {
            "question": "What are your strengths?",
            "answer": "I excel at communication.",
            "depth_score": 0.8
        },
        {
            "question": "What areas need improvement?",
            "answer": "I need to work on time management.",
            "depth_score": 0.7
        }
    ]
    
    doc_ids = mock_vector_store.add_qa_pairs(
        qa_pairs=qa_pairs,
        employee_id="emp_001",
        reviewer_type="self",
        review_cycle="2025-Q3"
    )
    
    assert len(doc_ids) == 3
    assert mock_vector_store.vector_store.add_documents.called


def test_search_reviews_with_filters(mock_vector_store):
    """Test searching reviews with metadata filters."""
    # Mock the similarity_search method
    mock_doc = Mock()
    mock_doc.page_content = "Test review content"
    mock_doc.metadata = {
        "employee_id": "emp_001",
        "reviewer_type": "self",
        "review_cycle": "2025-Q3"
    }
    
    mock_vector_store.vector_store.similarity_search = Mock(
        return_value=[mock_doc]
    )
    
    results = mock_vector_store.search_reviews(
        query="communication skills",
        employee_id="emp_001",
        reviewer_type="self",
        review_cycle="2025-Q3",
        k=10
    )
    
    assert len(results) == 1
    assert results[0].metadata["employee_id"] == "emp_001"
    assert mock_vector_store.vector_store.similarity_search.called


def test_get_all_reviews(mock_vector_store):
    """Test retrieving all reviews for an employee/cycle."""
    mock_docs = [Mock() for _ in range(5)]
    mock_vector_store.vector_store.similarity_search = Mock(
        return_value=mock_docs
    )
    
    results = mock_vector_store.get_all_reviews(
        employee_id="emp_001",
        review_cycle="2025-Q3"
    )
    
    assert len(results) == 5
    assert mock_vector_store.vector_store.similarity_search.called


def test_get_historical_reviews(mock_vector_store):
    """Test retrieving historical reviews across cycles."""
    mock_docs = [
        Mock(metadata={"review_cycle": "2025-Q3"}),
        Mock(metadata={"review_cycle": "2025-Q2"}),
        Mock(metadata={"review_cycle": "2025-Q1"}),
        Mock(metadata={"review_cycle": "2024-Q4"}),
    ]
    
    mock_vector_store.vector_store.similarity_search = Mock(
        return_value=mock_docs
    )
    
    results = mock_vector_store.get_historical_reviews(
        employee_id="emp_001",
        num_cycles=4
    )
    
    assert len(results) == 4
    assert "2025-Q3" in results
    assert "2024-Q4" in results