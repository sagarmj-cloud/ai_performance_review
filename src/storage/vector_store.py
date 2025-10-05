"""
Qdrant vector store for managing performance review sessions.

This module handles storage and retrieval of review sessions with rich metadata
for filtering, aggregation, and historical tracking across employees and cycles.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pydantic import SecretStr

from config.settings import Settings as settings


class ReviewVectorStore:
    """
    Manages vector storage for performance review sessions using Qdrant.
    
    Features:
    - Hybrid search combining dense (semantic) and sparse (keyword) vectors
    - Rich metadata indexing for efficient filtering
    - Hierarchical document organization (summaries, sections, Q&A pairs)
    - Historical tracking across multiple review cycles
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize the vector store with Qdrant client and embeddings.
        
        Args:
            collection_name: Name of the Qdrant collection (defaults to settings)
            qdrant_url: Qdrant server URL (defaults to settings)
            qdrant_api_key: API key for Qdrant Cloud (optional)
        """
        # Use settings as defaults
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        qdrant_url = qdrant_url or settings.QDRANT_URL
        qdrant_api_key = qdrant_api_key or settings.QDRANT_API_KEY
        # Initialize embeddings for semantic search
        # Dense vectors capture semantic meaning
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=SecretStr(settings.OPENAI_API_KEY)
        )
        
        # Sparse vectors for keyword matching (BM25-style)
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key if qdrant_api_key else None,
            timeout=30
        )
        
        # Create collection if it doesn't exist
        self._setup_collection()
        
        # Initialize the LangChain vector store wrapper
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,  # Use both dense and sparse
            vector_name="dense",
            sparse_vector_name="sparse"
        )
        
        # Create payload indexes for fast filtering
        self._setup_indexes()
    
    def _setup_collection(self):
        """
        Create the Qdrant collection with hybrid vector configuration.
        
        This sets up both dense vectors (for semantic similarity) and
        sparse vectors (for exact keyword matching).
        """
        # Check if collection already exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # Create collection with both vector types
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=3072,  # text-embedding-3-large dimension
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            print(f"Created collection: {self.collection_name}")
    
    def _setup_indexes(self):
        """
        Create payload indexes for frequently filtered fields.
        
        Indexes dramatically speed up filtered queries by allowing Qdrant
        to quickly narrow down candidates before computing vector similarity.
        """
        # Define fields that will be frequently filtered
        indexes = [
            ("metadata.employee_id", "keyword"),
            ("metadata.reviewer_type", "keyword"),
            ("metadata.review_cycle", "keyword"),
            ("metadata.section", "keyword"),
            ("metadata.department", "keyword"),
            ("metadata.year", {"type": "integer", "range": True}),
            ("metadata.rating", {"type": "integer", "range": True}),
            ("metadata.is_summary", "bool")
        ]
        
        # Create each index (skip if already exists)
        for field_name, field_schema in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
            except Exception:
                pass  # Index likely already exists
    
    def add_review_session(
        self,
        content: str,
        employee_id: str,
        reviewer_type: str,
        review_cycle: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Store a complete review session with rich metadata.
        
        Args:
            content: The review content (transcript, summary, or section)
            employee_id: Unique identifier for the employee being reviewed
            reviewer_type: Type of review (self, peer, manager)
            review_cycle: Review cycle identifier (e.g., "2025-Q3", "annual-2025")
            metadata: Dictionary containing optional fields like:
                - employee_name: Full name of employee
                - reviewer_id: Unique identifier for the reviewer
                - reviewer_name: Full name of reviewer
                - department: Department of employee
                - job_title: Job title of employee
                - section: Content section (full_interview, technical_skills, etc.)
                - competency: Specific competency area being addressed
                - rating: Numeric rating if applicable (1-5)
                - is_summary: Whether this is a summary vs. detailed content
            **kwargs: Additional metadata fields
            
        Returns:
            Document ID of the stored review
        """
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Parse review cycle for year/quarter if formatted as "YYYY-QN"
        year, quarter = self._parse_review_cycle(review_cycle)
        
        # Build comprehensive metadata
        full_metadata = {
            # Employee information
            "employee_id": employee_id,
            "employee_name": metadata.get("employee_name", ""),
            "department": metadata.get("department", ""),
            "job_title": metadata.get("job_title", ""),
            
            # Review context
            "reviewer_type": reviewer_type,
            "reviewer_id": metadata.get("reviewer_id", ""),
            "reviewer_name": metadata.get("reviewer_name", ""),
            
            # Temporal metadata
            "date": datetime.utcnow().isoformat(),
            "review_cycle": review_cycle,
            "year": year,
            "quarter": quarter,
            
            # Content categorization
            "section": metadata.get("section", "full_interview"),
            "competency": metadata.get("competency", ""),
            "rating": metadata.get("rating"),
            "is_summary": metadata.get("is_summary", False),
            
            # Additional fields
            **kwargs
        }
        
        # Create document with content and metadata
        doc = Document(page_content=content, metadata=full_metadata)
        doc_ids = self.vector_store.add_documents([doc])
        return doc_ids[0] if doc_ids else None
    
    def add_qa_pairs(
        self,
        qa_pairs: List[Dict],
        employee_id: str,
        reviewer_type: str,
        review_cycle: str,
        **kwargs
    ) -> List[str]:
        """
        Store individual Q&A pairs as separate searchable documents.
        
        This enables granular retrieval of specific exchanges while maintaining
        connections to the parent session through metadata.
        
        Args:
            qa_pairs: List of dicts with 'question', 'answer', and optional metadata
            employee_id: Employee identifier
            reviewer_type: Type of review
            review_cycle: Review cycle identifier
            **kwargs: Additional metadata for all Q&A pairs
            
        Returns:
            List of document IDs for stored Q&A pairs
        """
        documents = []
        
        for i, qa in enumerate(qa_pairs):
            # Format as question-answer pair
            content = f"Q: {qa['question']}\n\nA: {qa['answer']}"
            
            # Merge Q&A-specific metadata with session metadata
            qa_metadata = {
                "section": qa.get("section", "qa_pair"),
                "competency": qa.get("competency", ""),
                "topics": qa.get("topics", []),
                "depth_score": qa.get("depth_score", 0),
                "chunk_index": i,
                "total_chunks": len(qa_pairs),
                **kwargs
            }
            
            doc = Document(page_content=content, metadata=qa_metadata)
            documents.append(doc)
        
        # Batch add all Q&A pairs
        doc_ids = self.vector_store.add_documents(documents)
        return doc_ids
    
    def search_reviews(
        self,
        query: str,
        employee_id: Optional[str] = None,
        reviewer_type: Optional[str] = None,
        review_cycle: Optional[str] = None,
        department: Optional[str] = None,
        is_summary: Optional[bool] = None,
        k: int = 10
    ) -> List[Document]:
        """
        Search reviews with metadata filtering.
        
        This combines semantic search (via embeddings) with precise metadata
        filtering to find the most relevant review content.
        
        Args:
            query: Semantic search query
            employee_id: Filter by employee
            reviewer_type: Filter by review type (self, peer, manager)
            review_cycle: Filter by review cycle
            department: Filter by department
            is_summary: Filter by summary vs. detailed content
            k: Number of results to return
            
        Returns:
            List of matching documents with content and metadata
        """
        # Build filter conditions
        filter_conditions = []
        
        if employee_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.employee_id",
                    match=models.MatchValue(value=employee_id)
                )
            )
        
        if reviewer_type:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.reviewer_type",
                    match=models.MatchValue(value=reviewer_type)
                )
            )
        
        if review_cycle:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.review_cycle",
                    match=models.MatchValue(value=review_cycle)
                )
            )
        
        if department:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.department",
                    match=models.MatchValue(value=department)
                )
            )
        
        if is_summary is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.is_summary",
                    match=models.MatchValue(value=is_summary)
                )
            )
        
        # Create filter object if conditions exist
        filter_obj = models.Filter(must=filter_conditions) if filter_conditions else None
        
        # Execute hybrid search
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_obj
        )
        
        return results
    
    def get_all_reviews(
        self,
        employee_id: str,
        review_cycle: str
    ) -> List[Document]:
        """
        Retrieve all reviews for an employee in a specific cycle.
        
        This is useful for report generation where you need the complete
        set of reviews regardless of semantic relevance.
        
        Args:
            employee_id: Employee identifier
            review_cycle: Review cycle identifier
            
        Returns:
            All matching review documents
        """
        # Use an empty query to get all matching documents
        return self.search_reviews(
            query="",
            employee_id=employee_id,
            review_cycle=review_cycle,
            k=100  # High limit to get all
        )
    
    def get_historical_reviews(
        self,
        employee_id: str,
        num_cycles: int = 4
    ) -> Dict[str, List[Document]]:
        """
        Retrieve historical reviews across multiple cycles.
        
        Organizes reviews by cycle for trend analysis and progress tracking.
        
        Args:
            employee_id: Employee identifier
            num_cycles: Number of recent cycles to retrieve
            
        Returns:
            Dict mapping review_cycle to list of documents
        """
        # Get all reviews for employee
        all_reviews = self.search_reviews(
            query="",
            employee_id=employee_id,
            k=500  # Large limit for historical data
        )
        
        # Group by review cycle
        by_cycle = {}
        for doc in all_reviews:
            cycle = doc.metadata.get("review_cycle", "unknown")
            if cycle not in by_cycle:
                by_cycle[cycle] = []
            by_cycle[cycle].append(doc)
        
        # Sort cycles and return most recent
        sorted_cycles = sorted(by_cycle.keys(), reverse=True)[:num_cycles]
        return {cycle: by_cycle[cycle] for cycle in sorted_cycles}
    
    @staticmethod
    def _parse_review_cycle(review_cycle: str) -> tuple:
        """
        Parse review cycle string to extract year and quarter.
        
        Supports formats like "2025-Q3", "annual-2025", or "2025".
        
        Returns:
            Tuple of (year, quarter) where quarter is 0 for annual reviews
        """
        try:
            if "-Q" in review_cycle:
                # Format: "2025-Q3"
                year_str, quarter_str = review_cycle.split("-Q")
                return int(year_str), int(quarter_str)
            elif "annual" in review_cycle.lower():
                # Format: "annual-2025"
                year_str = review_cycle.split("-")[-1]
                return int(year_str), 0
            else:
                # Assume just year
                return int(review_cycle), 0
        except (ValueError, IndexError):
            # Default to current year, no quarter
            return datetime.now().year, 0