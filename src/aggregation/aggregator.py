"""
Performance review aggregator that synthesizes insights from multiple sessions.

This module consolidates self, peer, and manager reviews into comprehensive
reports using LangChain's document processing chains and custom prompts.
"""
import traceback
from typing import Dict, List
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from .prompts import (
    MAP_ANALYSIS_TEMPLATE,
    REDUCE_CONSOLIDATION_TEMPLATE,
    TRIANGULATION_TEMPLATE,
    TREND_ANALYSIS_TEMPLATE,
    SMART_RECOMMENDATIONS_TEMPLATE
)


class PerformanceReviewAggregator:
    """
    Aggregates and synthesizes performance reviews from multiple sources.
    
    This class implements several aggregation strategies:
    - Map-Reduce for analyzing many sessions at scale
    - Triangulation for comparing self/peer/manager perspectives
    - Trend analysis for tracking performance over time
    - SMART recommendation generation
    """
    
    def __init__(self, vector_store, llm):
        """
        Initialize the aggregator with storage and LLM.
        
        Args:
            vector_store: ReviewVectorStore instance for retrieving reviews
            llm: Language model for analysis and synthesis
        """
        self.vector_store = vector_store
        self.llm = llm
        
        # Build the map-reduce chain for processing multiple sessions
        self.map_reduce_chain = self._build_map_reduce_chain()
    
    def _build_map_reduce_chain(self) -> MapReduceDocumentsChain:
        """
        Construct a map-reduce chain for scalable review analysis.
        
        The map-reduce pattern is ideal when dealing with many review sessions.
        The map step analyzes each session independently to extract structured
        insights, then the reduce step consolidates all analyses into a single
        coherent report. This scales much better than trying to process all
        sessions in a single LLM call.
        
        Returns:
            Configured MapReduceDocumentsChain
        """
        # MAP STEP: Extract insights from each individual review session
        # This processes each document independently, which allows for
        # parallel processing and handles large numbers of reviews
        map_prompt = PromptTemplate.from_template(MAP_ANALYSIS_TEMPLATE)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
        
        # REDUCE STEP: Consolidate all individual analyses into final report
        # This takes the structured outputs from the map step and synthesizes
        # them into a coherent, actionable performance review
        reduce_prompt = PromptTemplate.from_template(REDUCE_CONSOLIDATION_TEMPLATE)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)
        
        # Combine documents chain handles the final synthesis
        # The "stuff" method simply concatenates all analyses for the reduce step
        combine_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="analyses"
        )
        
        # Reduce documents chain manages intermediate reduction if needed
        # If there are too many analyses to fit in context, this will
        # perform hierarchical reduction in multiple stages
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_chain,
            collapse_documents_chain=combine_chain,
            token_max=4000  # Adjust based on LLM context window
        )
        
        # Assemble the complete map-reduce chain
        return MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="content"
        )
    
    # In src/aggregation/aggregator.py

    def aggregate_reviews(
        self,
        employee_id: str,
        employee_name: str,  # Add this parameter
        review_cycle: str,
        include_trends: bool = True
    ) -> Dict:
        """
        Generate a complete performance review report from all sources.
        
        This method retrieves all review sessions for the specified employee
        and cycle, then uses various aggregation strategies to synthesize
        the information into a coherent, actionable report.
        
        Args:
            employee_id: Unique identifier for the employee
            employee_name: Full name of the employee for display purposes
            review_cycle: Review cycle identifier (e.g., "2025-Q3")
            include_trends: Whether to include historical trend analysis
            
        Returns:
            Dict containing the complete report with all sections
        """
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Diagnostic logging
        logger.info("aggregate_reviews called with:")
        logger.info(f"  employee_id: {employee_id}")
        logger.info(f"  employee_name: {employee_name}")
        logger.info(f"  review_cycle: {review_cycle}")
        logger.info(f"  include_trends: {include_trends}")
        
        # Retrieve all review sessions
        try:
            sessions = self.vector_store.get_all_reviews(
                employee_id=employee_id,
                review_cycle=review_cycle
            )
            
            logger.info(f"Retrieved {len(sessions)} sessions")
            for i, session in enumerate(sessions):
                logger.info(f"Session {i} metadata: {session.metadata}")
        except Exception as e:
            logger.error("Full traceback of error:")
            logger.error(traceback.format_exc())
            return {
                "error": f"Error generating report: {str(e)}",
                "employee_id": employee_id,
                "employee_name": employee_name,
                "review_cycle": review_cycle
            }
        
        if not sessions:
            return {
                "error": f"No reviews found for employee {employee_id} in cycle {review_cycle}",
                "employee_id": employee_id,
                "employee_name": employee_name,
                "review_cycle": review_cycle
            }
        
        # Organize sessions by review type
        by_type = {
            "self": [s for s in sessions if s.metadata.get("reviewer_type") == "self"],
            "peer": [s for s in sessions if s.metadata.get("reviewer_type") == "peer"],
            "manager": [s for s in sessions if s.metadata.get("reviewer_type") == "manager"]
        }
        
        # Extract themes across all sessions
        themes = self._extract_themes(sessions)
        
        # Perform triangulation analysis
        triangulation = self._triangulate_perspectives(by_type)
        
        # Add this right before: consolidated = self.map_reduce_chain.invoke(...)

        logger.info("=" * 60)
        logger.info("DEBUG: Template expectations vs. available data")
        logger.info(f"Template expects these variables: {self.map_reduce_chain.llm_chain.prompt.input_variables}")
        logger.info(f"First document has this metadata: {list(sessions[0].metadata.keys())}")

        # Now let's see what LangChain is actually passing to the template
        test_doc = sessions[0]
        test_inputs = {
            "content": test_doc.page_content,
            **test_doc.metadata  # This merges all metadata fields
        }
        logger.info(f"If we merged content + metadata, we'd have: {list(test_inputs.keys())}")
        logger.info("=" * 60)
        # Format documents to include metadata in accessible way
        inputs_list = []
        for session in sessions:
            inputs_list.append({
                "content": session.page_content,
                "reviewer_type": session.metadata.get("reviewer_type", "unknown"),
                "review_cycle": session.metadata.get("review_cycle", "unknown")
            })

        # Process with map chain directly
        map_results = [self.map_reduce_chain.llm_chain.invoke(inputs) for inputs in inputs_list]

        # Then reduce
        consolidated = self.map_reduce_chain.reduce_documents_chain.invoke({
            "input_documents": [Document(page_content=result["text"]) for result in map_results]
        })
        
        logger.info(f"Type of consolidated: {type(consolidated)}")
        logger.info(f"Consolidated keys: {consolidated.keys() if isinstance(consolidated, dict) else 'Not a dict'}")
        if isinstance(consolidated, dict):
            for key, value in consolidated.items():
                logger.info(f"  {key}: {type(value)} - {str(value)[:100]}...")
                
        # Extract the actual text content from the result dictionary
        consolidated_text = consolidated['output_text']

        logger.info(f"Successfully generated consolidated report, length: {len(consolidated_text)} characters")
                
        # Consolidate using map-reduce
        # Note: Changed from .run() to .invoke() to fix deprecation warning
        #consolidated = self.map_reduce_chain.invoke({"input_documents": sessions})
        
        # Generate SMART recommendations
        recommendations = self._generate_recommendations(consolidated_text)
        
        # Analyze historical trends if requested
        trends = None
        if include_trends:
            trends = self._analyze_trends(employee_id, num_cycles=4)
        
        # Assemble the complete report
        return {
            "employee_id": employee_id,
            "employee_name": employee_name,  # Now this is available
            "review_cycle": review_cycle,
            "executive_summary": self._extract_summary(consolidated_text),
            "triangulation": triangulation,
            "full_analysis": consolidated_text,
            "themes": themes,
            "recommendations": recommendations,
            "trends": trends,
            "session_count": {
                "self": len(by_type["self"]),
                "peer": len(by_type["peer"]),
                "manager": len(by_type["manager"]),
                "total": len(sessions)
            }
        }
    
    def _extract_themes(self, sessions: List[Document]) -> str:
        """
        Identify recurring themes across all review sessions.
        
        Themes help understand what topics are most frequently discussed
        and which areas receive the most attention across all reviews.
        
        Args:
            sessions: List of review documents
            
        Returns:
            Formatted string describing key themes with evidence
        """
        # Combine all session content for theme extraction
        # We limit the length to fit within context window
        combined_content = "\n\n---\n\n".join([
            f"[{s.metadata.get('reviewer_type', 'unknown')} review]\n{s.page_content}"
            for s in sessions[:20]  # Limit to avoid context overflow
        ])
        
        prompt = f"""Identify the five to seven most important recurring themes across these performance reviews.

                    REVIEWS:
                    {combined_content}

                    For each theme, provide:
                    - The theme name and description
                    - How frequently it appears across reviews (in how many reviews was it mentioned)
                    - The overall sentiment associated with this theme (positive strength, area for growth, or mixed)
                    - Key supporting evidence or quotes that illustrate this theme

                    Present the themes in order of importance, where importance is determined by frequency of mention, consistency across reviewer types, and impact on performance."""
        
        return self.llm.invoke(prompt).content
    
    def _triangulate_perspectives(self, by_type: Dict[str, List[Document]]) -> str:
        """
        Compare and contrast self, peer, and manager perspectives.
        
        Triangulation is crucial for understanding how self-perception aligns
        with how others perceive the employee's performance. Consensus across
        sources provides strong validation, while discrepancies highlight
        potential blind spots or communication issues.
        
        Args:
            by_type: Dict mapping review type to list of documents
            
        Returns:
            Formatted triangulation analysis
        """
        # Combine reviews within each type
        self_content = "\n\n".join([
            doc.page_content for doc in by_type.get("self", [])
        ]) or "No self-assessment provided"
        
        peer_content = "\n\n".join([
            doc.page_content for doc in by_type.get("peer", [])
        ]) or "No peer reviews provided"
        
        manager_content = "\n\n".join([
            doc.page_content for doc in by_type.get("manager", [])
        ]) or "No manager review provided"
        
        # Build the triangulation prompt with all perspectives
        prompt = TRIANGULATION_TEMPLATE.format(
            self_review=self_content[:2000],  # Limit length
            peer_count=len(by_type.get("peer", [])),
            peer_reviews=peer_content[:3000],
            manager_review=manager_content[:2000]
        )
        
        return self.llm.invoke(prompt).content
    
    def _analyze_trends(self, employee_id: str, num_cycles: int = 4) -> str:
        """
        Analyze performance trends across multiple review cycles.
        
        Historical trend analysis reveals patterns of growth, areas that
        have successfully improved, and persistent challenges that may need
        different development approaches.
        
        Args:
            employee_id: Employee identifier
            num_cycles: Number of historical cycles to analyze
            
        Returns:
            Formatted trend analysis
        """
        # Retrieve historical reviews organized by cycle
        historical_reviews = self.vector_store.get_historical_reviews(
            employee_id=employee_id,
            num_cycles=num_cycles
        )
        
        if not historical_reviews:
            return "Insufficient historical data for trend analysis."
        
        # Format historical data chronologically
        formatted_history = []
        for cycle in sorted(historical_reviews.keys()):
            cycle_reviews = historical_reviews[cycle]
            
            # Summarize this cycle's reviews
            cycle_summary = f"\n=== {cycle} ===\n"
            cycle_summary += f"Total Reviews: {len(cycle_reviews)}\n\n"
            
            # Include key excerpts from each review type
            for doc in cycle_reviews[:5]:  # Limit to avoid context overflow
                reviewer_type = doc.metadata.get("reviewer_type", "unknown")
                cycle_summary += f"[{reviewer_type}] {doc.page_content[:500]}...\n\n"
            
            formatted_history.append(cycle_summary)
        
        historical_data = "\n".join(formatted_history)
        
        # Generate trend analysis
        prompt = TREND_ANALYSIS_TEMPLATE.format(
            historical_data=historical_data[:6000],  # Limit for context
            num_cycles=num_cycles
        )
        
        return self.llm.invoke(prompt).content
    
    def _generate_recommendations(self, consolidated_report: str) -> str:
        """
        Generate SMART recommendations from the consolidated analysis.
        
        SMART recommendations provide specific, actionable next steps that
        the employee can take to address development areas and leverage
        strengths more effectively.
        
        Args:
            consolidated_report: The full consolidated performance analysis
            
        Returns:
            Formatted SMART recommendations
        """
        prompt = SMART_RECOMMENDATIONS_TEMPLATE.format(
            summary=consolidated_report[:4000]  # Use relevant portion
        )
        
        return self.llm.invoke(prompt).content
    
    def _extract_summary(self, consolidated_report: str) -> str:
        """
        Extract just the executive summary section from the full report.
        
        This provides a quick overview without requiring the reader to
        parse the entire detailed analysis.
        
        Args:
            consolidated_report: Full consolidated report
            
        Returns:
            Executive summary portion
        """
        # Try to extract the executive summary section
        # Look for common markers that indicate the summary
        lines = consolidated_report.split("\n")
        summary_lines = []
        in_summary = False
        
        for line in lines:
            # Start capturing when we find the summary header
            if "executive summary" in line.lower():
                in_summary = True
                continue
            
            # Stop when we hit the next major section
            if in_summary and (
                line.strip().startswith("#") or 
                "validated strengths" in line.lower() or
                "development areas" in line.lower()
            ):
                break
            
            # Capture summary lines
            if in_summary and line.strip():
                summary_lines.append(line)
        
        # If we found a summary section, return it
        if summary_lines:
            return "\n".join(summary_lines).strip()
        
        # Otherwise, return the first few sentences as a fallback
        sentences = consolidated_report.split(". ")[:3]
        return ". ".join(sentences) + "."