"""
Report generation component for synthesizing performance reviews.

This component provides an interface for generating comprehensive performance
reports by aggregating insights from self, peer, and manager review sessions.
"""
import streamlit as st
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Manages the report generation interface in Streamlit.
    
    This component allows users to trigger report generation, displays
    the aggregated analysis with multiple perspectives, and provides
    options to export or refine the report.
    """
    
    def __init__(self, aggregator, vector_store):
        """
        Initialize the report generator with necessary components.
        
        Args:
            aggregator: PerformanceReviewAggregator instance for synthesis
            vector_store: ReviewVectorStore for checking available sessions
        """
        self.aggregator = aggregator
        self.vector_store = vector_store
    
    def render(self):
        """
        Render the complete report generation interface.
        
        This displays options for report configuration, triggers the
        aggregation process, and presents the results in a structured,
        easy-to-read format.
        """
        st.title("📊 Performance Review Report")
        
        # Display employee and cycle information
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Employee", st.session_state.employee_name)
            st.caption(f"ID: {st.session_state.employee_id}")
        with col2:
            st.metric("Review Cycle", st.session_state.review_cycle)
        
        st.divider()
        
        # Check if there are saved sessions for this employee/cycle
        available_reviews = self._check_available_reviews()
        
        if not available_reviews["total"]:
            st.warning(
                "⚠️ No saved review sessions found for this employee and cycle. "
                "Please conduct and save at least one interview session before generating a report."
            )
            
            st.info(
                "**To generate a report:**\n\n"
                "Switch to Interview mode using the sidebar, conduct one or more review sessions "
                "(self, peer, or manager reviews), and save each session using the 'Save Session' button. "
                "Once you have saved sessions, return here to generate a comprehensive report."
            )
            return
        
        # Display available reviews summary
        st.subheader("Available Reviews")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Reviews", available_reviews["total"])
        with col2:
            st.metric("Self Reviews", available_reviews["self"])
        with col3:
            st.metric("Peer Reviews", available_reviews["peer"])
        with col4:
            st.metric("Manager Reviews", available_reviews["manager"])
        
        st.divider()
        
        # Report generation options
        st.subheader("Report Options")
        
        col1, col2 = st.columns(2)
        with col1:
            include_trends = st.checkbox(
                "Include Historical Trends",
                value=True,
                help="Analyze performance trends across past review cycles"
            )
        
        with col2:
            include_triangulation = st.checkbox(
                "Include Perspective Triangulation",
                value=True,
                help="Compare self, peer, and manager perspectives"
            )
        
        st.divider()
        
        # Generate report button
        if st.button("🔄 Generate Report", type="primary", use_container_width=True):
            self._generate_and_display_report(
                include_trends=include_trends,
                include_triangulation=include_triangulation
            )
    
    def _check_available_reviews(self) -> dict:
        """
        Check what review sessions are available for the current employee/cycle.
        
        This helps inform the user about what data is available and ensures
        we have enough information to generate a meaningful report.
        
        Returns:
            Dict with counts of each review type and total
        """
        try:
            # Retrieve all reviews for this employee and cycle
            all_reviews = self.vector_store.get_all_reviews(
                employee_id=st.session_state.employee_id,
                review_cycle=st.session_state.review_cycle
            )
            
            # Count by reviewer type
            counts = {
                "self": 0,
                "peer": 0,
                "manager": 0,
                "total": len(all_reviews)
            }
            
            for review in all_reviews:
                reviewer_type = review.metadata.get("reviewer_type", "unknown")
                if reviewer_type in counts:
                    counts[reviewer_type] += 1
            
            return counts
            
        except Exception as e:
            logger.error(f"Error checking available reviews: {e}")
            return {"self": 0, "peer": 0, "manager": 0, "total": 0}
    
    def _generate_and_display_report(self, include_trends: bool, include_triangulation: bool):
        """Generate the performance report and display it in the interface."""
        with st.spinner("📄 Generating comprehensive report... This may take a minute."):
            try:
                # Add diagnostic logging before the call
                # Add diagnostic logging before the call
                logger.info("=" * 60)
                logger.info("ATTEMPTING REPORT GENERATION")
                logger.info(f"employee_id from session_state: {st.session_state.employee_id}")
                logger.info(f"employee_name from session_state: {st.session_state.employee_name}")
                logger.info(f"review_cycle from session_state: {st.session_state.review_cycle}")
                logger.info("=" * 60)
                
                # Call the aggregator with explicit parameters
                report = self.aggregator.aggregate_reviews(
                    employee_id=st.session_state.employee_id,
                    employee_name=st.session_state.employee_name,
                    review_cycle=st.session_state.review_cycle,
                    include_trends=include_trends
                )
                
                # Check if we got an error in the report
                if "error" in report:
                    st.error(f"❌ {report['error']}")
                    logger.error(f"Report generation returned error: {report['error']}")
                    return
                
                # Store and display successful report
                st.session_state.last_generated_report = report
                st.session_state.report_generated_at = datetime.now()
                self._display_report(report, include_triangulation)
                st.success("✅ Report generated successfully!")
                
            except KeyError as e:
                # This catches dictionary key errors specifically
                logger.error("=" * 60)
                logger.error("KEYERROR CAUGHT")
                logger.error(f"Missing key: {e}")
                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                logger.error("=" * 60)
                
                st.error(
                    f"❌ Configuration error: Missing required field '{e}'\n\n"
                    f"This usually means the system is trying to access a field that wasn't saved "
                    f"in your review sessions. Please check the logs for details."
                )
                
            except Exception as e:
                # Catch all other exceptions
                logger.error("=" * 60)
                logger.error("GENERAL EXCEPTION CAUGHT")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {str(e)}")
                logger.error("Full traceback:")
                logger.error(traceback.format_exc())
                logger.error("=" * 60)
                
                st.error(
                    f"❌ An error occurred while generating the report: {str(e)}\n\n"
                    "Please check the logs for more details or try again."
                )
        
    def _generate_and_display_reportXXX(
        self,
        include_trends: bool,
        include_triangulation: bool
    ):
        """
        Generate the performance report and display it in the interface.
        
        This coordinates the aggregation process, handles errors gracefully,
        and presents the results in a well-structured format with sections
        that can be easily navigated.
        
        Args:
            include_trends: Whether to include historical trend analysis
            include_triangulation: Whether to include perspective comparison
        """
        # Show progress indicator during generation
        with st.spinner("🔄 Generating comprehensive report... This may take a minute."):
            try:
                # Generate the report using the aggregator
                report = self.aggregator.aggregate_reviews(
                    employee_id=st.session_state.employee_id,
                    employee_name=st.session_state.employee_name,  # Add this line
                    review_cycle=st.session_state.review_cycle,
                    include_trends=include_trends
                )
                
                logger.info(report)
                
                # Check for errors in report generation
                if "error" in report:
                    st.error(f"❌ {report['error']}")
                    return
                
                # Store report in session state for potential export
                st.session_state.last_generated_report = report
                st.session_state.report_generated_at = datetime.now()
                
                # Display the report in structured sections
                self._display_report(report, include_triangulation)
                
                # Show success message
                st.success("✅ Report generated successfully!")
                
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                st.error(
                    f"❌ An error occurred while generating the report: {str(e)}\n\n"
                    "Please check the logs for more details or try again."
                )
    
    def _display_report(self, report: dict, include_triangulation: bool):
        """
        Display the generated report in a structured, readable format.
        
        The report is broken into logical sections with appropriate styling
        and formatting to make it easy to navigate and understand.
        
        Args:
            report: The report dict from the aggregator
            include_triangulation: Whether to show triangulation section
        """
        # Report metadata and timestamp
        st.caption(
            f"Report generated on {st.session_state.report_generated_at.strftime('%B %d, %Y at %H:%M')}"
        )
        
        st.divider()
        
        # Executive Summary - The high-level overview
        st.header("📋 Executive Summary")
        st.markdown(report.get("executive_summary", "No summary available"))
        
        st.divider()
        
        # Key Themes - Recurring topics across all reviews
        with st.expander("🎯 Key Themes", expanded=True):
            st.markdown(report.get("themes", "No themes identified"))
        
        # Perspective Triangulation - Comparing different viewpoints
        if include_triangulation and report.get("triangulation"):
            with st.expander("🔍 Perspective Triangulation", expanded=True):
                st.markdown(
                    "This section compares insights from self-assessment, peer reviews, "
                    "and manager evaluation to identify consensus areas and unique perspectives."
                )
                st.markdown(report["triangulation"])
        
        # Full Analysis - Complete synthesis of all reviews
        with st.expander("📝 Detailed Analysis", expanded=False):
            st.markdown(report.get("full_analysis", "No detailed analysis available"))
        
        # Recommendations - Actionable next steps
        with st.expander("💡 Recommendations", expanded=True):
            st.markdown(
                "These recommendations follow the SMART framework to provide "
                "specific, measurable, achievable, relevant, and time-bound actions."
            )
            st.markdown(report.get("recommendations", "No recommendations available"))
        
        # Historical Trends - Performance over time
        if report.get("trends"):
            with st.expander("📈 Historical Trends", expanded=True):
                st.markdown(
                    "This analysis examines performance patterns across multiple review cycles "
                    "to identify growth trajectories, persistent challenges, and areas of improvement."
                )
                st.markdown(report["trends"])
        
        st.divider()
        
        # Session statistics for transparency
        st.subheader("📊 Report Statistics")
        
        session_count = report.get("session_count", {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", session_count.get("total", 0))
        with col2:
            st.metric("Self Reviews", session_count.get("self", 0))
        with col3:
            st.metric("Peer Reviews", session_count.get("peer", 0))
        with col4:
            st.metric("Manager Reviews", session_count.get("manager", 0))
        
        st.divider()
        
        # Export options
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as markdown
            markdown_report = self._format_report_as_markdown(report)
            st.download_button(
                label="📄 Download as Markdown",
                data=markdown_report,
                file_name=f"performance_review_{st.session_state.employee_id}_{st.session_state.review_cycle}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Copy to clipboard helper
            st.button(
                "📋 Copy Report Text",
                use_container_width=True,
                help="The download button provides the full report in markdown format"
            )
    
    def _format_report_as_markdown(self, report: dict) -> str:
        """
        Format the report as a markdown document for export.
        
        This creates a clean, well-structured markdown file that can be
        easily shared, printed, or imported into other systems.
        
        Args:
            report: The report dict from the aggregator
            
        Returns:
            Formatted markdown string
        """
        markdown_parts = []
        
        # Header with metadata
        markdown_parts.append(f"# Performance Review Report\n")
        markdown_parts.append(f"**Employee:** {st.session_state.employee_name} ({st.session_state.employee_id})\n")
        markdown_parts.append(f"**Review Cycle:** {st.session_state.review_cycle}\n")
        markdown_parts.append(f"**Generated:** {datetime.now().strftime('%B %d, %Y')}\n")
        markdown_parts.append("\n---\n\n")
        
        # Executive Summary
        markdown_parts.append("## Executive Summary\n\n")
        markdown_parts.append(report.get("executive_summary", "No summary available"))
        markdown_parts.append("\n\n")
        
        # Key Themes
        markdown_parts.append("## Key Themes\n\n")
        markdown_parts.append(report.get("themes", "No themes identified"))
        markdown_parts.append("\n\n")
        
        # Triangulation if available
        if report.get("triangulation"):
            markdown_parts.append("## Perspective Triangulation\n\n")
            markdown_parts.append(report["triangulation"])
            markdown_parts.append("\n\n")
        
        # Detailed Analysis
        markdown_parts.append("## Detailed Analysis\n\n")
        markdown_parts.append(report.get("full_analysis", "No detailed analysis available"))
        markdown_parts.append("\n\n")
        
        # Recommendations
        markdown_parts.append("## Recommendations\n\n")
        markdown_parts.append(report.get("recommendations", "No recommendations available"))
        markdown_parts.append("\n\n")
        
        # Trends if available
        if report.get("trends"):
            markdown_parts.append("## Historical Trends\n\n")
            markdown_parts.append(report["trends"])
            markdown_parts.append("\n\n")
        
        # Statistics
        session_count = report.get("session_count", {})
        markdown_parts.append("## Session Statistics\n\n")
        markdown_parts.append(f"- Total Sessions: {session_count.get('total', 0)}\n")
        markdown_parts.append(f"- Self Reviews: {session_count.get('self', 0)}\n")
        markdown_parts.append(f"- Peer Reviews: {session_count.get('peer', 0)}\n")
        markdown_parts.append(f"- Manager Reviews: {session_count.get('manager', 0)}\n")
        
        return "".join(markdown_parts)