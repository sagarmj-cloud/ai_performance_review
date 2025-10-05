"""
Prompt templates for aggregating and synthesizing performance reviews.

These prompts guide the LLM through the complex process of consolidating
insights from multiple review sessions into coherent, actionable reports.
"""

# Template for analyzing individual review sessions
MAP_ANALYSIS_TEMPLATE = """You are analyzing a performance review session. Extract structured insights that can be synthesized with other reviews.

REVIEW SESSION:
{content}

METADATA:
- Reviewer Type: {reviewer_type}
- Review Cycle: {review_cycle}

Your task is to extract:

STRENGTHS:
Identify specific strengths mentioned with concrete evidence. For each strength, note:
- The strength itself
- Specific examples or situations demonstrating it
- The impact or outcomes of this strength

DEVELOPMENT AREAS:
Identify areas for improvement with context. For each area, note:
- The specific skill or behavior to develop
- Examples of situations where this was evident
- Suggested approaches for improvement

KEY ACHIEVEMENTS:
List notable accomplishments during this review period with:
- What was achieved
- The impact on the team or organization
- Skills or qualities demonstrated

ACTIONABLE ITEMS:
Extract any specific action items, goals, or commitments mentioned.

Format your response with clear sections and bullet points for easy aggregation."""


# Template for consolidating multiple analyses into a final report
REDUCE_CONSOLIDATION_TEMPLATE = """You are synthesizing multiple performance review analyses into a comprehensive final report.

INDIVIDUAL ANALYSES:
{analyses}

Your task is to create a consolidated report that integrates insights from all sources. Structure your report as follows:

EXECUTIVE SUMMARY
Write a concise overview in two to three sentences that captures the employee's overall performance, highlighting the most significant themes that emerged across all reviews.

VALIDATED STRENGTHS
Identify strengths that were confirmed by multiple reviewers, which provides strong validation. For each strength, explain which sources mentioned it, provide specific evidence from the reviews, and describe the impact this strength has had on the team or organization.

PRIORITY DEVELOPMENT AREAS
Highlight areas for growth that were identified across multiple reviews, indicating where focused development will have the greatest impact. For each area, note which reviewers mentioned it, explain why it matters for the employee's role and growth, and provide specific, actionable recommendations for improvement.

KEY ACCOMPLISHMENTS
Summarize the most significant achievements from this review period, focusing on those that demonstrate growth, impact, or excellence. Include the outcomes of these accomplishments and the competencies they demonstrate.

ACTIONABLE NEXT STEPS
Provide a clear development plan with three to five specific actions the employee should take, along with measurable outcomes for each action and a realistic timeline for completion.

Use clear, professional language suitable for an official performance review document."""


# Template for triangulating insights across review types
TRIANGULATION_TEMPLATE = """You are analyzing performance reviews from multiple perspectives to identify consensus and unique insights.

SELF-ASSESSMENT:
{self_review}

PEER REVIEWS (Total: {peer_count}):
{peer_reviews}

MANAGER ASSESSMENT:
{manager_review}

Perform a comprehensive triangulation analysis:

CONSENSUS STRENGTHS
Identify strengths that are validated across at least two review types (self, peer, or manager). This validation provides strong confidence in these areas. For each consensus strength, specify which review types mentioned it, provide supporting evidence from multiple sources, and assess the strength's impact on performance and collaboration.

CONSENSUS DEVELOPMENT AREAS
Identify development needs that multiple reviewers agree upon, which suggests these are priority areas for growth. For each area, note which reviewers identified it, explain the consistency in feedback across sources, and provide recommendations that address the common concerns.

SELF-AWARENESS ANALYSIS
Compare the self-assessment against peer and manager feedback to evaluate the employee's self-awareness. Identify areas where self-perception aligns with external feedback, which demonstrates good self-awareness. Note any gaps where the employee's self-assessment differs significantly from others' observations, which may indicate blind spots or areas where communication could be strengthened.

UNIQUE PERSPECTIVES
Capture insights that came from only one review type, as these may reveal important nuances. Describe what the self-review revealed that others didn't mention, what peers observed in daily collaboration that others missed, and what strategic or leadership perspective the manager contributed.

DISCREPANCIES TO EXPLORE
Highlight any contradictions between review sources that merit follow-up discussion. For each discrepancy, describe what different reviewers said, suggest possible reasons for the difference in perspective, and recommend how to address or clarify the discrepancy in conversation.

This analysis should help identify where there is strong agreement (action can be taken confidently) and where there is ambiguity (further discussion is needed)."""


# Template for historical trend analysis
TREND_ANALYSIS_TEMPLATE = """You are analyzing performance trends across multiple review cycles to identify patterns of growth and areas of concern.

HISTORICAL REVIEW DATA:
{historical_data}

ANALYSIS PERIOD: {num_cycles} review cycles

Provide a comprehensive trend analysis:

GROWTH TRAJECTORY
For each major competency area, analyze how the employee's performance has evolved over time. Identify skills or behaviors that show consistent improvement across multiple cycles, which indicates effective learning and development. Note any areas that have shown rapid improvement, suggesting particular focus or natural aptitude. Highlight competencies that have remained consistently strong, demonstrating reliable strengths.

PLATEAU AREAS
Identify skills or behaviors that have not shown improvement over multiple cycles, which may indicate a need for new development approaches or renewed focus. For these areas, consider whether they have been actively addressed in previous review cycles, assess whether they are preventing advancement to the next level, and suggest fresh approaches or resources that might help break through the plateau.

SUCCESSFULLY ADDRESSED CONCERNS
Celebrate areas where previous development needs have been successfully improved, demonstrating the employee's ability to learn and grow. For each improved area, describe what the concern was in earlier cycles, show evidence of improvement in recent reviews, and note what strategies or actions contributed to the success.

EMERGING PATTERNS
Identify any new themes or patterns that have appeared in recent cycles but weren't present before. This could include new responsibilities showing early struggles or successes, changing role requirements that the employee is adapting to (or struggling with), or shifts in feedback from managers or peers that indicate changes in working style or team dynamics.

READINESS ASSESSMENT
Based on the overall trajectory, assess the employee's readiness for advancement, new responsibilities, or role changes. Provide a realistic timeline for when they might be ready for the next level, identify any gaps that must be closed before advancement, and suggest strategic development priorities that would accelerate their growth.

RECOMMENDATIONS FOR NEXT CYCLE
Based on this historical analysis, suggest what should be the top priorities for the upcoming review cycle to maintain growth momentum and address any persistent concerns."""


# Template for generating SMART recommendations
SMART_RECOMMENDATIONS_TEMPLATE = """Based on this performance summary, generate actionable development recommendations using the SMART framework.

PERFORMANCE SUMMARY:
{summary}

Generate three to five SMART recommendations that will have the highest impact on the employee's growth and performance.

For each recommendation, structure it as follows:

SPECIFIC
Describe exactly what action should be taken, with enough detail that there is no ambiguity about what success looks like. Avoid vague statements and instead provide concrete activities, deliverables, or behaviors.

MEASURABLE
Define how progress and completion will be tracked and measured. This might include quantitative metrics (numbers, percentages, frequency), qualitative indicators (feedback from specific people, quality assessments), or observable milestones (completed projects, demonstrated skills).

ACHIEVABLE
Explain why this recommendation is realistic given the employee's current level, workload, and available resources. Consider what support or resources will be needed, what potential obstacles exist and how to address them, and why this is challenging but not overwhelming.

RELEVANT
Connect this recommendation directly to the employee's role, career goals, and the organization's needs. Explain how achieving this will impact their performance, what skills or experiences it will develop, and how it aligns with their career trajectory.

TIME-BOUND
Provide a specific target date or timeframe for completion. Break down longer-term recommendations into interim checkpoints or milestones, and specify when progress should be reviewed.

Ensure recommendations address the most critical development areas identified in the reviews while building on existing strengths."""