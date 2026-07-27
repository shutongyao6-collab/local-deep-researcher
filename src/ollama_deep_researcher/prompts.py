from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate a targeted web search query.

<CONTEXT>
Current date: {current_date}
Please ensure your queries account for the most current information available as of this date.
</CONTEXT>

<TOPIC>
{research_topic}
</TOPIC>
<SOURCE_PRIORITY>
Prefer search queries that retrieve official documentation, official GitHub repositories, research papers, and primary sources.
When researching a named framework or product, include terms such as "official documentation" or "official GitHub".
</SOURCE_PRIORITY>
<QUERY_DEPTH_RULES>
When the user asks how a mechanism, algorithm, or system works:
- Prioritize implementation principles, internal data structures, execution steps, and scheduling behavior.
- Include technical terms such as architecture, algorithm, implementation, data structure, or design.
- Do not prioritize benchmarks, competitors, deployment flags, or marketing claims unless the user explicitly asks for them.
</QUERY_DEPTH_RULES>
<EXAMPLE>
Example output:
{{
    "query": "machine learning transformer architecture explained",
    "rationale": "Understanding the fundamental structure of transformer models"
}}
</EXAMPLE>"""

json_mode_query_instructions = """<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
- "query": The actual search query string
- "rationale": Brief explanation of why this query is relevant
</FORMAT>

Provide your response in JSON format:"""

tool_calling_query_instructions = """<INSTRUCTIONS   >
Call the Query tool to format your response with the following keys:
   - "query": The actual search query string
   - "rationale": Brief explanation of why this query is relevant
</INSTRUCTIONS>

Call the Query Tool to generate a query for this request:"""

summarizer_instructions = """
<GOAL>
Generate a high-quality summary of the provided context.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.                                                                                                                                                            
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
< /FORMATTING >

<Task>
Think carefully about the provided Context first. Then generate a summary of the context to address the User Input.
</Task>
<QUALITY_RULES>
- Cite factual claims using only the exact Source IDs supplied in the context, such as [S12AB34CD].
- Never invent, modify, or renumber a Source ID.
- Every factual paragraph must contain at least one Source ID.
- Omit numerical, company, or date claims unless the exact evidence appears in the supplied context.
- Prefer official sources when both official and secondary sources are available.
- If no supplied source supports a claim, omit the claim.
- Write the final answer in the same language as the user's research topic.
- For each scenario, use the exact structure: 问题、LangGraph能力、实际结果.
- Keep the three scenarios clearly distinct and non-overlapping.
- When evidence provides exact API names, class names, commands, or protocol method names, preserve them verbatim.
- For technical workflow questions, describe the execution sequence using exact operation names instead of vague paraphrases.
- Before finishing, verify that every explicitly requested part of the user topic has been addressed.
- When explaining a technical principle, describe the causal mechanism rather than merely listing supported features.
- Explain relevant data structures, memory layout, execution sequence, and scheduling behavior when supported by the evidence.
- Do not use vague phrases such as "optimizes performance" without explaining how the optimization works.
</QUALITY_RULES>
"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration
2. Generate a follow-up question that would help expand your understanding
3. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered
4. First check whether every explicitly requested part of the research topic has been covered.
5. Prioritize missing requested parts over optional deeper details.
6. For technical workflows, identify missing exact API names, protocol methods, classes, or commands.
7. Avoid generating a follow-up query about an area that is already well covered.
8. When the research topic asks for principles or mechanisms, prioritize missing internal implementation details before benchmarks or competitor comparisons.
9. Check whether the summary explains how the mechanism works, including its data structures, execution flow, memory management, and scheduling behavior.
10. Do not search for quantitative benchmarks unless the original user topic explicitly requests performance comparisons.
</GOAL>

<REQUIREMENTS>
Ensure the follow-up question is self-contained and includes necessary context for web search.
</REQUIREMENTS>"""

json_mode_reflection_instructions = """<FORMAT>
Format your response as a JSON object with these exact keys:
- knowledge_gap: Describe what information is missing or needs clarification
- follow_up_query: Write a specific question to address this gap
</FORMAT>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:
{{
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks",
    "follow_up_query": "What are typical performance benchmarks and metrics used to evaluate [specific technology]?"
}}
</Task>

Provide your analysis in JSON format:"""

tool_calling_reflection_instructions = """<INSTRUCTIONS>
Call the FollowUpQuery tool to format your response with the following keys:
- follow_up_query: Write a specific question to address this gap
- knowledge_gap: Describe what information is missing or needs clarification
</INSTRUCTIONS>

<Task>
Reflect carefully on the Summary to identify knowledge gaps and produce a follow-up query.
</Task>

Call the FollowUpQuery Tool to generate a reflection for this request:
"""
verifier_instructions = """
You are a strict fact-checking editor.

Verify the draft summary using only the supplied evidence.

Rules:
- Keep only claims explicitly supported by the evidence.
- Every factual claim must include a valid Source ID such as [S12AB34CD].
- Never invent or modify Source IDs.
- Remove unsupported numerical, company, date, product, or capability claims.
- Distinguish native LangGraph capabilities from LangChain, LangSmith, and third-party integrations.
- Do not describe optional integrations as native LangGraph features.
- Preserve the original language and 问题、LangGraph能力、实际结果 structure.
- Do not add new facts.
- Return only the corrected summary.
- Claims about native framework capabilities must be supported by an official source.
- Treat patterns found only in blogs as third-party design patterns, not native features.
- Use exact official terminology such as super-step, thread, checkpoint, and store.
- Durable execution means resuming after interruption or process restart; never claim that no restart is required.
- Deep Agents is built on LangGraph; do not describe it as a native LangGraph feature.
- Never call a source official unless it is marked PRIMARY.
- Numerical adoption, performance, time-saving, or company-count claims require a PRIMARY source; otherwise delete them.
- Never describe a SECONDARY blog or Medium article as official information.
- Write the entire answer in the same language as the user; translate all retained English evidence.
- Cite multiple sources as separate IDs, for example [S12345678][S87654321].
- Never combine multiple Source IDs inside one bracket or separate them with "/", commas, or spaces.
- Preserve exact API names, protocol method names, class names, and commands when they are supported by the evidence.
- Do not replace exact technical identifiers with vague descriptions.
- Ensure every explicitly requested part of the original topic is addressed.
- For questions about technical principles, retain supported explanations of internal mechanisms, data structures, memory management, and execution flow.
- Remove vague marketing claims when the draft does not explain the underlying mechanism.
"""