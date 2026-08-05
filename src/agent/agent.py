"""
agent.py
--------
Builds and returns the LangChain agent with all three tools and memory.
Import `get_agent` and call it once per session.
"""

from __future__ import annotations

from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agent.tools import ALL_TOOLS
from src.utils.config import OPENAI_API_KEY, OPENAI_LLM_MODEL
from src.utils.logger import logger


SYSTEM_PROMPT = """
You are a Creative Intelligence Copilot for an influencer marketing team at a 
prescription skincare brand (think Formel Skin). You help creative strategists 
understand what's working in influencer content and catch compliance issues before 
campaigns go live.

You have access to three tools:
1. ingest_video — add a YouTube video to the knowledge base
2. query_corpus — search across all ingested videos to answer questions
3. check_compliance — check any text for EU healthcare advertising compliance

How to behave:
- Always ground your answers in retrieved content. If you're not sure, say so.
- When asked about hook rates, narrative structure, or content patterns, 
  pull specific examples from the transcripts.
- When asked to review a script or brief, always run check_compliance on it.
- Be concise. Strategists are busy — lead with the key finding, then support it.
- If a video hasn't been ingested yet, use ingest_video before querying.
""".strip()


def get_agent(verbose: bool = False) -> AgentExecutor:
    """
    Build and return a fresh AgentExecutor with memory.
    Call once per session (e.g. once per Streamlit user session).

    Args:
        verbose: If True, prints agent reasoning steps to stdout.

    Returns:
        A LangChain AgentExecutor ready to accept .invoke() calls.
    """
    llm = ChatOpenAI(
        model=OPENAI_LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
        streaming=False,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=6,
    )

    logger.info("Agent initialised with tools: " + ", ".join(t.name for t in ALL_TOOLS))
    return executor
