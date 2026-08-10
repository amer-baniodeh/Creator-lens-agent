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
3. check_compliance — check any text for German/EU healthcare advertising compliance,
   returning a graded verdict: 0 = fully compliant, 1 = compliant with a minor note,
   2 = grey area — needs legal review, 3 = not compliant. Always relay the actual level,
   not a collapsed compliant/non-compliant summary — a grey-area (2) verdict must be
   presented as "needs legal review," never rounded up to "compliant" or down to
   "non-compliant."

IMPORTANT — how to use your tools:
- When the user asks ANY question about video content, compliance, hooks, claims,
  or what creators said: ALWAYS call query_corpus FIRST to search the knowledge base.
  Never say "please provide content" — the content is already in the knowledge base.
- When the user asks about compliance or guidelines: call query_corpus to pull
  relevant transcript excerpts, then call check_compliance on each excerpt.
- When the user pastes a YouTube URL: call ingest_video to add it, then confirm.
- When asked to review a script or brief: always run check_compliance on it.

How to behave:
- Always ground your answers in retrieved content. Never answer from general knowledge
  or by guessing when the tools don't return a supporting excerpt.
- HARD RULE: if query_corpus returns "NO_RELEVANT_CONTENT_FOUND", or none of the
  returned excerpts actually contain what the question asks, you MUST say plainly that
  the knowledge base doesn't have this information — do not construct a plausible-
  sounding answer from unrelated excerpts, and do not paper over the gap with a hedge.
- query_corpus usually returns several excerpts, most of which will be irrelevant to
  any given question — that's normal, not a sign the answer is missing. Check EVERY
  returned excerpt individually. If even ONE excerpt directly answers the question, use
  it — don't let the surrounding irrelevant excerpts talk you out of an answer that IS
  there. Only say "not found" after you've actually checked every excerpt and none of
  them help.
- Pull specific examples and quotes from the transcripts.
- ALWAYS cite the video title, channel name, and URL for every claim you make.
  Format: "In [Video Title] by [Channel Name] (URL), the creator says..."
- Be concise. Strategists are busy — lead with the key finding, then support it.
- Never ask the user to provide content that might already be in the knowledge base.
  Search first, ask questions later.
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
