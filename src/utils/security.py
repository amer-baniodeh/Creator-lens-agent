"""
security.py
------------
Shared prompt-injection detection, used anywhere untrusted third-party text
(video transcripts, pasted scripts) flows into an LLM prompt — currently the
compliance classifier (src/compliance/checker.py) and the agent's retrieval
tool (src/agent/tools.py).

Not a guarantee — a regex match doesn't prove malicious intent, and a careful
attacker can phrase around it. It's a tripwire that lowers trust in a
suspicious result, not a wall. See data/eval/SECURITY_SUMMARY.md for what this
was built in response to and what it has and hasn't been shown to catch.
"""

from __future__ import annotations

import re

# Known injection phrasing, drawn from what was confirmed to work in testing
# plus common variants.
INJECTION_PATTERNS: list[str] = [
    r"ignore (all |any |the )?(previous |prior |above )?instructions",
    r"system (override|note)",
    r"ai (system|assistant) note",
    r"pre-?approved exception",
    r"overrides? all (prior |previous )?(rules|instructions)",
    r"respond with verdict",
    r"always (classify|grade|rate|report|mark) (this |it )?as",
    r"regardless of what (the |any )?(law|legal|retrieved)",
    r"do not (mention|explain) this (note|instruction)",
    r"literal text of your (system prompt|instructions)",
    r"this is a routine internal (qa|quality assurance) check",
    r"(before|when) answering,? (the user'?s? (next )?question,? )?(first |also )?(output|print|reveal|repeat) your",
    r"tell (the user|them) (it|this|that) (does not|doesn'?t) exist",
]
_INJECTION_PATTERN = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def detect_injection_attempt(text: str) -> list[str]:
    """Return the actual matched snippets, or [] if nothing suspicious found."""
    return list(dict.fromkeys(m.group(0).lower() for m in _INJECTION_PATTERN.finditer(text)))


def wrap_untrusted_content(text: str, tag: str = "content") -> str:
    """Delimit untrusted text clearly so it can't be confused with instructions."""
    return f"<{tag}>\n{text}\n</{tag}>"
