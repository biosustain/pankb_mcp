"""Prompt registry — every version stays in git forever.

To add a new version:
  1. Copy prompts/vN.py to prompts/v{N+1}.py and edit.
  2. Add it to PROMPTS below.
  3. Set PROMPT_VERSION in config.py to switch.

The active version comes only from config.py.
"""
from config import PROMPT_VERSION

from . import v1, v2

PROMPTS = {
    "v1": v1.SYSTEM_PROMPT,
    "v2": v2.SYSTEM_PROMPT,
}


def get_prompt() -> tuple[str, str]:
    """Return (version, prompt_text) for config.PROMPT_VERSION.

    The version is also what gets stamped on agent.chat spans, so Phoenix
    always shows which prompt ran.
    """
    if PROMPT_VERSION not in PROMPTS:
        raise ValueError(
            f"config.PROMPT_VERSION={PROMPT_VERSION!r} is not registered "
            f"(available: {sorted(PROMPTS)})"
        )
    return PROMPT_VERSION, PROMPTS[PROMPT_VERSION]
