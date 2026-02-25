"""
rewrite_prompts.py

Prompting strategies for Greek Adaptive Rewriter.
Includes standard modes (surgical, simplify, retell) and new beginner modes (noob, ultra_noob).
"""

from typing import Sequence, Tuple
# Base system prompt used across all modes
_SYSTEM_COMMON = (
"You rewrite Modern Greek text.\n"
"Preserve meaning and named entities.\n"
"Strictly obey vocabulary constraints.\n"
"Output ONLY the rewritten Greek passage.\n"
"Before outputting, scan your output and ensure none of the forbidden lemmas/surface forms appear.\n"
)

def prompt_surgical(
    passage: str,
    target: float,
    banned_surface: Sequence[str],
) -> Tuple[str, str]:
    """Minimal edits mode - for high initial coverage"""
    system = _SYSTEM_COMMON + "First simplify meaning. Then adjust vocabulary to obey constraints.\n"
    user = (
        "Input passage:\n"
        f"{passage}\n"
        f"Target: ≥ {target:.3f} lemma coverage.\n"
        "Forbidden surface forms:\n"
        "\n".join(f"- {s}" for s in banned_surface) + "\n"
        "Keep length similar to input.\n"
        "Prefer minimal edits.\n"
    )
    return system, user


def prompt_simplify_then_enforce(
    passage: str,
    target: float,
    banned_surface: Sequence[str],
) -> Tuple[str, str]:
    """Two-stage simplification - for medium coverage (70-88%)"""
    system = _SYSTEM_COMMON + (
        "You may paraphrase more broadly, but must preserve all facts.\n"
        "Prefer shorter clauses and common vocabulary.\n"
    )
    user = (
        "Input passage:\n"
        f"{passage}\n"
        f"Target: ≥ {target:.3f} lemma coverage.\n"
        "Forbidden surface forms:\n"
        "\n".join(f"- {s}" for s in banned_surface) + "\n"
        "Keep length similar to input.\n"
        "Prefer simpler clauses and common words; rephrase freely if needed."

    )
    return system, user


def prompt_retell(
    passage: str,
    target: float,
    banned_surface: Sequence[str],
) -> Tuple[str, str]:
    """Retelling mode - for lower coverage (50-70%)"""
    system = _SYSTEM_COMMON + (
        "You may compress and reorder for clarity, but must keep key facts and causal relations.\n"
        "Write as much as needed to cover the main points using allowed vocabulary.\n"
    )
    user = (
        "Input passage:\n"
        f"{passage}\n"
        f"Target: ≥ {target:.3f} lemma coverage.\n"
        "Forbidden surface forms:\n"
        "\n".join(f"- {s}" for s in banned_surface) + "\n"
        "Keep length similar to input.\n"
        "You may reorder/retell; keep all key facts."
    )
    return system, user


def prompt_noob(
    passage: str,
    target: float,
    banned_surface: Sequence[str],
) -> Tuple[str, str]:
    """
    Noob mode - for beginners with low coverage (30-50%).
    Allows aggressive simplification and omission of details.
    """
    system = (
        "You rewrite Modern Greek text for absolute beginners.\n"
        "Rules:\n"
        "1) You MAY omit complex details, secondary information, and specific numbers.\n"
        "2) You MAY summarize and simplify aggressively.\n"
        "3) Focus on the MAIN idea and key events only.\n"
        "4) Use the simplest possible vocabulary.\n"
        "5) Obey banned lemma constraints strictly.\n"
        "6) Output ONLY the simplified Greek passage. No commentary.\n"
    )
    
    user = (
        "Input passage:\n"
        f"{passage}\n"
        f"Target: ≥ {target:.3f} lemma coverage.\n"
        "Forbidden surface forms:\n"
        "\n".join(f"- {s}" for s in banned_surface) + "\n"
        "Keep length similar to input.\n"
        "Aggressively simplify; it’s okay to omit secondary details."
    )
    return system, user


def prompt_ultra_noob(
    passage: str,
    target: float,
    banned_surface: Sequence[str],
) -> Tuple[str, str]:
    """
    Ultra-noob mode - for absolute beginners with very low coverage (<30%).
    Creates ultra-simple text focusing on core message with allowed vocabulary.
    """
    system = (
        "You create ultra-simple Greek text for absolute beginners.\n"
        "Rules:\n"
        "1) Extract the main ideas and express them simply.\n"
        "2) Use the absolute simplest Greek possible.\n"
        "3) Write as many sentences as needed to convey the content with allowed vocabulary.\n"
        "4) Avoid all banned vocabulary strictly.\n"
        "5) Output ONLY Greek. No commentary.\n"
    )
    
    user = (
        "Input passage:\n"
        f"{passage}\n"
        f"Target: ≥ {target:.3f} lemma coverage.\n"
        "Forbidden surface forms:\n"
        "\n".join(f"- {s}" for s in banned_surface) + "\n"
        "Keep length similar to input.\n"
        "Extreme simplification; focus on core message."
    )
    return system, user
