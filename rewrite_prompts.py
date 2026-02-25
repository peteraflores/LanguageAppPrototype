"""
rewrite_prompts.py

Prompting strategies for Greek Adaptive Rewriter.
Includes standard modes (surgical, simplify, retell) and new beginner modes (noob, ultra_noob).
"""

from typing import Sequence, Tuple


# Base system prompt used across all modes
_SYSTEM_COMMON = (
    "You rewrite Modern Greek text.\n"
    "Rules:\n"
    "1) Preserve meaning, facts, and named entities.\n"
    "2) Obey allowed/banned lemma constraints strictly.\n"
    "3) Output ONLY the rewritten Greek passage. No commentary.\n"
)


def prompt_surgical(
    passage: str,
    target_coverage: float,
    banned_lemmas: Sequence[str],
    banned_surface: Sequence[str],
) -> Tuple[str, str]:
    """Minimal edits mode - for high initial coverage (88%+)"""
    system = _SYSTEM_COMMON + (
        "Prefer minimal edits. Keep sentence count the same unless absolutely necessary.\n"
        "Do not delete key ideas.\n"
    )
    user = (
        "Input passage (Greek):\n"
        f"{passage}\n\n"
        f"Target: lemma-coverage ≥ {target_coverage:.3f} relative to the learner's known vocabulary.\n"
        "Avoid using the following banned lemmas and surface forms.\n\n"
        "Banned lemmas (must not appear in any inflected form):\n"
        + "\n".join(f"- {l}" for l in banned_lemmas) + "\n\n"
        "Banned surface forms (must not appear verbatim):\n"
        + "\n".join(f"- {s}" for s in banned_surface) + "\n\n"
        "Extra constraints:\n"
        "1) Keep sentence count the same unless absolutely necessary.\n"
        "2) Preserve tense/aspect and key relations.\n"
        "3) Keep register natural, modern.\n\n"
        "Return: rewritten passage only.\n"
    )
    return system, user


def prompt_simplify_then_enforce(
    passage: str,
    banned_lemmas: Sequence[str],
    banned_surface: Sequence[str],
) -> Tuple[str, str]:
    """Two-stage simplification - for medium coverage (70-88%)"""
    system = _SYSTEM_COMMON + (
        "You may paraphrase more broadly, but must preserve all facts.\n"
        "Prefer shorter clauses and common vocabulary.\n"
    )
    user = (
        "Task: two-stage rewrite in one output.\n\n"
        "Stage 1 (simplify): Rewrite the passage into simpler Modern Greek while preserving all facts and meaning.\n"
        "Prefer shorter clauses, common vocabulary, and explicit subject-verb-object structure.\n"
        "Do NOT remove important details.\n\n"
        "Stage 2 (enforce constraints): Adjust the simplified passage to avoid the banned vocabulary below.\n\n"
        "Input passage:\n"
        f"{passage}\n\n"
        "Banned lemmas (must not appear in any inflected form):\n"
        + "\n".join(f"- {l}" for l in banned_lemmas) + "\n\n"
        "Banned surface forms (must not appear verbatim):\n"
        + "\n".join(f"- {s}" for s in banned_surface) + "\n\n"
        "Hard constraints: do not use any banned lemma in any form. Output only the rewritten passage.\n"
    )
    return system, user


def prompt_retell(
    passage: str,
    banned_lemmas: Sequence[str],
    banned_surface: Sequence[str],
    target_sentences: int,
) -> Tuple[str, str]:
    """Retelling mode - for lower coverage (50-70%)"""
    system = _SYSTEM_COMMON + (
        "You may compress and reorder for clarity, but must keep key facts and causal relations.\n"
        "Write as much as needed to cover the main points using allowed vocabulary.\n"
    )
    user = (
        "Write a faithful retelling in Modern Greek.\n"
        "Keep the key facts and causal relations, but you may shorten and reorder for clarity.\n"
        "Use as many sentences as needed to express the content while avoiding banned vocabulary.\n\n"
        "Input passage:\n"
        f"{passage}\n\n"
        "Banned lemmas (must not appear in any inflected form):\n"
        + "\n".join(f"- {l}" for l in banned_lemmas) + "\n\n"
        "Banned surface forms (must not appear verbatim):\n"
        + "\n".join(f"- {s}" for s in banned_surface) + "\n\n"
        "Hard constraints: do not use any banned lemma in any form. Output only the retelling.\n"
    )
    return system, user


def prompt_noob(
    passage: str,
    target_coverage: float,
    banned_lemmas: Sequence[str],
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
        "Simplify this passage for a beginner who knows very limited Greek vocabulary.\n"
        "You are ALLOWED to:\n"
        "- Omit complex details and secondary information\n"
        "- Summarize events rather than describe them fully\n"
        "- Use general terms instead of specific ones\n"
        "- Write as many simple sentences as needed to reach the target coverage\n\n"
        
        "Input passage (Greek):\n"
        f"{passage}\n\n"
        
        f"Target: lemma-coverage ≥ {target_coverage:.3f} relative to the beginner's known vocabulary.\n"
        "The learner knows VERY FEW words, so be extremely aggressive in simplifying.\n\n"
        
        "Banned lemmas (must not appear in any form):\n"
        + "\n".join(f"- {l}" for l in banned_lemmas) + "\n\n"
        
        "Banned surface forms (must not appear verbatim):\n"
        + "\n".join(f"- {s}" for s in banned_surface) + "\n\n"
        
        "Guidelines:\n"
        "1) Preserve the essential information using allowed vocabulary\n"
        "2) A storm can become 'bad weather', specific places can become 'a place'\n"
        "3) Complex sequences can become multiple simple statements\n"
        "4) Use more sentences with simple vocabulary rather than fewer complex ones\n"
        "5) Repeat allowed words as much as needed to convey the meaning\n\n"
        
        "Return: simplified passage only.\n"
    )
    return system, user


def prompt_ultra_noob(
    passage: str,
    banned_lemmas: Sequence[str],
    banned_surface: Sequence[str],
    target_sentences: int = 3,
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
        "Create an ultra-simple version in basic Greek.\n"
        "Focus on the main ideas but use as many simple sentences as needed.\n"
        "It's better to use more simple sentences than to omit important information.\n\n"
        
        "Input passage:\n"
        f"{passage}\n\n"
        
        "Vocabulary constraints:\n"
        "Banned lemmas (NEVER use these in any form):\n"
        + "\n".join(f"- {l}" for l in banned_lemmas) + "\n\n"
        
        "Banned surface forms (must not appear verbatim):\n"
        + "\n".join(f"- {s}" for s in banned_surface) + "\n\n"
        
        "Guidelines:\n"
        "- Use multiple simple sentences rather than complex ones\n"
        "- Repeat allowed vocabulary as much as needed\n"
        "- Break complex ideas into simple statements\n"
        "- Keep writing until you've covered the main content\n\n"
        
        "Return: ultra-simple version only.\n"
    )
    return system, user
