# Greek Adaptive Rewriter

This project is an LLM-driven toolkit designed to adapt Modern Greek texts to a learner's specific vocabulary level. It analyzes the "lemma coverage" of a passage—comparing the words in the text against a user's known vocabulary—and uses an LLM to rewrite the passage so that it remains comprehensible while preserving the original meaning.

## Core Features

- **Personalized Lemmatization:** Uses a dual-engine approach (Stanza and UDPipe) combined with a local surface-form lexicon to accurately identify lemmas in Greek text.
- **Vocabulary Coverage Analysis:** Calculates what percentage of a text is composed of "known" lemmas versus "unknown" ones.
- **Adaptive Rewriting:** Orchestrates an iterative LLM process to rewrite passages using various strategies (Surgical, Simplify, Retell, Noob, Ultra-Noob) to meet a specific coverage target.
- **Lemma Salience Ranking:** Ranks unknown words by frequency and part-of-speech importance (e.g., favoring nouns and verbs) to help learners identify which new words are most worth learning.
- **Lexicon Management:** Tools to build lemma frequency tables from SUBTLEX-GR and manage a "needs review" workflow for ambiguous lemmatization cases.

## Project Structure

### Core Modules
- **`lemmatizer.py`**: The central engine. It handles text normalization, punctuation stripping, and tokenization using Stanza and UDPipe. It maintains a local lexicon (`surface_lemma_lexicon.csv`) for fast, reliable lookups.
- **`greek_adaptive_rewriter.py`**: The orchestrator that takes a passage and a target coverage level. It manages the loop between the LLM and the lemmatizer to ensure the output obeys vocabulary constraints.
- **`lemma_salience.py`**: Analyzes text to provide statistics on lemma frequency and coverage. It helps determine which "essential lemmas" should be kept or taught.
- **`rewrite_prompts.py`**: Contains the logic for different rewriting "modes," ranging from minimal surgical edits to "Ultra-Noob" mode for absolute beginners.

### Data Processing & Utilities
- **`main.py`**: The entry point for the pipeline, demonstrating how to load known lemmas, initialize the rewriter, and process a text.
- **`openai_llm_client.py`**: A robust implementation of an LLM client with built-in rate-limiting and retry logic.
- **`lingq_to_known_lemmas.py`**: A utility to import known vocabulary from LingQ exports into the project's internal format.
- **`build_lemma_frequency.py`**: Generates a frequency and rank table for Greek lemmas using the SUBTLEX-GR dataset.

### Human-in-the-Loop Workflow
- **`build_review_summary.py`**: Groups ambiguous lemmatization results (where tools disagree) into a CSV for manual review.
- **`promote_approved.py`**: Updates the master lexicon with manually approved lemma/UPOS assignments from the review summary.

## Setup and Requirements

### Dependencies
- **Python 3.8+**
- **Libraries**: `stanza`, `ufal.udpipe`, `openai`, `unicodedata`, `csv`
- **Models**:
  - Stanza Greek resources (stored in `stanza_resources/`)
  - UDPipe Greek model (e.g., `greek-gdt-ud-2.5-191206.udpipe`)

### Environment Variables
To use the rewriting features, you must set your OpenAI credentials:
```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o  # or your preferred model
```

## Usage

1.  **Prepare your vocabulary**: Generate a `known_lemmas.csv` file containing the lemmas you already know.
2.  **Initialize the Lexicon**: If you have a frequency list or existing lexicon, ensure `lemma_frequency.csv` and `surface_lemma_lexicon.csv` are in the project root.
3.  **Run the pipeline**:
    ```python
    from main import main
    # Configure your paths in main() or call the rewriter directly
    main()
    ```

## Workflow for Improving Accuracy

If the lemmatizer encounters words it isn't sure about:
1. Run your text through the system.
2. Instances requiring review are saved to `needs_review_instances.csv`.
3. Run `python build_review_summary.py` to create a summary.
4. Manually edit `needs_review_summary.csv` to approve correct lemmas.
5. Run `python promote_approved.py` to bake those corrections into the permanent lexicon.