"""
build_lemma_frequency.py

Processes SUBTLEX-GR_restricted.txt to create a lemma frequency table.
Uses fast lexicon lookup first, then falls back to full lemmatizer for unknown words.

Output: lemma_frequency.csv with columns: lemma, frequency, rank
"""

import csv
import os
from collections import Counter
from pathlib import Path

from lemmatizer import Lemmatizer, normalize_text


def _norm_lemma(s: str) -> str:
    """Use lemmatizer's normalize function"""
    return normalize_text(s)


def load_subtlex_frequencies(path: str) -> list[tuple[str, int]]:
    """
    Load SUBTLEX-GR_restricted.txt
    File has header lines, then tab-delimited data with columns:
    ID, Word, FREQcount, CD, SUBTLEX_WF, etc.
    Returns list of (surface_form, frequency) tuples
    """
    frequencies = []
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"SUBTLEX file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        # Skip header lines until we find the column header line
        for line in f:
            if line.startswith('"ID"'):
                break
        
        # Now read the data lines
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab and remove quotes
            parts = [p.strip('"') for p in line.split("\t")]
            
            if len(parts) >= 3:  # Need at least ID, Word, FREQcount
                try:
                    surface = parts[1]  # Word column
                    freq = int(parts[2])  # FREQcount column
                    frequencies.append((surface, freq))
                except (ValueError, IndexError):
                    # Skip lines with invalid data
                    continue
    
    return frequencies


def load_surface_lexicon_csv(path: str) -> dict[str, set[tuple[str, str]]]:
    """
    Load lexicon CSV directly (same format as Lemmatizer uses)
    CSV columns: surface, lemma, upos
    Returns dict mapping normalized surface -> set of (lemma, upos) tuples
    """
    # Just use the lemmatizer's lexicon loading method
    # Create a temporary lemmatizer just to load the lexicon
    temp_lemmatizer = Lemmatizer(
        surface_lexicon_path=path,
        udpipe_model_path="greek-gdt-ud-2.5-191206.udpipe",  # dummy, won't be used
        use_lexicon=True,
        skip_propn=False
    )
    return temp_lemmatizer.surface_lexicon


def build_lemma_frequency_table(
    subtlex_path: str,
    lexicon_path: str = "surface_lemma_lexicon.csv",
    output_path: str = "lemma_frequency.csv"
) -> None:
    """
    Main pipeline:
    1. Load SUBTLEX frequencies
    2. Load surface lexicon for fast lookup
    3. Try lexicon lookup first
    4. Only use full lemmatizer for unknown words
    5. Aggregate frequencies by lemma
    6. Output ranked frequency table
    """
    
    print(f"Loading SUBTLEX data from {subtlex_path}...")
    subtlex_data = load_subtlex_frequencies(subtlex_path)
    print(f"Loaded {len(subtlex_data)} surface forms with frequencies")
    
    print(f"Loading surface lexicon from {lexicon_path}...")
    surface_lexicon = load_surface_lexicon_csv(lexicon_path)
    print(f"Loaded lexicon with {len(surface_lexicon)} unique surface forms")
    
    # Collect unknown words that need full lemmatization
    unknown_words = []
    lemma_counter = Counter()
    lexicon_hits = 0
    
    print("Checking words against lexicon...")
    for surface, freq in subtlex_data:
        surface_norm = _norm_lemma(surface)
        
        # Check lexicon first
        lemma_options = surface_lexicon.get(surface_norm)
        if lemma_options:
            # Use first lemma if multiple options (sorted for consistency)
            lemma, upos = sorted(lemma_options)[0]
            lemma_counter[lemma] += freq
            lexicon_hits += 1
        else:
            # Save for batch processing with full lemmatizer
            unknown_words.append((surface, freq))
    
    print(f"Found {lexicon_hits} words in lexicon, {len(unknown_words)} need full lemmatization")
    
    # Process unknown words with full lemmatizer if needed
    if unknown_words:
        print(f"Initializing lemmatizer for {len(unknown_words)} unknown words...")
        udpipe_model_path = "greek-gdt-ud-2.5-191206.udpipe"
        stanza_model_dir = os.path.join(os.getcwd(), "stanza_resources")
        lemmatizer = Lemmatizer(
            surface_lexicon_path=lexicon_path,
            udpipe_model_path=udpipe_model_path,
            stanza_model_dir=stanza_model_dir
        )
        
        # Process in batches for efficiency
        batch_size = 100
        processed = 0
        skipped = 0
        
        print("Processing unknown words in batches...")
        for batch_start in range(0, len(unknown_words), batch_size):
            batch_end = min(batch_start + batch_size, len(unknown_words))
            batch = unknown_words[batch_start:batch_end]
            
            # Process each word individually for accurate matching
            for word, freq in batch:
                try:
                    lemma_data = lemmatizer.lemmatize_passage(word)
                    if lemma_data:
                        # Get the first (and should be only) lemma
                        _, lemma, _, _ = lemma_data[0]
                        if lemma:
                            lemma_norm = _norm_lemma(lemma)
                            lemma_counter[lemma_norm] += freq
                            processed += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
                except Exception as e:
                    # Skip problematic entries
                    skipped += 1
            
            # Progress update
            print(f"  Batch {batch_start//batch_size + 1}/{(len(unknown_words) + batch_size - 1)//batch_size} - Processed: {processed}, Skipped: {skipped}")
    
    print(f"\nTotal: Lexicon hits: {lexicon_hits}, Lemmatized: {processed}, Skipped: {skipped}")
    print(f"Found {len(lemma_counter)} unique lemmas")
    
    # Sort by frequency (descending) and assign ranks
    ranked_lemmas = []
    for rank, (lemma, freq) in enumerate(lemma_counter.most_common(), 1):
        ranked_lemmas.append({
            "lemma": lemma,
            "frequency": freq,
            "rank": rank
        })
    
    # Write output
    print(f"Writing frequency table to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lemma", "frequency", "rank"])
        writer.writeheader()
        writer.writerows(ranked_lemmas)
    
    print(f"Done! Wrote {len(ranked_lemmas)} lemmas to {output_path}")
    
    # Print top 20 most frequent lemmas as a sanity check
    print("\nTop 20 most frequent lemmas:")
    for entry in ranked_lemmas[:20]:
        print(f"  {entry['rank']:4d}. {entry['lemma']:20s} (freq: {entry['frequency']:,})")


def main():
    """
    Main entry point. Can be customized with different paths if needed.
    """
    subtlex_path = "SourceDataFiles/SUBTLEX-GR_restricted.txt"
    lexicon_path = "surface_lemma_lexicon.csv"
    output_path = "lemma_frequency.csv"
    
    # Check if surface_lemma_lexicon.csv exists
    if not os.path.exists(lexicon_path):
        print(f"Warning: {lexicon_path} not found. Building it first...")
        # Run the build script
        import subprocess
        subprocess.run(["python", "SourceDataFiles/build_surface_lexicon.py"], check=True)
    
    build_lemma_frequency_table(
        subtlex_path=subtlex_path,
        lexicon_path=lexicon_path,
        output_path=output_path
    )


if __name__ == "__main__":
    main()