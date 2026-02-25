from lemmatizer import Lemmatizer
from lemma_salience import LemmaSalienceRanker, load_known_lemmas_csv, load_lemma_frequency_csv
from greek_adaptive_rewriter import GreekAdaptiveRewriter
# from openai_llm_client import OpenAILLMClient
from corning_llm_client import CorningLLMClient
import os
import sys
import io

# Set UTF-8 encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("boot")

# --- pipeline pieces (your existing setup) ---
lemmatizer = Lemmatizer(
    surface_lexicon_path="surface_lexicon.csv",
    udpipe_model_path="greek-gdt-ud-2.5-191206.udpipe",
    auto_promote_agree=False,
    stanza_model_dir=r"stanza_resources",
    stanza_download_method=None,  # enforce offline/no-download behavior
)
print("lemmatizer ready") 

known = load_known_lemmas_csv("known_lemmas.csv")
print("known lemmas loaded:", len(known))

# Load frequency data if available
frequency_map = {}
frequency_file = "lemma_frequency.csv"
if os.path.exists(frequency_file):
    frequency_map = load_lemma_frequency_csv(frequency_file)
    print(f"frequency data loaded: {len(frequency_map)} lemmas")
    
    # Get top N frequent lemmas for threshold check
    # Sort by frequency and take top N
    sorted_lemmas = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)
    top_n = 500  # Match the config default
    top_frequent_lemmas = [lemma for lemma, freq in sorted_lemmas[:top_n]]
else:
    print(f"Warning: {frequency_file} not found. Frequency-based ranking will not be used.")
    top_frequent_lemmas = []

# Pass frequency map to ranker
ranker = LemmaSalienceRanker(
    lemmatizer,
    frequency_map=frequency_map,
    w_frequency=10.0  # Make frequency the dominant factor
)
print("ranker ready with frequency data")

# --- LLM backend ---
# llm = OpenAILLMClient()  # optionally: OpenAILLMClient(model="gpt-4.1-mini")
llm = CorningLLMClient()
print("llm client ready:", getattr(llm, "model", "<unknown>"))

# --- rewriter ---
rewriter = GreekAdaptiveRewriter(
    llm=llm, 
    ranker=ranker, 
    lemmatizer=lemmatizer,
    top_frequent_lemmas=top_frequent_lemmas  # Pass top frequent lemmas for threshold check
)
print("rewriter ready")

# --- test passage ---
passage = (
    "Χθες το απόγευμα μια δυνατή καταιγίδα έφτασε στα παράλια της Βόρειας Καρολίνας. "
    "Οι άνεμοι ενισχύθηκαν γρήγορα και η βροχή κράτησε για ώρες, με αποτέλεσμα να πλημμυρίσουν δρόμοι σε χαμηλές περιοχές. "
    "Σε αρκετές γειτονιές κόπηκε το ρεύμα, ενώ συνεργεία δούλευαν όλη τη νύχτα για να αποκαταστήσουν τις βλάβες. "
    "Οι αρχές ζήτησαν από τους κατοίκους να αποφύγουν τις άσκοπες μετακινήσεις και να προσέχουν τα πεσμένα κλαδιά. "
    "Σήμερα το πρωί η κατάσταση βελτιώθηκε, αλλά η προειδοποίηση για ισχυρούς ανέμους παραμένει και η θάλασσα είναι ακόμη φουρτουνιασμένη."
)
print("passage")
print(passage)

# --- sanity check lemmatizer output like you did before ---
rows = lemmatizer.lemmatize_passage(passage)
print("lemmatize_passage rows:", len(rows))
print("sample rows:", rows[:5])

analysis = ranker.analyze(passage, known_lemmas=known)
unknown_ranked = ranker.filter_known(analysis, known)

print("analysis total_tokens:", analysis.total_tokens)
print("ranked lemmas (all):", len(analysis.ranked))
print("ranked lemmas (unknown candidates):", len(unknown_ranked))

for s in unknown_ranked[:25]:
    print(s.score, s.lemma, s.upos, s.token_count, s.sentence_count, s.surface_forms[:3])

# --- now actually run the adaptive rewrite ---
print("\n--- ADAPT START ---")
result = rewriter.adapt(
    passage,
    known_lemmas=known,
    target_coverage=0.95,   # tune this
    temperature=0.2,        # tune this
)

print("\n--- ADAPT RESULT ---")
print("initial_coverage:", result.initial_coverage)
print("final_coverage:", result.final_coverage)
print("mode_used:", result.mode_used)
print("\nFINAL TEXT:\n")
print(result.final_text)

# --- debug trace so you can see what's happening round-by-round ---
print("\n--- ROUNDS ---")
for r in result.rounds:
    print(
        f"round={r.round_index} mode={r.mode} cov={r.coverage:.3f} "
        f"lemma_viol={len(r.violations_lemmas)} surf_viol={len(r.violations_surface)} "
        f"banned_lemmas={r.banned_lemmas_size} banned_surface={r.banned_surface_size}"
    )
    print(f"  essential_lemmas ({len(r.essential_lemmas)}): {r.essential_lemmas}")
    if r.violations_lemmas:
        print("  lemma viol sample:", r.violations_lemmas[:12])
    if r.violations_surface:
        print("  surface viol sample:", r.violations_surface[:12])

print("\n--- FINAL STATE ---")
print(f"Final keep-set ({len(result.essential_lemmas)}): {result.essential_lemmas}")
print(f"Final banned lemmas ({len(result.banned_lemmas)}): {result.banned_lemmas[:20]}...")
print(f"Final banned surface ({len(result.banned_surface_forms)}): {result.banned_surface_forms[:20]}...")

# Show which lemmas contribute to coverage
if result.final_text != passage:  # Only if text was rewritten
    final_analysis = ranker.analyze(result.final_text, known_lemmas=known)
    print(f"\n--- FINAL TEXT ANALYSIS ---")
    print(f"Total tokens: {final_analysis.total_tokens}")
    print(f"Coverage: {final_analysis.coverage:.3f}")
    
    # Show lemmas contributing to coverage
    contributing_lemmas = []
    for stat in final_analysis.ranked:
        if stat.lemma in known or stat.lemma in result.essential_lemmas:
            contributing_lemmas.append((stat.lemma, stat.token_count, 'known' if stat.lemma in known else 'keep'))
    
    print(f"\nLemmas contributing to coverage (top 20):")
    for lemma, count, source in sorted(contributing_lemmas, key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {lemma}: {count} tokens ({source})")
    
    # Show unknown lemmas that don't contribute
    unknown_present = []
    for stat in final_analysis.ranked:
        if stat.lemma not in known and stat.lemma not in result.essential_lemmas:
            unknown_present.append((stat.lemma, stat.token_count))
    
    if unknown_present:
        print(f"\nUnknown lemmas NOT contributing to coverage:")
        for lemma, count in sorted(unknown_present, key=lambda x: x[1], reverse=True):
            print(f"  {lemma}: {count} tokens")

print("\ndone")
