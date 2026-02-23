from lemmatizer import Lemmatizer
from lemma_salience import LemmaSalienceRanker, load_known_lemmas_csv
from greek_adaptive_rewriter import GreekAdaptiveRewriter
# from openai_llm_client import OpenAILLMClient
from corning_llm_client import CorningLLMClient

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

ranker = LemmaSalienceRanker(lemmatizer)
print("ranker ready")

# --- LLM backend ---
# llm = OpenAILLMClient()  # optionally: OpenAILLMClient(model="gpt-4.1-mini")
llm = CorningLLMClient()
print("llm client ready:", getattr(llm, "model", "<unknown>"))

# --- rewriter ---
rewriter = GreekAdaptiveRewriter(llm=llm, ranker=ranker, lemmatizer=lemmatizer)
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
        f"banned_lemmas={len(r.banned_lemmas)} banned_surface={len(r.banned_surface)}"
    )
    if r.violations_lemmas:
        print("  lemma viol sample:", r.violations_lemmas[:12])
    if r.violations_surface:
        print("  surface viol sample:", r.violations_surface[:12])

print("done")