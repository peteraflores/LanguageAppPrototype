from lemmatizer import Lemmatizer
from lemma_salience import LemmaSalienceRanker

print("boot")

lemmatizer = Lemmatizer(surface_lexicon_path="surface_lexicon.csv", udpipe_model_path="greek-gdt-ud-2.5-191206.udpipe")
print("lemmatizer ready")

ranker = LemmaSalienceRanker(lemmatizer)
print("ranker ready")

passage = "Χθες το απόγευμα μια δυνατή καταιγίδα έφτασε στα παράλια της Βόρειας Καρολίνας. Οι άνεμοι ενισχύθηκαν γρήγορα και η βροχή κράτησε για ώρες, με αποτέλεσμα να πλημμυρίσουν δρόμοι σε χαμηλές περιοχές. Σε αρκετές γειτονιές κόπηκε το ρεύμα, ενώ συνεργεία δούλευαν όλη τη νύχτα για να αποκαταστήσουν τις βλάβες. Οι αρχές ζήτησαν από τους κατοίκους να αποφύγουν τις άσκοπες μετακινήσεις και να προσέχουν τα πεσμένα κλαδιά. Σήμερα το πρωί η κατάσταση βελτιώθηκε, αλλά η προειδοποίηση για ισχυρούς ανέμους παραμένει και η θάλασσα είναι ακόμη φουρτουνιασμένη."
print("passage")
print(passage)
rows = lemmatizer.lemmatize_passage(passage)
print("lemmatize_passage rows:", len(rows))
print("sample rows:", rows[:5])

analysis = ranker.analyze(passage)
print("analysis total_tokens:", analysis.total_tokens)
print("ranked lemmas:", len(analysis.ranked))

for s in analysis.ranked[:25]:
    print(s.score, s.lemma, s.upos, s.token_count, s.sentence_count, s.surface_forms[:3])

print("done")