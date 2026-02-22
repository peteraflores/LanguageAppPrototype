from lemmatizer import Lemmatizer

# Initialize Lemmatizer with lexicon and UDPipe model path
lemmatizer = Lemmatizer(surface_lexicon_path="surface_lemma_lexicon.csv", udpipe_model_path="greek-gdt-ud-2.5-191206.udpipe")

# Lemmatize a sample passage
passage = "Ο Ξενοζούλης τσιμπολογάει."
lemmatized = lemmatizer.lemmatize_passage(passage)

# Print lemmatized output
for result in lemmatized:
    print(result)