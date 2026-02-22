from lemmatizer import Lemmatizer

# Initialize Lemmatizer with lexicon and UDPipe model path
lemmatizer = Lemmatizer(surface_lexicon_path="surface_lemma_lexicon.csv", udpipe_model_path="greek-gdt-ud-2.5-191206.udpipe")

# Lemmatize a sample passage
passage = "Η Ελλάδα είναι μια όμορφη χώρα με μεγάλη ιστορία και πολιτισμό. Στην Αθήνα, οι άνθρωποι ζουν την καθημερινότητά τους, και οι επισκέπτες μπορούν να απολαύσουν την εκπληκτική θέα από την Ακρόπολη. Πολλοί τουρίστες έρχονται από όλο τον κόσμο για να δουν τα μνημεία και τα μουσεία της πόλης. Η πόλη έχει πολλούς εστιατόρες που σερβίρουν παραδοσιακή ελληνική κουζίνα, και οι κάτοικοι της περιοχής είναι φιλόξενοι."
lemmatized = lemmatizer.lemmatize_passage(passage)

# Print lemmatized output
for result in lemmatized:
    print(result)