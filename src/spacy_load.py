import spacy

def load_spacy():
    nlp = spacy.load('pt_core_news_lg')
    return nlp