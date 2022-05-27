from data_collect import data_collect
from spacy_load import load_spacy

if __name__ == "__main__":
    # Coleta de dados
    df = data_collect()
    #Carregando o spacy
    nlp = load_spacy()
    
    print(df.head(5))