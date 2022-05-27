import re
import pickle
import numpy as np
from spacy_load import load_spacy
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

#Carregando o spacy
nlp = load_spacy()

def preprocessing(df):
    #Removendo os sentimentos neutros e não anotados
    df.drop(df.loc[df['sentiment'] == '0'].index, inplace = True)
    df.drop(df.loc[df['sentiment'] == '-'].index, inplace = True)
    #Transformando todas palavras pra minúsculo
    df['text'] = df['text'].apply(lambda x: x.lower())
    #Limpando os tweets    
    df['text'] = df['text'].apply((lambda x: re.sub('[^\w\s]','',x)))
    #Removendo stopwords
    df["text"] = df['text'].apply(lambda x: " ".join([y.lemma_ for y in nlp(x) if not y.is_stop]))
    #Transformando os sentiments -1(negativos) em 0
    df.loc[df['sentiment'] == '-1','sentiment'] = 0
    #Convertendo de string pra int
    df['sentiment'] = df.sentiment.astype(int)
    #Simplificando o df pra termos só as features de interesse
    df = df[['text','sentiment']]
    df.reset_index(drop=True, inplace=True)
    return df

def vetorizador_tfidf(df):
    tf_idf = TfidfVectorizer(
        encoding="latin-1",
        strip_accents='ascii',
        lowercase=True, 
        min_df= 0.01,
        ngram_range=(1,3)
    )
    X = np.array(tf_idf.fit_transform(df["text"]).todense())
    with open("models/tf_idf.pkl", "wb") as file:
        pickle.dump(obj=tf_idf, file=file)
    return X


def tokenizacao(df):
    max_features = 3000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(df['text'].values)
    X = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(X)
    with open("models/tokenizacao.pkl", "wb") as file:
        pickle.dump(obj=tokenizer, file=file)
    return X
  
def separa_datasets(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
    return X_train,X_test,Y_train,Y_test

