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
    #Transformando todas palavras pra minúsculo
    df['text'] = df['text'].apply(lambda x: x.lower())
    #Limpando os tweets    
    df['text'] = df['text'].apply((lambda x: re.sub('[^\w\s]','',x)))
    #Removendo stopwords
    df["text"] = df['text'].apply(lambda x: " ".join([y.lemma_ for y in nlp(x) if not y.is_stop]))
    #Transformando os sentiments -1(negativos) em 0
    df.loc[df['sentiment'] == -1,'sentiment'] = 0
    #Convertendo de string pra int
    df['sentiment'] = df.sentiment.astype(int)
    #Simplificando o df pra termos só as features de interesse
    df = df[['text','sentiment','group']]
    df.reset_index(drop=True, inplace=True)
    return df

def vetorizador_tfidf(df,df_treino):
    tf_idf = TfidfVectorizer(
        encoding="latin-1",
        strip_accents='ascii',
        lowercase=True, 
        min_df= 0.01,
        ngram_range=(1,3)
    )
    tf_idf.fit(df['text'])
    X = np.array(tf_idf.transform(df_treino["text"]).todense())
    with open("../models/tf_idf.pkl", "wb") as file:
        pickle.dump(obj=tf_idf, file=file)
    return X


def tokenizacao_fit(df):
    max_features = 3000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(df['text'].values)
    with open("models/tokenizacao.pkl", "wb") as file:
        pickle.dump(obj=tokenizer, file=file)
    return tokenizer

def tokenizacao_transform(df,tokenizer,ajust=False):
    X = tokenizer.texts_to_sequences(df['text'].values)
    X = pad_sequences(X)
    if ajust == True:
        X = np.lib.pad(X, ((0,0),(34 - X.shape[1],0)), 'constant', constant_values=(0))
    return X
  
def separa_datasets(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
    return X_train,X_test,Y_train,Y_test

def preprocessing_eleicoes(df):
    df['text'] = df['text'].apply(lambda x: x.lower())    
    df['text'] = df['text'].apply((lambda x: re.sub('[^\w\s]','',x)))
    df["text"] = df['text'].apply(lambda x: " ".join([y.lemma_ for y in nlp(x) if not y.is_stop]))
    df.loc[df['sentiment'] == -1,'sentiment'] = 0
    df['sentiment'] = df.sentiment.astype(int)
    df = df[['text','sentiment']]
    df.reset_index(drop=True, inplace=True)
    return df

