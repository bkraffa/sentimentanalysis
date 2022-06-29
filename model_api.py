from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np
import keras
from src.spacy_load import load_spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

nlp = load_spacy()

class Item(BaseModel):
    text: str

with open("models/rfc.pkl", "rb") as file:
    rfc = pickle.load(file)

with open("models/gnb.pkl", "rb") as file:
    gnb = pickle.load(file)

with open("models/xgb.pkl", "rb") as file:
    xgb = pickle.load(file)

with open("models/tf_idf.pkl", "rb") as file:
    tfidf = pickle.load(file)

path = 'models/lstm.h5'
lstm = keras.models.load_model(path)

with open("models/tokenizacao.pkl", "rb") as file:
    tokenizer = pickle.load(file)

app = FastAPI()

@app.post("/predict/{model}")
def predict(model: str, item: Item):
    
    if model == "lstm":
        X = item.text
        X = X.lower()
        X = re.sub('[^\w\s]','',X)
        X = [y.lemma_ for y in nlp(X) if not y.is_stop]
        X = tokenizer.texts_to_sequences(X)
        X = pad_sequences(X)
        print(X.shape)
        X = np.lib.pad(X, ((0,21-X.shape[0]),(0,0)), 'constant', constant_values=(0))
        print(X.shape)
        X = X.reshape(1,21)
        X = np.flip(X)
        pred = lstm.predict(X)
        pred = np.argmax(pred)
 
    elif model == "naive_bayes":
        X = tfidf.transform([item.text]).toarray()
        pred = gnb.predict(X)

    elif model == "xgboost":
        X = tfidf.transform([item.text]).toarray()
        pred = xgb.predict(X)
    
    elif model == "random_forest":
        X = tfidf.transform([item.text]).toarray()
        pred = rfc.predict(X)

    return {"model": model, "text": item.text, "prediction": pred.tolist()}