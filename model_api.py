from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np
import keras

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
        X = tokenizer.transform([item.text]).toarray()
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