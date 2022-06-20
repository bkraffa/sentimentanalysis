import streamlit as st
import requests
import json

MODEL_URL = f'http://localhost:8000/predict'

def predict_results(url,modelo,frase):
    result = requests.post(f'{url}/{modelo}', json = {'text':frase})
    if modelo == 'lstm':
        if json.loads(result.text)['prediction'] == 0:
            st.write('O sentimento previsto pelo modelo é: Negativo')
        else:
            st.write('O sentimento previsto pelo modelo é: Positivo')
    else:
        if json.loads(result.text)['prediction'][0] == 0:
            st.write('O sentimento previsto pelo modelo é: Negativo')
        else:
            st.write('O sentimento previsto pelo modelo é: Positivo')

with st.form('my_ml'):
    frase = st.text_input(label = 'Digite um tweet para avaliar o sentimento')
    modelo = st.radio('Escolha seu modelo', ('Naive Bayes','XGBoost','Random Forest Classifier','LSTM'))
    predict = st.form_submit_button(label = 'Prever sentimento!')

if predict and modelo == 'Naive Bayes':
    predict_results(MODEL_URL,'naive_bayes', frase)

elif predict and modelo == 'XGBoost':
    predict_results(MODEL_URL,'xgboost', frase)

elif predict and modelo == 'Random Forest Classifier':
    predict_results(MODEL_URL,'random_forest', frase)

elif predict and modelo == 'LSTM':
    predict_results(MODEL_URL,'lstm', frase)