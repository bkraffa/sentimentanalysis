from data_collect import data_collect
from preprocessing import preprocessing, separa_datasets, tokenizacao, vetorizador_tfidf, train_test_split
from evaluation import evaluate, evaluate_lstm
from modeling import modeling, modeling_lstm
from utils import *
import os
import numpy as np

if __name__ == "__main__":
    # Coleta de dados
    df = data_collect()
    #Ir para o diretório pai e salvar os dados
    os.chdir("..")
    save_data(df, 'data/raw_data.csv')
    #Pré-Processamento
    df = preprocessing(df)
    save_data(df, 'data/preprocessed_data.csv')
    #RF e XGBcd
    X1 = vetorizador_tfidf(df)
    Y = (df['sentiment']).values
    X_train1, X_test1, Y_train, Y_test = separa_datasets(X1,Y)


    gnb, rfc, xgb = modeling(X_train1, Y_train)

    #Pré-processamento do lstm
    X2 = tokenizacao(df)
    Y = np.array(pd.get_dummies((df['sentiment']).values))
    X_train2, X_test2, Y_train, Y_test = separa_datasets(X2,Y)

    #treino do lstm

    lstm = modeling_lstm(X_train2, Y_train)

    # Avaliação modelos
    acc, f1 = evaluate(gnb, X_test1, Y_test)
    print(f"Naive Bayes: acc = {acc:.3} / f1 = {f1:.3}")

    acc, f1 = evaluate(rfc, X_test1, Y_test)
    print(f"Random Forest Classification: acc = {acc:.3} / f1 = {f1:.3}")
    
    acc, f1 = evaluate(rfc, X_test1, Y_test)
    print(f"XGBoost com Tuning dos Hiperparametros: acc = {acc:.3} / f1 = {f1:.3}")

    acc, f1 = evaluate_lstm(lstm, X_test2, Y_test)
    print(f"LSTM: acc = {acc:.3} / f1 = {f1:.3}")
