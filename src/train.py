from data_collect import data_collect
from preprocessing import preprocessing, separa_datasets, tokenizacao, vetorizador_tfidf, train_test_split
from evaluation import evaluate
from modeling import modeling
from utils import *
import os

if __name__ == "__main__":
    # Coleta de dados
    df = data_collect()
    #Ir para o diretório pai e salvar os dados
    os.chdir("..")
    save_data(df, 'data/raw_data.csv')
    #Pré-Processamento
    df = preprocessing(df)
    save_data(df, 'data/preprocessed_data.csv')
    #RF e XGB
    X = vetorizador_tfidf(df)
    Y = (df['sentiment']).values
    X_train, X_test, Y_train, Y_test = separa_datasets(X,Y)


    gnb, rfc, xgb = modeling(X_train, Y_train)

    # Avaliação
    acc, f1 = evaluate(gnb, X_test, Y_test)
    print(f"Naive Bayes: acc = {acc:.3} / f1 = {f1:.3}")

    acc, f1 = evaluate(rfc, X_test, Y_test)
    print(f"Random Forest Classification: acc = {acc:.3} / f1 = {f1:.3}")
    
    acc, f1 = evaluate(rfc, X_test, Y_test)
    print(f"XGBoost com Tuning dos Hiperparametros: acc = {acc:.3} / f1 = {f1:.3}")


    #LSTM
    #Tokenizacao
    #X = tokenizacao(df)
    #Atribuindo a Y a nossa variável alvo
    Y = (df['sentiment']).values
    #Dividindo entre treino e teste
    #X_train, X_test, Y_train, Y_test = separa_datasets(X,Y)
    #print(X_train.shape,Y_train.shape)
    #print(X_test.shape,Y_test.shape)