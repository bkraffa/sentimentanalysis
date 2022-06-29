from data_collect import data_collect_treino, data_collect_teste
from eleicoes_collect import read_eleicoes_teste, read_eleicoes_treino
from preprocessing import preprocessing, separa_datasets, tokenizacao_fit, tokenizacao_transform, vetorizador_tfidf, preprocessing_eleicoes
from evaluation import evaluate, evaluate_lstm, figura_matriz_confusao, figura_matriz_confusao_lstm
from modeling import modeling, modeling_lstm
from utils import *
import os
import numpy as np

if __name__ == "__main__":
# Coleta e pré-processamento dos dados
    ttsbr = pd.read_json('..\data\corpusTT.json', orient='index')
    ttsbr = ttsbr.loc[ttsbr['sentiment'] != '-'].reset_index(drop=True)
    ttsbr = ttsbr.loc[ttsbr['sentiment'] != '0'].reset_index(drop=True)
    eleicoes_treino = read_eleicoes_treino()
    df = pd.concat([ttsbr, eleicoes_treino], axis=0)
    df.sentiment = df.sentiment.astype('int64')
    df = preprocessing(df)
    df_treino = df.loc[df['group'] != 'test'].reset_index(drop=True)
    ttsbr_teste = df.loc[df['group'] == 'test'].reset_index(drop=True)

#  Pré-processamento dos dados de teste com a vetorização tfidf do dataset df_treino (ttsbr + eleicoes_treino)

    X_treino1 = vetorizador_tfidf(df,df_treino)
    Y_treino1 = df_treino.sentiment

    X_teste1 = vetorizador_tfidf(df,ttsbr_teste)
    Y_teste1 = ttsbr_teste.sentiment

# Salvando o df_treino pré-processado

    os.chdir("..")
    save_data(df_treino, 'data/preprocessed_data.csv')

#Treino do GNB, RFC e XGB

    gnb, rfc, xgb = modeling(X_treino1, Y_treino1)

#  Pré-processamento dos dados de teste com a tokenização do dataset df_treino (ttsbr + eleicoes_treino)
    tokenizer = tokenizacao_fit(df_treino)
    X_treino2 = tokenizacao_transform(df_treino,tokenizer)
    print(f"X treino 2 depois da tokenizacao: {X_treino2.shape}")
    save_data(pd.DataFrame(X_treino2), 'data/preprocessed_datalstm.csv')
    Y_treino2 = np.array(pd.get_dummies((df_treino['sentiment']).values))

# Treino do LSTM

    lstm = modeling_lstm(X_treino2, Y_treino2)

#  Pré-processamento dos dados de teste com a tokenização do dataset ttsbr

    X_teste2 = tokenizacao_transform(ttsbr_teste,tokenizer,ajust=True)
    Y_teste2 = np.array(pd.get_dummies((ttsbr_teste['sentiment']).values))

# Avaliação do modelo usando como base de teste o dataset ttsbr

    acc, f1 = evaluate(gnb, X_teste1, Y_teste1)
    print(f"Naive Bayes: acc = {acc:.3} / f1 = {f1:.3}")
    matriz_confusao = figura_matriz_confusao(gnb, X_teste1, Y_teste1)
    save_plot(matriz_confusao, 'plots/gnb_ttsbr.jpg')

    acc, f1 = evaluate(rfc, X_teste1, Y_teste1)
    print(f"Random Forest Classification: acc = {acc:.3} / f1 = {f1:.3}")
    matriz_confusao = figura_matriz_confusao(rfc, X_teste1, Y_teste1)
    save_plot(matriz_confusao, 'plots/rfc_ttsbr.jpg')

    acc, f1 = evaluate(xgb, X_teste1, Y_teste1)
    print(f"XGBoost com Tuning dos Hiperparametros: acc = {acc:.3} / f1 = {f1:.3}")
    matriz_confusao = figura_matriz_confusao(xgb, X_teste1, Y_teste1)
    save_plot(matriz_confusao, 'plots/xgb_ttsbr.jpg')

    acc, f1 = evaluate_lstm(lstm, X_teste2, Y_teste2)
    print(f"LSTM: acc = {acc:.3} / f1 = {f1:.3}")
    matriz_confusao = figura_matriz_confusao_lstm(lstm, X_teste2, Y_teste2)
    save_plot(matriz_confusao, 'plots/lstm_ttsbr.jpg')

# Avaliação do modelo usando como base de validação o dataset de eleições

    eleicoes_teste = read_eleicoes_teste()
    print(eleicoes_teste)
    eleicoes_teste = preprocessing_eleicoes(eleicoes_teste)
    val_x = tokenizacao_transform(eleicoes_teste,tokenizer,ajust=True) 
    print("Shape validação X: ", val_x.shape)
    val_y = np.array(pd.get_dummies((eleicoes_teste['sentiment']).values))
    print("Shape validação Y: ", val_y.shape)
    
    acc, f1 = evaluate_lstm(lstm, val_x, val_y)
    print(f"LSTM: acc = {acc:.3} / f1 = {f1:.3}")
    matriz_confusao = figura_matriz_confusao_lstm(lstm, val_x, val_y)
    save_plot(matriz_confusao, 'plots/lstm_eleicoes.jpg')
