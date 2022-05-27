from data_collect import data_collect
from preprocessing import preprocessing, separa_datasets, tokenizacao, train_test_split
from spacy_load import load_spacy
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
    #Tokenizacao
    X = tokenizacao(df)
    #Atribuindo a Y a nossa variável alvo
    Y = (df['sentiment']).values
    #Dividindo entre treino e teste
    X_train, X_test, Y_train, Y_test = separa_datasets(X,Y)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)