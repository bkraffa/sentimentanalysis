from data_collect import data_collect
from preprocessing import preprocessing
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

    print(df.head(10))