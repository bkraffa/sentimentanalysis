import pandas as pd

def read_eleicoes_treino():
    df = pd.read_excel('..\data\eleicoes22_treino.xlsx', index_col=False)
    df.rename(columns = {'Sentimento':'sentiment', 'tweet':'text'}, inplace = True)
    return df 

def read_eleicoes_teste():
    df = pd.read_excel('data\eleicoes22_teste.xlsx', index_col=False)
    df.rename(columns = {'sentimento':'sentiment', 'tweet':'text'}, inplace=True)
    return df 