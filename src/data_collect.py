import pandas as pd
def data_collect():
    df = pd.read_json('..\data\corpusTT.json', orient = 'index')
    return df