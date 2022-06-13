import pandas as pd

def save_data(data,path):
    data.to_csv(path,index=False)

def save_plot(fig, path):
    fig.savefig(path)