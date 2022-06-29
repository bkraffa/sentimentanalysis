import pandas as pd


def data_collect_treino():
    ttsbr = pd.read_json('..\data\corpusTT.json', orient='index')
    ttsbr_treino = ttsbr.loc[ttsbr['group'] == 'train',
                            ['text', 'sentiment']].reset_index(drop=True)
    return ttsbr_treino


def data_collect_teste():
    ttsbr = pd.read_json('..\data\corpusTT.json', orient='index')
    ttsbr_teste = ttsbr.loc[ttsbr['group'] == 'test',
                            ['text', 'sentiment']].reset_index(drop=True)
    return ttsbr_teste


''' Dataset utilizado para o treino 1: 
  @InProceedings{BRUM18.389,
	  author = {Henrico Brum and Maria das Gra\c{c}as Volpe Nunes},
	  title = "{Building a Sentiment Corpus of Tweets in Brazilian Portuguese}",
	  booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
	  year = {2018},
	  month = {May 7-12, 2018},
	  address = {Miyazaki, Japan},
	  editor = {Nicoletta Calzolari (Conference chair) and Khalid Choukri and Christopher Cieri and Thierry Declerck and Sara Goggi and Koiti Hasida and Hitoshi Isahara and Bente Maegaard and Joseph Mariani and HÚlŔne Mazo and Asuncion Moreno and Jan Odijk and Stelios Piperidis and Takenobu Tokunaga},
	  publisher = {European Language Resources Association (ELRA)},
	  isbn = {979-10-95546-00-9},
	  language = {english}
	}
'''
