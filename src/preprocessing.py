import re
from spacy_load import load_spacy

#Carregando o spacy
nlp = load_spacy()

def preprocessing(df):
    #Removendo os sentimentos neutros e não anotados
    df.drop(df.loc[df['sentiment'] == '0'].index, inplace = True)
    df.drop(df.loc[df['sentiment'] == '-'].index, inplace = True)
    #Transformando todas palavras pra minúsculo
    df['text'] = df['text'].apply(lambda x: x.lower())
    #Limpando os tweets    
    df['text'] = df['text'].apply((lambda x: re.sub('[^\w\s]','',x)))
    #Removendo stopwords
    df["text"] = df['text'].apply(lambda x: " ".join([y.lemma_ for y in nlp(x) if not y.is_stop]))
    #Transformando os sentiments -1(negativos) em 0
    df.loc[df['sentiment'] == '-1','sentiment'] = 0
    #Convertendo de string pra int
    df['sentiment'] = df.sentiment.astype(int)
    #Simplificando o df pra termos só as features de interesse
    df = df[['text','sentiment']]
    df.reset_index(drop=True, inplace=True)
    return df


