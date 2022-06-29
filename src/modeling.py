import pickle
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

def modeling(X_train, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    with open("models/gnb.pkl", "wb") as file:
        pickle.dump(obj=gnb, file=file)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    with open("models/rfc.pkl", "wb") as file:
        pickle.dump(obj=rfc, file=file)
    
    param_dist = {"learning_rate": uniform(0, 2),
              "gamma": uniform(1, 0.000001),
              "max_depth": range(1,50),
              "n_estimators": range(1,300),
              "min_child_weight": range(1,10),
              'n_jobs': range(1,5)}
    
    rs = RandomizedSearchCV(XGBClassifier(), param_distributions=param_dist, n_iter=3)

    xgb  = Pipeline([
    ('model', rs)
        ])

    xgb.fit(X_train, y_train)

    with open("models/xgb.pkl", "wb") as file:
        pickle.dump(obj=xgb, file=file)       

    return gnb, rfc, xgb

def modeling_lstm(X_train,y_train):
    embed_dim = 128
    lstm_out = 256
    max_features = 3000

    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X_train.shape[1]))
    model.add(SpatialDropout1D(0.15))
    model.add(LSTM(lstm_out, dropout=0.15, recurrent_dropout=0.15))
    model.add(Dense(2,activation='sigmoid'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    batch_size = 32
    model.fit(X_train, y_train, epochs = 20, batch_size=batch_size, verbose = 2, shuffle=False)

    path = 'models/lstm.h5'
    model.save(path)

    return model