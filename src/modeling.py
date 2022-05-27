import pickle
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

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
