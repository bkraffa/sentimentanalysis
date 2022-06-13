from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, pos_label=1)

def evaluate_lstm(model, X_test,Y_test):

    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    resultados = []

    for x in range(len(X_test)):
    
        result = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 3)[0]
    
        if np.argmax(result) == np.argmax(X_test[x]):
            if np.argmax(X_test[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1
        
        if np.argmax(X_test[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1
        
        resultados.append(np.argmax(result))
    
    Y_test_argmax = np.argmax(Y_test,axis=1)
    Y_test_argmax  = Y_test_argmax.reshape(-1,1)
    resultados = np.asarray(resultados)
    resultados  = resultados.reshape(-1,1)
 
    return accuracy_score(Y_test_argmax, resultados), f1_score(Y_test_argmax, resultados, pos_label=1)

def figura_matriz_confusao(model, X_test, y_test):
    Y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, Y_pred)
    fig = plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap = 'GnBu');
    model_name = type(model).__name__
    plt.title(f"Matriz de confusão {model_name}")
    plt.ylabel('Classe Correta')
    plt.xlabel('Classe Prevista')
    return fig

def figura_matriz_confusao_lstm(model, X_test, y_test):

    predict_x=model.predict(X_test) 
    classes_x=np.argmax(predict_x,axis=1)
    y_test_max=np.argmax(y_test,axis=1) 
    conf_matrix = confusion_matrix(y_test_max, classes_x)
    fig = plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap = 'GnBu');
    model_name = type(model).__name__
    plt.title(f"Matriz de confusão {model_name}")
    plt.ylabel('Classe Correta')
    plt.xlabel('Classe Prevista')
    return fig