import numpy as np
import pandas as pd
import joblib
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import json
import os


# Parâmetros para a construção 
PARAM_DT = {
    'criterion': ['squared_error'],  # Mede o erro quadrático médio
    'splitter': ['best'],  # Escolhe a melhor divisão dos dados
    'max_depth': [16],  # Define a profundidade máxima da árvore
    'min_samples_split': [2],  # Mínimo de amostras para dividir um nó
    'min_samples_leaf': [2],  # Mínimo de amostras por folha
    'random_state': [42],  # Garante reprodutibilidade
    'max_leaf_nodes': [100, 200],  # Número máximo de folhas na árvore
}

def import_files_txt(file_path, header=None):
    df = pd.read_csv(file_path, delimiter=",", header=header)
    print(f"Arquivo carregado com sucesso: {file_path}")
    return df

def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(random_state=42)  # Inicializa a árvore
    
    clf = GridSearchCV(model, PARAM_DT, cv=3)  # Busca os melhores hiperparâmetros
    clf.fit(X_train, y_train)  # Treina a árvore com os dados de treino

    best_model = clf.best_estimator_  # Modelo com melhor configuração encontrada
    y_pred = best_model.predict(X_test)  # Faz previsões nos dados de teste
    mse = mean_squared_error(y_test, y_pred)  # Calcula o erro quadrático médio

    print("Best model MSE:", mse)

    train_accuracy = best_model.score(X_train, y_train)  # Precisão nos dados de treino
    test_accuracy = best_model.score(X_test, y_test)  # Precisão nos dados de teste

    return clf, best_model, mse, train_accuracy, test_accuracy

