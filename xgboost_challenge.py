# Importação de bibliotecas
import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Carregar e Preparar os Dados ---
print("1. Carregando e preparando o dataset de câncer de mama...")

# Carrega o dataset de exemplo do scikit-learn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Divide os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cria DMatrix, o formato de dados otimizado para XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# --- 2. Definir Parâmetros do Modelo ---
print("2. Definindo parâmetros e treinando o modelo...")

# Parâmetros otimizados para classificação binária
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,  # Taxa de aprendizado
    'max_depth': 4, # Profundidade máxima da árvore
    'seed': 42
}

# Número de rodadas de boosting (iterações)
num_round = 100

# --- 3. Treinar o Modelo ---
start_time = time.time()
bst = xgb.train(params, dtrain, num_round)
end_time = time.time()

training_time = end_time - start_time
print(f"   -> Treinamento concluído em {training_time:.2f} segundos.")

# --- 4. Avaliar o Modelo ---
print("4. Avaliando o desempenho do modelo...")

# Previsão das probabilidades no conjunto de teste
y_pred_proba = bst.predict(dtest)

# Converte probabilidades em classes binárias (0 ou 1)
y_pred = (y_pred_proba > 0.5).astype(int)

# Calcula as métricas
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\n--- Resultados da Classificação XGBoost ---")
print(f"Acurácia (Accuracy): {accuracy * 100:.2f}%")
print("-" * 40)
print("Matriz de Confusão:")
print(conf_matrix)
print("-" * 40)
print("Relatório de Classificação:")
print(class_report)
print("-" * 40)
print(f"Tempo total de execução: {training_time:.2f}s")