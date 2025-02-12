import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# Função para carregar os dados
def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    return data

# Carregamento dos dados
data_train = load_data('4000vit.txt')
data_test = load_data('800vit.txt')

# Preparação dos dados
X_train = data_train[:, 3:6]  # qPA, pulso, frequencia respiratoria
y_train = data_train[:, -1]   # Valor contínuo (não subtrair 1)

# Divisão dos dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Construção do modelo de árvore de regressão
model = DecisionTreeRegressor(random_state=42)

# Treinamento do modelo
model.fit(X_train, y_train)

# Validação
val_predictions = model.predict(X_val)
mse_val = mean_squared_error(y_val, val_predictions)
mae_val = mean_absolute_error(y_val, val_predictions)

print("Validação (com 4.000 vítimas):")
print(f"MSE: {mse_val}, MAE: {mae_val}")

# Teste
X_test = data_test[:, 3:6]
y_test = data_test[:, -1]  # Valor contínuo (não subtrair 1)

test_predictions = model.predict(X_test)
mse_test = mean_squared_error(y_test, test_predictions)
mae_test = mean_absolute_error(y_test, test_predictions)

print("Teste (com 800 vítimas):")
print(f"MSE: {mse_test}, MAE: {mae_test}")

# Salvamento do modelo (opcional)
import joblib
joblib.dump(model, 'modelo_arvore_regressor.pkl')
print("Modelo salvo com sucesso!")