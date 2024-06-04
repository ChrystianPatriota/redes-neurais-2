import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# Função para criar os dados de entrada/saída
def generate_data(n):
    return np.sqrt(1 + np.sin(n + np.square(np.sin(n))))


# Gerando dados
points = 1000
n = np.arange(points)
x = generate_data(n)

# Dividindo os dados em treino e teste
split = int(0.8 * points)
x_train, y_train = n[:split], x[:split]
x_test, y_test = n[split:], x[split:]

# Criando a rede NARX
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation="linear")
])

# Compilando o modelo
model.compile(optimizer='adam', loss='mse')

# Treinando o modelo
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Fazendo previsões
predictions = model.predict(x_test)

# Visualizando os resultados
import matplotlib.pyplot as plt

plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.scatter(x_test, predictions, color='red', label='Predicted')
plt.legend()
plt.show()
