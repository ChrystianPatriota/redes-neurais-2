import numpy as np
from keras import Input, Model
from keras.src.layers import Dense
from matplotlib import pyplot as plt

mean1 = [0, 0, 0, 0, 0, 0, 0, 0]
mean2 = [4, 0, 0, 0, 0, 0, 0, 0]
mean3 = [0, 0, 0, 4, 0, 0, 0, 0]
mean4 = [0, 0, 0, 0, 0, 0, 0, 4]

samples = 10000

c1 = np.random.normal(mean1, 1, size=(samples, 8))
c2 = np.random.normal(mean2, 1, size=(samples, 8))
c3 = np.random.normal(mean3, 1, size=(samples, 8))
c4 = np.random.normal(mean4, 1, size=(samples, 8))

X = np.vstack((c1, c2, c3, c4))

input_data = Input(shape=(8,))
encoded = Dense(2, activation='linear')(input_data)
decoded = Dense(8, activation='linear')(encoded)

autoencoder = Model(input_data, decoded)

# Compilando o modelo
autoencoder.compile(optimizer='adam', loss="mean_squared_error")

# Treinando o modelo
autoencoder.fit(X, X, epochs=5, batch_size=4, validation_split=0.1)

encoder = Model(input_data, encoded)
encoded_data_m1 = encoder.predict(c1)
encoded_data_m2 = encoder.predict(c2)
encoded_data_m3 = encoder.predict(c3)
encoded_data_m4 = encoder.predict(c4)

print(np.mean(encoded_data_m1, axis=0))
print(np.var(encoded_data_m1, axis=0))
print(np.mean(encoded_data_m2, axis=0))
print(np.var(encoded_data_m2, axis=0))
print(np.mean(encoded_data_m3, axis=0))
print(np.var(encoded_data_m3, axis=0))
print(np.mean(encoded_data_m4, axis=0))
print(np.var(encoded_data_m4, axis=0))


plt.scatter(encoded_data_m2[:, 0], encoded_data_m2[:, 1], c="yellow")
plt.scatter(encoded_data_m3[:, 0], encoded_data_m3[:, 1], c="green")
plt.scatter(encoded_data_m4[:, 0], encoded_data_m4[:, 1], c="red")
plt.scatter(encoded_data_m1[:, 0], encoded_data_m1[:, 1])
plt.title('Dados Reduzidos usando Autoencoder')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()