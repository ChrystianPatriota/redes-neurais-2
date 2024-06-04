import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense

dataset = pd.read_csv('../assets/Bitcoin Historical Data.csv')
dataset["Open"] = dataset["Open"].str.replace(",","")
dataset["Open"] = dataset["Open"].astype(float)
time_series_data = dataset["Open"].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(time_series_data).reshape(-1, 1))


# Function to create dataset with look back
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# Create the dataset with look back
look_back = 1  # You can adjust this value based on your data
X, Y = create_dataset(scaled_data, look_back)

# Reshape input data to be 3D [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split the data into training and testing sets
train_size = int(len(X) * 0.7)  # 70% training data
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions to original scale
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# Plotting
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(np.arange(look_back, look_back + len(train_predict)), train_predict,
         label='Training Predictions')
plt.plot(
    np.arange(look_back + len(train_predict), len(train_predict) + len(test_predict) + look_back),
    test_predict, label='Testing Predictions')
plt.legend()
plt.show()
