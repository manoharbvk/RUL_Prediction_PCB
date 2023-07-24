
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the Lorenz system equations


def lorenz(x, y, z, sigma=10, rho=28, beta=8/3):
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return x_dot, y_dot, z_dot


# Define the time steps
dt = 0.01
t = np.arange(0, 20, dt)

# Define the initial conditions
x0 = np.array([0.1, 0.1, 0.1])

# Simulate the Lorenz system
x = np.zeros((len(t), 3))
x[0, :] = x0
for i in range(1, len(t)):
    x_dot = lorenz(x[i-1, 0], x[i-1, 1], x[i-1, 2])
    x[i, :] = x[i-1, :] + x_dot * dt

# Split the data into training and validation sets
train_data = x[:int(0.8 * len(t)), :]
val_data = x[int(0.8 * len(t)):, :]

# Normalize the data to have zero mean and unit variance
mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
train_data = (train_data - mean) / std
val_data = (val_data - mean) / std

# Function to generate input-output pairs for the RNN model


def generate_data(data, seq_length):
    inputs = []
    outputs = []
    for i in range(seq_length, len(data)):
        inputs.append(np.hstack((data[i-seq_length:i, :], i*dt)))
        outputs.append(data[i, :])
    return np.array(inputs), np.array(outputs)


# Set the sequence length for the input data
seq_length = 50

# Generate input-output pairs for the training and validation sets
train_inputs, train_outputs = generate_data(train_data, seq_length)
val_inputs, val_outputs = generate_data(val_data, seq_length)

# Define the Koopman Operator function using the EDMD algorithm


def compute_koopman_operator(data, d=3, k=20):


    # Compute the data matrices
    X = np.zeros((len(data)-d, d))
    Y = np.zeros((len(data)-d, d))
    for i in range(d):
        X[:, i] = data[i:len(data)-d+i, 0]
        Y[:, i] = data[i+1:len(data)-d+i+1, 0]
# Compute the SVD of the data matrices
    Ux, Sx, Vx = np.linalg.svd(X)
    Uy, Sy, Vy = np.linalg.svd(Y)
# Truncate the SVD matrices
    Ux = Ux[:, :k]
    Sx = Sx[:k]
    Vx = Vx[:k, :]
    Uy = Uy[:, :k]
    Sy = Sy[:k]
    Vy = Vy[:k, :]
# Compute the extended observables matrix
    psi_x = Ux @ np.diag(np.sqrt(Sx))
    psi_y = Uy @ np.diag(np.sqrt(Sy))
    phi = Y @ Vx.T @ np.linalg.inv(np.diag(Sx)) @ psi_x
# Compute the Koopman operator
    K = np.linalg.pinv(psi_x) @ phi
    return K
# Compute the Koopman operator for the training data
K = compute_koopman_operator(train_data)

# Define the input shape and the number of output features for the RNN model
input_shape = (seq_length, 4)
num_outputs = 3

# Define the RNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=64, input_shape=input_shape),
    tf.keras.layers.Dense(units=num_outputs)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(train_inputs, train_outputs, validation_data=(
    val_inputs, val_outputs), epochs=100, batch_size=32)

# Evaluate the model on the validation data
loss = model.evaluate(val_inputs, val_outputs)

# Generate predictions for the validation data
predictions = model.predict(val_inputs)

# Plot the predictions against the true values
fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
for i in range(num_outputs):
    axs[i].plot(val_outputs[:, i], label='True')
    axs[i].plot(predictions[:, i], label='Predicted')
    axs[i].legend()
    axs[i].set_ylabel(f'x_{i+1}')
    axs[num_outputs-1].set_xlabel('Time step')
plt.show()




























import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('data.csv', index_col=0)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Split data into training and test sets
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Define model architecture
window_size = 10

# Model 1
model1 = Sequential()
model1.add(LSTM(100, activation='relu', input_shape=(window_size, 1)))
model1.add(Dense(1))
model1.compile(optimizer='adam', loss='mse')

# Model 2
model2 = Sequential()
model2.add(LSTM(50, activation='relu', input_shape=(window_size, 1), return_sequences=True))
model2.add(LSTM(50, activation='relu'))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mse')

# Model 3
model3 = Sequential()
model3.add(LSTM(100, activation='relu', input_shape=(window_size, 1), return_sequences=True))
model3.add(LSTM(50, activation='relu', return_sequences=True))
model3.add(LSTM(50, activation='relu'))
model3.add(Dense(1))
model3.compile(optimizer='adam', loss='mse')

# Train models
X_train, y_train = [], []
for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model1.fit(X_train, y_train, epochs=100, batch_size=32)
model2.fit(X_train, y_train, epochs=100, batch_size=32)
model3.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate models on test data
X_test, y_test = [], []
for i in range(window_size, len(test_data)):
    X_test.append(test_data[i-window_size:i, 0])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)

rmse1 = np.sqrt(np.mean((y_test - y_pred1)**2))
rmse2 = np.sqrt(np.mean((y_test - y_pred2)**2))
rmse3 = np.sqrt(np.mean((y_test - y_pred3)**2))

mae1 = np.mean(np.abs(y_test - y_pred1))
mae2 = np.mean(np.abs(y_test - y_pred2))
mae3 = np.mean(np.abs(y_test - y_pred3))

r2_1 = 1 - np.sum((y_test - y_pred1)**2) / np.sum((y_test - np.mean(y_test))**2)
r2_2 = 1 - np.sum((y_test - y_pred2)**2) / np.sum






















