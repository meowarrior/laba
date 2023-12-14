import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense


data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_y_train = encoder.transform(y_train)
encoded_y_test = encoder.transform(y_test)

hot_encoded_y_train = to_categorical(encoded_y_train)
hot_encoded_y_test = to_categorical(encoded_y_test)


def mse(y_true, y_pred):
    return np.mean(np.square(np.subtract(y_true, y_pred)))


model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

history = model.fit(X_train, hot_encoded_y_train, validation_data=(X_test, hot_encoded_y_test), epochs=60, batch_size=10)

y_pred = model.predict(X_test)
y_pred_encoded = np.argmax(y_pred, axis=1)
y_pred_decoded = encoder.inverse_transform(y_pred_encoded)

mse_score = mse(y_test, y_pred_decoded)
accuracy_score = np.sum(y_test == y_pred_decoded) / len(y_test)
print("MSE: ", mse_score)
print("Accuracy: ", accuracy_score)

model.save('my_model.keras')