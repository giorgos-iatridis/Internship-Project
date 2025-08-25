import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

import tensorflow as tf

np.random.seed(145)

#Parameters
batch_size = 64

#Load train and validation data
with open('train_val_all.p', 'rb') as f:
    X_train, X_val, y_train, y_val = pickle.load(f)

with open('test_all.p', 'rb') as f:
    X_test, y_test = pickle.load(f)

#Scaling
scaler_train = MinMaxScaler()
X_train = scaler_train.fit_transform(X_train)
X_val = scaler_train.transform(X_val)

with open('scaler.p', 'wb') as f:
    pickle.dump(scaler_train, f)

#Model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(10, activation='sigmoid', batch_input_shape = (None, X_train.shape[1])),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#Training parameters
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose=1)
#apothikeusi modelou meta apo kathe epoch
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.p', monitor='val_loss', save_best_only=True, verbose=1)
#Train - Validate model
history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), batch_size = batch_size, epochs=20, shuffle=True, verbose=2, callbacks=[checkpointer])

X_test = scaler_train.transform(X_test)

res= model.evaluate(x=X_test, y=y_test, batch_size=64, verbose=1)
print(res)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss over epochs')
plt.legend(['train','validation'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy over epochs')
plt.legend(['train','validation'])
plt.show()