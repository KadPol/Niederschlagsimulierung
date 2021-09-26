import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def kompilierung(model: tf.keras.Model, inputs: np.ndarray, targets: np.ndarray,
                    validation: tuple, batch_size: int, epochs: int):
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.losses.MeanSquaredError(), tf.metrics.MeanAbsoluteError()])

    history = model.fit(x=inputs, y=targets,
                        validation_data=validation,
                        batch_size=batch_size,
                        epochs=epochs)
    return history

def loss(history: tf.keras.callbacks.History):
    plt.figure(figsize=(8, 5), dpi=100)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoche')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

def model_LSTM(days,features):
    return tf.keras.Sequential([
    tf.keras.layers.LSTM(50,  input_shape=(days,features), return_sequences=True),
    tf.keras.layers.LSTM(30, return_sequences=True),
    tf.keras.layers.LSTM(10),
    tf.keras.layers.Dense(1)
])

def model_CNN_LSTM(days,x_shape,y_shape,features):
    return tf.keras.Sequential([
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(64,(3,3),strides=2,input_shape=(days,x_shape,y_shape,features))),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(4)),
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32,(2,2),strides=2)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
    tf.keras.layers.LSTM(units=30,dropout=0.2,return_sequences=True),
    tf.keras.layers.LSTM(units=10,dropout=0.1),
    tf.keras.layers.Dense(units=1)
])