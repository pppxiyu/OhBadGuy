import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train(TrainX, TrainY, ValX, ValY, epochs = 12, batch_size = 12):
    inputs = keras.layers.Input(shape = (TrainX.shape[1], TrainX.shape[2], TrainX.shape[3], TrainX.shape[4]))
    x = keras.layers.ConvLSTM2D(20, 3, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x) 
    outputs = keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    model.summary()

    # Model training
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Fit the model to the training data
    model.fit(
        TrainX,
        TrainY,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(ValX, ValY),
        callbacks=[early_stopping, reduce_lr],)
        
    return model

if __name__ == '__main__':
    model = train()