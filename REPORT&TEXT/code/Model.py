model = Sequential([

    Conv2D(64, kernel_size=3, padding="same", activation="relu", input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),

    Conv2D(128, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),

    Conv2D(128, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    UpSampling2D(size=2),

    Conv2D(64, kernel_size=3, padding="same", activation="relu"),
    BatchNormalization(),
    UpSampling2D(size=2),

    Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")
])

model.compile(optimizer=SGD(learning_rate=0.01), loss="mse", metrics=["mae"])
model.summary()