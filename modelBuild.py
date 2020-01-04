import tensorflow as tf

class myModel():
    def getModel():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5,5), activation="relu", input_shape=(28,28,1)),
            tf.keras.layers.Conv2D(32, (5,5), activation="relu"),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(.2),

            tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
            tf.keras.layers.Conv2D(128, (2,2), activation="relu"),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Dropout(.15),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])

        model.summary()

        optimizer = tf.keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)

        model.compile(optimize=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])

        return model
