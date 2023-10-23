import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

class CustomResNet50:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = self.build_model()

    def load_pretrained_model(self):
        pretrained_model = tf.keras.applications.ResNet50(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet',
            pooling='max'
        )
        for layer in pretrained_model.layers:
            layer.trainable = False
        return pretrained_model

    def augmentation_layer(self):
        augment = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.15),
            layers.experimental.preprocessing.RandomZoom(0.15),
            layers.experimental.preprocessing.RandomContrast(0.15),
        ], name='AugmentationLayer')
        return augment

    def build_model(self):
        pretrained_model = self.load_pretrained_model()
        augment = self.augmentation_layer()

        inputs = layers.Input(shape=self.input_shape, name='inputLayer')
        x = augment(inputs)
        pretrain_out = pretrained_model(x, training=False)
        x = layers.Dense(256)(pretrain_out)
        x = layers.Activation(activation="relu")(x) 
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.45)(x)
        x = layers.Dense(self.num_classes)(x)
        outputs = layers.Activation(activation="softmax", dtype=tf.float32, name='activationLayer')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def get_model(self):
        return self.model

    def summary(self):
        return self.model.summary()


