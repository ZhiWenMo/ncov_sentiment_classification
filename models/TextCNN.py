import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, GlobalAveragePooling1D, concatenate


class TextCNN(tf.keras.layers.Layer):
    def __init__(self, kernel_sizes=(8, 16, 32), filter_size=256, strides=1, activation='elu', dropout_rate=0.15,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_sizes
        self.filter_size = filter_size
        self.strides = strides
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.convs = [Conv1D(filters=self.filter_size, kernel_size=kernel_size, strides=self.strides,
                             padding='valid', activation=self.activation)
                      for kernel_size in self.kernel_size]
        self.dropout = Dropout(self.dropout_rate)
        self.globalavgpooling = GlobalAveragePooling1D()

    def call(self, inputs):
        grams = []

        for conv in self.convs:
            gram = conv(inputs)
            gram = self.globalavgpooling(gram)
            gram = self.dropout(gram)
            grams.append(gram)

        out = concatenate(grams, axis=-1)

        return out

    def compute_output_shape(self, input_shape):
        return (None, len(self.kernel_size) * self.filter_size)

