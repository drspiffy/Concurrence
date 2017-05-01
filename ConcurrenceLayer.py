from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints

class Concurrence(Layer):
    def __init__(self, kernel_initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        super(Concurrence, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        input_dim = input_shape[2]
        self.kernel = self.add_weight(shape=(input_dim, 1),
                                 initializer=self.kernel_initializer,
                                 name='kernel',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        super(Concurrence, self).build(input_shape)

    def call(self, x, mask=None):
        focus = K.softmax(K.squeeze(K.dot(x, self.kernel), 2))
        return K.squeeze(K.batch_dot(x, K.expand_dims(focus, axis=-1), axes=[1, 1]), 2)

    def get_config(self):
        config = {'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint)}
        base_config = super(Concurrence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
