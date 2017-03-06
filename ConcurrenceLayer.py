from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import initializations
from keras import regularizers, constraints


class Concurrence(Layer):
    def __init__(self, weights=None, init='glorot_uniform',
                 W_regularizer=None, activity_regularizer=None, W_constraint=None, **kwargs):
        self.init = initializations.get(init)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.initial_weights = weights
        super(Concurrence, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]

        self.W = self.add_weight((self.input_dim, 1),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(Concurrence, self).build(input_shape)

    def call(self, x, mask=None):
        focus = K.softmax(K.squeeze(K.dot(x, self.W), 2))
        return K.batch_dot(x, focus, (1, 1))

    def get_config(self):
        config = {'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'input_dim': self.input_dim,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None}
        base_config = super(Concurrence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[2]
