from keras.layers.core import Layer
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.callbacks import History

MAX_LR = 1e-3
MIN_LR = 1e-5
loss_prev = 1e10

class LRN(Layer):
    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, X, mask=None):
        b, ch, r, c = X._keras_shape
        half_n = self.n // 2
        input_sqr = K.square(X)
        extra_channels = tf.zeros_like(X)  # make an empty tensor with zero pads along channel dimension
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :], input_sqr, extra_channels[:, half_n:2*half_n, :, :]], axis = 1)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
            scale = scale ** self.beta
        return X / scale

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PoolHelper(Layer):
    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :, 1:, 1:]

    def get_config(self):
        config = {}
        base_config = super(PoolHelper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LossLearningRateScheduler(History):
    """
    A learning rate scheduler that relies on changes in loss function
    value to dictate whether learning rate is decayed or not.
    LossLearningRateScheduler has the following properties:
    base_lr: the starting learning rate
    lookback_epochs: the number of epochs in the past to compare with the loss function at the current epoch to determine if progress is being made.
    decay_threshold / decay_multiple: if loss function has not improved by a factor of decay_threshold * lookback_epochs, then decay_multiple will be applied to the learning rate.
    spike_epochs: list of the epoch numbers where you want to spike the learning rate.
    spike_multiple: the multiple applied to the current learning rate for a spike.
    """

    def __init__(self, spike_epochs=None, spike_multiple=10, decay_threshold=0.002, decay_multiple=0.5):

        super(LossLearningRateScheduler, self).__init__()

        self.base_lr = MAX_LR
        self.spike_epochs = spike_epochs
        self.spike_multiple = spike_multiple
        self.decay_threshold = decay_threshold
        self.decay_multiple = decay_multiple

    def on_train_begin(self, logs={}):
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        global loss_prev
        self.logs.append(logs)

        current_lr = K.get_value(self.model.optimizer.lr)

        target_loss = logs.get('loss')

        loss_diff = np.abs(loss_prev - target_loss)
        
        loss_prev = target_loss

        if loss_diff <= np.abs(loss_prev) * (self.decay_threshold):

            print(' '.join(
                ('Changing learning rate from', str(current_lr), 'to', str(current_lr * self.decay_multiple))))
            K.set_value(self.model.optimizer.lr, current_lr * self.decay_multiple)
            current_lr = current_lr * self.decay_multiple

        else:

            print(' '.join(('Learning rate:', str(current_lr))))

        if self.spike_epochs is not None and len(self.epoch) in self.spike_epochs:
            print(' '.join(
                ('Spiking learning rate from', str(current_lr), 'to', str(current_lr * self.spike_multiple))))
            K.set_value(self.model.optimizer.lr, current_lr * self.spike_multiple)

        return K.get_value(self.model.optimizer.lr)
