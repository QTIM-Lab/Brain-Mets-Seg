"""Custom LR scheduler that includes a learning rate warmup."""

import tensorflow as tf

class PolynomialDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    #
    def __init__(self, initial_learning_rate, factor_decrease_learning_rate, total_steps, warmup_iterations=0, power=2.5):
        self.initial_learning_rate = initial_learning_rate
        self.end_learning_rate = initial_learning_rate / factor_decrease_learning_rate
        self.total_steps = total_steps
        self.total_steps_without_warmup = total_steps - warmup_iterations
        self.warmup_iterations = warmup_iterations
        self.power = power
        #convert to tensors
        self.initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate)
        self.dtype = self.initial_learning_rate.dtype
        self.end_learning_rate = tf.cast(self.end_learning_rate, self.dtype)
        self.total_steps = tf.cast(self.total_steps, self.dtype)
        self.total_steps_without_warmup = tf.cast(self.total_steps_without_warmup, self.dtype)
        self.warmup_iterations = tf.cast(self.warmup_iterations, self.dtype)
        self.power = tf.cast(self.power, self.dtype)
    #
    def __call__(self, step):
        float_step = tf.cast(step, self.dtype)
        return tf.cond(float_step < self.warmup_iterations, lambda: self._warmup_function(float_step), lambda: self._poly_function(float_step))
    #
    def _warmup_function(self, float_step):
        return self.initial_learning_rate * (float_step / self.warmup_iterations)
    #
    def _poly_function(self, float_step):
        current_position = tf.math.minimum((float_step - self.warmup_iterations) / self.total_steps_without_warmup, 1.0)
        return self.end_learning_rate + (self.initial_learning_rate - self.end_learning_rate) * ((1 - current_position) ** self.power)

