"""Custom SGDW optimizer implementation."""

import re
import tensorflow as tf

class SGDW(tf.keras.optimizers.experimental.Optimizer):
    r"""Optimizer that implements the SGDW algorithm.

    Args:
      learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.01.
      weight_decay: A `tf.Tensor`, floating point value. The weight decay.
        factor to be applied to parameters. Defaults to 0.00003.
      momentum: float hyperparameter >= 0 that accelerates gradient descent in
        the relevant direction and dampens oscillations. Defaults to 0.99, i.e.,
        vanilla gradient descent.
      dampening: float hyperparameter >= 0 to determine how much to dampen 
        momentum with current gradient update. Defaults to 0.0, i.e., entire 
        current gradient added to momentum variable, though it could be prudent
        to set this equal to momentum (similar to how ADAM works).
      nesterov: boolean. Whether to apply Nesterov momentum. Defaults to True.
      decoupled: boolean. Whether weight decay should be applied in a decoupled
        manner. Defaults to False.
    
    Notes:
      This follows the pytorch implementation wherein there is a dampening factor
      on the momentum step as opposed to a direct multiplication with the learning
      rate. Weight decay can be added normally or in a decoupled manner.
    """

    def __init__(
        self,
        learning_rate=0.01,
        weight_decay=0.00003,
        momentum=0.99,
        dampening=0.0,
        nesterov=True,
        decoupled=False,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.999,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="sgdw",
        **kwargs
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.decoupled = decoupled
        #
        if isinstance(learning_rate, (int, float)) and (
            learning_rate < 0
        ):
            raise ValueError("`learning_rate` must be between [0, inf].")
        #
        if isinstance(momentum, (int, float)) and (
            momentum < 0 or momentum > 1
        ):
            raise ValueError("`momentum` must be between [0, 1].")
        #
        if isinstance(dampening, (int, float)) and (
            dampening < 0 or dampening > 1
        ):
            raise ValueError("`dampening` must be between [0, 1].")
        #
        if isinstance(weight_decay, (int, float)) and (
            weight_decay < 0
        ):
            raise ValueError("`weight_decay` must be between [0, inf].")
    
    def build(self, var_list):
        """Initialize optimizer variables.

        SGDW optimizer has one variable `momentums`, only set if `self.momentum`
        is not 0.

        Args:
          var_list: list of model variables to build SGDW variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        if self.momentum != 0:
            for var in var_list:
                self.momentums.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="m"
                    )
                )
        self._built = True

    def _use_weight_decay(self, variable):
        exclude_from_weight_decay = getattr(
            self, "_exclude_from_weight_decay", []
        )
        exclude_from_weight_decay_names = getattr(
            self, "_exclude_from_weight_decay_names", []
        )
        if variable in exclude_from_weight_decay:
            return False
        for name in exclude_from_weight_decay_names:
            if re.search(name, variable.name) is not None:
                return False
        return True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        wd = tf.cast(self.weight_decay, variable.dtype)
        damp = tf.cast(self.dampening, variable.dtype)
        m = None
        var_key = self._var_key(variable)
        if self.momentum != 0:
            momentum = tf.cast(self.momentum, variable.dtype)
            m = self.momentums[self._index_dict[var_key]]
        # Dense gradients
        #add weight decay early if not using decoupled formulation
        if self.decoupled == False and self._use_weight_decay(variable):
            gradient_buffer = gradient + (variable * wd)
        else:
            gradient_buffer = gradient
        #check if momentum is being used
        if m is not None:
            m.assign((m * momentum) + (gradient_buffer * (1.0 - damp)))
            #check if nesterov momentum is being used
            if self.nesterov == True:
                gradient_buffer = gradient_buffer + (m * momentum)
            else:
                gradient_buffer = m
        #update parameter values
        variable.assign(variable - (gradient_buffer * lr))
        #check if decoupled weight decay is being used
        if self.decoupled == True and self._use_weight_decay(variable):
            variable.assign(variable - (variable * wd * lr))
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
                "dampening": self.dampening,
                "nesterov": self.nesterov,
                "decoupled": self.decoupled,
            }
        )
        return config

    def exclude_from_weight_decay(self, var_list=None, var_names=None):
        """Exclude variables from weight decays.

        This method must be called before the optimizer's `build` method is
        called. You can set specific variables to exclude out, or set a list of
        strings as the anchor words, if any of which appear in a variable's
        name, then the variable is excluded.

        Args:
            var_list: A list of `tf.Variable`s to exclude from weight decay.
            var_names: A list of strings. If any string in `var_names` appear
                in the model variable's name, then this model variable is
                excluded from weight decay. For example, `var_names=['bias']`
                excludes all bias variables from weight decay.
        """
        if hasattr(self, "_built") and self._built:
            raise ValueError(
                "`exclude_from_weight_decay()` can only be configued before "
                "the optimizer is built."
            )

        self._exclude_from_weight_decay = var_list or []
        self._exclude_from_weight_decay_names = var_names or []
