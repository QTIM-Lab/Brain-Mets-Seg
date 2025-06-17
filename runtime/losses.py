# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from itertools import product

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, y_one_hot=True, reduce_batch=False, eps=1e-6, include_background=False):
        super().__init__()
        self.y_one_hot = y_one_hot
        self.reduce_batch = reduce_batch
        self.eps = eps
        self.include_background = include_background

    def dice_coef(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred, axis=1)
        pred_sum = tf.reduce_sum(y_pred, axis=1)
        true_sum = tf.reduce_sum(y_true, axis=1)
        dice = (2.0 * intersection + self.eps) / (pred_sum + true_sum + self.eps)
        return tf.reduce_mean(dice, axis=0)

    @tf.function
    def call(self, y_true, y_pred):
        n_class = y_pred.shape[-1]
        if self.reduce_batch:
            flat_shape = (1, -1, n_class)
        else:
            flat_shape = (y_pred.shape[0], -1, n_class)
        if self.y_one_hot:
            y_true = tf.one_hot(y_true, n_class)

        flat_pred = tf.reshape(tf.cast(y_pred, tf.float32), flat_shape)
        flat_true = tf.reshape(y_true, flat_shape)

        dice_coefs = self.dice_coef(flat_true, tf.keras.activations.softmax(flat_pred, axis=-1))
        if not self.include_background:
            dice_coefs = dice_coefs[1:]
        dice_loss = tf.reduce_mean(1 - dice_coefs)

        return dice_loss


class DiceCELoss(tf.keras.losses.Loss):
    def __init__(self, y_one_hot=True, **dice_kwargs):
        super().__init__()
        self.y_one_hot = y_one_hot
        self.dice_loss = DiceLoss(y_one_hot=False, **dice_kwargs)

    @tf.function
    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        n_class = y_pred.shape[-1]
        if self.y_one_hot:
            y_true = tf.one_hot(y_true, n_class)
        dice_loss = self.dice_loss(y_true, y_pred)
        ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_true,
                logits=y_pred,
            )
        )
        return dice_loss + ce_loss


class WeightDecay:
    def __init__(self, factor):
        self.factor = factor

    @tf.function
    def __call__(self, model):
        return self.factor * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if "norm" not in v.name])


class Soft_Dice_Coef_Metric(tf.keras.losses.Loss):
    def __init__(self, smooth = 0.00001, reduce_batch=False, include_background=False, batch_size=2):
        super().__init__()
        self.smooth = smooth
        self.reduce_batch = reduce_batch
        self.include_background = include_background
        n_class = 1 if self.include_background == False else 2
        self.flat_shape = (batch_size, -1, n_class) if self.reduce_batch == False else (1, -1, n_class)
    #
    @tf.function
    def call(self, y_true, y_pred):
        #activate outputs
        y_pred = tf.keras.activations.sigmoid(y_pred)
        #can block gradients for operations acting only on y_true
        if self.include_background == True:
            #one-hot encode input (only if need background dice)
            y_true = tf.stop_gradient(tf.concat([tf.stop_gradient(1.0 - y_true), y_true], axis=-1))
            #one-hot encode outputs (only if need background dice)
            y_pred = tf.concat([1.0 - y_pred, y_pred], axis=-1)
        #reshape input
        y_true_flat = tf.stop_gradient(tf.reshape(y_true, self.flat_shape))
        #reshape output
        y_pred_flat = tf.reshape(y_pred, self.flat_shape)
        #calculate dice metric per class
        true_pred_sum = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        true_sum = tf.reduce_sum(y_true_flat, axis=1)
        pred_sum = tf.reduce_sum(y_pred_flat, axis=1)
        dice_metric_per_class = (2.0 * true_pred_sum + self.smooth) / (true_sum + pred_sum + self.smooth)
        #reduce across batch and class dimension
        return tf.reduce_mean(dice_metric_per_class)

class Soft_Dice_Coef_Loss(tf.keras.losses.Loss):
    def __init__(self,  **dice_kwargs):
        super().__init__()
        self.soft_dice_coef_metric = Soft_Dice_Coef_Metric(**dice_kwargs)
    #
    @tf.function
    def call(self, y_true, y_pred):
        return -self.soft_dice_coef_metric(y_true, y_pred)

class Weighted_Binary_Cross_Entropy_Loss(tf.keras.losses.Loss):
    def __init__(self, positive_class_weight=3.):
        super().__init__()
        self.positive_class_weight = positive_class_weight
        self.weights = tf.constant([[1.,1.],[self.positive_class_weight,self.positive_class_weight]])
    #
    @tf.function
    def call(self, y_true, y_pred):
        #compute un-weighted cross-entropy
        cross_entropy_matrix = tf.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
        #one-hot encode inputs
        y_true_one_hot = tf.concat([tf.subtract(1., y_true), y_true], axis=-1)
        #activate and one-hot encode outputs
        y_pred = tf.cast(y_pred >= 0, tf.float32)
        y_pred_one_hot = tf.concat([tf.subtract(1., y_pred), y_pred], axis=-1)
        #
        final_mask = tf.zeros_like(y_pred_one_hot[...,0])
        #make per-voxel weight mask given what our network predicted
        for (i,j) in product(range(0, 2), range(0, 2)):
            w = self.weights[i][j]
            y_t = y_true_one_hot[...,i]
            y_p = y_pred_one_hot[...,j]
            final_mask = tf.add(final_mask, tf.multiply(w, tf.multiply(y_t, y_p)))
        return tf.reduce_mean(tf.multiply(final_mask, cross_entropy_matrix))

class Joint_Dice_Weighted_BinaryCE_Loss(tf.keras.losses.Loss):
    def __init__(self, **dice_kwargs):
        super().__init__()
        self.soft_dice_coef_loss = Soft_Dice_Coef_Loss(**dice_kwargs)
        self.weighted_binary_cross_entropy_loss = Weighted_Binary_Cross_Entropy_Loss()
    #
    @tf.function
    def call(self, y_true, y_pred):
        #split apart ground truth label and distance transform map
        dist_map = y_true[1]
        y_true = y_true[0]
        #
        #cast to tf.float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        dist_map = tf.cast(dist_map, tf.float32)
        #compute dice loss
        dice_loss = self.soft_dice_coef_loss(y_true, y_pred)
        #compute cross entropy loss
        ce_loss = self.weighted_binary_cross_entropy_loss(y_true, y_pred)
        return dice_loss + ce_loss

class Joint_Dice_BinaryCE_Loss(tf.keras.losses.Loss):
    def __init__(self, **dice_kwargs):
        super().__init__()
        self.soft_dice_coef_loss = Soft_Dice_Coef_Loss(**dice_kwargs)
        self.binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #
    @tf.function
    def call(self, y_true, y_pred):
        #split apart ground truth label and distance transform map
        dist_map = y_true[1]
        y_true = y_true[0]
        #
        #cast to tf.float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        dist_map = tf.cast(dist_map, tf.float32)
        #compute dice loss
        dice_loss = self.soft_dice_coef_loss(y_true, y_pred)
        #compute cross entropy loss
        ce_loss = self.binary_cross_entropy_loss(y_true, y_pred)
        return dice_loss + ce_loss

class Joint_Dice_Boundary_Weighted_CE_Loss(tf.keras.losses.Loss):
    def __init__(self, **dice_kwargs):
        super().__init__()
        self.soft_dice_coef_loss = Soft_Dice_Coef_Loss(**dice_kwargs)
        self.binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
    #
    @tf.function
    def call(self, y_true, y_pred):
        #split apart ground truth label and distance transform map
        dist_map = y_true[1]
        y_true = y_true[0]
        #
        #cast to tf.float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        dist_map = tf.cast(dist_map, tf.float32)
        #compute dice loss
        dice_loss = self.soft_dice_coef_loss(y_true, y_pred)
        #compute cross entropy loss
        ce_loss = tf.reduce_mean(tf.squeeze(dist_map, axis=-1) * self.binary_cross_entropy_loss(y_true, y_pred))
        return dice_loss + ce_loss

class Joint_Dice_Focal_CE_Loss(tf.keras.losses.Loss):
    def __init__(self, **dice_kwargs):
        super().__init__()
        self.soft_dice_coef_loss = Soft_Dice_Coef_Loss(**dice_kwargs)
        self.binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
        self.gamma = 2.0
    #
    @tf.function
    def call(self, y_true, y_pred):
        #split apart ground truth label and distance transform map
        dist_map = y_true[1]
        y_true = y_true[0]
        #
        #cast to tf.float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        dist_map = tf.cast(dist_map, tf.float32)
        #compute dice loss
        dice_loss = self.soft_dice_coef_loss(y_true, y_pred)
        #compute cross entropy loss
        y_pred_sigmoid = tf.keras.activations.sigmoid(y_pred)
        y_pred_prob = tf.concat([tf.subtract(1., y_pred_sigmoid), y_pred_sigmoid], axis=-1)
        y_true_one_hot = tf.concat([tf.subtract(1., y_true), y_true], axis=-1)
        p_t = tf.reduce_sum(tf.multiply(y_true_one_hot, y_pred_prob), axis=-1)
        focal_weight = tf.pow(tf.subtract(1., p_t), self.gamma)
        ce_loss = tf.reduce_mean(focal_weight * self.binary_cross_entropy_loss(y_true, y_pred))
        return dice_loss + ce_loss

class Joint_Dice_Focal_Boundary_CE_Loss(tf.keras.losses.Loss):
    def __init__(self, **dice_kwargs):
        super().__init__()
        self.soft_dice_coef_loss = Soft_Dice_Coef_Loss(**dice_kwargs)
        self.binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")
        self.gamma = 2.0
    #
    @tf.function
    def call(self, y_true, y_pred):
        #split apart ground truth label and distance transform map
        dist_map = y_true[1]
        y_true = y_true[0]
        #
        #cast to tf.float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        dist_map = tf.cast(dist_map, tf.float32)
        #compute dice loss
        dice_loss = self.soft_dice_coef_loss(y_true, y_pred)
        #compute cross entropy loss
        y_pred_sigmoid = tf.keras.activations.sigmoid(y_pred)
        y_pred_prob = tf.concat([tf.subtract(1., y_pred_sigmoid), y_pred_sigmoid], axis=-1)
        y_true_one_hot = tf.concat([tf.subtract(1., y_true), y_true], axis=-1)
        p_t = tf.reduce_sum(tf.multiply(y_true_one_hot, y_pred_prob), axis=-1)
        focal_weight = tf.pow(tf.subtract(1., p_t), self.gamma)
        ce_loss = tf.reduce_mean(tf.squeeze(dist_map, axis=-1) * focal_weight * self.binary_cross_entropy_loss(y_true, y_pred))
        return dice_loss + ce_loss