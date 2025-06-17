# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import os

import numpy as np
import tensorflow as tf
from runtime.utils import get_config_file, get_tta_flips, is_main_process
from skimage.transform import resize

from models.sliding_window import get_importance_kernel, sliding_window_inference, sliding_window_inference_batched
from models.unet import UNet, UNet_Jay

class NNUnet(tf.keras.Model):
    def __init__(self, args, loaded_model=None):
        super(NNUnet, self).__init__()
        self.args = args
        in_channels, n_class, kernels, strides, self.patch_size = self.get_unet_params(self.args)
        self.n_class = n_class
        input_shape = (None, None, None, in_channels)
        if self.args.dim == 3:
            input_shape = (None,) + input_shape
        if loaded_model is not None:
            input_dtype = tf.float16 if args.amp else tf.float32

            @tf.function
            def wrapped_model(inputs, *args, **kwargs):
                return loaded_model(tf.cast(inputs, dtype=input_dtype), *args, **kwargs)

            self.model = wrapped_model
        else:
            if not self.args.xla and self.args.norm == "instance":
                self.args.norm = "atex_instance"
            self.model = UNet(
                input_shape=input_shape,
                n_class=n_class,
                kernels=kernels,
                strides=strides,
                dimension=self.args.dim,
                normalization_layer=self.args.norm,
                negative_slope=self.args.negative_slope,
                deep_supervision=self.args.deep_supervision,
                max_pool=False,
            )
            if is_main_process():
                print(f"Filters: {self.model.filters}")
        self.tta_flips = get_tta_flips(self.args.dim)
        if self.args.dim == 3:
            self.predictor = self.sw_inference
        elif self.args.benchmark:
            self.predictor = self.call
        else:
            self.predictor = self.call_2d

        if args.dim == 3:
            importance_kernel = get_importance_kernel(self.patch_size, args.blend_mode, 0.125)
            self.importance_map = tf.tile(
                tf.reshape(importance_kernel, shape=[1, *self.patch_size, 1]),
                multiples=[1, 1, 1, 1, n_class],
            )

    @tf.function
    def call(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @tf.function(reduce_retracing=True)
    def call_2d(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @tf.function
    def compute_loss(self, loss_fn, label, dist_maps, preds):
        #if self.args.deep_supervision:
        #    upsample_layer = tf.keras.layers.UpSampling3D if self.args.dim == 3 else tf.keras.layers.UpSampling2D
        #    loss = loss_fn(label, preds[0])
        #    upsample_factor = np.ones(self.args.dim, dtype=np.uint8)
        #    for i, pred in enumerate(preds[1:]):
        #        upsample_factor = upsample_factor * self.model.strides[i + 1]
        #        upsampled_pred = upsample_layer(upsample_factor)(pred)
        #        loss += 0.5 ** (i + 1) * loss_fn(label, upsampled_pred)
        #    c_norm = 1 / (2 - 2 ** (-len(preds)))
        #    return c_norm * loss
        #return loss_fn(label, preds)
        if self.args.deep_supervision:
            loss = loss_fn((label, dist_maps), preds[0])
            downsample_factor = np.ones(self.args.dim, dtype=np.uint8)
            for i, pred in enumerate(preds[1:]):
                downsample_factor = downsample_factor * self.model.strides[i + 1]
                downsampled_label = label[:, ::downsample_factor[0], ::downsample_factor[1], ::downsample_factor[2], :]
                downsampled_dist_map = dist_maps[:, ::downsample_factor[0], ::downsample_factor[1], ::downsample_factor[2], :]
                loss += 0.5 ** (i + 1) * loss_fn((downsampled_label, downsampled_dist_map), pred)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return loss_fn((label, dist_maps), preds)

    def sw_inference(self, img, **kwargs):
        return sliding_window_inference(
            inputs=img,
            roi_size=self.patch_size,
            model=self.model,
            overlap=self.args.overlap,
            n_class=self.n_class,
            importance_map=self.importance_map,
            **kwargs,
        )

    def inference(self, img, training=False):
        pred = self.predictor(img, training=training)
        if self.args.tta:
            for flip_axes in self.tta_flips:
                flipped_img = tf.reverse(img, axis=flip_axes)
                flipped_pred = self.predictor(flipped_img, training=training)
                pred = pred + tf.reverse(flipped_pred, axis=flip_axes)
            pred = pred / (len(self.tta_flips) + 1)
        return pred

    @staticmethod
    def get_unet_params(args):
        #config = get_config_file(args)
        config = {'patch_size': [128]*3, 'spacings': [1.0]*3, 'n_class': 1, 'in_channels': 1}
        patch_size, spacings = config["patch_size"], config["spacings"]
        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == 5:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return config["in_channels"], config["n_class"], kernels, strides, patch_size

    @staticmethod
    def layout_2d(x):
        if x is None:
            return None
        batch_size, depth, height, width, channels = x.shape
        return tf.reshape(x, (batch_size * depth, height, width, channels))

    def adjust_batch(self, features, labels):
        if self.args.dim == 2:
            features, labels = self.layout_2d(features), self.layout_2d(labels)
        return features, labels

    def save_pred(self, pred, idx, data_module, save_dir):
        pred = tf.cast(pred[0, ..., 0] >= 0, tf.float32).numpy()
        fname = data_module.test_fname(idx)
        output_fname = os.path.basename(fname).replace("_x", "")
        np.save(os.path.join(save_dir, output_fname), pred, allow_pickle=False)

'''
import os

import numpy as np
import tensorflow as tf
from runtime.utils import get_config_file, get_tta_flips, is_main_process
from skimage.transform import resize

from models.sliding_window import get_importance_kernel, sliding_window_inference, sliding_window_inference_batched
from models.unet import UNet, UNet_Jay

config = {'patch_size': [128]*3, 'spacings': [1.0]*3, 'n_class': 1, 'in_channels': 1}
patch_size, spacings = config["patch_size"], config["spacings"]
strides, kernels, sizes = [], [], patch_size[:]
while True:
    spacing_ratio = [spacing / min(spacings) for spacing in spacings]
    stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
    kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
    if all(s == 1 for s in stride):
        break
    sizes = [i / j for i, j in zip(sizes, stride)]
    spacings = [i * j for i, j in zip(spacings, stride)]
    kernels.append(kernel)
    strides.append(stride)
    if len(strides) == 5:
        break


strides.insert(0, len(spacings) * [1])
kernels.append(len(spacings) * [3])

in_channels = 1 
n_class = 1
input_shape = (None, None, None, in_channels)
input_shape = (None,) + input_shape

model = UNet(input_shape=input_shape,n_class=n_class,kernels=kernels,strides=strides,dimension=3,normalization_layer='instance',negative_slope=0.01,deep_supervision=True)
model_jay = UNet_Jay(input_shape=input_shape,n_class=n_class,kernels=kernels,strides=strides,dimension=3,normalization_layer='instance',negative_slope=0.01,deep_supervision=True)
x = tf.constant(np.random.random((1,128,128,128,1)))
y = model(x, training=False)
y_jay = model_jay(x, training=False)

model.summary()
model_jay.summary()
'''