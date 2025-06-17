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

#TODO: See if it is worth only applying transforms to non-zero voxels (i.e. don't add gaussian noise to area outside brain)
#TODO: Optimized affine transform without gamma correction takes about 3 seconds to initialize; with one gamma correction takes about 25 seconds; with both gamma correction takes about 770 seconds
#NOTE: If have small-ish volumes and lots of extra GPU space, minor to no difference between computationally expensive and optimized affine transform (expensive operation may perform better since optimized version has lots of overhead); if have really large volumes or limited GPU space, optimized version will perform better

#fn.shapes(), crop_shape, and anchor are in width, height, depth format!
#keypoints (and other points / transforms) are in depth, height, width format!  

import itertools

import horovod.tensorflow as hvd
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.math as math
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as types
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline


def get_numpy_reader(files, shard_id, num_shards, seed, shuffle):
    return ops.readers.Numpy(
        seed=seed,
        files=files,
        device="cpu",
        read_ahead=True,
        shard_id=shard_id,
        pad_last_batch=True,
        num_shards=num_shards,
        dont_use_mmap=True,
        shuffle_after_epoch=shuffle,
    )


def random_augmentation(probability, augmented, original):
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original


class GenericPipeline(Pipeline):
    def __init__(
        self,
        batch_size,
        num_threads,
        shard_id,
        seed,
        num_gpus,
        dim,
        patch_size,
        shuffle_input=True,
        input_x_files=None,
        input_y_files=None,
        input_z_files=None,
        batch_size_2d=None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            #device_id=hvd.rank(),
            device_id=0,
            seed=seed,
        )

        if input_x_files is not None:
            self.input_x = get_numpy_reader(
                files=input_x_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )
        if input_y_files is not None:
            self.input_y = get_numpy_reader(
                files=input_y_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )
        if input_z_files is not None:
            self.input_z = get_numpy_reader(
                files=input_z_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )

        self.patch_size = patch_size
        self.dim = dim
        if self.dim == 2 and batch_size_2d is not None:
            self.patch_size = [batch_size_2d] + self.patch_size
        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)
        self.internal_seed = seed
        self.constant_1 = types.Constant(1.0, dtype=types.DALIDataType.FLOAT)
        self.constant_0 = types.Constant(0.0, dtype=types.DALIDataType.FLOAT)
    

class TrainPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, dist_maps, oversampling, patch_size, batch_size_2d=None, **kwargs):
        super().__init__(patch_size=patch_size, batch_size_2d=batch_size_2d, input_x_files=imgs, input_y_files=lbls, input_z_files=dist_maps, shuffle_input=True, **kwargs)
        self.oversampling = oversampling

    def load_data(self):
        img, lbl, dist_map = self.input_x(name="ReaderX"), self.input_y(name="ReaderY"), self.input_z(name="ReaderZ")
        img, lbl, dist_map = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC"), fn.reshape(dist_map, layout="DHWC")
        image_shapes = fn.cast(fn.shapes(img)[:-1], dtype=types.DALIDataType.FLOAT)
        return img, lbl, dist_map, image_shapes

    def biased_crop_fn(self, lbl):
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            lbl,
            format="start_end",
            foreground_prob=self.oversampling,
            k_largest=None,
            device="cpu",
            cache_objects=True,
        )
        anchor = fn.roi_random_crop(
            lbl,
            roi_start=roi_start,
            roi_end=roi_end,
            crop_shape=[*self.patch_size, 1],
        )
        anchor = fn.slice(anchor, 0, 3, axes=[0])
        return anchor
    
    def move_to_gpu_fn(self, img, lbl, dist_map):
        img, lbl, dist_map = img.gpu(), lbl.gpu(), dist_map.gpu()
        return img, lbl, dist_map
    
    def generate_affine_centering_transforms(self, anchor):
        center = anchor + (self.crop_shape_float / 2)
        center = fn.stack(center[2], center[1], center[0])
        translation_center_1 = fn.transforms.translation(offset=-center)
        translation_center_2 = fn.transforms.translation(offset=center)
        return translation_center_1, translation_center_2
    
    def generate_affine_transform_matrix(self):
        #generate scale transform if requested
        scale_factors = random_augmentation(0.125, fn.random.uniform(range=(0.7, 1.4)), 1.0)
        scale_x = fn.transforms.scale(scale=fn.stack(scale_factors, self.constant_1, self.constant_1))
        scale_factors = random_augmentation(0.125, fn.random.uniform(range=(0.7, 1.4)), 1.0)
        scale_y = fn.transforms.scale(scale=fn.stack(self.constant_1, scale_factors, self.constant_1))
        scale_factors = random_augmentation(0.125, fn.random.uniform(range=(0.7, 1.4)), 1.0)
        scale_z = fn.transforms.scale(scale=fn.stack(self.constant_1, self.constant_1, scale_factors))
        #generate rotation transform if requested
        rotation_factors = random_augmentation(0.125, fn.random.uniform(range=(-30., 30.)), 0.0)
        rotation_x = fn.transforms.rotation(angle=rotation_factors, axis=[1,0,0])
        rotation_factors = random_augmentation(0.125, fn.random.uniform(range=(-30., 30.)), 0.0)
        rotation_y = fn.transforms.rotation(angle=rotation_factors, axis=[0,1,0])
        rotation_factors = random_augmentation(0.125, fn.random.uniform(range=(-30., 30.)), 0.0)
        rotation_z = fn.transforms.rotation(angle=rotation_factors, axis=[0,0,1])
        #generate shear transform if requested
        shear_factors = random_augmentation(0.0625, fn.random.uniform(range=(-0.15, 0.15)), 0.0)
        shear_xy = fn.transforms.shear(shear=fn.stack(fn.stack(shear_factors, self.constant_0), fn.stack(self.constant_0, self.constant_0), fn.stack(self.constant_0, self.constant_0)))
        shear_factors = random_augmentation(0.0625, fn.random.uniform(range=(-0.15, 0.15)), 0.0)
        shear_xz = fn.transforms.shear(shear=fn.stack(fn.stack(self.constant_0, shear_factors), fn.stack(self.constant_0, self.constant_0), fn.stack(self.constant_0, self.constant_0)))
        shear_factors = random_augmentation(0.0625, fn.random.uniform(range=(-0.15, 0.15)), 0.0)
        shear_yx = fn.transforms.shear(shear=fn.stack(fn.stack(self.constant_0, self.constant_0), fn.stack(shear_factors, self.constant_0), fn.stack(self.constant_0, self.constant_0)))
        shear_factors = random_augmentation(0.0625, fn.random.uniform(range=(-0.15, 0.15)), 0.0)
        shear_yz = fn.transforms.shear(shear=fn.stack(fn.stack(self.constant_0, self.constant_0), fn.stack(self.constant_0, shear_factors), fn.stack(self.constant_0, self.constant_0)))
        shear_factors = random_augmentation(0.0625, fn.random.uniform(range=(-0.15, 0.15)), 0.0)
        shear_zx = fn.transforms.shear(shear=fn.stack(fn.stack(self.constant_0, self.constant_0), fn.stack(self.constant_0, self.constant_0), fn.stack(shear_factors, self.constant_0)))
        shear_factors = random_augmentation(0.0625, fn.random.uniform(range=(-0.15, 0.15)), 0.0)
        shear_zy = fn.transforms.shear(shear=fn.stack(fn.stack(self.constant_0, self.constant_0), fn.stack(self.constant_0, self.constant_0), fn.stack(self.constant_0, shear_factors)))
        #generate translation transform if requested
        translation_factors = random_augmentation(0.0, fn.random.uniform(range=(-5, 5)), 0.0)
        translation_x = fn.transforms.translation(offset=fn.stack(translation_factors, self.constant_0, self.constant_0))
        translation_factors = random_augmentation(0.0, fn.random.uniform(range=(-5, 5)), 0.0)
        translation_y = fn.transforms.translation(offset=fn.stack(self.constant_0, translation_factors, self.constant_0))
        translation_factors = random_augmentation(0.0, fn.random.uniform(range=(-5, 5)), 0.0)
        translation_z = fn.transforms.translation(offset=fn.stack(self.constant_0, self.constant_0, translation_factors))
        #compose transforms together
        affine_transform_matrix = fn.transforms.combine(scale_x, scale_y, scale_z, rotation_x, rotation_y, rotation_z, shear_xy, shear_xz, shear_yx, shear_yz, shear_zx, shear_zy, translation_x, translation_y, translation_z)
        return affine_transform_matrix
    
    def generate_affine_transform_simple_matrix(self):
        #generate scale transform if requested (mostly isotropic with slight jittering)
        scale_factors = random_augmentation(0.20, fn.random.uniform(range=(0.7, 1.4)) + fn.random.normal(mean=0, stddev=0.01, shape=[3]), 1.0)
        scale_transform = fn.transforms.scale(scale=scale_factors)
        #generate rotation transform if requested (random rotation applied around random rotation axis)
        rotation_axis = fn.random.uniform(range=(-1., 1.), shape=[3])
        rotation_factors = random_augmentation(0.20, fn.random.uniform(range=(-30., 30.)), 0.0)
        rotation_transform = fn.transforms.rotation(angle=rotation_factors, axis=rotation_axis)
        #compose transforms together
        affine_transform_matrix = fn.transforms.combine(scale_transform, rotation_transform)
        return affine_transform_matrix
    
    def centered_affine_transform_matrix(self, anchor, uncentered_affine_transform_matrix):
        #find center of patch, which will be used as origin for transformations
        center = anchor + (self.crop_shape_float / 2)
        center = fn.stack(center[2], center[1], center[0])
        #center the affine transform
        return fn.transforms.combine(fn.transforms.translation(offset=-center), uncentered_affine_transform_matrix, fn.transforms.translation(offset=center))

    def slice_fn(self, img, lbl, dist_map, anchor, crop_shape):
        img, lbl, dist_map = fn.slice(
            [img, lbl, dist_map],
            anchor,
            crop_shape,
            axis_names="DHW",
            out_of_bounds_policy="pad",
            normalized_anchor=False,
            normalized_shape=False,
        )
        return img, lbl, dist_map
    
    def affine_transform_computationally_expensive_fn(self, img, lbl, dist_map, anchor):
        #generate affine transform
        #uncentered_affine_transform_matrix = self.generate_affine_transform_matrix()
        uncentered_affine_transform_matrix = self.generate_affine_transform_simple_matrix()
        #compose transforms together
        affine_transform_matrix = self.centered_affine_transform_matrix(anchor, uncentered_affine_transform_matrix)
        #warp image and label
        img = fn.warp_affine(img, matrix=affine_transform_matrix, fill_value=0, inverse_map=True, interp_type=types.DALIInterpType.INTERP_LINEAR)
        lbl = fn.warp_affine(lbl, matrix=affine_transform_matrix, fill_value=0, inverse_map=True, interp_type=types.DALIInterpType.INTERP_NN)
        dist_map = fn.warp_affine(dist_map, matrix=affine_transform_matrix, fill_value=0, inverse_map=True, interp_type=types.DALIInterpType.INTERP_NN)
        #crop image to patch size
        img, lbl, dist_map = self.slice_fn(img, lbl, dist_map, anchor, self.crop_shape)
        return img, lbl, dist_map
    
    def affine_transform_fn(self, img, lbl, dist_map, anchor, image_shapes):
        #generate affine transform
        #uncentered_affine_transform_matrix = self.generate_affine_transform_matrix()
        uncentered_affine_transform_matrix = self.generate_affine_transform_simple_matrix()
        #compose transforms together
        affine_transform_matrix = self.centered_affine_transform_matrix(anchor, uncentered_affine_transform_matrix)
        #generate bounding box keypoints from anchor
        keypoints = fn.cast(fn.stack(
            fn.stack(anchor[2]                     , anchor[1]                     , anchor[0]                     ),
            fn.stack(anchor[2]                     , anchor[1]                     , anchor[0] + self.crop_shape[0]),
            fn.stack(anchor[2]                     , anchor[1] + self.crop_shape[1], anchor[0]                     ),
            fn.stack(anchor[2]                     , anchor[1] + self.crop_shape[1], anchor[0] + self.crop_shape[0]),
            fn.stack(anchor[2] + self.crop_shape[2], anchor[1]                     , anchor[0]                     ),
            fn.stack(anchor[2] + self.crop_shape[2], anchor[1]                     , anchor[0] + self.crop_shape[0]),
            fn.stack(anchor[2] + self.crop_shape[2], anchor[1] + self.crop_shape[1], anchor[0]                     ),
            fn.stack(anchor[2] + self.crop_shape[2], anchor[1] + self.crop_shape[1], anchor[0] + self.crop_shape[0]),
            ), dtype=types.DALIDataType.FLOAT)
        #transform bounding box keypoints to see what extent we need for the affine transformation
        keypoints = fn.coord_transform(keypoints, MT=affine_transform_matrix)
        #convert back to width, height, depth format
        keypoints = fn.stack(keypoints[:,2], keypoints[:,1], keypoints[:,0], axis=1)
        #find lowest and highest extent of transformed bounding box (and clamp bottom and top extents)
        lo = math.floor(math.max(fn.reductions.min(keypoints, axes=0), 0.))
        hi = math.ceil(math.min(fn.reductions.max(keypoints, axes=0), image_shapes))
        #the transformed bounding box shape should not be smaller than the original patch size
        padding_value = math.max((self.crop_shape_float - (hi - lo)) / 2., 0.)
        lo = lo - math.floor(padding_value)
        hi = hi + math.ceil(padding_value)
        #crop image to new shape
        img, lbl = self.slice_fn(img, lbl, lo, hi - lo)
        #update anchor so that the affine transformation works correctly on smaller patch
        anchor = anchor - lo
        #compose transforms together
        affine_transform_matrix = self.centered_affine_transform_matrix(anchor, uncentered_affine_transform_matrix)
        #warp image and label
        img = fn.warp_affine(img, matrix=affine_transform_matrix, fill_value=0, inverse_map=True, interp_type=types.DALIInterpType.INTERP_LINEAR)
        lbl = fn.warp_affine(lbl, matrix=affine_transform_matrix, fill_value=0, inverse_map=True, interp_type=types.DALIInterpType.INTERP_NN)
        dist_map = fn.warp_affine(dist_map, matrix=affine_transform_matrix, fill_value=0, inverse_map=True, interp_type=types.DALIInterpType.INTERP_NN)
        #crop image to patch size
        img, lbl, dist_map = self.slice_fn(img, lbl, dist_map, anchor, self.crop_shape_float)
        return img, lbl, dist_map
        
    def noise_fn(self, img):
        img = fn.noise.gaussian(img, stddev=random_augmentation(0.1, fn.random.uniform(range=(0.0, 0.3)), 0.0))
        return img

    def blur_fn(self, img):
        img = fn.gaussian_blur(img, sigma=random_augmentation(0.1, fn.random.uniform(range=(0.5, 1.0)), 1e-7))
        return img
    
    def intensity_factor_generator(self, prob, low, high):
        intensity_factor = random_augmentation(0.5, fn.random.uniform(range=(low, 1.0)), fn.random.uniform(range=(1.0, high)))
        return random_augmentation(prob, intensity_factor, 1.0)

    def brightness_fn(self, img):
        img = img * self.intensity_factor_generator(0.15, 0.75, 1.25)
        return img

    def contrast_fn(self, img):
        img_mean = fn.reductions.mean(img)
        img = math.clamp(((img - img_mean) * self.intensity_factor_generator(0.15, 0.75, 1.25)) + img_mean, fn.reductions.min(img), fn.reductions.max(img))
        return img

    def flips_fn(self, img, lbl, dist_map):
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
            "depthwise": fn.random.coin_flip(probability=0.5)
            }
        img = fn.flip(img, **kwargs)
        lbl = fn.flip(lbl, **kwargs)
        dist_map = fn.flip(dist_map, **kwargs)
        return img, lbl, dist_map
    
    def simulate_low_resolution_fn(self, img, lbl, dist_map):
        #random size augmentation
        random_size = random_augmentation(0.125, fn.random.uniform(range=(0.5, 1.0)), 1.0) * self.crop_shape_float
        #resize the image/label to a smaller size using nearest neighbor interpolation
        img = fn.resize(img, size=random_size, interp_type=types.DALIInterpType.INTERP_NN)
        lbl = fn.resize(lbl, size=random_size, interp_type=types.DALIInterpType.INTERP_NN)
        dist_map = fn.resize(dist_map, size=random_size, interp_type=types.DALIInterpType.INTERP_NN)
        #resize the image/label back to original resolution
        img = fn.resize(img, size=self.crop_shape_float, interp_type=types.DALIInterpType.INTERP_LINEAR)
        lbl = fn.resize(lbl, size=self.crop_shape_float, interp_type=types.DALIInterpType.INTERP_NN)
        dist_map = fn.resize(dist_map, size=self.crop_shape_float, interp_type=types.DALIInterpType.INTERP_NN)
        return img, lbl, dist_map
    
    def gamma_correct_helper_fn(self, img, prob):
        #compute mean, standard deviation, min, and range of intensities in original image
        orig_mean = fn.reductions.mean(img)
        orig_std_dev = fn.reductions.std_dev(img, orig_mean)
        orig_min = fn.reductions.min(img)
        orig_range = fn.reductions.max(img) - orig_min
        #compute gamma transform
        img_augmented = math.pow((img - orig_min) / math.max(orig_range, 1e-7), self.intensity_factor_generator(prob, 0.7, 1.5)) * orig_range + orig_min
        #compute mean and standard deviation of intensities in new image
        current_mean = fn.reductions.mean(img_augmented)
        current_std_dev = fn.reductions.std_dev(img_augmented, current_mean)
        #rescale range back to the original mean and standard deviation
        img_augmented = ((img_augmented - current_mean) * (orig_std_dev / math.max(current_std_dev, 1e-7))) + orig_mean
        return img_augmented
    
    def gamma_correct_invert_intensities_fn(self, img):
        #gamma correction on intensity inverted image
        img = self.gamma_correct_helper_fn(img * -1.0, 0.1) * -1.0
        return img
    
    def gamma_correct_fn(self, img):
        #gamma correction on original image
        img = self.gamma_correct_helper_fn(img, 0.3)
        return img
    
    def define_graph(self):
        #load batch images/labels
        img, lbl, dist_map, image_shapes = self.load_data()
        #find anchor of patch based on segmentation labels
        anchor = self.biased_crop_fn(lbl)
        #move image and label to gpu to speed up augmentations
        img, lbl, dist_map = self.move_to_gpu_fn(img, lbl, dist_map)
        #random affine transformation
        img, lbl, dist_map = self.affine_transform_computationally_expensive_fn(img, lbl, dist_map, anchor)
        #img, lbl, dist_map = self.affine_transform_fn(img, lbl, dist_map, anchor, image_shapes)
        #random gaussian noise
        img = self.noise_fn(img)
        #random blurring
        img = self.blur_fn(img)
        #random brightness adjustment
        img = self.brightness_fn(img)
        #random contrast adjustment
        img = self.contrast_fn(img)
        #random simulation of low resolution
        img, lbl, dist_map = self.simulate_low_resolution_fn(img, lbl, dist_map)
        #random gamma correction with intensity inversion
        img = self.gamma_correct_invert_intensities_fn(img)
        #random gamma correction without intensity inversion
        img = self.gamma_correct_fn(img)
        #random flipping
        img, lbl, dist_map = self.flips_fn(img, lbl, dist_map)
        #return augmented images/labels
        return img, lbl, dist_map


class EvalPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, dist_maps, patch_size, **kwargs):
        super().__init__(patch_size=patch_size, input_x_files=imgs, input_y_files=lbls, input_z_files=dist_maps, shuffle_input=False, **kwargs)
    
    def define_graph(self):
        img, lbl, dist_map = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu(), self.input_z(name="ReaderZ").gpu()
        img, lbl, dist_map = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC"), fn.reshape(dist_map, layout="DHWC")
        return img, lbl, dist_map


class TestPipeline(GenericPipeline):
    def __init__(self, imgs, patch_size, **kwargs):
        super().__init__(patch_size=patch_size, input_x_files=imgs, shuffle_input=False, **kwargs)

    def define_graph(self):
        img = self.input_x(name="ReaderX").gpu()
        img = fn.reshape(img, layout="DHWC")
        return img


class BenchmarkPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, dist_maps, patch_size, batch_size_2d=None, **kwargs):
        super().__init__(patch_size=patch_size, input_x_files=imgs, input_y_files=lbls, input_z_files=dist_maps, shuffle_input=False, **kwargs)
        self.patch_size = patch_size
        if self.dim == 2 and batch_size_2d is not None:
            self.patch_size = [batch_size_2d] + self.patch_size

    def crop_fn(self, img, lbl, dist_map):
        img = fn.crop(img, crop=self.patch_size, out_of_bounds_policy="pad")
        lbl = fn.crop(lbl, crop=self.patch_size, out_of_bounds_policy="pad")
        dist_map = fn.crop(dist_map, crop=self.patch_size, out_of_bounds_policy="pad")
        return img, lbl, dist_map

    def define_graph(self):
        img, lbl, dist_map = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu(), self.input_z(name="ReaderZ").gpu()
        img, lbl, dist_map = self.crop_fn(img, lbl, dist_map)
        img, lbl, dist_map = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC"), fn.reshape(dist_map, layout="DHWC")
        return img, lbl, dist_map


def fetch_dali_loader(imgs, lbls, dist_maps, batch_size, mode, use_hvd, **kwargs):
    assert len(imgs) > 0, "No images found"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Got {len(imgs)} images but {len(lbls)} lables"
    
    if use_hvd == True:
        gpus = hvd.size()
        device_id = hvd.rank()
    else:
        gpus = 1
        device_id = 0
    if kwargs["benchmark"]:
        # Just to make sure the number of examples is large enough for benchmark run.
        nbs = kwargs["bench_steps"]
        if kwargs["dim"] == 3:
            nbs *= batch_size
        imgs = list(itertools.chain(*(100 * [imgs])))[: nbs * gpus]
        lbls = list(itertools.chain(*(100 * [lbls])))[: nbs * gpus]

    pipe_kwargs = {
        "dim": kwargs["dim"],
        "num_gpus": gpus,
        "seed": kwargs["seed"],
        "batch_size": batch_size,
        "num_threads": kwargs["num_workers"],
        "shard_id": device_id,
    }
    if kwargs["dim"] == 2:
        if kwargs["benchmark"]:
            pipe_kwargs.update({"batch_size_2d": batch_size})
            batch_size = 1
        elif mode == "train":
            pipe_kwargs.update({"batch_size_2d": batch_size // kwargs["nvol"]})
            batch_size = kwargs["nvol"]
    if mode == "eval":  # Validation data is manually sharded beforehand.
        pipe_kwargs["shard_id"] = 0
        pipe_kwargs["num_gpus"] = 1

    output_dtypes = (tf.float32, tf.uint8, tf.float32)
    if kwargs["benchmark"]:
        pipeline = BenchmarkPipeline(imgs, lbls, dist_maps, kwargs["patch_size"], **pipe_kwargs)
    elif mode == "train":
        pipeline = TrainPipeline(imgs, lbls, dist_maps, kwargs["oversampling"], kwargs["patch_size"], **pipe_kwargs)
    elif mode == "eval":
        pipeline = EvalPipeline(imgs, lbls, dist_maps, kwargs["patch_size"], **pipe_kwargs)
    else:
        pipeline = TestPipeline(imgs, kwargs["patch_size"], **pipe_kwargs)
        output_dtypes = tf.float32

    tf_pipe = dali_tf.DALIDataset(pipeline, batch_size=batch_size, device_id=device_id, output_dtypes=output_dtypes)
    return tf_pipe
