import itertools

import numpy as np
import tensorflow as tf
from scipy import signal

#NOTE: Batched sliding window inference is peforming slower non-batched version for some reason

def get_window_slices(image_size, roi_size, overlap, strategy):
    dim_starts = []
    for image_x, roi_x in zip(image_size, roi_size):
        interval = roi_x if roi_x == image_x else int(roi_x * (1 - overlap))
        starts = list(range(0, image_x - roi_x + 1, interval))
        if strategy == "overlap_inside" and starts[-1] + roi_x < image_x:
            starts.append(image_x - roi_x)
        dim_starts.append(starts)
    slices = [(starts + (0,), roi_size + (-1,)) for starts in itertools.product(*dim_starts)]
    batched_window_slices = [((0,) + start, (1,) + roi_size) for start, roi_size in slices]
    return batched_window_slices


@tf.function
def gaussian_kernel(roi_size, sigma):
    gauss = signal.windows.gaussian(roi_size[0], std=sigma * roi_size[0])
    for s in roi_size[1:]:
        gauss = np.outer(gauss, signal.windows.gaussian(s, std=sigma * s))
    #
    gauss = np.reshape(gauss, roi_size)
    gauss = np.power(gauss, 1 / len(roi_size))
    gauss /= gauss.max()
    #
    return tf.convert_to_tensor(gauss, dtype=tf.float32)


def get_importance_kernel(roi_size, blend_mode, sigma):
    if blend_mode == "constant":
        return tf.ones(roi_size, dtype=tf.float32)
    elif blend_mode == "gaussian":
        return gaussian_kernel(roi_size, sigma)
    else:
        raise ValueError(f'Invalid blend mode: {blend_mode}. Use either "constant" or "gaussian".')


@tf.function
def run_model(x, model, importance_map, **kwargs):
    return tf.cast(model(x, **kwargs), dtype=tf.float32) * importance_map


def sliding_window_inference(
    inputs,
    roi_size,
    model,
    overlap,
    n_class,
    importance_map,
    strategy="overlap_inside",
    **kwargs,
    ):
    image_size = tuple(inputs.shape[1:-1])
    roi_size = tuple(roi_size)

    output_shape = (1, *image_size, n_class)
    output_sum = tf.zeros(output_shape, dtype=tf.float32)
    output_weight_sum = tf.ones(output_shape, dtype=tf.float32)
    window_slices = get_window_slices(image_size, roi_size, overlap, strategy)

    for window_slice in window_slices:
        window = tf.slice(inputs, begin=window_slice[0], size=window_slice[1])
        pred = run_model(window, model, importance_map, **kwargs)
        padding = [
            [start, output_size - (start + size)] for start, size, output_size in zip(*window_slice, output_shape)
        ]
        padding = padding[:-1] + [[0, 0]]
        output_sum = output_sum + tf.pad(pred, padding)
        output_weight_sum = output_weight_sum + tf.pad(importance_map, padding)

    return output_sum / output_weight_sum

def sliding_window_inference_batched(
    inputs,
    roi_size,
    model,
    overlap,
    n_class,
    importance_map,
    strategy="overlap_inside",
    inference_batch_size=2,
    **kwargs,
    ):
    image_size = tuple(inputs.shape[1:-1])
    roi_size = tuple(roi_size)
    #
    output_shape = (1, *image_size, n_class)
    output_sum = tf.zeros(output_shape, dtype=tf.float32)
    output_weight_sum = tf.ones(output_shape, dtype=tf.float32)
    window_slices = get_window_slices(image_size, roi_size, overlap, strategy)
    #
    window_slices = tf.stack(window_slices)
    num_windows = window_slices.shape[0]
    windows = tf.concat([tf.slice(inputs, begin=window_slices[i][0,:], size=window_slices[i][1,:]) for i in range(0,num_windows)], axis=0)
    #
    pred = tf.concat([run_model(windows[i:i+inference_batch_size, ...], model, importance_map, **kwargs) for i in range(0, num_windows, inference_batch_size)], axis=0)
    for i in range(0, num_windows):
        padding = [[start, output_size - (start + size)] for start, size, output_size in zip(window_slices[i][0,:], window_slices[i][1,:], output_shape)]
        padding = padding[:-1] + [[0, 0]]
        output_sum = output_sum + tf.pad(pred[i:i+1,...], padding)
        output_weight_sum = output_weight_sum + tf.pad(importance_map, padding)
    #
    return output_sum / output_weight_sum