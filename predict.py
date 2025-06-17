#!/usr/local/bin/python
import os
import time
import scipy
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf

from glob import glob
from scipy.ndimage import find_objects
from data_loading.data_module import DataModule
from models.nn_unet import NNUnet
from runtime.args import get_main_args
from runtime.checkpoint import load_model
from runtime.logging import get_logger
from runtime.run import evaluate, export_model, predict, train
from runtime.utils import hvd_init, set_seed, set_tf_flags

def predict_on_patient(full_patient_path, vol_name, index_patient, num_patients, patch_size=np.array([128,128,128]), final_ensemble_label_name='model_ensemble-label.nii.gz', final_ensemble_probability_name='model_ensemble_probability_map.nii.gz', final_uncertainty_entropy_name = 'model_ensemble_uncertainty_entropy_map.nii.gz'):
    start_prediction_time = time.time()
    #
    nib_vol = nib.load(full_patient_path + vol_name)
    T1_im = nib_vol.get_fdata()
    #crop to nonzero foreground
    if np.any(T1_im != 0):
        volume_location = find_objects(T1_im != 0)[0]
    else:
        volume_location = find_objects(T1_im == 0)[0]
    T1_im_crop = T1_im[volume_location]
    background_mask = np.ones_like(T1_im)
    background_mask[volume_location] = 0
    #pad all axes to the patch size in case cropped volume is smaller than size of patch along an axis or axes
    extra_padding = np.maximum(patch_size - np.array(T1_im_crop.shape), 0)
    pad_tuple_initial = tuple([(int(np.floor(i / 2)), int(np.ceil(i / 2))) for i in extra_padding])
    T1_im_crop = np.pad(T1_im_crop, pad_tuple_initial, mode='constant')
    #
    x = tf.cast(tf.constant(np.expand_dims(T1_im_crop, axis=(0,-1))), tf.float32)
    #
    for i in range(0, 5):
        args.ckpt_dir = '/workspace/brain_mets_seg/trained_models/Model_' + str(i)
        checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_dir)).expect_partial()
        y = model.inference(x)[0,...,0].numpy()
        #
        remove_padding_initial_index = tuple([slice(start_index, y.shape[i] - end_index, 1) for i, (start_index, end_index) in enumerate(pad_tuple_initial)])
        y = y[remove_padding_initial_index]
        pad_bounds = [(j.start, T1_im.shape[i]-j.stop) for i,j in enumerate(volume_location)]
        y = np.pad(y, pad_bounds, mode='constant')
        if i == 0:
            logits_y = np.zeros((5,) + y.shape)
        logits_y[i, ...] = y
    #
    ensemble_logit = np.mean(logits_y, axis=0)
    ensemble_label = (ensemble_logit > 0).astype(int)
    ensemble_label[background_mask == 1] = 0
    nib_vol_save = nib.Nifti1Image(ensemble_label, affine=nib_vol.affine, header=nib_vol.header)
    nib.save(nib_vol_save, full_patient_path + final_ensemble_label_name)
    #
    all_prob = scipy.special.expit(logits_y)
    ensemble_prob = np.mean(all_prob, axis=0)
    ensemble_prob[background_mask == 1] = 0
    nib_vol_save = nib.Nifti1Image(ensemble_prob, affine=nib_vol.affine, header=nib_vol.header)
    nib.save(nib_vol_save, full_patient_path + final_ensemble_probability_name)
    #
    uncertainity_entropy_map = np.mean(-((all_prob * np.log2(all_prob, out=np.zeros_like(all_prob, dtype=np.float32), where=(all_prob != 0))) + ((1 - all_prob) * np.log2(1 - all_prob, out=np.zeros_like(all_prob, dtype=np.float32), where=((1 - all_prob) != 0)))), axis=0)
    uncertainity_entropy_map[background_mask == 1] = 0
    nib_vol_save = nib.Nifti1Image(uncertainity_entropy_map, affine=nib_vol.affine, header=nib_vol.header)
    nib.save(nib_vol_save, full_patient_path + final_uncertainty_entropy_name)
    #
    time_for_prediction = np.round(time.time() - start_prediction_time, 2)
    print(str(index_patient+1) + '/' + str(num_patients) + ': Prediction for patient ' + full_patient_path + ' is complete (' + str(time_for_prediction) + 's) \n')

if __name__ == "__main__":
    args = get_main_args()
    args.fold = 0
    args.exec_mode = 'predict'
    args.gpus = 1
    args.dim = 3
    args.seed = 12345
    args.amp = True
    args.xla = True
    args.batch_size = 2
    args.steps_per_epoch = 250
    args.epochs = 1000
    args.skip_eval = 0
    args.skip_train_eval = 0
    args.optimizer = 'sgdw' 
    args.learning_rate = 0.01
    args.end_learning_rate = 0.0
    args.weight_decay = 0.00003
    args.momentum = 0.99
    args.dampening = 0.0
    args.nesterov = True
    args.decoupled = False
    args.scheduler = 'poly'
    args.power = 0.9
    args.deep_supervision = True
    args.loss_batch_reduction = False
    args.overlap = 0.75 
    args.tta = True
    args.save_preds = True
    args.blend_mode = 'gaussian'
    args.negative_slope = 0.01 
    args.use_hvd = False
    args.results = '/workspace/brain_mets_seg/trained_models'
    hvd_init(args.use_hvd)
    if args.seed is not None:
        set_seed(args.seed)
    set_tf_flags(args)
    #
    model = NNUnet(args)
    checkpoint = tf.train.Checkpoint(model=model)
    #PREDICT ON IMAGES FROM CSV
    df_csv = pd.read_csv('/workspace/brain_mets_seg/data/files_to_segment.csv')
    patients = list(df_csv['Full_Patient_File_Path'])
    vol_names = list(df_csv['T1_CE_Volume_Name'])
    num_patients = len(patients)
    for i, (patient, vol_name) in enumerate(zip(patients, vol_names)):
        if os.path.exists(patient + '/' + vol_name):
            predict_on_patient(patient + '/', vol_name, i, num_patients)