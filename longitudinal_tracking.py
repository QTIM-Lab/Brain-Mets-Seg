import os
import numpy as np
import pandas as pd
import nibabel as nib
import multiprocessing

from skimage.morphology import label
from joblib import Parallel, delayed

#create tracked METS labels
def track_patient_across_visits(patient, track_ground_truth=True, track_prediction=True, BIDS_format_names=False):
    patient_dir = nifti_dir + patient + '/'
    os.chdir(patient_dir)
    visits = sorted(next(os.walk('.'))[1])
    if BIDS_format_names == True:
        visits = [visit + '/anat' for visit in visits]
    #check that ground truth and/or prediction files exist
    visits = verify_segmentations_exist(patient, visits, track_ground_truth, ground_truth_name, track_prediction, model_prediction_label_name, BIDS_format_names)
    num_time_points = len(visits)
    if num_time_points > 0:
        #ground truth
        if track_ground_truth == True:
            conn_comp_ground_truth, final_tracked_ground_truth_label, total_num_tracked_METS_ground_truth = initialize_variables_for_track_mets(patient, visits, num_time_points, ground_truth_name, BIDS_format_names)
            #track mets from time point 2 onwards for ground truth
            for i in range(1, num_time_points):
                final_tracked_ground_truth_label[i,...], total_num_tracked_METS_ground_truth = track_mets(final_tracked_ground_truth_label[:i,...], conn_comp_ground_truth[i,...], total_num_tracked_METS_ground_truth)
            #save tracked label map
            save_tracked_mets(patient, visits, ground_truth_name, final_tracked_ground_truth_label, save_name_ground_truth, BIDS_format_names)
        #prediction
        if track_prediction == True:
            conn_comp_prediction, final_tracked_prediction_label, total_num_tracked_METS_prediction = initialize_variables_for_track_mets(patient, visits, num_time_points, model_prediction_label_name, BIDS_format_names)
            #track mets from time point 2 onwards for prediction
            for i in range(1, num_time_points):
                final_tracked_prediction_label[i,...], total_num_tracked_METS_prediction = track_mets(final_tracked_prediction_label[:i,...], conn_comp_prediction[i,...], total_num_tracked_METS_prediction)
            #save tracked label map
            save_tracked_mets(patient, visits, model_prediction_label_name, final_tracked_prediction_label, save_name_prediction, BIDS_format_names)
        #
        if (track_ground_truth == True) and (track_prediction == True):
            #track mets between prediction and ground truth to determine correspondances
            conn_comp_prediction_to_ground_truth = np.copy(conn_comp_prediction)
            final_tracked_prediction_to_ground_truth_label = np.zeros(conn_comp_prediction_to_ground_truth.shape)
            total_num_tracked_METS_prediction_to_ground_truth = np.unique(final_tracked_ground_truth_label).shape[0] - 1
            for i in range(0, num_time_points):
                final_tracked_prediction_to_ground_truth_label[i,...], total_num_tracked_METS_prediction_to_ground_truth = track_mets(np.expand_dims(final_tracked_ground_truth_label[i,...], axis=0), conn_comp_prediction_to_ground_truth[i,...], total_num_tracked_METS_prediction_to_ground_truth)
            #save tracked label map
            save_tracked_mets(patient, visits, model_prediction_label_name, final_tracked_prediction_to_ground_truth_label, save_name_prediction_to_ground_truth, BIDS_format_names)

def verify_segmentations_exist(patient, visits, track_ground_truth, ground_truth_name, track_prediction, model_prediction_label_name, BIDS_format_names):
    patient_dir = nifti_dir + patient + '/'
    verified_visits = []
    for i, visit in enumerate(visits):
        os.chdir(patient_dir + visit)
        found_ground_truth = False
        found_prediction = False
        found_ground_truth = verify_segmentations_exists_helper(patient, visit, BIDS_format_names, track_ground_truth, ground_truth_name, found_ground_truth)
        found_prediction = verify_segmentations_exists_helper(patient, visit, BIDS_format_names, track_prediction, model_prediction_label_name, found_prediction)
        if (found_ground_truth == True) and (found_prediction == True):
            verified_visits.append(visit)
    return verified_visits

def verify_segmentations_exists_helper(patient, visit, BIDS_format_names, track_file, label_file_name, found_file):
    if track_file == True:
        if BIDS_format_names == True:
            label_file_name = os.path.split(patient + '_' + visit)[0] + label_file_name
        if os.path.exists(label_file_name):
            found_file = True
    else:
        found_file = True
    return found_file

def initialize_variables_for_track_mets(patient, visits, num_time_points, label_file_name, BIDS_format_names):
    patient_dir = nifti_dir + patient + '/'
    #load in untracked label for patient
    for i, visit in enumerate(visits):
        os.chdir(patient_dir + visit)
        if BIDS_format_names == True:
            if i == 0:
                temp_label_file_name = label_file_name
            label_file_name = os.path.split(patient + '_' + visit)[0] + temp_label_file_name
        temp_seg = (nib.load(label_file_name).get_fdata() > 0).astype(int)
        if i == 0:
            seg = np.zeros((num_time_points,) + temp_seg.shape)
        seg[i, ...] = temp_seg
    #generate connected components
    conn_comp = np.zeros(seg.shape)
    for i in range(0, num_time_points):
        conn_comp[i,...] = label(seg[i,...], connectivity=connectivity)
    #total number of tracked mets is initialized as number in first time point
    total_num_tracked_METS = np.unique(conn_comp[0,...]).shape[0] - 1
    #initialize array for final_tracked labels
    final_tracked_label = np.zeros(conn_comp.shape)
    final_tracked_label[0,...] = conn_comp[0,...]
    return conn_comp, final_tracked_label, total_num_tracked_METS

def track_mets(reference_vol, conn_comp_vol, total_num_tracked_METS):
    #get all unique METS in previous time points
    unique_vals = np.unique(reference_vol)
    #create new volume to hold tracked METS
    tracked_METS = np.zeros(conn_comp_vol.shape)
    #copy connected components volume
    current_time_point_CC = np.copy(conn_comp_vol)
    #first val is background so ignore
    for j, unique_val in enumerate(unique_vals[1:]):
        temp_seg = np.copy(reference_vol[-1,...])
        temp_seg = (temp_seg == unique_val).astype(int)
        #tells us what values are overlapping
        overlapping_METS = np.multiply(temp_seg, current_time_point_CC)
        unique_vals_prediction = np.unique(overlapping_METS)
        #if one overlapping value, then know which lesion to track
        tracked_lesion = unique_vals_prediction[-1]
        #if more than one overlapping value, find MET with largest overlap area
        if len(unique_vals_prediction) > 2:
            ground_truth_seg = np.copy(reference_vol[-1,...])
            ground_truth_seg = (ground_truth_seg == unique_val).astype(int)
            temp_seg = np.copy(conn_comp_vol)
            max_overlap_region = 0
            for k, unique_val_prediction in enumerate(unique_vals_prediction[1:]):
                temp_lesion = (temp_seg == unique_val_prediction).astype(int)
                #find overlap between this lesion and current lesion
                overlap_region = np.sum(np.multiply(ground_truth_seg, temp_lesion))
                if overlap_region > max_overlap_region:
                    max_overlap_region = overlap_region
                    tracked_lesion = unique_val_prediction
        #by here have found maximum overlap lesion so use that as tracked lesion
        #first check to make sure that there is at least one overlapping value (other than background)
        if len(unique_vals_prediction) > 1:
            #add tracked METS to new volume
            temp_seg = np.copy(conn_comp_vol)
            temp_seg = (temp_seg == tracked_lesion).astype(int) * unique_val
            #only add regions that are not overlapping
            ####tracked_METS[(temp_seg > 0) & (tracked_METS == 0)] = unique_val
            tracked_METS = tracked_METS + temp_seg
            #remove this tracked METS from the connected components volume so that it doesn't get re-selected in future loop iterations
            current_time_point_CC[current_time_point_CC == tracked_lesion] = 0
    #finished tracking all mets for this time point.
    #there may be new mets in this time point that should be given unique tracking number
    if np.any(current_time_point_CC):
        remaining_METS = np.unique(current_time_point_CC)
        for l, remaining_MET in enumerate(remaining_METS[1:]):
            #increment number of tracked mets
            total_num_tracked_METS = total_num_tracked_METS + 1
            #add (un)tracked METS to new volume
            temp_seg = np.copy(conn_comp_vol)
            temp_seg = (temp_seg == remaining_MET).astype(int) * total_num_tracked_METS
            #only add regions that are not overlapping
            ####tracked_METS[(temp_seg > 0) & (tracked_METS == 0)] = total_num_tracked_METS
            tracked_METS = tracked_METS + temp_seg
    #return outputs
    return tracked_METS, total_num_tracked_METS

def save_tracked_mets(patient, visits, label_file_name, final_tracked_label, save_name, BIDS_format_names):
    patient_dir = nifti_dir + patient + '/'
    final_tracked_label = final_tracked_label.astype(int)
    #loop through visits and save out final tracked labels
    for i,visit in enumerate(visits):
        os.chdir(patient_dir + visit)
        if BIDS_format_names == True:
            if i == 0:
                temp_label_file_name = label_file_name
                temp_save_name = save_name
            label_file_name = os.path.split(patient + '_' + visit)[0] + temp_label_file_name
            save_name = label_file_name[:-13] + temp_save_name
        nib_vol = nib.load(label_file_name)
        visit_tracked_label = nib.Nifti1Image(final_tracked_label[i,...], affine=nib_vol.affine, header=nib_vol.header)
        nib.save(visit_tracked_label, save_name)

if __name__ == "__main__":
    #Directories
    base_dir = '/workspace/brain_mets_seg/data/'
    nifti_dir = base_dir + 'files_to_track/'
    ground_truth_name = ''
    model_prediction_label_name = 'model_ensemble-label.nii.gz'
    save_name_ground_truth = ''
    save_name_prediction = model_prediction_label_name[:-13] + '_TRACKED-label.nii.gz'
    save_name_prediction_to_ground_truth = model_prediction_label_name[:-13] + '_TRACKED_TO_GROUND_TRUTH-label.nii.gz'
    track_ground_truth = False
    track_prediction = True
    BIDS_format_names = False
    connectivity = 3 #full 3D connectivity for connected components
    #
    os.chdir(nifti_dir)
    patients = sorted(next(os.walk('.'))[1])
    num_cores = int(multiprocessing.cpu_count() * 0.25)
    Parallel(n_jobs=num_cores)(delayed(track_patient_across_visits)(patient, track_ground_truth=track_ground_truth, track_prediction=track_prediction, BIDS_format_names=BIDS_format_names) for patient in patients)
