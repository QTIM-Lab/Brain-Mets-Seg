import os
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from auto_rano_bm.largest_sum_axial_diameter import met_maxes, rano_measures

def response_category(baseline_rano_measure, visit_rano_measure, nadir_rano_measure, remaining_target_lesions):
    response = 'N/A'
    #progressive disease
    if nadir_rano_measure == 0:
        if visit_rano_measure > 0:
            response = 'PD'
        else:
            response = 'SD'
    elif (visit_rano_measure / nadir_rano_measure) >= 1.2:
        response = 'PD'
    #stable disease
    elif ((visit_rano_measure / nadir_rano_measure) < 1.2) and ((1 - (visit_rano_measure / baseline_rano_measure)) < 0.3):
        response = 'SD'
    #partial response or complete response
    elif (1 - (visit_rano_measure / baseline_rano_measure)) >= 0.3:
        if visit_rano_measure == 0:
            response = 'CR'
        else:
            response = 'PR'
    elif len(remaining_target_lesions) > 0:
        response = 'PR'
    #
    return response

def auto_rano_bm_all_patients():
    #measure the diameters of all lesions, and then filter for only those that are larger than the requested threshold
    lesion_measure_threshold = 1
    rano_threshold = 10
    df_final = pd.DataFrame()
    for patient in patients:
        patient_dir = nifti_dir + patient + '/'
        os.chdir(patient_dir)
        visits = next(os.walk('.'))[1]
        visits.sort()
        nadir_rano_measure_true = 0
        nadir_rano_measure_pred = 0
        patient_target_lesions_true = []
        patient_target_lesions_pred = []
        for i, visit in enumerate(visits):
            visit_dict = {}
            visit_dir = patient_dir + visit + '/'
            os.chdir(visit_dir)
            visit_dict['patient_id'] = patient
            visit_dict['patient_visit'] = visit
            visit_dict['full_patient_path'] = visit_dir
            visit_roi_path_true = visit_dir + ROI_names[0]
            visit_roi_path_pred = visit_dir + ROI_names[1]
            if i == 0:
                #baseline visit true mask
                baseline_mets_true, baseline_diams_true = met_maxes(None, visit_roi_path_true, None, None, vox_x = 1, tol = 1, thres = lesion_measure_threshold, output_images = False)
                baseline_rano_measure_true, baseline_total_volume_true, baseline_target_mets_volume_true, baseline_num_target_lesions_true, baseline_num_total_lesions_true, baseline_new_mets_true = rano_measures(baseline_mets_true, baseline_diams_true, visit_roi_path_true, baseline_thres = rano_threshold)
                #baseline visit pred mask
                baseline_mets_pred, baseline_diams_pred = met_maxes(None, visit_roi_path_pred, None, None, vox_x = 1, tol = 1, thres = lesion_measure_threshold, output_images = False)
                baseline_rano_measure_pred, baseline_total_volume_pred, baseline_target_mets_volume_pred, baseline_num_target_lesions_pred, baseline_num_total_lesions_pred, baseline_new_mets_pred = rano_measures(baseline_mets_pred, baseline_diams_pred, visit_roi_path_pred, baseline_thres = rano_threshold)
                #
                visit_dict['total_volume_true'] = baseline_total_volume_true
                visit_dict['target_mets_volume_true'] = baseline_target_mets_volume_true
                visit_dict['total_volume_pred'] = baseline_total_volume_pred
                visit_dict['target_mets_volume_pred'] = baseline_target_mets_volume_pred
                visit_dict['rano_measure_true'] = baseline_rano_measure_true
                visit_dict['rano_measure_pred'] = baseline_rano_measure_pred
                visit_dict['new_target_lesions_true'] = baseline_new_mets_true
                visit_dict['new_target_lesions_pred'] = baseline_new_mets_pred
                visit_dict['num_target_lesions_true'] = baseline_num_target_lesions_true
                visit_dict['num_target_lesions_pred'] = baseline_num_target_lesions_pred
                visit_dict['num_total_lesions_true'] = baseline_num_total_lesions_true
                visit_dict['num_total_lesions_pred'] = baseline_num_total_lesions_pred
                #
                nadir_rano_measure_true = baseline_rano_measure_true
                nadir_rano_measure_pred = baseline_rano_measure_pred
                #
                patient_target_lesions_true = list(set(patient_target_lesions_true + list(baseline_mets_true)))
                patient_target_lesions_pred = list(set(patient_target_lesions_pred + list(baseline_mets_pred)))
            else:
                #visit true mask
                visit_mets_true, visit_diams_true = met_maxes(None, visit_roi_path_true, None, None, vox_x = 1, tol = 1, thres = lesion_measure_threshold, output_images = False)
                visit_rano_measure_true, visit_total_volume_true, visit_target_mets_volume_true, remaining_target_lesions_true, visit_num_target_lesions_true, visit_num_total_lesions_true, visit_new_mets_true = rano_measures(baseline_mets_true, baseline_diams_true, None, baseline_thres = rano_threshold, visit_mets = visit_mets_true, visit_diams = visit_diams_true, visit_roi_path = visit_roi_path_true, visit_thres = rano_threshold, patient_target_lesions = patient_target_lesions_true)
                #visit pred mask
                visit_mets_pred, visit_diams_pred = met_maxes(None, visit_roi_path_pred, None, None, vox_x = 1, tol = 1, thres = lesion_measure_threshold, output_images = False)
                visit_rano_measure_pred, visit_total_volume_pred, visit_target_mets_volume_pred, remaining_target_lesions_pred, visit_num_target_lesions_pred, visit_num_total_lesions_pred, visit_new_mets_pred = rano_measures(baseline_mets_pred, baseline_diams_pred, None, baseline_thres = rano_threshold, visit_mets = visit_mets_pred, visit_diams = visit_diams_pred, visit_roi_path = visit_roi_path_pred, visit_thres = rano_threshold, patient_target_lesions = patient_target_lesions_pred)
                #
                visit_dict['total_volume_true'] = visit_total_volume_true
                visit_dict['target_mets_volume_true'] = visit_target_mets_volume_true
                visit_dict['total_volume_pred'] = visit_total_volume_pred
                visit_dict['target_mets_volume_pred'] = visit_target_mets_volume_pred
                visit_dict['rano_measure_true'] = visit_rano_measure_true
                visit_dict['rano_measure_pred'] = visit_rano_measure_pred
                visit_dict['new_target_lesions_true'] = visit_new_mets_true
                visit_dict['new_target_lesions_pred'] = visit_new_mets_pred
                visit_dict['num_target_lesions_true'] = visit_num_target_lesions_true
                visit_dict['num_target_lesions_pred'] = visit_num_target_lesions_pred
                visit_dict['num_total_lesions_true'] = visit_num_total_lesions_true
                visit_dict['num_total_lesions_pred'] = visit_num_total_lesions_pred
                #
                if nadir_rano_measure_true >= visit_rano_measure_true:
                    nadir_rano_measure_true = visit_rano_measure_true
                if nadir_rano_measure_pred >= visit_rano_measure_pred:
                    nadir_rano_measure_pred = visit_rano_measure_pred
                #
                patient_target_lesions_true = list(set(patient_target_lesions_true + list(visit_mets_true)))
                patient_target_lesions_pred = list(set(patient_target_lesions_pred + list(visit_mets_pred)))
            #
            #calculate response category if this is not baseline
            if i == 0:
                visit_dict['response_category_true'] = 'N/A'
                visit_dict['response_category_pred'] = 'N/A'
            else:
                visit_dict['response_category_true'] = response_category(baseline_rano_measure_true, visit_rano_measure_true, nadir_rano_measure_true, remaining_target_lesions_true)
                visit_dict['response_category_pred'] = response_category(baseline_rano_measure_pred, visit_rano_measure_pred, nadir_rano_measure_pred, remaining_target_lesions_pred)
            #
            df_final = df_final.append(visit_dict, ignore_index=True)
    #
    df_final.to_csv(base_dir + folder + '_rano.csv')

if __name__ == "__main__":
    #Directories
    base_dir = '/workspace/brain_mets_seg/data/'
    folder = 'files_to_track'
    nifti_dir = base_dir + folder + '/'
    #
    os.chdir(nifti_dir)
    patients = sorted(next(os.walk('.'))[1])
    ROI_names = ['ROI_RAI_RESAMPLED_BINARY_REG_TRACKED-label.nii.gz', 'model_ensemble_TRACKED_TO_GROUND_TRUTH-label.nii.gz']
    auto_rano_bm_all_patients()