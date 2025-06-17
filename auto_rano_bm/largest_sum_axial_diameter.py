#imports
import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk

from rano_function import rano

def met_maxes(img_path, roi_path, output_dir, outfile, vox_x = 1, tol = 1, thres = 10, output_images = False):
	roi_arr = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))
	met_maxes = []
	met_ids = np.unique(roi_arr)[1:]
	for met in met_ids:
		axial_maxes = []
		clip_roi_arr = np.zeros(roi_arr.shape)
		clip_roi_arr[np.where(roi_arr == met)] = 1
		clip_roi_arr = clip_roi_arr.astype(np.int)
		for slice in np.unique(np.where(clip_roi_arr>0)[0]):
			curr_maj, curr_min = rano(clip_roi_arr[slice,:,:], tol = tol, output_file=None, background_image=None, vox_x = vox_x, thres = thres)
			axial_maxes.append(round(curr_maj,4))
		if output_images:
			img_arr = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
			maximal_slice_idx = np.where(axial_maxes == np.max(axial_maxes))[0][0]
			maximal_slice = np.unique(np.where(clip_roi_arr>0)[0])[maximal_slice_idx]
			curr_maj, curr_min = rano(clip_roi_arr[maximal_slice,:,:], tol = tol, output_file=os.path.join(output_dir,outfile+'_'+str(met)+'.jpg'), background_image=img_arr[maximal_slice,:,:], vox_x = vox_x, thres = 10)
		met_maxes.append(np.max(axial_maxes))
	return np.array(met_ids), np.array(met_maxes)
	
def single_visit_diams(mets, diams, thres):
	target_mets = mets[np.where(diams >= thres)[0]]
	target_diams = diams[np.where(diams >= thres)[0]]
	LSAD = sum(sorted(target_diams, reverse=True)[0:5])
	return LSAD
    
def single_visit_vols(mets, diams, roi_path, thres):
    target_mets = mets[np.where(diams >= thres)[0]]
    roi_arr = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))
    met_ids, met_quants = np.unique(roi_arr, return_counts = True)
    met_vols = np.zeros(len(target_mets))
    for i, met in enumerate(target_mets):
        met_vols[i] = met_quants[np.where(met_ids == met)]
    return np.sum(met_vols)
	
def rano_measures(baseline_mets, baseline_diams, baseline_roi_path, baseline_thres=10, visit_mets=None, visit_diams=None, visit_roi_path=None, visit_thres=10, patient_target_lesions=None):
	baseline_target_mets = baseline_mets[np.where(baseline_diams >= baseline_thres)[0]]
	baseline_target_met_diams = baseline_diams[np.where(baseline_diams >= baseline_thres)[0]]
	indices_top_five_biggest_baseline_lesions = np.argsort(baseline_target_met_diams)[::-1][:5]
	diam_1 = np.sum(baseline_target_met_diams[indices_top_five_biggest_baseline_lesions])
	if np.all(visit_mets) == None:
		#only calculate volumes if this is the baseline visit to reduce computation
		baseline_roi_arr = sitk.GetArrayFromImage(sitk.ReadImage(baseline_roi_path))
		baseline_total_volume = np.sum((baseline_roi_arr > 0).astype(int))
		baseline_target_mets_volume = 0
		for temp_ind in indices_top_five_biggest_baseline_lesions:
			baseline_target_mets_volume = baseline_target_mets_volume + np.sum((baseline_roi_arr == baseline_target_mets[temp_ind]).astype(int))
		if diam_1 == 0:
			new_mets = False
		else:
			new_mets = True
		baseline_num_target_lesions = len(indices_top_five_biggest_baseline_lesions)
		baseline_num_total_lesions = len(np.unique(baseline_roi_arr)) - 1
		return diam_1, baseline_total_volume, baseline_target_mets_volume, baseline_num_target_lesions, baseline_num_total_lesions, new_mets
	else:
		visit_target_mets = visit_mets[np.where(visit_diams >= visit_thres)[0]]
		visit_new_target_mets = np.array([f for f in visit_target_mets if f not in patient_target_lesions])
		new_mets = len(visit_new_target_mets)>0
		remaining_target_lesions = np.array([f for f in visit_mets if f in patient_target_lesions])
		visit_include_mets = np.unique(np.concatenate([visit_target_mets, remaining_target_lesions]))
		visit_include_met_diams = visit_diams[np.in1d(visit_mets,visit_include_mets).nonzero()[0]]
		visit_roi_arr = sitk.GetArrayFromImage(sitk.ReadImage(visit_roi_path))
		indices_top_five_biggest_visit_lesions = np.argsort(visit_include_met_diams)[::-1][:5]
		visit_total_volume = np.sum((visit_roi_arr > 0).astype(int))
		visit_target_mets_volume = 0
		for temp_ind in indices_top_five_biggest_visit_lesions:
			visit_target_mets_volume = visit_target_mets_volume + np.sum((visit_roi_arr == visit_include_mets[temp_ind]).astype(int))
		#
		diam_2 = np.sum(visit_include_met_diams[indices_top_five_biggest_visit_lesions])
		visit_num_target_lesions = len(indices_top_five_biggest_visit_lesions)
		visit_num_total_lesions = len(np.unique(visit_roi_arr)) - 1
		return diam_2, visit_total_volume, visit_target_mets_volume, remaining_target_lesions, visit_num_target_lesions, visit_num_total_lesions, new_mets