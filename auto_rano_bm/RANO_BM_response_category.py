#imports
import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd

os.chdir('/root/AUTO_RANO_BM/code') 
from largest_sum_axial_diameter import met_maxes, dual_visit_diams

def category_logic(baseline_img_path, baseline_roi_path, img_path, roi_path, output_dir, output_images, immunotherapy = True):
	baseline_mets, baseline_diams = met_maxes(baseline_img_path, baseline_roi_path, output_dir, vox_x = 1, tol = 1, thres = 1, output_images = output_images)
	visit_mets, visit_diams = met_maxes(img_path, roi_path, output_dir, vox_x = 1, tol = 1, thres = 1, output_images = output_images)
	diam_1, diam_2, new_mets = dual_visit_diams(baseline_mets, baseline_diams, visit_mets, visit_diams, baseline_thres = 10, visit_thres = 10)
	
	if immunotherapy:
		if diam_1 == 0 and diam_2 != 0: #added this to avoid division by 0 
			response = 'PD'
		elif diam_1 == 0 and diam_2 == 0: #added this to avoid division by 0 
			response = 'SD'
		elif (diam_2/diam_1)>=1.2:
			response = 'PD'
		elif 1.2>(diam_2/diam_1) and 0.3>(1-(diam_2/diam_1)):
			response = 'SD'
		elif 0.3<=(1-(diam_2/diam_1))<1:
			response = 'PR'
		elif remaining_target_lesions:
			response = 'PR'
		elif diam_2 == 0:
			response = 'CR'
		else:
			print(p)
			print('New mets? ',new_mets)
			print('Baseline Diameter: ',diam_1)
			print('New Visit Diameter: ',diam_2)
	elif not immunotherapy:
		if new_mets:
			response = 'PD'
		elif not new_mets and (diam_2/diam_1)>=1.2:
			resposne = 'PD'
		elif not new_mets and 1.2>(diam_2/diam_1) and 0.3>(1-(diam_2/diam_1)):
			response = 'SD'
		elif not new_mets and 0.3<=(1-(diam_2/diam_1))<1:
			response = 'PR'
		elif remaining_target_lesions:
			response = 'PR'
		elif not new_mets and diam_2 == 0:
			response = 'CR'
		else:
			print(p)
			print('New mets? ',new_mets)
			print('Baseline Diameter: ',diam_1)
			print('New Visit Diameter: ',diam_2)
	
	return response