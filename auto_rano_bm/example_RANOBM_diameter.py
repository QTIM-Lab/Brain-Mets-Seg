#imports
import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd

os.chdir('/root/AUTO_RANO_BM/code') 
from largest_sum_axial_diameter import met_maxes,single_visit_diams

mets, diams = met_maxes('/root/AUTO_RANO_BM/Sample_patient/img.nii.gz', '/root/AUTO_RANO_BM/Sample_patient/roi.nii.gz', '/root/AUTO_RANO_BM/Sample_output/', 'example_vis', vox_x = 1, tol = 1, thres = 10, output_images = True)
LSAD = single_visit_diams(mets,diams,10)
print(LSAD)