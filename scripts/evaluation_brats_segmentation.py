from nnunet.evaluation.evaluator import NiftiEvaluator, run_evaluation
from nnunet.evaluation.metrics import dice
import glob
import numpy as np

ref = sorted(glob.glob('/mnt/sdb/nnUnet/data_training/nnUNet_raw/Task05_GTV_2modalities/labelsTs/*.nii.gz'))
test = sorted(glob.glob('/mnt/sdb/nnUnet/data_training/nnUNet_raw/Task05_GTV_2modalities/imagesTs_predicted/*.nii.gz'))
# ref = sorted(glob.glob('/mnt/sdb/nnUnet/data_training/nnUNet_raw_splitted/Task02_BraTS2015/labelsTs/*.nii.gz'))
# test = sorted(glob.glob('/mnt/sdb/nnUnet/data_training/nnUNet_raw_splitted/Task02_BraTS2015/imagesTs_res_new/*.nii.gz'))
all_dices = {}
labels = ['0', '1']
# labels = ['0', '1', '2', '3', '4']
for l in labels:
    all_dices[l] = []

for im, r in zip(test, ref):
    scores = run_evaluation([im, r, NiftiEvaluator(), {}])
    for label in scores:
        try:
            all_dices[label].append(scores[label]['Dice'])
        except:
            pass
for lab in all_dices:
    print('Median DSC for label {0}: {1}'.format(lab, np.median(all_dices[lab])))
