import sys
sys.path.append('../')

import pandas as pd
import sys
from AttackStagesCondensed import MicroAttackStage
from AttackStagesCondensed import MacroAttackStage
from AttackStagesCondensed import MicroToMacroMapping
from AttackStagesCondensed import MicroToMacroCondensedMapping
from AttackStagesCondensed import MicroAttackStageCondensed
import pickle
from fastai import *
from fastai.text import *
import re
import torch
from AIF_Mappings import RecentAlertsMapping
import random
import LearningUtils
from SignatureAttackStagePredictor import SignatureAttackStagePredictor
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

out_path = './'
if not torch.cuda.is_available():
	print("ERROR: CUDA not available.")
else :
	torch.cuda.set_device(1)
	torch.multiprocessing.freeze_support()

def box_plot(class_uncerts, filename, title='') :
	matplotlib.use('Agg')
	data = []
	names = []
	for stg, uncerts in class_uncerts.items() :
		names.append(stg.name)
		data.append(uncerts)
		
	plt.clf()
	plt.boxplot(data, labels=names)
	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.title(title)
	plt.ylim(0, .4)
	plt.savefig(filename, dpi=400)
	return 

if __name__ == '__main__' :

	cptc_data_cond, class_labels, class_map = LearningUtils.signature_to_data_frame_condensed(transform_labels=False, output_dict=True)
	ccdc_data_cond = LearningUtils.transform_dataset_condensed(RecentAlertsMapping().ccdc_combined, class_map, output_dict=True)
	
	main_data = []
	main_data.extend(cptc_data_cond)
	main_data.extend(ccdc_data_cond)
	
	classifier = SignatureAttackStagePredictor() 
	
	top1_uncerts = []
	topk_uncerts = []
	miss_uncerts = []
	
	top1_class_uncerts = {}
	topk_class_uncerts = {}
	miss_class_uncerts = {}
	
	top1_gt_uncerts = {}
	topk_gt_uncerts = {}
	miss_gt_uncerts = {}
	overall_gt_uncerts = {}
	
	for test_sig in tqdm(main_data) :
		gt = MicroAttackStageCondensed(test_sig[0])
		in_text = test_sig[1]
		
		k_preds, uncert_val = classifier.predict_mc_uncertainty(in_text)
		
		if gt not in top1_class_uncerts : 
			top1_class_uncerts[gt] = []
			topk_class_uncerts[gt] = []
			miss_class_uncerts[gt] = []
			top1_gt_uncerts[gt] = []
			topk_gt_uncerts[gt] = []
			miss_gt_uncerts[gt] = []
			overall_gt_uncerts[gt] = []
			
		overall_gt_uncerts[gt].append([gt, in_text ,uncert_val])
		
		if k_preds[0] == gt :
			top1_uncerts.append(uncert_val)
			top1_class_uncerts[gt].append(uncert_val)
			top1_gt_uncerts[gt].append([gt, in_text ,uncert_val, k_preds])
			
		elif gt in k_preds :
			topk_uncerts.append(uncert_val)
			topk_class_uncerts[gt].append(uncert_val)
			topk_gt_uncerts[gt].append([gt, in_text ,uncert_val, k_preds])
		else :
			miss_uncerts.append(uncert_val)
			miss_class_uncerts[gt].append(uncert_val)
			miss_gt_uncerts[gt].append([gt, in_text ,uncert_val, k_preds])
			
	