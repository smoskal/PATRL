import sys
sys.path.append('../')

import pandas as pd
from aif.AttackStagesCondensed import MicroAttackStageCondensed
import torch
from SignatureAttackStagePredictor import SignatureAttackStagePredictor
from tqdm import tqdm
from interfaces import SnortRuleParser

out_path = './'
if not torch.cuda.is_available():
	print("ERROR: CUDA not available.")
else :
	torch.cuda.set_device(0)
	torch.multiprocessing.freeze_support()
	
def process_uncerts(overall_uncerts, num_uncerts=10) :
	low_uncerts = []
	high_uncerts = []
	for stg, uncerts in overall_uncerts.items() :
		temp_sort_low = sorted(uncerts, key=lambda x: x[2][0])
		temp_sort_high = sorted(uncerts, key=lambda x:x[2][0], reverse=True)
		
		for pred in temp_sort_low[0:num_uncerts] :
			temp = [pred[1][0], pred[0][0].name, pred[2][0], pred[0][1].name, pred[0][2].name, pred[2][1], pred[2][2]]
			low_uncerts.append(temp)

		for pred in temp_sort_high[0:num_uncerts] :
			temp = [pred[1][0], pred[0][0].name, pred[2][0], pred[0][1].name, pred[0][2].name, pred[2][1], pred[2][2]]
			high_uncerts.append(temp)
			
	columns = ['Signature', 'Top Pred.', 'Pred. Uncert.', '2nd Pred.', '3rd Pred.', '2nd Pred Uncert.', '3rd Pred Uncert.']	
	low_uncertainty = pd.DataFrame(low_uncerts, columns=columns)
	high_uncertainty = pd.DataFrame(high_uncerts, columns=columns)

	return low_uncertainty, high_uncertainty

if __name__ == '__main__' :

	main_data = SnortRuleParser.collect_signature_msg(as_list=True)
	classifier = SignatureAttackStagePredictor(pred_model='CUSTOM', model_loc='./cve_lang_model_oct22')
	
	overall_uncerts = {}
	
	for test_sig in tqdm(main_data) :
		in_text = test_sig
		#k_preds, uncert_val = classifier.predict_mc_uncertainty(in_text)
		k_preds, uncert_vals = classifier.predict_k_mc(in_text, k=5)
		top1 = MicroAttackStageCondensed(k_preds[0])
		
		if top1 not in overall_uncerts :
			overall_uncerts[top1] = []
			
		overall_uncerts[top1].append([k_preds, in_text ,uncert_vals])
		