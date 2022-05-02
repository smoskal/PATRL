import sys
sys.path.append('../')

from aif.AttackStagesCondensed import MicroAttackStageCondensed
import torch
from aif.AIF_Mappings import RecentAlertsMapping
from utils import LearningUtils
from SignatureAttackStagePredictor import SignatureAttackStagePredictor
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
	
	filtered_data = []
	target_filter = MicroAttackStageCondensed.CREDENTIAL_ACCESS.value
	for sig in main_data :
		if sig[0] == target_filter : 
			filtered_data.append(sig)
	
	classifier = SignatureAttackStagePredictor(pred_model='CUSTOM', model_loc='./cve_lang_model_2') 
	
	
	
	