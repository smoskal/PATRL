import sys
sys.path.append('../')

from scipy.stats import entropy
import numpy as np
from aif.AttackStagesCondensed import MicroAttackStageCondensed
import torch
from aif.AIF_Mappings import RecentAlertsMapping
from utils import LearningUtils
from SignatureAttackStagePredictor import SignatureAttackStagePredictor
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def jsd(p, q):
	p = np.asarray(p, dtype=np.float64)
	q = np.asarray(q, dtype=np.float64)
	# normalize
	p /= p.sum()
	q /= q.sum()
	m = (p + q) / 2
	return (entropy(p, m) + entropy(q, m)) / 2
	
def tensor2list(data) :
	return [float(x) for x in data]
	
def assess_predicition_similarity(text, classifier, transform_proba=True) :

	normal_pred, normal_proba = classifier.predict_proba(text, transform_proba=transform_proba)
	mc_pred, mc_proba = classifier.predict_mc_proba(text, transform_proba=transform_proba, num_iters=20)
	normal_topk = torch.topk(normal_proba, 3) 
	normal_k_stgs = [MicroAttackStageCondensed(int(x)).name for x in normal_topk[1]]
	
	mc_k_stg, mc_k_uncert = classifier.mc_dropout_topk_uncertainty(mc_proba)
	mc_stgs = [MicroAttackStageCondensed(x).name for x in mc_k_stg]
	
	normal_proba_dist = tensor2list(list(normal_proba))
	flattened_mc_probs = []
	jsd_vals = []
	for mc_iter in mc_proba :
		iter_probs = tensor2list(list(mc_iter[2]))
		flattened_mc_probs.append(iter_probs)
		
		jsd_val = jsd(normal_proba_dist, iter_probs)
		jsd_vals.append(jsd_val)
		
	jsd_vals = np.asarray(jsd_vals)
	mc_pred_means = np.mean(flattened_mc_probs, axis=0)
	mean_jsd = jsd(normal_proba_dist, mc_pred_means)
	
	jsd_stats = [text, normal_pred.name, mc_pred.name, mean_jsd, np.mean(jsd_vals), jsd_vals.min(), jsd_vals.max(), np.std(jsd_vals), normal_k_stgs[0], normal_k_stgs[1], normal_k_stgs[2], float(normal_proba[int(normal_topk[1][0])]), float(normal_proba[int(normal_topk[1][1])]),float(normal_proba[int(normal_topk[1][2])]), mc_stgs[0],mc_stgs[1], mc_stgs[2],  mc_k_uncert[0],mc_k_uncert[1],mc_k_uncert[2],]

	return jsd_stats


if __name__ == '__main__' :

	cptc_data_cond, class_labels, class_map = LearningUtils.signature_to_data_frame_condensed(transform_labels=False, output_dict=True)
	ccdc_data_cond = LearningUtils.transform_dataset_condensed(RecentAlertsMapping().ccdc_combined, class_map, output_dict=True)
	
	main_data = []
	main_data.extend(cptc_data_cond)
	main_data.extend(ccdc_data_cond)
	
	classifier = SignatureAttackStagePredictor() 
	columns = ['Signature', 'No MC Predict', 'MC Predict', 'Mean MC Prediction JSD','Mean of Per Iter. JSD', 'Iter. Min JSD', 'Iter. Max JSD', 'Iter. STDEV JSD','1st Predict()','2nd Predict()','3rd Predict()', '1st Pred() Conf.', '2nd Pred() Conf.', '3rd Pred() Conf.', '1st MC_Pred()', '2nd MC_Pred()', '3rd MC_Pred()', '1st MC Uncert', '2nd MC Uncert', '3rd MC Uncert','GT Label']
	
	overall_jsd_stats = []
	for sig_sample in tqdm(main_data) :
		jsd_stats = assess_predicition_similarity(sig_sample[1], classifier)
		jsd_stats.append(MicroAttackStageCondensed(sig_sample[0]).name)
		overall_jsd_stats.append(jsd_stats)
		