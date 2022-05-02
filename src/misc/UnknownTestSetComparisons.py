import sys

import pandas as pd

sys.path.append('../')
from aif.AttackStagesCondensed import MicroAttackStageCondensed
import torch
import statistics
import pickle as pkl
from aif.AIF_Mappings import RecentAlertsMapping
from utils import LearningUtils
import SignatureTransferLearningNewData
from SignatureAttackStagePredictor import SignatureAttackStagePredictor
from tqdm import tqdm 

out_path = '../data_store/'
if not torch.cuda.is_available():
	print("ERROR: CUDA not available.")
else :
	torch.cuda.set_device(2)
	torch.multiprocessing.freeze_support()
	
def process_uncertainty_stats(uncertainty_preds, output_df=True, remove_deleted=True) :

	output_preds = []
	for stg, preds in uncertainty_preds.items() :
		for pred in preds : 
			sig = pred[1][0]
			if remove_deleted :
				
				if "DELETED" in sig :
					pass
				else :
					temp_data = [sig, stg.value, pred[2][0]]
				
			else :
				temp_data = [sig, stg.value, pred[2][0]]
			output_preds.append(temp_data)
			
	if output_df :
		output_df = pd.DataFrame(output_preds, columns=['Sig', 'Label', 'Uncert'])
		return output_df
	else : return output_preds
	
def process_uncertainty_stats_class(uncertainty_preds) :

	output_preds = {}
	for stg, preds in uncertainty_preds.items() :
		class_preds = []
		for pred in preds : 
			temp_data = [pred[1][0], stg.value, pred[2][0]]
			class_preds.append(temp_data)
		temp_df = pd.DataFrame(class_preds, columns=['Sig', 'Label', 'Uncert'])
		output_preds[stg] = temp_df
		
	return output_preds
	
def calculate_mc_preds(classifier_model, test_data) :
	classifier = SignatureAttackStagePredictor(pred_model=classifier_model)
	signatures = test_data['Sig'].values.tolist()
	predictions = []
	
	for sig in tqdm(signatures):
		mc_k_pred, mc_conf = classifier.predict_k_mc(sig)
		temp_pred = [sig, mc_k_pred[0], mc_conf[0]]
		predictions.append(temp_pred)
		
	out_df = pd.DataFrame(predictions, columns=['Sig', 'Label', 'Uncert'])
	return out_df
	
def uncertainty_statistics(uncertainty_preds) :
	uncerts = uncertainty_preds['Uncert'].values.tolist()
	
	mean = statistics.mean(uncerts)
	min_val = min(uncerts)
	max_val = max(uncerts)
	std = statistics.stdev(uncerts)
	output_stats = [mean, std, min_val, max_val]
	return output_stats
	
def sample_uncertainty_df(data_df, sample_size, mode='RANDOM') : 

	if mode == 'RANDOM' :
		sampled_data = data_df.sample(sample_size)
		data_df = data_df.drop(sampled_data.index)
	elif mode == 'TOP_UNCERT' :
		sorted_df = data_df.sort_values(by='Uncert',ascending=False)
		sampled_data = sorted_df[:sample_size]
		data_df = data_df.drop(sampled_data.index)
	elif mode == 'LEAST_UNCERT' :
		sorted_df = data_df.sort_values(by='Uncert',ascending=True)
		sampled_data = sorted_df[:sample_size]
		data_df = data_df.drop(sampled_data.index)
	
	return sampled_data
	
def sample_uncertainty_df_per_class(uncertainty_preds, sample_size, mode='RANDOM') :

	temp_preds = []
	
	for stg, preds in uncertainty_preds.items() :
		sample_pred = sample_uncertainty_df(preds, sample_size, mode=mode)
		temp_preds.append(sample_pred)
		
	out_df = pd.concat(temp_preds)
	return out_df
	
def sample_recomputed_uncertainty(recomputed_uncertainty, uncertainty_preds, sample_size, mode='RANDOM') :

	recomputed_sample = sample_uncertainty_df(recomputed_uncertainty, sample_size, mode=mode) 
	temp_recomputed = recomputed_uncertainty.drop(recomputed_sample.index)
	additional_samples = sample_uncertainty_df(uncertainty_preds, sample_size, mode=mode)
	
	to_be_recomputed_samples = [temp_recomputed, additional_samples]
	recomputed_uncertainty_new = pd.concat(to_be_recomputed_samples)
	
	return recomputed_sample, recomputed_uncertainty_new
	
def update_pseudo_labels(label_df, model:SignatureAttackStagePredictor) :
	columns = list(label_df)
	data = label_df.values.tolist()
	out_data = []
	
	
	for sig in data :
		new_pred = model.predict(sig[0])
		temp_data = [sig[0], new_pred.value, sig[2]]
		out_data.append(temp_data)
		
	out_df = pd.DataFrame(out_data, columns=columns)
	return out_df
	
def get_label_stats(data) :
	stats = data['Label'].value_counts().sort_index()
	return stats
	
def recompute_mc_uncertainty(pred_model, recomputed_uncertainty) :
	
	mc_list = recomputed_uncertainty.values.tolist()
	predictor = SignatureAttackStagePredictor(pred_model=pred_model)
	mc_list_new = []
	
	for sample in tqdm(mc_list) :
		k_stg, k_conf = predictor.predict_k_mc(sample[0])
		temp_sample = [sample[0], k_stg[0].value, k_conf[0]]
		mc_list_new.append(temp_sample) 
		
	out_df = pd.DataFrame(mc_list_new, columns=['Sig','Label','Uncert'])
	return out_df
	
def get_label_stats(data) :
	stats = data['Label'].value_counts().sort_index()
	return stats
	
def process_testing_set(test_set) :
	
	temp_list = test_set.values.tolist()
	out_list = []
	
	for val in temp_list :
		out_list.append([val[0], MicroAttackStageCondensed[val[1]].value])
		
	out_df = pd.DataFrame(out_list, columns=['Sig', 'Label'])
	
	return out_df
	
if __name__ == "__main__" :

	LABELED_DATA_SPLIT = .2
	SEED = 69
	SAMPLE_MODE = 'TOP_UNCERT'
	RECOMPUTE_UNCERT = True
	RECOMPUTE_SIZE = 1500
	ITERATION_SIZE = 250
	RECOMPUTE_ITER_STEP = 500
	RELABEL_PSEUDO = True
	tok = SpacyTokenizer('en')
	the_tokenizer = Tokenizer()

	cptc_data_cond, class_labels, class_map = LearningUtils.signature_to_data_frame_condensed(transform_labels=False)
	ccdc_data_cond = LearningUtils.transform_dataset_condensed(RecentAlertsMapping().ccdc_combined, class_map)
	all_data = [cptc_data_cond, ccdc_data_cond]
	main_df = pd.concat(all_data)
	main_df['i'] = main_df.index
	
	uncertainty_preds = pkl.load(open('suricata_uncertainty_predictions_sept22.pkl', 'rb'))
	class_suricata_mc_preds =process_uncertainty_stats_class(uncertainty_preds)
	suricata_mc_preds = process_uncertainty_stats(uncertainty_preds, output_df=True)
	
	train_df_sig, test_df_sig = LearningUtils.split_train_test(main_df, split=LABELED_DATA_SPLIT, seed=SEED)
	suricata_main_preds, suricata_testing_preds = LearningUtils.split_train_test(suricata_mc_preds, split=.015, seed=SEED)
	suricata_testing_preds = pd.read_pickle('./suricata_unknown_testing_dataset.pkl')
	
	recomputed_mc_preds = sample_uncertainty_df(suricata_mc_preds, RECOMPUTE_SIZE, mode=SAMPLE_MODE)
	
	unk_test_gt = process_testing_set(pd.read_csv('./unknown_gt_labels.csv'))
	
	# lang_model, data_lm, encoder_name = SignatureTransferLearningNewData.signature_transfer_learning_language_model(mode='normal', additional_samples=None, bs_lang=64, test_mode=False, random_seed=0)
	
	# gt_model_1, top1_gt, topk_gt,_,_,_,_,_  = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(main_df, main_df, encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
	
	# gt_model_2, top1_gt_unk, topk_gt_unk ,_,_,_,_,_  = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(main_df, unk_test_gt, encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
	
	# unk_model_1, top1_unk, topk_unk,_,_,_,_,_  = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(unk_test_gt, unk_test_gt, encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
	
	# unk_model_2, top1_unk_gt, topk_unk_gt,_,_,_,_,_  = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(unk_test_gt, main_df, encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
	
	unk_model_2, top1_unk_gt, topk_unk_gt,_,_,_,_,_  = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(main_df, unk_test_gt, None, None, class_map, class_labels, bs_sig=16, test_mode=False)
	
	#gt_model, top1_gt, topk_gt ,class_map_gt, miss_counts_gt, miss_stats_gt, data_stats_gt, miss_record_gt = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(train_df_sig, test_df_sig,encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)