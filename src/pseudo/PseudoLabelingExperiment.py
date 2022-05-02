import sys

import pandas as pd
from fastai.text.core import SpacyTokenizer, Tokenizer

sys.path.append('../')
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
		sampled_data = suricata_main_preds.sample(sample_size)
		data_df.drop(sampled_data.index)
	elif mode == 'TOP_UNCERT' :
		sorted_df = data_df.sort_values(by='Uncert',ascending=False)
		sampled_data = sorted_df[:sample_size]
		data_df.drop(sampled_data.index)
	elif mode == 'LEAST_UNCERT' :
		sorted_df = data_df.sort_values(by='Uncert',ascending=True)
		sampled_data = sorted_df[:sample_size]
		data_df.drop(sampled_data.index)
	
	return sampled_data
	
def sample_uncertainty_df_per_class(uncertainty_preds, sample_size, mode='RANDOM') :

	temp_preds = []
	
	#NOTE THIS DIVIDES THE ITERATION SIZE BY THE NUMBER OF CLASSES, DIRTY I KNOW
	class_size = int(sample_size/11)
	
	for stg, preds in uncertainty_preds.items() :
		sample_pred = sample_uncertainty_df(preds, class_size, mode=mode)
		temp_preds.append(sample_pred)
		
	out_df = pd.concat(temp_preds)
	return out_df
	
def get_label_stats(data) :
	stats = data['Label'].value_counts().sort_index()
	return stats
	
if __name__ == "__main__" :

	LABELED_DATA_SPLIT = .2
	SEED = 69
	SAMPLE_MODE = 'LEAST_UNCERT'
	BALANCE_CLASS_LABELS = True


	tok = SpacyTokenizer('en')
	the_tokenizer = Tokenizer()

	cptc_data_cond, class_labels, class_map = LearningUtils.signature_to_data_frame_condensed(transform_labels=False)
	ccdc_data_cond = LearningUtils.transform_dataset_condensed(RecentAlertsMapping().ccdc_combined, class_map)
	all_data = [cptc_data_cond, ccdc_data_cond]
	main_df = pd.concat(all_data)
	main_df['i'] = main_df.index
	
	uncertainty_preds = pkl.load(open('suricata_uncertainty_predictions_oct22.pkl', 'rb'))
	class_suricata_mc_preds =process_uncertainty_stats_class(uncertainty_preds)
	suricata_mc_preds = process_uncertainty_stats(uncertainty_preds, output_df=True)
	
	train_df_sig, test_df_sig = LearningUtils.split_train_test(main_df, split=LABELED_DATA_SPLIT, seed=SEED)
	suricata_main_preds, suricata_testing_preds = LearningUtils.split_train_test(suricata_mc_preds, split=.015, seed=SEED)
	
	lang_model, data_lm, encoder_name = SignatureTransferLearningNewData.signature_transfer_learning_language_model(mode='all', additional_samples=None, bs_lang=64, test_mode=False, random_seed=0)
	
	# gt_model, top1_gt, topk_gt ,class_map_gt, miss_counts_gt, miss_stats_gt, data_stats_gt, miss_record_gt = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(main_df, main_df,None, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
	
	labeled_set_uncert_stats = []
	unknown_uncert_stats = []
	
	og_test_iter_preds = pd.DataFrame()
	unk_test_iter_preds = pd.DataFrame()
	
	og_test_iter_mc = pd.DataFrame()
	unk_test_iter_mc = pd.DataFrame()
	
	pl_class_dist = pd.DataFrame()
	
	for iter in range(0, 5001, 250) :
		print(f"Iteration Step: {iter}, {SAMPLE_MODE}, Class Bal: {BALANCE_CLASS_LABELS}")
		
		if iter > 0 :
			if not BALANCE_CLASS_LABELS :
				pseudo_labels = sample_uncertainty_df(suricata_main_preds, iter, mode=SAMPLE_MODE)
			else :
				print('CLASS BALANCING ACTIVE')
				pseudo_labels = sample_uncertainty_df_per_class(class_suricata_mc_preds, iter, mode=SAMPLE_MODE)
			#suricata_main_preds.drop(pseudo_labels.index)
		else : pseudo_labels=pd.DataFrame()
			
		new_training_iter = pd.concat([main_df, pseudo_labels])
		print(f'Length of training size: {len(new_training_iter)}')
		
		label_stats = get_label_stats(new_training_iter)
		if 'Label' not in pl_class_dist :
			pl_class_dist['Label'] = label_stats.index
		pl_class_dist[iter] = label_stats
		
		iter_model, top1_iter, topk_iter ,class_map_iter, miss_counts_iter, miss_stats_iter1, data_stats_iter, miss_record_iter = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(new_training_iter, main_df,encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
		
		og_test_mc = calculate_mc_preds(iter_model, main_df)
		if 'Sig' not in og_test_iter_preds :
			og_test_iter_preds['Sig'] = og_test_mc['Sig']
		og_test_iter_preds[iter] = og_test_mc['Label']
		
		if 'Sig' not in og_test_iter_mc :
			og_test_iter_mc['Sig'] = og_test_mc['Sig']
		og_test_iter_mc[iter] = og_test_mc['Uncert']
		
		og_uncert_stats = uncertainty_statistics(og_test_mc)
		og_uncert_stats.insert(0, top1_iter)
		og_uncert_stats.insert(0, iter)
		labeled_set_uncert_stats.append(og_uncert_stats)
		print(f'Labeled data set uncerts: {str(og_uncert_stats)}')
		
		unk_test_mc = calculate_mc_preds(iter_model, suricata_testing_preds)
		if 'Sig' not in unk_test_iter_preds :
			unk_test_iter_preds['Sig'] = unk_test_mc['Sig']
		unk_test_iter_preds[iter] = unk_test_mc['Label']
		
		if 'Sig' not in unk_test_iter_mc :
			unk_test_iter_mc['Sig'] = unk_test_mc['Sig']
		unk_test_iter_mc[iter] = unk_test_mc['Uncert']
		
		unk_uncert_stats = uncertainty_statistics(unk_test_mc)
		unk_uncert_stats.insert(0, top1_iter)
		unk_uncert_stats.insert(0, iter)
		unknown_uncert_stats.append(unk_uncert_stats)
		print(f'Unlabeled data set uncerts: {str(unk_uncert_stats)}')
		
	labeled_data = pd.DataFrame(labeled_set_uncert_stats, columns=['iter', 'acc', 'mean mc', 'std mc', 'min mc', 'max mc'])
	#labeled_data.to_csv(open('class_based_psuedo_labels_labeled.csv', 'w'))

	unlabeled_data = pd.DataFrame(unknown_uncert_stats, columns=['iter', 'acc', 'mean mc', 'std mc', 'min mc', 'max mc'])
	#unlabeled_data.to_csv(open('low_cert_psuedo_labels_unlabeled_oct27.csv', 'w'))
	
	
	unk_test_iter_preds.to_csv(open('./nov16_data/highcert_pseudolabel_unk_BAL_nov16.csv', 'w'))
	og_test_iter_preds.to_csv(open('./nov16_data/higcert_pseudolabel_labeled_BAL_nov16.csv', 'w'))
	unk_test_iter_mc.to_csv(open('./nov16_data/highcert_pseudolabel_unk_mc_BAL_nov16.csv', 'w'))
	og_test_iter_mc.to_csv(open('./nov16_data/highcert_pseudolabel_labeled_BAL_mc_nov16.csv', 'w'))
	labeled_data.to_csv(open('./nov16_data/highcert_iterstats_labeled_mc_BAL_nov16.csv', 'w'))
	unlabeled_data.to_csv(open('./nov16_data/highcert_iterstats_unk_mc_BAL_nov16.csv', 'w'))
	pl_class_dist.to_csv(open('./nov16_data/highcert_iterstats_classstats_BAL_nov16.csv', 'w'))
	
	# pseudo_labels_iter1 = suricata_main_preds.sample(1000)
	# suricata_main_preds.drop(pseudo_labels_iter1.index)
	# new_training_iter1 = pd.concat([train_df_sig, pseudo_labels_iter1])
	
	# iter1_model, top1_iter1, topk_iter1 ,class_map_iter1, miss_counts_iter1, miss_stats_iter1, data_stats_iter1, miss_record_iter1 = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(new_training_iter1, test_df_sig,encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
	
	# pseudo_labels_iter2 = suricata_main_preds.sample(10000)
	# suricata_main_preds.drop(pseudo_labels_iter1.index)
	# new_training_iter2 = pd.concat([pseudo_labels_iter1, pseudo_labels_iter2 ])
	
	# iter2_model, top1_iter2, topk_iter2 ,class_map_iter1, miss_counts_iter2, miss_stats_iter2, data_stats_iter2, miss_record_iter2 = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(new_training_iter2, test_df_sig,encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
	
	# pseudo_labels_iter3 = suricata_main_preds.sample(40000)
	# suricata_main_preds.drop(pseudo_labels_iter1.index)
	# new_training_iter3 = pd.concat([pseudo_labels_iter2, pseudo_labels_iter3])

	# iter3_model, top1_iter3, topk_iter3 ,class_map_iter1, miss_counts_iter3, miss_stats_iter3, data_stats_iter3, miss_record_iter3 = SignatureTransferLearningNewData.signature_transfer_learning_classifier_model_testing(new_training_iter3, test_df_sig,encoder_name, data_lm, class_map, class_labels, bs_sig=16, test_mode=False)
	