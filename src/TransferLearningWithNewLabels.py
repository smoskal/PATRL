import random

import pandas as pd
import torch

from misc import ClassifierAnalysis
from interfaces import SnortRuleParser, MitreAttackInterface
import TransferLearningDemo
from utils import signatureWithCVE as cve_interface
from aif.AIF_Mappings import RecentAlertsMapping
from aif.AttackStages import MicroAttackStage
from aif.AttackStages import SensorObservability
from aif.AttackStages import SignatureMapping as attkstg_map

out_path = '../data_store/'
if not torch.cuda.is_available():
	print("ERROR: CUDA not available.")
else :
	torch.cuda.set_device(0)
	torch.multiprocessing.freeze_support()

def train_language_model(lm_data_bunch, dropout=0.5, pre_trained=True, test_mode=False) :

	learn = language_model_learner(lm_data_bunch, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)

	if not test_mode :
		learn.freeze()
		learn.fit_one_cycle(cyc_len=8, max_lr=1e-2, moms=(0.8, 0.7))

		learn.freeze_to(-2)
		learn.fit_one_cycle(5, slice(1e-4, 1e-2), moms=(0.8, 0.7))

		learn.freeze_to(-3)
		learn.fit_one_cycle(5, slice(1e-5, 5e-3), moms=(0.8, 0.7))

		learn.unfreeze()
		learn.fit_one_cycle(5, slice(1e-5, 1e-3), moms=(0.8, 0.7))
	else :
		learn.unfreeze()
		learn.fit_one_cycle(2, slice(1e-5, 1e-3), moms=(0.8, 0.7))

	return learn

def train_classifier_model(class_data_bunch, encoder=None,dropout=0.5, pre_trained=True, test_mode=False) :
	learn_class = text_classifier_learner(class_data_bunch, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
	if encoder is not None : learn_class.load_encoder(encoder)

	if not test_mode :
		learn_class.freeze()
		learn_class.fit_one_cycle(cyc_len=8, max_lr=1e-2, moms=(0.8, 0.7))

		learn_class.freeze_to(-2)
		learn_class.fit_one_cycle(3, max_lr=1e-3, moms=(0.8, 0.7))

		learn_class.freeze_to(-3)
		learn_class.fit_one_cycle(3, max_lr=1e-4, moms=(0.8, 0.7))

		learn_class.unfreeze()
		learn_class.fit_one_cycle(8, slice(1e-5, 1e-3), moms=(0.8, 0.7))
	else :
		learn_class.unfreeze()  
		learn_class.fit_one_cycle(2, slice(1e-5, 1e-3), moms=(0.8, 0.7))

	return learn_class

'''
Splits the results tuples into individual lists
'''
def result_to_lists(classifier_results) :

	x_vals = []
	y_gt = []
	y_pred = []
	y_pred_conf = []

	for result in classifier_results :
		x_vals.append(result[0])
		y_gt.append(result[3])
		y_pred.append(result[2][0])
		y_pred_conf.append(result[1][0])

	return x_vals, y_gt, y_pred, y_pred_conf

def signature_transfer_learning_language_model(mode='normal', additional_samples=None, pre_trained_lang=True, pre_trained_sig=True,data_split_text=.25,data_split_class=0.25, bs_lang=32,
									 bs_sig=16,dropout=0.5, include_unknown_lang_data=False, random_seed=None, dataset=URLs.WIKITEXT_TINY,
									 output_acc=False, test_mode=False) :
	text_data, class_labels, class_map = TransferLearningDemo.signature_to_data_frame(transform_labels=True)
	class_labels.append(len(class_map))
	class_map[len(class_map)] = MicroAttackStage.PHISHING.value

	reverse_class_map = TransferLearningDemo.reverse_dict(class_map)
	x_unknown, y = attkstg_map().get_unknown_mapping()
	y_unknown = TransferLearningDemo.transform_gt(y, reverse_class_map)

	the_tokenizer = Tokenizer()
	path = untar_data(dataset)

	if mode=='normal' :

		train_df_lang, test_df_lang = TransferLearningDemo.split_train_test(text_data, split=data_split_text,
																			  seed=random_seed)
		if output_acc:
			unique_words_lang = ClassifierAnalysis.get_unique_words_from_df(train_df)

		if include_unknown_lang_data:
			train_df_lang = train_df_lang.append(pd.DataFrame(x_unknown, columns=['Sig']), sort=False)


	elif mode=='large' :
		print('Retrieving signature database for large configuration...')
		sig_data = SnortRuleParser.collect_signature_msg()
		train_df_lang, test_df_lang = TransferLearningDemo.split_train_test(sig_data, split=data_split_text)

		if output_acc :
			unique_words_lang = ClassifierAnalysis.get_unique_words_from_df(train_df_lang)

		if include_unknown_lang_data:
			train_df_lang = train_df_lang.append(pd.DataFrame(x_unknown, columns=['Sig']), sort=False)

	elif mode == 'snort' :
		print('Retrieving signature database for Snort configuration...')
		sig_data = SnortRuleParser.collect_snort_msg()
		if output_acc :
			unique_words_lang = ClassifierAnalysis.get_unique_words_from_df(sig_data)

		train_df_lang, test_df_lang= TransferLearningDemo.split_train_test(sig_data, split=data_split_text)

	elif mode == 'cve' :
		print('Getting the CVE data..')
		cve_text = cve_interface.get_all_cve_summaries()
		train_df_lang, test_df_lang = TransferLearningDemo.split_train_test(cve_text, split=data_split_text)

	elif mode == 'all' :
		print('--- COMBINED DATA SET ---')
		print('Retrieving signature database for large configuration...')
		sig_data = SnortRuleParser.collect_signature_msg()

		print('Retrieving signature database for Snort configuration...')
		snort_data = SnortRuleParser.collect_snort_msg()

		print('Getting the MITRE Att&ck data..')
		mitre_data = MitreAttackInterface.get_patterns_df(observability_list=SensorObservability.signature_based,
														  combine_priv_esc=True)
		print('Getting the CVE data..')
		cve_text = cve_interface.get_all_cve_summaries()

		print('Combining all data sets..')
		lang_data = cve_text.append(text_data, sort=False)
		lang_data = lang_data.append(mitre_data, sort=False)
		lang_data = lang_data.append(snort_data, sort=False)
		lang_data = lang_data.append(sig_data, sort=False)
		train_df_lang, test_df_lang = TransferLearningDemo.split_train_test(lang_data, split=data_split_text,
																			seed=random_seed)
		print('...Done!')

	else :
		print('Signature Lang Model Test- Unsupported mode:', mode)
		return
	
	if additional_samples is not None :
		for sig, stg in additional_samples.items() :
			train_df_lang = train_df_lang.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)


	print('Loading in data to data bunch...')
	data_lm = TextLMDataBunch.from_df(path=path, train_df=train_df_lang, valid_df=test_df_lang, bs=bs_lang,
									  text_cols='Sig', tokenizer=the_tokenizer)

	print('Signature Language Model Training -- Mode:', mode)
	lm_learn = train_language_model(data_lm, pre_trained=pre_trained_lang, test_mode=test_mode)

	if mode=='normal':
		lm_learn.save_encoder('ft_enc_sig_normal')
		out_encoder = 'ft_enc_sig_normal'
	elif mode=='large' or mode=='snort':
		lm_learn.save_encoder('ft_enc_sig_large')
		out_encoder = 'ft_enc_sig_large'
	elif mode == 'cve' :
		lm_learn.save_encoder('ft_enc_cve')
		out_encoder = 'ft_enc_cve'
	elif mode == 'all' :
		lm_learn.save_encoder('ft_enc_all')
		out_encoder = 'ft_enc_all'

	return  lm_learn, data_lm, out_encoder

def signature_transfer_learning_classifier_model(encoder_name, data_lm,mode='normal',additional_samples=None,pre_trained_lang=True, pre_trained_sig=True,data_split_text=.25,data_split_class=0.25, bs_lang=32,
									 bs_sig=16,dropout=0.5, include_unknown_lang_data=False, random_seed=None, dataset=URLs.WIKITEXT_TINY,
									 output_acc=False, test_mode=False) :

	text_data, class_labels, class_map = TransferLearningDemo.signature_to_data_frame(transform_labels=True)
	class_labels.append(len(class_map))
	class_map[len(class_map)] = MicroAttackStage.PHISHING.value
	reverse_class_map = TransferLearningDemo.reverse_dict(class_map)


	x_unknown, y = attkstg_map().get_unknown_mapping()
	y_unknown = TransferLearningDemo.transform_gt(y, reverse_class_map)
	path = untar_data(dataset)
	the_tokenizer = Tokenizer()

	print('Signature Classifier Training -- Mode:', mode)
	train_df_sig, test_df_sig = TransferLearningDemo.split_train_test(text_data, split=data_split_class,
																	  seed=random_seed)
	#Assumes the structure of {Sig:AIS value}
	if additional_samples is not None :
		for sig, stg in additional_samples.items() :
			train_df_sig = train_df_sig.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)

	if data_lm is not None :
		data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df_sig, valid_df=test_df_sig,
											  classes=class_labels,
											  vocab=data_lm.train_ds.vocab, bs=bs_sig, text_cols='Sig',
											  label_cols='Label', tokenizer=the_tokenizer)
	else :
		data_clas = TextClasDataBunch.from_df(path='./', train_df=train_df_sig, valid_df=test_df_sig, classes=class_labels,
										  vocab=None, bs=bs_sig, text_cols='Sig',
										  label_cols='Label', tokenizer=the_tokenizer)

	learn_classifier = train_classifier_model(data_clas, encoder=encoder_name, dropout=dropout, pre_trained=pre_trained_sig, test_mode=test_mode)


	top1_acc, topk_acc, hit_record, topk_hits, full_misses, all_samples = TransferLearningDemo.get_topk(learn_classifier,
																										test_df_sig[
																											'Sig'],
																										test_df_sig[
																											'Label'],
																										class_map)

	print('Pre-trained Lang: ', str(pre_trained_lang), "  Pre-trained Sig: ", str(pre_trained_sig), ' Data split: ', str(data_split_text))
	print('Signature Model Top 1: ', str(top1_acc))
	print('Signature Model Top k: ', str(topk_acc))

	if output_acc :
		return top1_acc, topk_acc,unique_words_lang, unique_words_class

	return learn_classifier, class_map

#This is to get the class map out of the 'recent' dataset.  Super dirty code but lets gooooo	
def transform_labels(data_dict) :

	out_df = pd.DataFrame(columns=['Sig', 'Label'])
	temp_class_map = {}
	out_class_labels = []
	
	class_count = 0
	for sig, label in data_dict.items() :
		if label not in temp_class_map :
			temp_class_map[label] = class_count 
			out_class_labels.append(class_count)
			class_count += 1
			
		out_df = out_df.append({'Sig': sig, 'Label': temp_class_map[label]}, ignore_index=True)	

	return out_df, temp_class_map, out_class_labels
	
#Custom class that allows me to input the phishing or non phishing sets in here.  this could be done better but w/e
def custom_signature_classifier_model(data_dict, encoder_name, data_lm,mode='normal',additional_samples=None, additional_testing=None, pre_trained_lang=True, pre_trained_sig=True,data_split_text=.25,data_split_class=0.25, bs_lang=32,
									 bs_sig=16,dropout=0.5, include_unknown_lang_data=False, random_seed=None, dataset=URLs.WIKITEXT_TINY,
									 output_acc=False, test_mode=False) :


	text_data, reverse_class_map, class_labels = transform_labels(data_dict)
	class_map = TransferLearningDemo.reverse_dict(reverse_class_map)

	path = untar_data(dataset)
	the_tokenizer = Tokenizer()

	print('Signature Classifier Training -- Mode:', mode)
	train_df_sig, test_df_sig = TransferLearningDemo.split_train_test(text_data, split=data_split_class,
																	  seed=random_seed)
	#Assumes the structure of {Sig:AIS value}
	if additional_samples is not None :
		for sig, stg in additional_samples.items() :
			train_df_sig = train_df_sig.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)
			
	if additional_testing is not None :
		for sig, stg in additional_testing.items() :
			test_df_sig = test_df_sig.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)

	if data_lm is not None :
		data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df_sig, valid_df=test_df_sig,
											  classes=class_labels,
											  vocab=data_lm.train_ds.vocab, bs=bs_sig, text_cols='Sig',
											  label_cols='Label', tokenizer=the_tokenizer)
	else :
		data_clas = TextClasDataBunch.from_df(path='./', train_df=train_df_sig, valid_df=test_df_sig, classes=class_labels,
										  vocab=None, bs=bs_sig, text_cols='Sig',
										  label_cols='Label', tokenizer=the_tokenizer)

	learn_classifier = train_classifier_model(data_clas, encoder=encoder_name, dropout=dropout, pre_trained=pre_trained_sig, test_mode=test_mode)


	top1_acc, topk_acc, hit_record, topk_hits, full_misses, all_samples = TransferLearningDemo.get_topk(learn_classifier,
																										test_df_sig[
																											'Sig'],
																										test_df_sig[
																											'Label'],
																										class_map)

	print('Pre-trained Lang: ', str(pre_trained_lang), "  Pre-trained Sig: ", str(pre_trained_sig), ' Data split: ', str(data_split_text))
	print('Signature Model Top 1: ', str(top1_acc))
	print('Signature Model Top k: ', str(topk_acc))

	if output_acc :
		return top1_acc, topk_acc,unique_words_lang, unique_words_class

	return learn_classifier, class_map

def check_for_phishing(sig_model, class_map,test_sigs) :
	phish_label = None
	count = 0
	hits = []
	misses = []

	for trans_label, real_label in class_map.items() :
		if real_label == MicroAttackStage.PHISHING.value :
			phish_label = trans_label
			break
	if phish_label is None :
		print('No Phishing in Class Map')
		return None

	for sig in test_sigs :
		pred = sig_model.predict(sig)

		pred_class = int(pred[1])
		if pred_class == phish_label :
			count += 1
			hits.append((sig, pred_class))
		else :
			misses.append((sig, pred_class))

	return count, hits, misses
	
def recent_alerts_training_test(samples_per_class=1, random_selection=True) :

	not_phishing_sigs = RecentAlertsMapping().get_not_phishing()
	sorted_sigs = RecentAlertsMapping().sort_by_class()

	testing_df = pd.DataFrame(columns=['Sig', 'Label'])
	training_samples = {} #aka additional samples
	
	for label, sigs in sorted_sigs.items() :
		if len(sigs) < samples_per_class :
			print('Too many samples requested for class (Skipping): ', str(label))
			for sample in sigs :
				training_samples[sample] = label
				#remove it from the eventual testing set.
				#del not_phishing_sigs[sample]
		else :
			samples = random.sample(sigs, k=samples_per_class)
			
			for sample in samples :
				training_samples[sample] = label
				#remove it from the eventual testing set.
				del not_phishing_sigs[sample]
				
	
	return training_samples, not_phishing_sigs
	
#Converts the train/test dictionary from the AIF_Mapping using a class map, and then puts it into a DF
def convert_labels_df(sig_dict, reverse_class_map) :

	out_df = pd.DataFrame(columns=['Sig', 'Label'])
	
	for sig, label in sig_dict.items() :
	
		if not isinstance(label, Enum) :
			out_df = out_df.append({'Sig': sig, 'Label': reverse_class_map[label]}, ignore_index=True)
		else:	
			out_df = out_df.append({'Sig': sig, 'Label': reverse_class_map[label.value]}, ignore_index=True)

	return out_df

def recent_alerts_split_test(recent_split=.9, class_split=.2, mode='normal', add_testing=False) :
	not_phishing_sigs = RecentAlertsMapping().get_not_phishing()
	recent_additional_training, recent_testing = RecentAlertsMapping().split_data_set(not_phishing_sigs, test_split=recent_split)
	repeated_recent_sigs = RecentAlertsMapping().repeated_recent_alerts
	
	if not add_testing :
		lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode=mode,additional_samples=not_phishing_sigs, pre_trained_lang=True,bs_lang=32, bs_sig=16, test_mode=False)
	else :
		lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode=mode,additional_samples=not_phishing_sigs, additional_testing=recent_testing,pre_trained_lang=True,bs_lang=32, bs_sig=16, test_mode=False)
	
	sig_model, class_map = signature_transfer_learning_classifier_model(
		encoder_name, data_lm, mode=mode, data_split_class=class_split, bs_lang=32, bs_sig=16, test_mode=False, additional_samples=recent_additional_training)
		
	reverse_class_map = TransferLearningDemo.reverse_dict(class_map)
	recent_testing_df = convert_labels_df(recent_testing, reverse_class_map)
	repeated_recent_testing = convert_labels_df(repeated_recent_sigs, reverse_class_map)
	
	top1_acc_recent, topk_acc_recent, hit_record_recent, topk_hits_recent, full_misses_recent, all_samples_recent = TransferLearningDemo.get_topk(sig_model, recent_testing_df['Sig'],recent_testing_df['Label'],class_map)
	
	print('***** SPLIT SIZE: ', str(recent_split), ' ******') 
	
	print('Recent Sample Top 1: ', str(top1_acc_recent))
	print('Recent Sample Top k: ', str(topk_acc_recent))
	
	top1_acc_repeat, topk_acc_repeat, hit_record_repeat, topk_hits_repeat, full_misses_repeat, all_samples_repeat = TransferLearningDemo.get_topk(sig_model, repeated_recent_testing['Sig'],repeated_recent_testing['Label'],class_map)
	
	print('Repeat Samples Top 1: ', str(top1_acc_repeat))
	print('Repeat Samples Top k: ', str(topk_acc_repeat))

	return sig_model, top1_acc_recent,top1_acc_repeat

if __name__ == '__main__' :
	phishing_sigs = RecentAlertsMapping().get_phishing()
	not_phishing_sigs = RecentAlertsMapping().get_not_phishing()

	'''
	SIGNATURE MODEL TESTING STARTS HERE
	'''

	# lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode='normal',additional_samples=not_phishing_sigs,bs_lang=32, bs_sig=16, test_mode=False)

	# sig_model, class_map = signature_transfer_learning_classifier_model(
		# encoder_name, data_lm, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=None)

	# count, hits, misses = check_for_phishing(sig_model, class_map, phishing_sigs)

	# print('NO PHISHING SIG: ', str(count))

	# additional_sig = {}
	# for sig in phishing_sigs[0:1]:
		# additional_sig[sig] = MicroAttackStage.PHISHING.value

	# sig_model1, class_map = signature_transfer_learning_classifier_model(
		# encoder_name, data_lm ,mode='normal',bs_lang=32, bs_sig=16, test_mode=False, additional_samples=additional_sig)

	# count1, hits1, misses1 = check_for_phishing(sig_model1, class_map, phishing_sigs)
	# print('1 PHISHING SIG: ', str(count1))


	# additional_sig = {}
	# for sig in phishing_sigs[0:2]:
		# additional_sig[sig] = MicroAttackStage.PHISHING.value

	# sig_model2, class_map = signature_transfer_learning_classifier_model(
		# encoder_name, data_lm ,mode='normal',bs_lang=32, bs_sig=16, test_mode=False, additional_samples=additional_sig)

	# count2, hits2, misses2 = check_for_phishing(sig_model2, class_map, phishing_sigs)
	# print('2 PHISHING SIG: ', str(count2))

	# additional_sig = {}
	# for sig in phishing_sigs[0:3]:
		# additional_sig[sig] = MicroAttackStage.PHISHING.value

	# sig_model3, class_map = signature_transfer_learning_classifier_model(
		# encoder_name, data_lm, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=additional_sig)

	# count3, hits3, misses3 = check_for_phishing(sig_model3, class_map, phishing_sigs)
	# print('3 PHISHING SIG: ', str(count3))

	# additional_sig = {}
	# for sig in phishing_sigs[0:4]:
		# additional_sig[sig] = MicroAttackStage.PHISHING.value

	# sig_model4, class_map = signature_transfer_learning_classifier_model(
		# encoder_name, data_lm, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=additional_sig)

	# count4, hits4, misses4 = check_for_phishing(sig_model4, class_map, phishing_sigs)
	# print('4 PHISHING SIG: ', str(count4))

	# additional_sig = {}
	# for sig in phishing_sigs[0:4]:
		# additional_sig[sig] = MicroAttackStage.PHISHING.value

	# sig_model5, class_map = signature_transfer_learning_classifier_model(
		# encoder_name, data_lm, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=additional_sig)

	# count5, hits5, misses5 = check_for_phishing(sig_model5, class_map, phishing_sigs)
	# print('5 PHISHING SIG: ', str(count5))

	# additional_sig = {}
	# for sig in phishing_sigs[0:10]:
		# additional_sig[sig] = MicroAttackStage.PHISHING.value

	# sig_model10, class_map = signature_transfer_learning_classifier_model(
		# encoder_name, data_lm, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=additional_sig)

	# count10, hits10, misses10 = check_for_phishing(sig_model10, class_map, phishing_sigs)
	# print('10 PHISHING SIG: ', str(count10))

	# additional_sig = {}
	# for sig in phishing_sigs[0:50]:
		# additional_sig[sig] = MicroAttackStage.PHISHING.value

	# sig_model50, class_map = signature_transfer_learning_classifier_model(
		# encoder_name, data_lm ,mode='normal',bs_lang=32, bs_sig=16, test_mode=False, additional_samples=additional_sig)

	# count50, hits50, misses50 = check_for_phishing(sig_model50, class_map, phishing_sigs)
	# print('50 PHISHING SIG: ', str(count50))

	# print(str(count), ' - ',str(count1),' - ',str(count2),' - ',str(count3),' - ',str(count4),' - ',str(count5),' - ',str(count10),' - ', str(count50))
	
	'''
	SIGNATURE MODEL ENDS HERE BRUV
	'''

	# text_data, class_labels, class_map2 = TransferLearningDemo.signature_to_data_frame(transform_labels=True)
	
	# train_df_sig, test_df_sig = TransferLearningDemo.split_train_test(text_data)

	#***********************

	# recent_additional_samples, recent_testing_samples = recent_alerts_training_test(samples_per_class=0, random_selection=True)

	# lang_model_before, data_lm_before, encoder_name_before = signature_transfer_learning_language_model(mode='normal',additional_samples=None, pre_trained_lang=True,bs_lang=32, bs_sig=16, test_mode=False)

	# sig_model_before, class_map = signature_transfer_learning_classifier_model(
		# encoder_name_before, data_lm_before, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=recent_additional_samples)
		
	# reverse_class_map = TransferLearningDemo.reverse_dict(class_map)
	# not_phishing_df = pd.DataFrame(columns=['Sig', 'Label'])
	# for sig, stg in recent_testing_samples.items() :
			# not_phishing_df = not_phishing_df.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)	
			
	# top1_acc_before, topk_acc_before, hit_record_before, topk_hits_before, full_misses_before, all_samples_before = TransferLearningDemo.get_topk(sig_model_before,
																										# not_phishing_df[
																											# 'Sig'],
																										# not_phishing_df[
																											# 'Label'],
																										# class_map)
																
	# print('Recent Sample Top 1: ', str(top1_acc_before))
	# print('Recent Sample Top k: ', str(topk_acc_before))
	
	# sig_model_cust, class_map_cust = custom_signature_classifier_model(not_phishing_sigs,
		# encoder_name_before, data_lm_before, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=None)
	
	
    #**************************
	
	# lang_model_after, data_lm_after, encoder_name_after = signature_transfer_learning_language_model(mode='normal',pre_trained_lang=True,additional_samples=not_phishing_sigs,
	# bs_lang=32, bs_sig=16, test_mode=False)

	# sig_model_after, class_map_after = signature_transfer_learning_classifier_model(
		# encoder_name_after, data_lm_after, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=None)
		
	# top1_acc_after, topk_acc_after, hit_record_after, topk_hits_after, full_misses_after, all_samples_after = TransferLearningDemo.get_topk(sig_model_after,
																										# not_phishing_df[
																											# 'Sig'],
																										# not_phishing_df[
																											# 'Label'],
																										# class_map)
																
	# print('Recent Sample Top 1: ', str(top1_acc_after))
	# print('Recent Sample Top k: ', str(topk_acc_after))
	
	#correct_attention, incorrect_attention, correct_analysis, incorrect_analysis = ChangeInIntrinsicAttention.get_intrinisic_analysis(sig_model_before,all_samples_before)
	
	
	'''
	SIGNATURE MODEL TESTING STARTS HERE
	'''

	lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode='normal',additional_samples=None,bs_lang=32, bs_sig=16, test_mode=False)

	sig_model, class_map = signature_transfer_learning_classifier_model(
		encoder_name, data_lm, mode='normal', bs_lang=32, bs_sig=16, test_mode=False, additional_samples=None)

	count, hits, misses = check_for_phishing(sig_model, class_map, phishing_sigs)
	
	
	# schedule = [.95,.9, .8,.7,.6,.5,.4,.3,.2,.1]
	# num_runs = 1
	
	# result_recent = {}
	# result_repeated = {}
	
	# for test_split in schedule :
		
		# recent_res = []
		# repeat_res = []
		
		# for i in range(num_runs) :
			# print('TEST ', str(i), ' FOR SPLIT: ', str(test_split))
			# sig_model, top1_acc_recent,top1_acc_repeat = recent_alerts_split_test(recent_split=.25, class_split=test_split)
			
			# recent_res.append(top1_acc_recent)
			# repeat_res.append(top1_acc_repeat)
			
		# result_recent[test_split] = recent_res
		# result_repeated[test_split] = repeat_res
		
	# print('RECENT: ')
	# print(result_recent)
	# print('REPEATED: ')
	# print(result_repeated)
	
	
	
	

	


