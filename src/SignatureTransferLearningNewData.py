import random

import torch
from fastai.text.core import Tokenizer

from misc import ClassifierAnalysis
from utils import LearningUtils, signatureWithCVE as cve_interface
from interfaces import SnortRuleParser, MitreAttackInterface
from aif.AIF_Mappings import RecentAlertsMapping
from aif.AttackStages import SensorObservability

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
		learn.fit_one_cycle(cyc_len=2, max_lr=1e-2, moms=(0.8, 0.7))

		learn.freeze_to(-2)
		learn.fit_one_cycle(2, slice(1e-4, 1e-2), moms=(0.8, 0.7))

		learn.freeze_to(-3)
		learn.fit_one_cycle(2, slice(1e-5, 5e-3), moms=(0.8, 0.7))

		learn.unfreeze()
		learn.fit_one_cycle(2, slice(1e-5, 1e-3), moms=(0.8, 0.7))
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
	
def signature_transfer_learning_language_model(mode='normal', other_data=None,additional_samples=None, pre_trained_lang=True,
	data_split_text=.25, bs_lang=32,dropout=0.5, random_seed=None, dataset=URLs.WIKITEXT_TINY,output_acc=False, 
	test_mode=False, out_path='./') :
									 
									
	the_tokenizer = Tokenizer()
	path = out_path

	if mode=='normal' :

		text_data, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=True)
		train_df_lang, test_df_lang = LearningUtils.split_train_test(text_data, split=data_split_text, seed=random_seed)
		
		if output_acc:
			unique_words_lang = ClassifierAnalysis.get_unique_words_from_df(train_df)

	elif mode=='large' :
		print('Retrieving signature database for large configuration...')
		sig_data = SnortRuleParser.collect_signature_msg()
		train_df_lang, test_df_lang = LearningUtils.split_train_test(sig_data, split=data_split_text)

		if output_acc :
			unique_words_lang = ClassifierAnalysis.get_unique_words_from_df(train_df_lang)

	elif mode == 'snort' :
		print('Retrieving signature database for Snort configuration...')
		sig_data = SnortRuleParser.collect_snort_msg()
		if output_acc :
			unique_words_lang = ClassifierAnalysis.get_unique_words_from_df(sig_data)

		train_df_lang, test_df_lang= LearningUtils.split_train_test(sig_data, split=data_split_text)

	elif mode == 'cve' :
		print('Getting the CVE data..')
		cve_text = cve_interface.get_all_cve_summaries()
		train_df_lang, test_df_lang = LearningUtils.split_train_test(cve_text, split=data_split_text)

	elif mode == 'all' :
		print('--- COMBINED DATA SET ---')
		text_data, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=True)
		
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
		train_df_lang, test_df_lang = LearningUtils.split_train_test(lang_data, split=data_split_text,
																	 seed=random_seed)
		print('...Done!')
	elif mode == 'other' :
		if other_data is None :
			print('Language Model:  Other mode selected but no data was provided.')
			return
		
		train_df_lang, test_df_lang= LearningUtils.split_train_test(other_data, split=data_split_text)
		
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
	else :
		lm_learn.save_encoder('ft_enc_other')
		out_encoder = 'ft_enc_other'

	return  lm_learn, data_lm, out_encoder
	
def signature_transfer_learning_classifier_model(data_df, encoder_name, data_lm, class_map, class_labels,
		additional_samples=None, additional_testing=None, pre_trained_sig=True,data_split_class=0.25, bs_sig=16,dropout=0.5,
		random_seed=None, output_acc=False, test_mode=False, output_miss_stats=False, path='./', binary_mode=False) :


	#text_data, reverse_class_map, class_labels = transform_labels(data_dict)
	reverse_class_map = LearningUtils.reverse_dict(class_map)
	the_tokenizer = Tokenizer()

	print('Signature Classifier Training')
	train_df_sig, test_df_sig = LearningUtils.split_train_test(data_df, split=data_split_class,
															   seed=random_seed)
	#Assumes the structure of {Sig:AIS value}
	if additional_samples is not None :
		for sig, stg in additional_samples.items() :
			train_df_sig = train_df_sig.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)
			
	if additional_testing is not None :
		for sig, stg in additional_testing.items() :
			test_df_sig = test_df_sig.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)

	if data_lm is not None :

		data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df_sig, valid_df=test_df_sig,classes=class_labels,vocab=data_lm.train_ds.vocab, bs=bs_sig, text_cols='Sig',label_cols='Label', tokenizer=the_tokenizer)
	else :
		data_clas = TextClasDataBunch.from_df(path='./', train_df=train_df_sig, valid_df=test_df_sig, classes=class_labels,
										  vocab=None, bs=bs_sig, text_cols='Sig',
										  label_cols='Label', tokenizer=the_tokenizer)

	learn_classifier = train_classifier_model(data_clas, encoder=encoder_name, dropout=dropout, pre_trained=pre_trained_sig, test_mode=test_mode)


	if not binary_mode : 
		top1_acc, topk_acc, miss_records, data_counts, miss_counts, miss_stats = LearningUtils.get_topk(learn_classifier, test_df_sig['Sig'], test_df_sig['Label'], class_map)
	else :

		test_stats = LearningUtils.binary_perf_stats(learn_classifier, test_df_sig['Sig'], test_df_sig['Label'])
		return learn_classifier, test_stats
																								

	print('Signature Model Top 1: ', str(top1_acc))
	print('Signature Model Top k: ', str(topk_acc))

	if output_acc :
		return top1_acc, topk_acc
	if output_miss_stats :
		return learn_classifier, top1_acc, topk_acc, class_map, miss_counts, miss_stats, data_counts, miss_records

	return learn_classifier, top1_acc, topk_acc, class_map
	
def signature_transfer_learning_classifier_model_testing(data_df, test_df,encoder_name, data_lm, class_map, class_labels,
		pre_trained_sig=False,data_split_class=0.25, bs_sig=16,dropout=0.5,
		random_seed=None, output_acc=False, test_mode=False) :


	#text_data, reverse_class_map, class_labels = transform_labels(data_dict)
	reverse_class_map = LearningUtils.reverse_dict(class_map)

	path = './'
	the_tokenizer = Tokenizer()

	if data_lm is not None :

		data_clas = TextClasDataBunch.from_df(path='./', train_df=data_df, valid_df=test_df,classes=class_labels,vocab=data_lm.train_ds.vocab, bs=bs_sig, text_cols='Sig',label_cols='Label', tokenizer=the_tokenizer, include_bos=False)
		
		data_clas.vocab.itos = data_lm.vocab.itos
		
	else :
		data_clas = TextClasDataBunch.from_df(path='./', train_df=data_df, valid_df=test_df, classes=class_labels,
										  vocab=None, bs=bs_sig, text_cols='Sig',
										  label_cols='Label', tokenizer=the_tokenizer, include_bos=False)
										  
	

	learn_classifier = train_classifier_model(data_clas, encoder=encoder_name, dropout=dropout, pre_trained=pre_trained_sig, test_mode=test_mode)


	top1_acc, topk_acc, miss_records, data_counts, miss_counts, miss_stats = LearningUtils.get_topk(learn_classifier, test_df['Sig'], test_df['Label'], class_map)

	print('Signature Model Top 1: ', str(top1_acc))
	print('Signature Model Top k: ', str(topk_acc))

	if output_acc :
		return top1_acc, topk_acc

	return learn_classifier, top1_acc, topk_acc, class_map, miss_counts, miss_stats, data_counts, miss_records
	
def signature_transfer_learning_classifier_model_kfold(data_df, test_df, encoder_name, data_lm, class_map, class_labels,
		pre_trained_sig=False,data_split_class=0.25, bs_sig=16,dropout=0.5,
		random_seed=None, output_acc=False, test_mode=False, output_miss_stats=True, vocab=None) :


	#text_data, reverse_class_map, class_labels = transform_labels(data_dict)
	reverse_class_map = LearningUtils.reverse_dict(class_map)

	path = './'
	the_tokenizer = Tokenizer()

	if vocab is not None :
		data_clas = TextClasDataBunch.from_df(path=path, train_df=data_df, valid_df=test_df,classes=class_labels,vocab=vocab, bs=bs_sig, text_cols='Sig',label_cols='Label', tokenizer=the_tokenizer, include_bos=False)
	elif data_lm is not None :

		data_clas = TextClasDataBunch.from_df(path=path, train_df=data_df, valid_df=test_df,classes=class_labels,vocab=data_lm.train_ds.vocab, bs=bs_sig, text_cols='Sig',label_cols='Label', tokenizer=the_tokenizer, include_bos=False)
	else :
		data_clas = TextClasDataBunch.from_df(path='./', train_df=data_df, valid_df=test_df, classes=class_labels,
										  vocab=None, bs=bs_sig, text_cols='Sig',
										  label_cols='Label', tokenizer=the_tokenizer, include_bos=False)

	learn_classifier = train_classifier_model(data_clas, encoder=encoder_name, dropout=dropout, pre_trained=pre_trained_sig, test_mode=test_mode)


	top1_acc, topk_acc, miss_records, data_counts, miss_counts, miss_stats = LearningUtils.get_topk(learn_classifier, test_df['Sig'], test_df['Label'], class_map)

	print('Signature Model Top 1: ', str(top1_acc))
	print('Signature Model Top k: ', str(topk_acc))

	if output_acc :
		return top1_acc, topk_acc
	if output_miss_stats :
		return learn_classifier, top1_acc, topk_acc, class_map, miss_counts, miss_stats, data_counts, miss_records

	return learn_classifier, top1_acc, topk_acc, class_map
	
def signature_transfer_learning_classifier_incremental(data_df, test_df, classifier_model, data_lm, class_map, class_labels,
		pre_trained_sig=True,data_split_class=0.25, bs_sig=16,dropout=0.5,
		random_seed=None, output_acc=False, test_mode=False) :


	#text_data, reverse_class_map, class_labels = transform_labels(data_dict)
	reverse_class_map = LearningUtils.reverse_dict(class_map)

	path = untar_data(URLs.WIKITEXT_TINY)
	the_tokenizer = Tokenizer()

	if data_lm is not None :

		data_clas = TextClasDataBunch.from_df(path=path, train_df=data_df, valid_df=test_df,classes=class_labels,vocab=data_lm.train_ds.vocab, bs=bs_sig, text_cols='Sig',label_cols='Label', tokenizer=the_tokenizer, include_bos=False)
	else :
		data_clas = TextClasDataBunch.from_df(path='./', train_df=data_df, valid_df=test_df, classes=class_labels,
										  vocab=None, bs=bs_sig, text_cols='Sig',
										  label_cols='Label', tokenizer=the_tokenizer, include_bos=False)

	#learn_classifier = train_classifier_model(data_clas, encoder=encoder_name, dropout=dropout, pre_trained=pre_trained_sig, test_mode=test_mode)
	classifier_model.data = data_clas
	classifier_model.fit_one_cycle(7, slice(1e-5, 1e-3), moms=(0.8, 0.7))
	

	top1_acc, topk_acc, hit_record, topk_hits, full_misses, all_samples = LearningUtils.get_topk(classifier_model,
																								 test_df[
																											'Sig'],
																								 test_df[
																											'Label'],
																								 class_map)

	print('Signature Model Top 1: ', str(top1_acc))
	print('Signature Model Top k: ', str(topk_acc))

	if output_acc :
		return top1_acc, topk_acc

	return classifier_model, top1_acc, topk_acc, class_map
	
def model_performance_check(pretrained_lang=True, pretrained_class=True ,data_mode='cptc', lang_mode='normal',num_runs=5, bs_lang=32, bs_class=16) :

	text_data, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=True)

	print('Data mode: ', str(data_mode))
	if data_mode == 'cptc' : 
		data, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=True)
	elif data_mode == 'ccdc'	:
		data = LearningUtils.transform_dataset(RecentAlertsMapping().ccdc_combined, class_map)
	elif data_mode == 'update' :
		data = LearningUtils.transform_dataset(RecentAlertsMapping().get_not_phishing(), class_map)
	else :
		print('Performace check: Data mode type not specificied - ', str(data_mode))
	
	lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode=lang_mode,other_data=data,
	additional_samples=None,bs_lang=bs_lang, test_mode=False, random_seed=None, pre_trained_lang=pretrained_lang)
	
	out_top1 = []
	out_topk = []
	
	for i in range(num_runs) :
			top1_acc, topk_acc = signature_transfer_learning_classifier_model(data, encoder_name, data_lm, class_map, class_labels, bs_sig=bs_class, pre_trained_sig=pretrained_class, output_acc=True)
			
			out_top1.append(top1_acc)
			out_topk.append(topk_acc)
	
	return out_top1, out_topk
	
	
	
	
def predicting_others_test(num_runs=5, lang_mode='large') :

	text_data, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=True)

	
	lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode=lang_mode,other_data=text_data,
	additional_samples=None,bs_lang=32, test_mode=False, random_seed=0)
	
	top1_out = []
	topk_out = []
	
	for i in range(num_runs) :
		random_value = random.randint(1,1000000)
		classifier_top1, classifier_topk = train_test_classifiers(lang_model, data_lm, encoder_name, random_value)
		
		top1_out.append(classifier_top1)
		topk_out.append(classifier_topk)
	
	return top1_out, topk_out

def train_test_classifiers(lang_model, data_lm, encoder_name, random_value) :
	
	#Get all of the data sets.
	cptc_data, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=True)
	ccdc_data = LearningUtils.transform_dataset(RecentAlertsMapping().ccdc_combined, class_map)
	update_data = LearningUtils.transform_dataset(RecentAlertsMapping().get_not_phishing(), class_map)
	
	sig_model_cptc, top1_cptc, topk_cptc ,class_map_cptc = signature_transfer_learning_classifier_model(cptc_data, encoder_name, data_lm, class_map, class_labels,bs_sig=16, test_mode=False, additional_samples=None, random_seed=random_value)
	
	sig_model_ccdc, top1_ccdc, topk_ccdc ,class_map_ccdc = signature_transfer_learning_classifier_model(ccdc_data, encoder_name, data_lm, class_map, class_labels,bs_sig=16, test_mode=False, additional_samples=None, random_seed=random_value)
	
	sig_model_update, top1_update, topk_update ,class_map_update = signature_transfer_learning_classifier_model(update_data, encoder_name, data_lm, class_map, class_labels,bs_sig=16, test_mode=False, additional_samples=None, random_seed=random_value)
	
	
	
	top1_cptc_ccdc, topk_cptc_ccdc = LearningUtils.get_topk(sig_model_cptc, ccdc_data['Sig'], ccdc_data['Label'], class_map, reduced_output=True)
	top1_cptc_update, topk_cptc_update = LearningUtils.get_topk(sig_model_cptc, update_data['Sig'], update_data['Label'], class_map, reduced_output=True)
	
	top1_ccdc_cptc, topk_ccdc_cptc = LearningUtils.get_topk(sig_model_ccdc, cptc_data['Sig'], cptc_data['Label'], class_map, reduced_output=True)
	top1_ccdc_update, topk_ccdc_update = LearningUtils.get_topk(sig_model_ccdc, update_data['Sig'], update_data['Label'], class_map, reduced_output=True)
	
	top1_update_cptc, topk_update_cptc = LearningUtils.get_topk(sig_model_update, cptc_data['Sig'], cptc_data['Label'], class_map, reduced_output=True)
	top1_update_ccdc, topk_update_ccdc = LearningUtils.get_topk(sig_model_update, ccdc_data['Sig'], ccdc_data['Label'], class_map, reduced_output=True)
	
	cptc_results =   [top1_cptc, top1_cptc_ccdc, top1_cptc_update]
	ccdc_results =   [top1_ccdc_cptc, top1_ccdc, top1_ccdc_update]
	update_results = [top1_update_cptc, top1_update_ccdc, top1_update]
	
	cptc_results_k =   [topk_cptc, topk_cptc_ccdc, topk_cptc_update]
	ccdc_results_k =   [topk_ccdc_cptc, topk_ccdc, topk_ccdc_update]
	update_results_k  = [topk_update_cptc, topk_update_ccdc, topk_update]
	
	out_result = [cptc_results, ccdc_results, update_results]
	out_result_k = [cptc_results_k, ccdc_results_k, update_results_k]
	
	return out_result, out_result_k
	
if __name__ == '__main__' :

	# text_data, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=True)
	# ccdc_data = LearningUtils.transform_dataset(RecentAlertsMapping().ccdc_combined, class_map)
	# temp = [text_data, ccdc_data]
	# combined_data = pd.concat(temp)
	
	# lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode='cve',other_data=text_data,
	# additional_samples=None,pre_trained_lang=False, bs_lang=64, test_mode=False, random_seed=None)
	
	# top1 = []
	# topk = []
	# for i in range(7) :
		# sig_model_comb, top1_comb, topk_comb ,class_map_comb = signature_transfer_learning_classifier_model(combined_data, encoder_name, data_lm, class_map, class_labels,bs_sig=16, test_mode=False, additional_samples=None, random_seed=None)
		# top1.append(top1_comb)
		# topk.append(topk_comb)
		

	
	
	# random_value = random.randint(1,1000000)
	# #random_value = 833874
	# print(random_value)

	# lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode='large',other_data=text_data,
	# additional_samples=None,bs_lang=32, test_mode=False, random_seed=random_value)

	# sig_model_cptc, top1_cptc, topk_cptc ,class_map_cptc = signature_transfer_learning_classifier_model(text_data, encoder_name, data_lm, class_map, class_labels,bs_sig=16, test_mode=False, additional_samples=None, random_seed=random_value)
	
	# ccdc2017_data = LearningUtils.transform_dataset(RecentAlertsMapping().ccdc2017, class_map)
	
	# sig_model_ccdc, top1_ccdc, topk_ccdc ,class_map_ccdc = signature_transfer_learning_classifier_model(ccdc2017_data, encoder_name, data_lm, class_map, class_labels,bs_sig=16, test_mode=False, additional_samples=None, random_seed=random_value)
	
	# ccdc2015_data = LearningUtils.transform_dataset(RecentAlertsMapping().ccdc2015, class_map)
	
	# sig_model_ccdc15, top1_ccdc15, topk_ccdc15 ,class_map_ccdc15 = signature_transfer_learning_classifier_model(ccdc2015_data, encoder_name, data_lm, class_map, class_labels,bs_sig=16, test_mode=False, additional_samples=None, random_seed=random_value)
	
	# ids_update = LearningUtils.transform_dataset(RecentAlertsMapping().get_not_phishing(), class_map)
	
	# sig_model_update, top1_update, topk_update ,class_map_update = signature_transfer_learning_classifier_model(ids_update, encoder_name, data_lm, class_map, class_labels,bs_sig=16, test_mode=False, additional_samples=None, random_seed=random_value)
	
	# classifier_top1, classifier_topk = predicting_others_test(num_runs=10, lang_mode='large') 
	
	# top1 = np.array(classifier_top1)
	# topk = np.array(classifier_topk)
	# avg_top1 = np.average(top1, axis=0)
	# avg_topk = np.average(topk, axis=0)
	
	# top1_cptc_pt, topk_cptc_pt = model_performance_check(pretrained_lang=True, pretrained_class=True ,data_mode='cptc', lang_mode='cve',num_runs=10, bs_lang=32, bs_class=16)
	# top1_cptc_class_npt, topk_cptc_class_npt = model_performance_check(pretrained_lang=True, pretrained_class=False ,data_mode='cptc', lang_mode='cve',num_runs=10, bs_lang=32, bs_class=16)
	# top1_cptc_lang_npt, topk_cptc_lang_npt = model_performance_check(pretrained_lang=False, pretrained_class=True ,data_mode='cptc', lang_mode='cve',num_runs=10, bs_lang=32, bs_class=16)
	# top1_cptc_npt, topk_cptc_npt = model_performance_check(pretrained_lang=False, pretrained_class=False ,data_mode='cptc', lang_mode='cve',num_runs=10, bs_lang=32, bs_class=16)
	
	import time
	split_tests = [.9, .8, .7, .6,.5,.4,.3,.2,.1,.05,.01]
	timing = []
	
	for split in split_tests : 
		start_time = time.time()
		lang_model, data_lm, encoder_name = signature_transfer_learning_language_model(mode='cve',data_split_text=split, additional_samples=None,bs_lang=32, test_mode=False, random_seed=None)
		end_time = time.time()
		total_time = end_time - start_time
		timing.append(total_time)
	
	
	
	
	
	