import pandas as pd
import torch
from utils.NaiveBayesTextClassifcation import train_x_valid as train_x_valid_nb
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from interfaces import MitreAttackInterface
from utils import signatureWithCVE as cve_interface
from aif.AttackStages import MicroAttackStage
from aif.AttackStages import SensorObservability
from aif.AttackStages import SignatureMapping as attkstg_map

out_path = '../data_store/'
sig_classifier = None
lang_model = None
class_model = None
signature_classes = None
class_mapping = None
test_data = None

def signature_to_data_frame(use_observability=True, observability_type='signature', combine_priv_esc=True, transform_labels=True) :
	data = []
	classes = []

	if use_observability :
		if observability_type == 'signature' :
			obsv_data = SensorObservability.signature_based
		elif observability_type == 'binary' :
			obsv_data = SensorObservability.binary_testing
		else :
			raise Exception('Unsupported observability type: ' + observability_type)

		for signature, attk_stage in attkstg_map.mapping.items() :
			if attk_stage in obsv_data :
				if combine_priv_esc and (attk_stage == MicroAttackStage.USER_PRIV_ESC or attk_stage == MicroAttackStage.ROOT_PRIV_ESC) :
					data.append([MicroAttackStage.PRIV_ESC.value, signature])
				else :
					data.append([attk_stage.value, signature])

				if attk_stage.value not in classes:
					classes.append(attk_stage.value)

		if combine_priv_esc and MicroAttackStage.PRIV_ESC.value not in classes:
			classes.append(MicroAttackStage.PRIV_ESC.value)
			classes.remove(MicroAttackStage.ROOT_PRIV_ESC.value)
			classes.remove(MicroAttackStage.USER_PRIV_ESC.value)

		for obvs_stage in obsv_data :
			#For each stage defined in the observability set, add all the signatures in the list.
			if obvs_stage in attkstg_map.attack_stage_mapping :
				for signature in attkstg_map.attack_stage_mapping[obvs_stage] :
					#Combine Root and User privs into one class if needed
					if combine_priv_esc and (
							obvs_stage == MicroAttackStage.USER_PRIV_ESC or obvs_stage == MicroAttackStage.ROOT_PRIV_ESC):
						data.append([MicroAttackStage.PRIV_ESC.value, signature])
					else:
						data.append([obvs_stage.value, signature])

	else :
		for signature, attk_stage in attkstg_map.mapping.items():
			data.append([attk_stage.value, signature])
			if attk_stage.value not in classes:
				classes.append(attk_stage.value)

		for attk_stage, sig_list in attkstg_map.attack_stage_mapping.items() :
			for signature in sig_list :
				data.append([attk_stage.value, signature])

	class_map = {}
	if transform_labels :
		classes.sort()
		new_labels = []
		transform_map = {}
		class_count = 0
		for class_label in classes :
			class_map[class_count] = class_label
			transform_map[class_label] = class_count
			new_labels.append(class_count)
			class_count += 1

		for sample in data :
			sample[0] = transform_map[sample[0]]

		classes=new_labels
	else :
		for stage in MicroAttackStage :
			class_map[stage.value] = stage.value


	out_df = pd.DataFrame(data, columns=['Label', 'Sig'])
	return out_df, classes, class_map

def split_train_test(df, split=0.2, seed=None) :
	train, test = train_test_split(df, test_size=split,random_state=seed)
	return train, test

def transform_gt(y_labels, class_map) :
	output = []
	for label in y_labels :
		output.append(class_map[label])

	return output

def reverse_dict(in_dict) :
	return dict([(value, key) for key, value in in_dict.items()])

def main_part1() :
	torch.multiprocessing.freeze_support()
	text_data, sig_classes = signature_to_data_frame()
	mitre_attk_patterns = MitreAttackInterface.get_patterns_df(SensorObservability.signature_based)

	stop = set(stopwords.words('english'))
	stop.update(['ET', 'CVE', 'GPL', 'Microsoft', 'Windows', 'mitre', 'citation', 'code', 'org', 'attack', 'system',
	             'used', 'using', 'may'])
	stop.update(['ATTACK_RESPONSE', 'CURRENT_EVENTS', 'DOS', 'INFO', 'POLICY', 'NETBIOS', 'TROJAN', 'WEB_SERVER',
	             'WEB_SPECIFIC_APPS'])
	text_data['Filtered_Signature'] = text_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

	train_df, test_df = split_train_test(text_data)
	train_df_mit, test_df_mit = split_train_test(mitre_attk_patterns)
	data_lm = TextLMDataBunch.from_df(path='../data_store/',train_df=train_df, valid_df=test_df)
	data_clas = TextClasDataBunch.from_df(path=out_path, train_df=train_df, valid_df=test_df, classes=sig_classes, vocab=data_lm.train_ds.vocab, bs=32)

	data_lm_mit = TextLMDataBunch.from_df(path='../data_store/', train_df=train_df_mit, valid_df=test_df_mit)
	data_clas_mit = TextClasDataBunch.from_df(path=out_path, train_df=train_df_mit, valid_df=test_df_mit, classes=sig_classes,
	                                      vocab=data_lm.train_ds.vocab, bs=32)

	print("Signature Language Model Training:")
	learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
	learn.fit_one_cycle(1, 1e-2)
	learn.save_encoder('ft_enc')

	print("MITRE Attack Language Model Training:")
	learn = language_model_learner(data_lm_mit, AWD_LSTM, drop_mult=0.3)
	learn.fit_one_cycle(1, 1e-2)
	learn.save_encoder('ft_enc_mitre')

	print("Classifier Training")
	learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
	learn_class.load_encoder('ft_enc')
	learn_class.fit_one_cycle(1, 1e-2)
	learn_class.fit_one_cycle(1, 1e-3)
	learn_class.fit_one_cycle(1, 1e-4)
	learn_class.fit_one_cycle(1, 1e-5)
	learn_class.fit_one_cycle(1, 1e-6)
	learn_class.fit_one_cycle(1, 1e-7)
	learn_class.fit_one_cycle(1, 1e-8)


	#learn_class.lr_find(start_lr=1e-10, end_lr=1e2)
	#learn_class.recorder.plot()

def main_part2() :
	stop = set(stopwords.words('english'))
	stop.update(['ET', 'CVE', 'GPL', 'Microsoft', 'Windows', 'mitre', 'citation', 'code', 'org', 'attack', 'system',
	             'used', 'using', 'may'])
	#stop.update(['ATTACK_RESPONSE', 'CURRENT_EVENTS', 'DOS', 'INFO', 'POLICY', 'NETBIOS', 'TROJAN', 'WEB_SERVER',
	#             'WEB_SPECIFIC_APPS'])

	torch.multiprocessing.freeze_support()
	path = untar_data(URLs.WIKITEXT_TINY)
	text_data, sig_classes, class_map = signature_to_data_frame()
	sig_classes.sort()

	text_data['Filtered_Signature'] = text_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	text_data = text_data.drop(['Sig'], axis=1)

	the_tokenizer = Tokenizer()

	train_df, test_df = split_train_test(text_data)
	data_lm = TextLMDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, bs=16, text_cols='Filtered_Signature', label_cols='Label', tokenizer=the_tokenizer)
	data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, classes=sig_classes,
	                                      vocab=data_lm.train_ds.vocab, bs=16, text_cols='Filtered_Signature', label_cols='Label', tokenizer=the_tokenizer)

	print("Signature Language Model Training:")
	learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
	learn.fit_one_cycle(20, 1e-1)
	#learn.fit_one_cycle(10, 0.5e-2)
	learn.save_encoder('ft_enc')

	print("Classifier Training")
	learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
	learn_class.load_encoder('ft_enc')
	learn_class.unfreeze()
	learn_class.fit_one_cycle(5, 1e-3)
	#learn.fit_one_cycle(20, 1e-1)

	global sig_classifier, signature_classes, lang_model, class_mapping
	sig_classifier = learn_class
	signature_classes = sig_classes
	lang_model = learn
	class_mapping = class_map

	pass

def main_gpu(pre_trained=True) :
	if not torch.cuda.is_available() :
		print("ERROR: CUDA not available.")
		return

	torch.cuda.set_device(0)
	stop = set(stopwords.words('english'))
	stop.update(['ET', 'CVE', 'GPL', 'Microsoft', 'Windows', 'mitre', 'citation', 'code', 'org', 'attack', 'system',
	             'used', 'using', 'may'])
	# stop.update(['ATTACK_RESPONSE', 'CURRENT_EVENTS', 'DOS', 'INFO', 'POLICY', 'NETBIOS', 'TROJAN', 'WEB_SERVER',
	#             'WEB_SPECIFIC_APPS'])

	torch.multiprocessing.freeze_support()
	path = untar_data(URLs.WIKITEXT_TINY)
	text_data, sig_classes, class_map = signature_to_data_frame()
	sig_classes.sort()

	text_data['Filtered_Signature'] = text_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	text_data = text_data.drop(['Sig'], axis=1)

	the_tokenizer = Tokenizer()

	train_df, test_df = split_train_test(text_data, split=0.25)
	data_lm = TextLMDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, bs=16,
	                                  text_cols='Filtered_Signature', label_cols='Label', tokenizer=the_tokenizer)
	data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, classes=sig_classes,
	                                      vocab=data_lm.train_ds.vocab, bs=16, text_cols='Filtered_Signature',
	                                      label_cols='Label', tokenizer=the_tokenizer)

	print("Signature Language Model Training:")
	learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5, pretrained=pre_trained)
	learn.freeze()
	learn.fit_one_cycle(cyc_len=10, max_lr=1e-1, moms=(0.8, 0.7))

	learn.freeze_to(-2)
	learn.fit_one_cycle(5, slice(1e-4, 1e-2), moms=(0.8, 0.7))

	learn.freeze_to(-3)
	learn.fit_one_cycle(5, slice(1e-5, 5e-3), moms=(0.8, 0.7))

	learn.unfreeze()
	learn.fit_one_cycle(5, slice(1e-5, 1e-3), moms=(0.8, 0.7))
	learn.save_encoder('ft_enc')
	#
	# learn.lr_find(start_lr=1e-8, end_lr=1e2)
	# learn.recorder.plot()

	print("Classifier Training")
	learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3, pretrained=pre_trained)
	learn_class.load_encoder('ft_enc')

	learn_class.freeze()
	learn_class.fit_one_cycle(cyc_len=10, max_lr=1e-3, moms=(0.8, 0.7))

	learn_class.freeze_to(-3)
	learn_class.fit_one_cycle(3, max_lr=1e-4, moms=(0.8, 0.7))

	learn_class.unfreeze()
	learn_class.fit_one_cycle(10, slice(1e-5, 1e-3), moms=(0.8, 0.7))




	# learn_class.unfreeze()
	# learn_class.fit_one_cycle(5, 1e-3)
	# # learn.fit_one_cycle(20, 1e-1)
	#
	global sig_classifier, signature_classes, lang_model, class_mapping, class_model, test_data
	sig_classifier = learn_class
	signature_classes = sig_classes
	lang_model = learn
	class_mapping = class_map
	test_data = test_df

'''
Training Set: Signatures + MITRE Attack, Testing Set: Signatures
'''
def main_gpu_mitre(pre_trained=True) :
	if not torch.cuda.is_available() :
		print("ERROR: CUDA not available.")
		return

	torch.cuda.set_device(0)
	stop = set(stopwords.words('english'))
	stop.update(['ET', 'CVE', 'GPL', 'Microsoft', 'Windows', 'mitre', 'citation', 'code', 'org', 'attack', 'system',
	             'used', 'using', 'may'])
	# stop.update(['ATTACK_RESPONSE', 'CURRENT_EVENTS', 'DOS', 'INFO', 'POLICY', 'NETBIOS', 'TROJAN', 'WEB_SERVER',
	#             'WEB_SPECIFIC_APPS'])

	mitre_data = MitreAttackInterface.get_patterns_df(observability_list=SensorObservability.signature_based, combine_priv_esc=True)

	torch.multiprocessing.freeze_support()
	path = untar_data(URLs.WIKITEXT_TINY)
	text_data, sig_classes, class_map = signature_to_data_frame(transform_labels=False)
	sig_classes.sort()

	text_data['Filtered_Signature'] = text_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	text_data = text_data.drop(['Sig'], axis=1)

	mitre_data['Filtered_Signature'] = mitre_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	mitre_data = mitre_data.drop(['Sig'], axis=1)

	the_tokenizer = Tokenizer()

	train_df, test_df = split_train_test(text_data, split=0.25)
	train_df = train_df.append(mitre_data)

	data_lm = TextLMDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, bs=16,
	                                  text_cols='Filtered_Signature', label_cols='Label', tokenizer=the_tokenizer)
	data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, classes=sig_classes,
	                                      vocab=data_lm.train_ds.vocab, bs=16, text_cols='Filtered_Signature',
	                                      label_cols='Label', tokenizer=the_tokenizer)

	print("Signature Language Model Training:")
	learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5, pretrained=pre_trained)
	learn.freeze()
	learn.fit_one_cycle(cyc_len=10, max_lr=1e-1, moms=(0.8, 0.7))

	learn.freeze_to(-2)
	learn.fit_one_cycle(10, slice(1e-4, 1e-2), moms=(0.8, 0.7))

	learn.freeze_to(-3)
	learn.fit_one_cycle(10, slice(1e-5, 5e-3), moms=(0.8, 0.7))

	learn.unfreeze()
	learn.fit_one_cycle(10, slice(1e-5, 1e-3), moms=(0.8, 0.7))
	learn.save_encoder('ft_enc')


	print("Classifier Training")
	learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, pretrained=pre_trained)
	learn_class.load_encoder('ft_enc')
	learn_class.freeze()
	learn_class.fit_one_cycle(cyc_len=10, max_lr=1e-3, moms=(0.8, 0.7))

	learn_class.freeze_to(-3)
	learn_class.fit_one_cycle(3, max_lr=1e-5, moms=(0.8, 0.7))

	learn_class.unfreeze()
	learn_class.fit_one_cycle(3, slice(1e-5, 1e-3), moms=(0.8, 0.7))

'''
Training set- MITRE Att&ck, Testing Set- Signatures
'''
def main_gpu_mitre_only() :
	if not torch.cuda.is_available() :
		print("ERROR: CUDA not available.")
		return

	torch.cuda.set_device(0)
	stop = set(stopwords.words('english'))
	stop.update(['ET', 'CVE', 'GPL', 'Microsoft', 'Windows', 'mitre', 'citation', 'code', 'org', 'attack', 'system',
	             'used', 'using', 'may'])
	# stop.update(['ATTACK_RESPONSE', 'CURRENT_EVENTS', 'DOS', 'INFO', 'POLICY', 'NETBIOS', 'TROJAN', 'WEB_SERVER',
	#             'WEB_SPECIFIC_APPS'])

	mitre_data = MitreAttackInterface.get_patterns_df(observability_list=SensorObservability.signature_based, combine_priv_esc=True)

	torch.multiprocessing.freeze_support()
	path = untar_data(URLs.WIKITEXT_TINY)
	text_data, sig_classes, class_map = signature_to_data_frame(transform_labels=False)
	sig_classes.sort()

	text_data['Filtered_Signature'] = text_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	text_data = text_data.drop(['Sig'], axis=1)

	mitre_data['Filtered_Signature'] = mitre_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	mitre_data = mitre_data.drop(['Sig'], axis=1)

	the_tokenizer = Tokenizer()

	train_df = mitre_data
	test_df = text_data

	data_lm = TextLMDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, bs=16,
	                                  text_cols='Filtered_Signature', label_cols='Label', tokenizer=the_tokenizer)
	data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, classes=sig_classes,
	                                      vocab=data_lm.train_ds.vocab, bs=16, text_cols='Filtered_Signature',
	                                      label_cols='Label', tokenizer=the_tokenizer)

	print("Signature Language Model Training:")
	learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

	learn.lr_find(start_lr=1e-8, end_lr=1e2)
	learn.recorder.plot()

	learn.freeze()
	learn.fit_one_cycle(cyc_len=10, max_lr=1e-1, moms=(0.8, 0.7))

	learn.unfreeze()
	learn.fit_one_cycle(10, max_lr=1e-2, moms=(0.8, 0.7))
	learn.save_encoder('ft_enc')


	print("Classifier Training")
	learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
	learn_class.load_encoder('ft_enc')
	learn_class.freeze()
	learn_class.fit_one_cycle(cyc_len=3, max_lr=1e-1, moms=(0.8, 0.7))

	learn_class.unfreeze()
	learn_class.fit_one_cycle(3, max_lr=1e-2, moms=(0.8, 0.7))


def get_topk(test_model, test_input, test_labels, class_map ,k=3) :

	test_size = len(test_input)
	hit_record = []
	full_misses = []
	topk_hits = []
	all_records = []

	top1_count = 0
	topk_count = 0

	for i, input in enumerate(test_input) :
		truth = int(test_labels.iloc[i])
		pred = test_model.predict(input)

		top_k = torch.topk(pred[2], k)

		#Get the actual attack stage labels from the class map
		actual_truth = class_map[truth]
		actual_labels = []

		try:
			for p in top_k[1] : actual_labels.append(class_map[int(p)])
		except KeyError as e:
			print(input, " -- ", top_k[1] )



		#Transform the predictions probs out of tensors
		pred_vals = []
		for p in top_k[0] : pred_vals.append(float(p))


		all_records.append((input, pred_vals, actual_labels, actual_truth))

		if int(pred[1]) == truth :
			top1_count += 1
			topk_count += 1
			hit_record.append((input, pred_vals, actual_labels, actual_truth))
		else :
			k_result = top_k[1]
			if truth in k_result :
				topk_count += 1
				topk_hits.append((input, pred_vals, actual_labels, actual_truth))
			else :
				full_misses.append((input, pred_vals, actual_labels, actual_truth))

	top1_acc = float(top1_count) /  float(test_size)
	topk_acc = float(topk_count) / float(test_size)

	return top1_acc, topk_acc, hit_record,topk_hits, full_misses, all_records

def train_model(train_df, test_df, class_labels, bs=16, dropout=0.5, pre_trained=True, debug_mode=False) :

	if not torch.cuda.is_available() :
		print("ERROR: CUDA not available.")
		return

	torch.cuda.set_device(0)
	torch.multiprocessing.freeze_support()


	path = untar_data(URLs.WIKITEXT_TINY)
	the_tokenizer = Tokenizer()

	data_lm = TextLMDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, bs=bs,
	                                  text_cols='Filtered_Signature', label_cols='Label', tokenizer=the_tokenizer)
	data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df, valid_df=test_df, classes=class_labels,
	                                      vocab=data_lm.train_ds.vocab, bs=bs, text_cols='Filtered_Signature',
	                                      label_cols='Label', tokenizer=the_tokenizer)
	if not debug_mode :
		print("Signature Language Model Training:")
		learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
		learn.freeze()
		learn.fit_one_cycle(cyc_len=10, max_lr=1e-1, moms=(0.8, 0.7))

		learn.freeze_to(-2)
		learn.fit_one_cycle(5, slice(1e-4, 1e-2), moms=(0.8, 0.7))

		learn.freeze_to(-3)
		learn.fit_one_cycle(5, slice(1e-5, 5e-3), moms=(0.8, 0.7))

		learn.unfreeze()
		learn.fit_one_cycle(5, slice(1e-5, 1e-3), moms=(0.8, 0.7))
		learn.save_encoder('ft_enc')

		print("Classifier Training")
		learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
		learn_class.load_encoder('ft_enc')

		learn_class.freeze()
		learn_class.fit_one_cycle(cyc_len=8, max_lr=1e-3, moms=(0.8, 0.7))

		learn_class.freeze_to(-3)
		learn_class.fit_one_cycle(2, max_lr=1e-4, moms=(0.8, 0.7))

		learn_class.unfreeze()
		learn_class.fit_one_cycle(6, slice(1e-5, 1e-3), moms=(0.8, 0.7))
	else :
		print('Debug mode enabled.')
		learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
		learn.save_encoder('ft_enc')
		learn.fit_one_cycle(cyc_len=10, max_lr=1e-1, moms=(0.8, 0.7))

		learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
		learn_class.load_encoder('ft_enc')
		learn_class.unfreeze()
		learn_class.fit_one_cycle(1, slice(1e-5, 1e-3), moms=(0.8, 0.7))

	return learn_class, learn

def init_sig_data() :
	stop = set(stopwords.words('english'))
	stop.update(['ET', 'CVE', 'GPL', 'Microsoft', 'Windows', 'mitre', 'citation', 'code', 'org', 'attack', 'system',
	             'used', 'using', 'may'])
	text_data, sig_classes, class_map = signature_to_data_frame()
	sig_classes.sort()

	text_data['Filtered_Signature'] = text_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	text_data = text_data.drop(['Sig'], axis=1)

	return text_data, sig_classes, class_map

def x_valid_models(text_data, n_splits=4) :

	folds = []
	kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)

	for fold in kf.split(text_data) :
		folds.append(fold)

	return folds

'''
Note the difference betwee class_labels and class_map.  Class Labels is just a list of the labels, where the map is
label-> Attack Stage.  Labels are 0-15. 
'''
def train_x_valid(text_data, class_labels, k_folds, class_map, topk=3, debug_mode=False) :

	classification_models = []
	top1_accuracies = []
	topk_accuracies = []

	top1_record = []
	topk_hit_record = []
	miss_record = []
	all_records = []

	for fold in k_folds :
		train_df = text_data.iloc[fold[0]]
		test_df = text_data.iloc[fold[1]]

		learn_class, learn_lang = train_model(train_df, test_df, class_labels=class_labels, debug_mode=debug_mode)
		classification_models.append(learn_class)

		top1_acc, topk_acc, hit_record, topk_hits, full_misses, all_samples = get_topk(learn_class, test_df['Filtered_Signature'], test_df['Label'], class_map,k=topk)

		top1_accuracies.append(top1_acc)
		topk_accuracies.append(topk_acc)

		top1_record.append(hit_record)
		topk_hit_record.append(topk_hits)
		miss_record.append(full_misses)
		all_records.append(all_samples)

	return classification_models, top1_accuracies, topk_accuracies, top1_record ,topk_hit_record, miss_record, all_records

def x_valid_main() :

	text_data, class_labels, class_map = init_sig_data()
	fold_info = x_valid_models(text_data)
	classification_models, top1_acc, topk_acc, top1_record,topk_hit_record, miss_record, all_samples = train_x_valid(text_data, class_labels, fold_info, class_map, debug_mode=False)

	print('Top 1: ', top1_acc, "\nTop k: ", topk_acc)
	return classification_models, top1_record, topk_hit_record, miss_record, all_samples

def common_folds_classification() :

	text_data, class_labels, class_map = init_sig_data()
	fold_info = x_valid_models(text_data)

	cv2 = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True,
						  stop_words='english')
	cv2.fit_transform(text_data['Filtered_Signature'])
	nb_models, nb_top1_acc, nb_topk_acc, nb_top1_hits, nb_topk_hits, nb_misses, nb_all_samples = train_x_valid_nb(text_data,
																										 fold_info, cv2)

	lstm_models, lstm_top1_acc, lstm_topk_acc, lstm_top1_record, lstm_topk_hit_record, lstm_miss_record, lstm_all_samples = train_x_valid(
		text_data, class_labels, fold_info, class_map, debug_mode=False)

	lstm_output = {
		'models': lstm_models,
		'top1' : lstm_top1_acc,
		'topk' : lstm_topk_acc,
		'top1_record' : lstm_top1_record,
		'topk_record' : lstm_topk_hit_record,
		'miss_record' : lstm_miss_record,
		'all' : lstm_all_samples
	}

	nb_output = {
		'models': nb_models,
		'top1': nb_top1_acc,
		'topk': nb_topk_acc,
		'top1_record': nb_top1_hits,
		'topk_record': nb_topk_hits,
		'miss_record': nb_misses,
		'all': nb_all_samples
	}

	return lstm_output, nb_output

def train_model_cve(bs=16, dropout=0.5, pre_trained=True, debug_mode=False) :

	if not torch.cuda.is_available() :
		print("ERROR: CUDA not available.")
		return

	torch.cuda.set_device(0)
	torch.multiprocessing.freeze_support()

	stop = set(stopwords.words('english'))
	stop.update(['ET', 'CVE', 'GPL', 'Microsoft', 'Windows', 'mitre', 'citation', 'code', 'org', 'attack', 'system',
				 'used', 'using', 'may'])
	text_data, class_labels, class_map = signature_to_data_frame()
	class_labels.sort()
	text_data['Filtered_Signature'] = text_data['Sig'].apply(
		lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
	text_data = text_data.drop(['Sig'], axis=1)
	train_df_sig, test_df_sig = split_train_test(text_data, split=0.25)

	print('Getting the CVE data..')
	cve_text = cve_interface.get_all_cve_summaries()
	train_df_cve, test_df_cve = split_train_test(cve_text, split=0.15)
	print('Done!')

	path = untar_data(URLs.WIKITEXT_TINY)
	the_tokenizer = Tokenizer()

	data_lm = TextLMDataBunch.from_df(path=path, train_df=train_df_cve, valid_df=test_df_cve, bs=bs,
	                                  text_cols='Sig', tokenizer=the_tokenizer)

	data_clas = TextClasDataBunch.from_df(path=path, train_df=train_df_sig, valid_df=test_df_sig, classes=class_labels,
	                                      vocab=data_lm.train_ds.vocab, bs=bs, text_cols='Filtered_Signature',
	                                      label_cols='Label', tokenizer=the_tokenizer)
	if not debug_mode :
		print("Signature Language Model Training:")
		learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
		learn.freeze()
		learn.fit_one_cycle(cyc_len=5, max_lr=1e-2, moms=(0.8, 0.7))

		learn.freeze_to(-2)
		learn.fit_one_cycle(5, slice(1e-4, 1e-2), moms=(0.8, 0.7))

		learn.freeze_to(-3)
		learn.fit_one_cycle(5, slice(1e-5, 5e-3), moms=(0.8, 0.7))

		learn.unfreeze()
		learn.fit_one_cycle(5, slice(1e-5, 1e-3), moms=(0.8, 0.7))
		learn.save_encoder('ft_enc_cve')

		print("Classifier Training")
		learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
		learn_class.load_encoder('ft_enc_cve')

		learn_class.freeze()
		learn_class.fit_one_cycle(cyc_len=8, max_lr=1e-3, moms=(0.8, 0.7))

		learn_class.freeze_to(-3)
		learn_class.fit_one_cycle(2, max_lr=1e-4, moms=(0.8, 0.7))

		learn_class.unfreeze()
		learn_class.fit_one_cycle(6, slice(1e-5, 1e-3), moms=(0.8, 0.7))
	else :
		print('Debug mode enabled.')
		learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
		learn.save_encoder('ft_enc')
		learn.fit_one_cycle(cyc_len=10, max_lr=1e-1, moms=(0.8, 0.7))

		learn_class = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=dropout, pretrained=pre_trained)
		learn_class.load_encoder('ft_enc')
		learn_class.unfreeze()
		learn_class.fit_one_cycle(1, slice(1e-5, 1e-3), moms=(0.8, 0.7))

	return learn_class, learn


if __name__ == '__main__' :
	#main()
	#main_part2()
	#main_gpu(pre_trained=True)
	#main_gpu_mitre(pre_trained=True)
	#main_gpu_mitre_only()
	#class_models, top1_record,topk_hit_record, miss_record, all_samples = x_valid_main()
	#lstm_output, nb_output = common_folds_classification()
	signature_model, language_model = train_model_cve()

# tokenizer = Tokenizer()
# tok = SpacyTokenizer('en')
# token = ' '.join(tokenizer.process_text("DOS Microsoft Remote Desktop Protocol (RDP) maxChannelIds Integer indef DoS Attempt", tok))
