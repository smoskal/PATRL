import collections
import pickle

import pandas as pd

from aif.AttackStages import MicroAttackStage
from aif.AttackStages import SensorObservability
from aif.AttackStages import SignatureMapping as attkstg_map


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


	out_df = pd.DataFrame(data, columns=['Label', 'Sig'])
	return out_df, classes, class_map

'''
Assumes the tuple structure of (result, result probs, class prediction (topk), and top1)
'''
def topk_confidence_stats(result) :

	topk_confidence = []

	#Top1 class-specific confidence
	top1_pred = {}
	#Sum of the topk confidences WHEN correctly classifiying
	top1_confidence = []
	top1_probs = []

	# Sum of the topk confidences when outside of topk
	miss_confidence = []
	#Contains the a mapping of the truth to the count of misclassified top1
	correct_labels = {}

	for val in result :
		#Asumes that the prob array is in pos 1 of result
		topk_confidence.append(val[1].sum())

		#Now looking at if a correctly classified value has a high confidence
		if val[2][0] == val[3] :
			top1_probs.append(val[1][0])
			top1_confidence.append(val[1].sum())
			if val[3] not in top1_pred :
				top1_pred[val[3]] = [val[1][0]]
			else :
				top1_pred[val[3]].append(val[1][0])
		#Addressing the misses.
		elif val[3] not in val[2] :
			miss_confidence.append(val[1].sum())

			if val[3] not in correct_labels :
				correct_labels[val[3]] = {}
			if val[2][0] not in correct_labels[val[3]] :
				correct_labels[val[3]][val[2][0]] = 1
			else :
				correct_labels[val[3]][val[2][0]] += 1

	output = {
		'top1_conf' : top1_confidence,
		'top1_class_conf' : top1_pred,
		'top1_probabilities' : top1_probs,
		'topk_conf' : topk_confidence,
		'miss_conf' : miss_confidence,
		'missed_label_counts' : correct_labels
	}

	return output

def stage_result_stats(results) :

	attk_stage_stats = {}
	for res in results :
		#assumes the truth is the last value
		if res[3] not in attk_stage_stats :
			attk_stage_stats[res[3]] = 1
		else :
			attk_stage_stats[res[3]] += 1

	return attk_stage_stats



'''
Combines the output from multiple 'k-folds' and puts it into one single list
'''
def compress_folds(outputs) :
	unfolded_data = []
	for fold in outputs :
		unfolded_data.extend(fold)
	return unfolded_data

def get_tuple_col(tuple_list, index=0) :
	return [i[index] for i in tuple_list]

def split_by_common_elements(result, common_elements_list) :

	commonalities = []
	differences = []

	for pred in result :
		if pred[0] in common_elements_list :
			commonalities.append(pred)
		else :
			differences.append(pred)

	return commonalities, differences

def compare_prediction_results(res1, res2, index=0) :
	res1_list = get_tuple_col(res1, index=index)
	res2_list = get_tuple_col(res2, index=index)

	common_elements = list(set(res1_list).intersection(set(res2_list)))

	res1_common, res1_diff = split_by_common_elements(res1, common_elements)
	res2_common, res2_diff = split_by_common_elements(res2, common_elements)

	result1_split = {
		'common' : res1_common,
		'diff' : res1_diff
	}

	result2_split = {
		'common' : res2_common,
		'diff' : res2_diff
	}

	return common_elements, result1_split, result2_split

def save_result(filename, ser_obj) :
	try:
		pickle.dump(ser_obj, open(filename, 'wb'))
		return filename
	except Exception as e:
		print('Pickle Error: ', e)
		return -1

def load_result(filename) :
	try:
		ser_obj = pickle.load(open(filename, 'rb'))
		return ser_obj
	except Exception as e :
		print('Pickle Load Error: ', e)
		return -1

def count_attack_stage_signatures(sig_list) :

	mapping = attkstg_map().get_signature_mapping()

	output = {}
	for sig in sig_list :
		attk_stage = mapping[sig]

		if attk_stage not in output :
			output[attk_stage] = 1
		else:
			output[attk_stage] += 1

	return output

'''
Evaluates how each word in the input contributes to the classifcation and sorts them 
(input, processed string, attention values, y (gt), y (predicted)
'''
def intrinsic_attention_analysis(learner, x, y, y_pred) :

	try :
		txt_ci = TextClassificationInterpretation.from_learner(learner)
	except Exception as e :
		print('ClassifierAnalysis: Error in creation of TextClassificationInterpreation -- ', e)
		return None

	output = []

	for i, input in enumerate(x) :

		attention = txt_ci.intrinsic_attention(input)
		transformed_string = str(attention[0]).split()
		attention_values = attention[1].tolist()

		sorted_attention_string = [x for _, x in sorted(zip(attention_values, transformed_string), reverse=True)]
		sorted_attention = sorted(attention_values, reverse=True)

		if not y_pred :
			data = (input, sorted_attention_string, sorted_attention, y[i])
		else :
			#If we have the data for the predicted value of the model
			data = (input, sorted_attention_string, sorted_attention, y[i], y_pred[i])

		output.append(data)

	return output

'''
Sort the examples by correctly classified, and incorrectly.  Within, sort them by class.  
We then will analyize the specific words that caused the answer.  
'''
def attention_analysis_by_class(int_atten_output) :

	correct_answers = {}
	incorrect_answers = {}

	for atten in int_atten_output :

		if atten[3] == atten[4] :
			if atten[3] not in correct_answers :
				correct_answers[atten[3]] = []
			correct_answers[atten[3]].append(atten)
		else :
			if atten[4] not in incorrect_answers :
				incorrect_answers[atten[4]] = []
			incorrect_answers[atten[4]].append(atten)

	return correct_answers, incorrect_answers

'''
Find the most common words to classify or miss-classify results
'''
def attention_word_analysis(attention_set, normalize=False) :

	word_analysis = {}
	word_count = {}

	for atten in attention_set :

		#Iterate through all of the words in the attention set and sum up their contributions
		for i, val in enumerate(atten[2]) :
			if atten[1][i] not in word_analysis :
				word_analysis[atten[1][i]] = val
				word_count[atten[1][i]] = 1
			else :
				word_analysis[atten[1][i]] += val
				word_count[atten[1][i]] += 1


	if normalize :
		for key, val in word_analysis.items() :
			word_analysis[key] = float(val)/float(word_count[key])

	sort_word_analysis = sorted(word_analysis.items(), key=lambda kv: kv[1], reverse=True)
	sorted_word_analysis = collections.OrderedDict(sort_word_analysis)

	return sorted_word_analysis, word_count

def process_word_analysis(attention_answers, normalize=False) :
	class_word_analysis = {}

	for gt_class, attention_set in attention_answers.items() :
		word_analysis, count = attention_word_analysis(attention_set, normalize=normalize)
		class_word_analysis[gt_class] = word_analysis

	return class_word_analysis

'''
Gets all the unique words from a dataframe so that we can measure the overlap.
We are assuming the format where all the text is under "sig"
'''
def get_unique_words_from_df(in_df) :

	unique_words = list(in_df['Sig'].str.split(' ', expand=True).stack().unique())
	return unique_words


def intersection(lst1, lst2):
	# Use of hybrid method
	temp = set(lst2)
	lst3 = [value for value in lst1 if value in temp]
	return lst3



