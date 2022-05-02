from enum import Enum

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from aif.AttackStages import MacroAttackStage
from aif.AttackStages import MicroAttackStage
from aif.AttackStages import MicroAttackStageCondensed
from aif.AttackStages import MicroToMacroCondensedMapping
from aif.AttackStages import MicroToMacroMapping
from aif.AttackStages import MicroToMitreTactics
from aif.AttackStages import MitreTactics
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
			
				if obvs_stage.value not in classes:
					classes.append(obvs_stage.value)
			
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
	
def signature_to_data_frame_macro(use_observability=True, observability_type='signature', combine_priv_esc=True, transform_labels=True) :
	data = []
	classes = []
	class_map = {}
	
	if use_observability :
		if observability_type == 'signature' :
			obsv_data = SensorObservability.signature_based
		elif observability_type == 'binary' :
			obsv_data = SensorObservability.binary_testing
		else :
			raise Exception('Unsupported observability type: ' + observability_type)
			
	for signature, attk_stage in attkstg_map.mapping.items() :
		if attk_stage in obsv_data :
			macro_stg = MicroToMacroMapping.mapping[attk_stage] 
			data.append([macro_stg.value, signature]) 
			
	for obvs_stage in obsv_data :
		#For each stage defined in the observability set, add all the signatures in the list.
		if obvs_stage in attkstg_map.attack_stage_mapping :
			
			for signature in attkstg_map.attack_stage_mapping[obvs_stage] :
				data.append([MicroToMacroMapping.mapping[obvs_stage].value, signature])
			
		# if macro_stg.value not in classes:
			# classes.append(macro_stg.value)
			
		# if attk_stage.value not in class_map :
			# class_map[attk_stage.value] = MicroToMacroMapping.mapping[attk_stage].value 
			
	class_map = {}
	# for micro, macro in MicroToMacroMapping.mapping.items() :
		# class_map[micro.value] = macro.value
	for macro in MacroAttackStage :
		class_map[macro.value] = macro.value
		classes.append(macro.value)
		
	out_df = pd.DataFrame(data, columns=['Label', 'Sig'])
	return out_df, classes, class_map
	
def signature_to_data_frame_mitre(use_observability=True, observability_type='signature', combine_priv_esc=True, transform_labels=True) :
	data = []
	classes = []
	class_map = {}
	
	if use_observability :
		if observability_type == 'signature' :
			obsv_data = SensorObservability.signature_based
		elif observability_type == 'binary' :
			obsv_data = SensorObservability.binary_testing
		else :
			raise Exception('Unsupported observability type: ' + observability_type)
			
	for signature, attk_stage in attkstg_map.mapping.items() :
		if attk_stage in obsv_data :
			macro_stg = MicroToMitreTactics.mapping[attk_stage] 
			data.append([macro_stg.value, signature]) 
			
	for obvs_stage in obsv_data :
		#For each stage defined in the observability set, add all the signatures in the list.
		if obvs_stage in attkstg_map.attack_stage_mapping :
			
			for signature in attkstg_map.attack_stage_mapping[obvs_stage] :
				data.append([MicroToMitreTactics.mapping[obvs_stage].value, signature])
			
		# if macro_stg.value not in classes:
			# classes.append(macro_stg.value)
			
		# if attk_stage.value not in class_map :
			# class_map[attk_stage.value] = MicroToMacroMapping.mapping[attk_stage].value 
			
	class_map = {}
	# for micro, macro in MicroToMacroMapping.mapping.items() :
		# class_map[micro.value] = macro.value
	for macro in MitreTactics :
		class_map[macro.value] = macro.value
		classes.append(macro.value)
		
		
	out_df = pd.DataFrame(data, columns=['Label', 'Sig'])
	return out_df, classes, class_map
	
def signature_to_data_frame_condensed(use_observability=True, observability_type='signature', combine_priv_esc=True, transform_labels=False, output_dict=False) :
	data = []
	classes = []
	class_map = {}
	
	
	if use_observability :
		if observability_type == 'signature' :
			obsv_data = SensorObservability.signature_based
		elif observability_type == 'binary' :
			obsv_data = SensorObservability.binary_testing
		else :
			raise Exception('Unsupported observability type: ' + observability_type)
			
	for signature, attk_stage in attkstg_map.mapping.items() :
		if attk_stage in obsv_data :
			macro_stg = MicroToMacroCondensedMapping.mapping[attk_stage] 
			data.append([macro_stg.value, signature]) 
			
	for obvs_stage in obsv_data :
		#For each stage defined in the observability set, add all the signatures in the list.
		if obvs_stage in attkstg_map.attack_stage_mapping :
			
			for signature in attkstg_map.attack_stage_mapping[obvs_stage] :
				data.append([MicroToMacroCondensedMapping.mapping[obvs_stage].value, signature])
			
		# if macro_stg.value not in classes:
			# classes.append(macro_stg.value)
			
		# if attk_stage.value not in class_map :
			# class_map[attk_stage.value] = MicroToMacroMapping.mapping[attk_stage].value 
			
	class_map = {}
	# for micro, macro in MicroToMacroMapping.mapping.items() :
		# class_map[micro.value] = macro.value
	for macro in MicroAttackStageCondensed :
		class_map[macro.value] = macro.value
		classes.append(macro.value)
		
	if output_dict :
		return data, classes, class_map
		
	out_df = pd.DataFrame(data, columns=['Label', 'Sig'])
	return out_df, classes, class_map
	
'''
Formats the data as a binary classifier, one-vs-rest..   Define the Condensed Attack sTage
'''
def signature_to_data_frame_binary(test_attk_stg:MicroAttackStageCondensed, use_observability=True, observability_type='signature', combine_priv_esc=True, transform_labels=False, output_dict=False) :
	data = []
	classes = []
	class_map = {}
	
	
	if use_observability :
		if observability_type == 'signature' :
			obsv_data = SensorObservability.signature_based
		elif observability_type == 'binary' :
			obsv_data = SensorObservability.binary_testing
		else :
			raise Exception('Unsupported observability type: ' + observability_type)
			
	for signature, attk_stage in attkstg_map.mapping.items() :
		if attk_stage in obsv_data :
			macro_stg = MicroToMacroCondensedMapping.mapping[attk_stage] 
			
			if macro_stg == test_attk_stg : data.append([1, signature]) 
			else : data.append([0, signature]) 
			
			
			
	for obvs_stage in obsv_data :
		#For each stage defined in the observability set, add all the signatures in the list.
		if obvs_stage in attkstg_map.attack_stage_mapping :
			
			for signature in attkstg_map.attack_stage_mapping[obvs_stage] :
				if MicroToMacroCondensedMapping.mapping[obvs_stage] == test_attk_stg : data.append([1, signature])
				else : data.append([0, signature])		
			
	class_map = {0:0, 1:1}
	classes = [0,1]
		
	if output_dict :
		return data, classes, class_map
		
	out_df = pd.DataFrame(data, columns=['Label', 'Sig'])
	return out_df, classes, class_map
	
def signature_to_data_frame_one2one(pos_class:MicroAttackStageCondensed, neg_class:MicroAttackStageCondensed,use_observability=True, observability_type='signature', combine_priv_esc=True, transform_labels=False, output_dict=False) :
	data = []
	classes = []
	class_map = {}
	
	
	if use_observability :
		if observability_type == 'signature' :
			obsv_data = SensorObservability.signature_based
		elif observability_type == 'binary' :
			obsv_data = SensorObservability.binary_testing
		else :
			raise Exception('Unsupported observability type: ' + observability_type)
			
	for signature, attk_stage in attkstg_map.mapping.items() :
		if attk_stage in obsv_data :
			macro_stg = MicroToMacroCondensedMapping.mapping[attk_stage] 
			
			if macro_stg != pos_class and macro_stg != neg_class : pass
			else :
				if macro_stg == pos_class : data.append([1, signature]) 
				else : data.append([0, signature]) 
			
			
			
	for obvs_stage in obsv_data :
		#For each stage defined in the observability set, add all the signatures in the list.
		if obvs_stage in attkstg_map.attack_stage_mapping :
			macro_stg = MicroToMacroCondensedMapping.mapping[obvs_stage]
			for signature in attkstg_map.attack_stage_mapping[obvs_stage] :
				if macro_stg != pos_class and macro_stg != neg_class : pass
				else :
					if macro_stg == pos_class : data.append([1, signature]) 
					else : data.append([0, signature]) 		
			
	class_map = {0:0, 1:1}
	classes = [0,1]
		
	if output_dict :
		return data, classes, class_map
		
	out_df = pd.DataFrame(data, columns=['Label', 'Sig'])
	return out_df, classes, class_map

def binary_perf_stats(test_model, test_input, test_labels) :

	test_size = len(test_input)
	
	total_pos_truth = 0
	total_neg_truth = 0
	
	pred_pos = 0
	pred_neg = 0
	
	true_pos = 0
	true_neg = 0
	
	false_pos = 0
	false_neg = 0
	
	
	for i, input in enumerate(test_input) :
		truth = int(test_labels.iloc[i])
		
		if truth == 1 : total_pos_truth +=1
		else : total_neg_truth += 1
		
		pred = test_model.predict(input)
		actual_pred = int(pred[1])
		
		if actual_pred == 1 : pred_pos += 1
		else : pred_neg += 1
		
		if truth == actual_pred and truth == 1: true_pos += 1
		elif truth == actual_pred and truth == 0 : true_neg += 1
		elif truth != actual_pred and truth == 0 and actual_pred == 1 : false_pos += 1
		elif truth != actual_pred and truth == 1 and actual_pred == 0 : false_neg += 1
		
	try : 
		precision = true_pos/(true_pos+false_pos)
	except Exception : 
		precision = -1
	recall = true_pos/(true_pos+false_neg)
	accuracy = (true_pos+true_neg)/test_size
	f1 = true_pos/(true_pos+.5*(false_pos+false_neg))
	
	out_stats = [('test size', test_size),
	('truth pos', total_pos_truth),
	('truth neg', total_neg_truth),
	('pred pos' , pred_pos),
	('pred_neg' , pred_neg),
	('tp', true_pos),
	('tn', true_neg),
	('fp', false_pos),
	('fn', false_neg),
	('precision' , precision),
	('recall', recall),
	('accuracy', accuracy),
	('f1', f1)
	]
	
	return out_stats
	
	
def get_topk(test_model, test_input, test_labels, class_map ,k=3, reduced_output=False) :

	test_size = len(test_input)
	hit_record = []
	full_misses = []
	topk_hits = []
	all_records = []
	any_misses = {}
	
	#new addition.  truth will be the key, dict for the stats.
	miss_counts = {}
	miss_stats = {}
	data_counts = {}

	top1_count = 0
	topk_count = 0

	for i, input in enumerate(test_input) :
		truth = int(test_labels.iloc[i])
		pred = test_model.predict(input)
		

		top_k = torch.topk(pred[2], k)

		#Get the actual attack stage labels from the class map
		actual_truth = class_map[truth]
		actual_labels = []
		actual_pred = class_map[int(pred[1])]
		
		if actual_truth not in data_counts :
			data_counts[actual_truth] = 0
			
		data_counts[actual_truth] += 1

		try:
			for p in top_k[1] : actual_labels.append(class_map[int(p)])
		except KeyError as e:
			print('Learning Utils Label Error: ', input, " -- ", top_k[1] )
			#ipdb.set_trace()



		#Transform the predictions probs out of tensors
		pred_vals = []
		for p in top_k[0] : pred_vals.append(float(p))

		if actual_truth not in miss_stats :
			miss_counts[actual_truth] = 0
			miss_stats[actual_truth] = {}
			
		

		all_records.append((input, pred_vals, actual_labels, actual_truth))

		if int(pred[1]) == truth :
			top1_count += 1
			topk_count += 1
			hit_record.append((input, pred_vals, actual_labels, actual_truth))
		else :
		
			miss_counts[actual_truth] += 1
			if actual_pred not in miss_stats[actual_truth] :
				miss_stats[actual_truth][actual_pred] = 1
			else :
				miss_stats[actual_truth][actual_pred] += 1 
				
			if actual_truth not in any_misses : any_misses[actual_truth] = []
				
			any_misses[actual_truth].append((input, actual_pred ,pred_vals, actual_labels))
				
			k_result = top_k[1]
			if truth in k_result :
				topk_count += 1
				topk_hits.append((input, pred_vals, actual_labels, actual_truth))
			else :
				full_misses.append((input, pred_vals, actual_labels, actual_truth))

	top1_acc = float(top1_count) /  float(test_size)
	topk_acc = float(topk_count) / float(test_size)
	
	if not reduced_output :
		return top1_acc, topk_acc, any_misses, data_counts, miss_counts, miss_stats	
	else :
		return top1_acc, topk_acc
	
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
	
def transform_dataset(data_dict, class_map) :
	reverse_class_map = reverse_dict(class_map)
	temp_data = []
	
	for sig, label in data_dict.items() :
		if not isinstance(label, Enum) :
			temp_data.append([sig, reverse_class_map[label]])
		else:	
			temp_data.append([sig, reverse_class_map[label.value]])
			
	out_df = pd.DataFrame(temp_data, columns=['Sig', 'Label'])
	
	return out_df
	
def transform_dataset_macro(data_dict, class_map) :
	reverse_class_map = reverse_dict(class_map)
	temp_data = []
	
	for sig, label in data_dict.items() :
		if not isinstance(label, Enum) :
			temp_data.append([sig, MicroToMacroMapping.mapping[MicroAttackStage(label)].value])
		else:	
			temp_data.append([sig, MicroToMacroMapping.mapping[MicroAttackStage(label.value)].value])
			
	out_df = pd.DataFrame(temp_data, columns=['Sig', 'Label'])
	
	return out_df
	
def transform_dataset_mitre(data_dict, class_map) :
	reverse_class_map = reverse_dict(class_map)
	temp_data = []
	
	for sig, label in data_dict.items() :
		if not isinstance(label, Enum) :
			temp_data.append([sig, MicroToMitreTactics.mapping[MicroAttackStage(label)].value])
		else:	
			temp_data.append([sig, MicroToMitreTactics.mapping[MicroAttackStage(label.value)].value])
			
	out_df = pd.DataFrame(temp_data, columns=['Sig', 'Label'])
	
	return out_df
	
def transform_dataset_condensed(data_dict, class_map, output_dict=False) :
	reverse_class_map = reverse_dict(class_map)
	temp_data = []
	
	for sig, label in data_dict.items() :
		if not isinstance(label, Enum) :
			temp_data.append([MicroToMacroCondensedMapping.mapping[MicroAttackStage(label)].value, sig])
		else:	
			temp_data.append([MicroToMacroCondensedMapping.mapping[MicroAttackStage(label.value)].value, sig])
	
	if output_dict :
		return temp_data
	
	out_df = pd.DataFrame(temp_data, columns=['Label', 'Sig'])
	
	return out_df
	
def transform_dataset_binary(data_dict, test_attk_stg:MicroAttackStageCondensed, output_dict=False) :
	temp_data = []
	
	for sig, label in data_dict.items() :
		if not isinstance(label, Enum) :
			stg = MicroToMacroCondensedMapping.mapping[MicroAttackStage(label)]
		
			if stg == test_attk_stg : temp_data.append([1, sig])
			else : temp_data.append([0, sig])
		
		else:	
			stg = MicroToMacroCondensedMapping.mapping[MicroAttackStage(label.value)]
		
			if stg == test_attk_stg : temp_data.append([1, sig])
			else : temp_data.append([0, sig])
	
	if output_dict :
		return temp_data
	
	out_df = pd.DataFrame(temp_data, columns=['Label', 'Sig'])
	
	return out_df
	
def transform_dataset_one2one(data_dict, pos_class:MicroAttackStageCondensed, neg_class:MicroAttackStageCondensed,output_dict=False) :
	temp_data = []
	
	for sig, label in data_dict.items() :
		if not isinstance(label, Enum) :
			stg = MicroToMacroCondensedMapping.mapping[MicroAttackStage(label)]
		
			if stg != pos_class and stg != neg_class : pass
			else :
				if stg == pos_class : temp_data.append([1, sig]) 
				else : temp_data.append([0, sig])
		
		else:	 
			stg = MicroToMacroCondensedMapping.mapping[MicroAttackStage(label.value)]
		
			if stg != pos_class and stg != neg_class : pass
			else :
				if stg == pos_class : temp_data.append([1, sig]) 
				else : temp_data.append([0, sig])
	
	if output_dict :
		return temp_data
	
	out_df = pd.DataFrame(temp_data, columns=['Label', 'Sig'])
	
	return out_df
	
	
def count_classes(dict) :
	output = {}
	
	for sig, stg in dict.items() :
		if stg not in output : output[stg] = 1
		else : output[stg] = output[stg]+1
		
	return output
	
	
	