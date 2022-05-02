import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from AttackStages import SignatureMapping as attkstg_map
from AttackStages import MicroAttackStage
from AttackStages import MacroAttackStage
from AttackStages import MicroToMacroMapping
import re
from sklearn.model_selection import train_test_split
from AttackStages import SensorObservability
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import MitreAttackInterface
from nltk.corpus import stopwords
from statistics import mean
from statistics import stdev
import ClassifierAnalysis
import copy
from sklearn.model_selection import KFold
from AIF_Mappings import RecentAlertsMapping
import LearningUtils

def x_valid_models(text_data, n_splits=4) :

	folds = []
	kf = KFold(n_splits=n_splits, shuffle=True)

	for fold in kf.split(text_data) :
		folds.append(fold)

	return folds

def train_model(X_train, y_train, cv) :
	X_train_cv = cv.transform(X_train)
	word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
	top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)

	naive_bayes = MultinomialNB()
	naive_bayes.fit(X_train_cv, y_train)

	return naive_bayes, top_words_df

def get_topk(test_model, test_input, test_labels, cv,k=3) :

	test_size = len(test_input)
	full_misses = []
	topk_hits = []
	hit_record = []
	all_records = []

	top1_count = 0
	topk_count = 0

	transformed_input = cv.transform(test_input)
	all_probs = test_model.predict_proba(transformed_input)
	class_labels = list(test_model.classes_)


	for i, pred in enumerate(all_probs) :
		truth = class_labels.index(test_labels.iloc[i])

		top_preds = pred.argsort()[-k:][::-1]
		pred_vals = pred[top_preds]

		actual_truth = test_labels.iloc[i]
		actual_labels = []

		for val in top_preds : actual_labels.append(class_labels[val])

		all_records.append((test_input.iloc[i], pred_vals, actual_labels, actual_truth))

		if top_preds[0] == truth :
			top1_count += 1
			topk_count += 1
			hit_record.append((test_input.iloc[i], pred_vals, actual_labels, actual_truth))
		else :

			if truth in top_preds :
				topk_count += 1
				topk_hits.append((test_input.iloc[i], pred_vals, actual_labels, actual_truth))
			else :
				full_misses.append((test_input.iloc[i], pred_vals, actual_labels, actual_truth))

	top1_acc = float(top1_count) /  float(test_size)
	topk_acc = float(topk_count) / float(test_size)

	return top1_acc, topk_acc, hit_record,topk_hits, full_misses, all_records

def transform_gt(y_labels, class_map) :
	output = []
	for label in y_labels :
		output.append(class_map[label])

	return output

def reverse_dict(in_dict) :
	return dict([(value, key) for key, value in in_dict.items()])

def train_x_valid(text_data, k_folds, cv ,topk=3) :

	classification_models = []
	top1_accuracies = []
	topk_accuracies = []

	top1_hit_record = []
	topk_hit_record = []
	miss_record = []
	all_records = []

	for fold in k_folds :
		train_df = text_data.iloc[fold[0]]
		test_df = text_data.iloc[fold[1]]

		data_model, top_words = train_model(train_df['Filtered_Signature'], train_df['Label'], cv)
		classification_models.append(data_model)

		top1_acc, topk_acc, hit_record, topk_hits, full_misses, all_samples = get_topk(data_model, test_df['Filtered_Signature'], test_df['Label'], cv, k=topk)

		top1_accuracies.append(top1_acc)
		topk_accuracies.append(topk_acc)

		top1_hit_record.append(hit_record)
		topk_hit_record.append(topk_hits)
		miss_record.append(full_misses)
		all_records.append(all_samples)

	return classification_models, top1_accuracies, topk_accuracies, top1_hit_record, topk_hit_record, miss_record, all_records

def word_count_analysis(X_train_cv, y_train, cv) :
	y_reset = y_train.reset_index()
	comb_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
	comb_df['Label'] = y_reset['Label']

	unique_labels = comb_df.Label.unique()
	output_df = pd.DataFrame()

	for label in unique_labels :

		temp_df = comb_df[comb_df['Label'] == label]
		temp_df = temp_df.drop('Label', axis=1)

		temp_count = temp_df.sum(axis=0)
		temp_frame = temp_count.to_frame()
		temp_frame = temp_frame.T
		temp_frame['Label'] = label

		output_df = output_df.append(temp_frame)

	output_df = output_df.set_index('Label')
	return output_df

def label_word_uniqueness(wordcount_df) :
	cols = wordcount_df.columns
	bt = wordcount_df.apply(lambda x: x > 0)
	word_lists = bt.apply(lambda x: list(cols[x.values]), axis=1)

	unique_word_set = {}
	#First lets find the purely unique set of words on a per class basis
	for i, row in word_lists.iteritems():
		unique_words = copy.deepcopy(row)
		for j, next_row in word_lists.iteritems() :
			if i == j : pass
			else :
				unique_words = list(set(unique_words) - set(next_row))
		unique_word_set[i] = unique_words

	#Now find all of the common words between all of the classes
	word_intersection = {}
	for i, row in word_lists.iteritems():
		if i not in word_intersection : word_intersection[i] = {}
		for j, next_row in word_lists.iteritems() :
			if i == j : pass
			else :
				intersection = list(set(row) & set(next_row))
				word_intersection[i][j] = len(intersection)

	return unique_word_set, word_intersection

def word_analysis_df(unique_words, word_intersect) :
	key_list = list(unique_words.keys())
	key_list.sort()
	col_vals = copy.deepcopy(key_list)
	col_vals.insert(0, 'Unique')
	col_vals.insert(0, 'Label')
	result = []
	for val_1 in key_list :
		temp_row = []
		temp_row.append(val_1)
		temp_row.append(len(unique_words[val_1]))

		for val_2 in key_list :
			if val_1 == val_2 : temp_row.append(0)
			else :
				temp_row.append(word_intersect[val_1][val_2])
		result.append(temp_row)

	output_df = pd.DataFrame(result, columns=col_vals)

	return output_df



def test_recent_alerts() :

	not_phishing_sigs = RecentAlertsMapping().get_not_phishing()
	text_data, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=False)
	reverse_class_map = LearningUtils.reverse_dict(class_map)
	
	recent_alerts = pd.DataFrame(columns=['Sig', 'Label'])
	
	for sig, stg in not_phishing_sigs.items() :
		text_data = text_data.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)
		recent_alerts = recent_alerts.append({'Sig': sig, 'Label': reverse_class_map[stg]}, ignore_index=True)
		
	cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
	text_data_cv = cv.fit_transform(list(text_data['Sig']))
	
	x_test_cv = cv.transform(list(recent_alerts['Sig']))
	
	word_counts_recent = word_count_analysis(x_test_cv, recent_alerts['Label'], cv)
	unique_recent, intersect_recent = label_word_uniqueness(word_counts_recent)
	word_df_recent = word_analysis_df(unique_recent, intersect_recent)
	
	text_data_normal, class_labels, class_map = LearningUtils.signature_to_data_frame(transform_labels=False)
	
	text_data_norm_cv = cv.fit_transform(list(text_data_normal['Sig']))
	
	word_counts_norm = word_count_analysis(text_data_norm_cv, text_data_normal['Label'], cv)
	unique_norm, intersect_norm = label_word_uniqueness(word_counts_norm)
	word_df_norm = word_analysis_df(unique_norm, intersect_norm)

	return unique_recent, intersect_recent, unique_norm, intersect_norm

if __name__ == '__main__' :

	# # mitre_attk_patterns = MitreAttackInterface.get_patterns_df(SensorObservability.signature_based)
	text_data, sig_classes, class_map = ClassifierAnalysis.signature_to_data_frame(transform_labels=False)
	stop = set(stopwords.words('english'))
	stop.update(['ET', 'CVE', 'GPL', 'Microsoft', 'Windows', 'mitre', 'citation', 'code', 'org', 'attack', 'system',
					  'used', 'using', 'may'])
	stop.update(['ATTACK_RESPONSE', 'CURRENT_EVENTS', 'DOS', 'INFO', 'POLICY', 'NETBIOS', 'TROJAN', 'WEB_SERVER', 'WEB_SPECIFIC_APPS'])
	text_data['Filtered_Signature'] = text_data['Sig'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


	X_train, X_test, y_train, y_test = train_test_split(text_data['Filtered_Signature'], text_data['Label'], random_state=1)
	# X_train2, X_test2, y_train2, y_test2 = train_test_split(mitre_attk_patterns['Sig'], mitre_attk_patterns['Label'], random_state=1)
	#
	#

	cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
	X_train_cv = cv.fit_transform(X_train)
	X_test_cv = cv.transform(X_test)
	word_freq_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())
	top_words_df = pd.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)
	naive_bayes = MultinomialNB()
	naive_bayes.fit(X_train_cv, y_train)
	predictions = naive_bayes.predict(X_test_cv)
	print('Signature Classification Test:')
	print('Accuracy score: ', accuracy_score(y_test, predictions))
	print('Precision score: ', precision_score(y_test, predictions,average='micro'))
	print('Recall score: ', recall_score(y_test, predictions,average='micro'))
	# #
	#
	# cv2 = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
	# cv2.fit_transform(text_data['Filtered_Signature'])
	# fold_info = x_valid_models(text_data)
	# classification_models, top1_acc, topk_acc, top1_hits, topk_hits, misses, all_samples = train_x_valid(text_data, fold_info, cv2)
	# print('Top 1: ', mean(top1_acc), "\nTop k: ", mean(topk_acc))
	#
	# all_data = []
	# for hit_list in all_samples:
	# 	all_data.extend(hit_list)
	# data_stats = ClassifierAnalysis.topk_confidence_stats(all_data)
	#
	# miss_data = []
	# for hit_list in misses:
	# 	miss_data.extend(hit_list)
	#
	# word_counts = word_count_analysis(X_train_cv, y_train, cv)
	# unique, intersect = label_word_uniqueness(word_counts)
	# word_df = word_analysis_df(unique, intersect)
	#
	#
	#
	# # cv2 = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
	# # X_train_cv2 = cv2.fit_transform(X_train2)
	# # X_test_cv2 = cv2.transform(X_test2)
	# # word_freq_df2 = pd.DataFrame(X_train_cv2.toarray(), columns=cv2.get_feature_names())
	# # top_words_df2 = pd.DataFrame(word_freq_df2.sum()).sort_values(0, ascending=False)
	# # naive_bayes2 = MultinomialNB()
	# # naive_bayes2.fit(X_train_cv2, y_train2)
	# # predictions2 = naive_bayes2.predict(X_test_cv2)
	# #
	# # print('\nMITRE Attack Classification Test: ')
	# # print('Accuracy score 2: ', accuracy_score(y_test2, predictions2))
	# # print('Precision score 2: ', precision_score(y_test2, predictions2,average='micro'))
	# # print('Recall score 2: ', recall_score(y_test2, predictions2,average='micro'))
	#
	#
	# sig_classes.sort()
	# cm = confusion_matrix(y_test, predictions, labels=[x for x in sig_classes])
	# sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,xticklabels=[x for x in sig_classes], yticklabels=[x for x in sig_classes])
	# plt.xlabel('true label')
	# plt.ylabel('predicted label')
	
	unique_recent, intersect_recent, unique_norm, intersect_norm = test_recent_alerts()
	
	intersect_sets = {}
	for stage, word_list in unique_recent.items() :
		intersect = set(word_list).intersection(set(unique_norm[stage]))
		intersect_sets[stage] = intersect
		
