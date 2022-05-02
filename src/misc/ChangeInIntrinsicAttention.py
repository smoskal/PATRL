from misc import ClassifierAnalysis
import TransferLearningDemo
import TransferLearningWithNewLabels as trans_learners

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

def get_intrinisic_analysis(classifier, classifier_results, normalize_attention=True) :
	#txt_ci = TextClassificationInterpretation.from_learner(classifier)
	x_vals, y_gt, y_pred, y_pred_conf = result_to_lists(classifier_results)
	int_analysis = ClassifierAnalysis.intrinsic_attention_analysis(classifier, x_vals, y_gt, y_pred)
	correct_answers, incorrect_answers = ClassifierAnalysis.attention_analysis_by_class(int_analysis)

	correct_word_analysis = ClassifierAnalysis.process_word_analysis(correct_answers, normalize=normalize_attention)
	incorrect_word_analysis = ClassifierAnalysis.process_word_analysis(incorrect_answers, normalize=normalize_attention)

	return correct_answers, incorrect_answers, correct_word_analysis, incorrect_word_analysis
	
def combine_attention(attention_dict) :

	combined_attention = {}
	for i, attention in attention_dict.items() :
	
		for id, attention_array in attention.items() :
			if id not in combined_attention : combined_attention[id] = {}
			
			curr_atten = combined_attention[id] 
			
			#Assume the word_atten is a tuple where (word, attention value))
			for word, atten_val in attention_array.items() :
				if word not in curr_atten :
					curr_atten[word] = atten_val
				else :
					curr_atten[word] += atten_val
					
					
	num_runs = float(len(attention_dict))
	#Now normalize the attention with respect to the number of runs
	for id, attention in combined_attention.items() :
		for word in attention.keys() :
			combined_attention[id][word] = combined_attention[id][word] / num_runs
		
		
	return combined_attention
	
def no_lang_classifier_IA(data_split_class=.2, bs_sig=16, num_iters=5, random_seed=None) :

	text_data, class_labels, class_map = TransferLearningDemo.signature_to_data_frame(transform_labels=True)
	train_df_sig, test_df_sig = TransferLearningDemo.split_train_test(text_data, split=data_split_class,
																	  seed=random_seed)
																	  
	correct_attention = {}
	incorrect_attention = {}
	
	for i in range(num_iters) :
		
		sig_model, class_map = trans_learners.signature_transfer_learning_classifier_model(
		None, None, mode='normal',data_split_class=data_split_class, bs_lang=32, bs_sig=16, 
		test_mode=False, additional_samples=None, random_seed=random_seed)
		
		top1_acc, topk_acc, hit_record, topk_hits, full_misses, all_samples = TransferLearningDemo.get_topk(sig_model,test_df_sig['Sig'],test_df_sig['Label'],class_map)
		
		correct_answers, incorrect_answers, correct_word_analysis, incorrect_word_analysis = get_intrinisic_analysis(sig_model, all_samples)
			
		correct_attention[i] = correct_word_analysis
		incorrect_attention[i] = incorrect_word_analysis
		
		
	combined_correct_attention = combine_attention(correct_attention)
	combined_incorrect_attention = combine_attention(incorrect_attention)
		
	return combined_correct_attention, combined_incorrect_attention
	
def basic_lang_classifier_IA(mode='normal', pre_trained_lang=False, data_split_class=.2, bs_sig=16, num_iters=5, random_seed=None) :

	text_data, class_labels, class_map = TransferLearningDemo.signature_to_data_frame(transform_labels=True)
	train_df_sig, test_df_sig = TransferLearningDemo.split_train_test(text_data, split=data_split_class,
																	  seed=random_seed)
																	  
	correct_attention = {}
	incorrect_attention = {}
	
	for i in range(num_iters) :
	
		lang_model, data_lm, encoder_name = trans_learners.signature_transfer_learning_language_model(mode=mode,additional_samples=None, pre_trained_lang=pre_trained_lang,bs_lang=32, bs_sig=16, test_mode=False)
		
		sig_model, class_map = trans_learners.signature_transfer_learning_classifier_model(
		encoder_name, data_lm, mode=mode,data_split_class=data_split_class, bs_lang=32, bs_sig=16, 
		test_mode=False, additional_samples=None, random_seed=random_seed)
		
		top1_acc, topk_acc, hit_record, topk_hits, full_misses, all_samples = TransferLearningDemo.get_topk(sig_model,test_df_sig['Sig'],test_df_sig['Label'],class_map)
		
		correct_answers, incorrect_answers, correct_word_analysis, incorrect_word_analysis = get_intrinisic_analysis(sig_model, all_samples)
			
		correct_attention[i] = correct_word_analysis
		incorrect_attention[i] = incorrect_word_analysis
		
		
	combined_correct_attention = combine_attention(correct_attention)
	combined_incorrect_attention = combine_attention(incorrect_attention)
		
	return combined_correct_attention, combined_incorrect_attention
	
if __name__ == '__main__' :

	correct_attention_no_lang, incorrect_attention_no_lang = no_lang_classifier_IA(num_iters=10)
	correct_attention_no_pt, incorrect_attention_no_pt = basic_lang_classifier_IA(num_iters=10, pre_trained_lang=False)
	correct_attention_pt, incorrect_attention_pt = basic_lang_classifier_IA(num_iters=10, pre_trained_lang=True)
	correct_attention_large, incorrect_attention_large = basic_lang_classifier_IA(mode='large', num_iters=10, pre_trained_lang=False)
	correct_attention_large_pt, incorrect_attention_large_pt = basic_lang_classifier_IA(mode='large',num_iters=10, pre_trained_lang=True)
	
	
	test_analysis = {0:{10:[('noice', .5), ('test', .2)]},	1:{10:[('noice',.7)], 20:[('nice',1)]}}

	
	
	
	
	
	
	
