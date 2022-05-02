import torch

from aif.AttackStagesCondensed import MicroAttackStageCondensed

out_path = './'
if not torch.cuda.is_available():
	print("ERROR: CUDA not available.")
else :
	torch.cuda.set_device(1)
	torch.multiprocessing.freeze_support()
	
DEFAULT_CLASSIFIER_LOCATION = './default_class_model/'
CUSTOM_CLASSIFIER_LOCATION = ''

'''
Creates a alert signature->Condensed Attack Stages predictor,   
'''
class SignatureAttackStagePredictor :

	'''
	Loads in the classifier model using the supplied models or a custom model.
	
	pred_model={"CVE", "CUSTOM"}
	You can also pass in a RNNLearner object in pred_model to use this as a MCD wrapper class
	CVE will supply a model trained with the CVE texts
	Use the global constants to define where these models live.  
	
	'''
	def __init__(self, pred_model='DEFAULT', model_loc=CUSTOM_CLASSIFIER_LOCATION) :
	
		if pred_model == 'DEFAULT' :
			self.classifier_model = load_learner(DEFAULT_CLASSIFIER_LOCATION)
		elif pred_model == 'CUSTOM' :
			self.classifier_model = load_learner(model_loc)
		elif isinstance(pred_model, text.learner.RNNLearner) :
			self.classifier_model = pred_model
		else :
			raise Exception('Unsupported prediction model type.  Consider CUSTOM.')

		return
	
	'''
	Predicts the top-1 given the input text.  Outputs AIF stage.  
	'''
	def predict(self, text:str) -> MicroAttackStageCondensed:
		pred = self.classifier_model.predict(text)
		stg = MicroAttackStageCondensed(int(pred[1]))
		return stg
		
	
	'''
	Same a predict() but outputs the 
	'''
	def predict_proba(self, text:str, transform_proba=False): 
		pred = self.classifier_model.predict(text)
		stg = MicroAttackStageCondensed(int(pred[1]))
		probs = pred[2]
		if transform_proba : probs = self.transform_probs(probs)
		return stg, probs
		
	def predict_mc_proba(self, text:str, transform_proba=False, num_iters=10) -> MicroAttackStageCondensed :
		mcd_preds = self.predict_mc_prob(text, num_iters=num_iters)
		if transform_proba : mcd_preds = self.transform_mc_probs(mcd_preds)
		
		mcd_uncert,pred = self.monte_carlo_dropout_uncertainty(mcd_preds)
		stg = MicroAttackStageCondensed(pred)
		
		return stg, mcd_preds
		
	'''
	Predicts the top-k given the input text.  Outputs a list of AIF stage, length k.
	If output_probs=True, the probilities of the top-k preds are given also.
	'''
	def predict_k(self, text, k=3, output_probs=False) :
	
		if k > len(MicroAttackStageCondensed) :
			raise Exception('K Value too large for classifer.  Please use less than: ', int(len(MicroAttackStageCondensed)))
	
		pred = self.classifier_model.predict(text)
		top_k = torch.topk(pred[2], k)
		
		#Transform the predictions probs out of tensors
		pred_vals = []
		for p in top_k[0] : pred_vals.append(float(p))
		
		#Convert the class labels into the AIF
		k_stg = []
		for stg_val in top_k[1] : k_stg.append(MicroAttackStageCondensed(int(stg_val)))
		
		if not output_probs : return k_stg
		else : return k_stg, pred_vals
		
	def predict_k_mc(self, text, k=3, num_iters=10, transform_proba=False, output_mc_preds=False) :
		if k > len(MicroAttackStageCondensed) :
			raise Exception('K Value too large for classifer.  Please use less than: ', int(len(MicroAttackStageCondensed)))
			
		mcd_preds = self.predict_mc_prob(text, num_iters=num_iters, transform_proba=transform_proba)
		top_k, k_conf = self.mc_dropout_topk_uncertainty(mcd_preds, k=k)
		
		#Convert the labels into the AIF
		k_stg = []
		for stg_val in top_k : k_stg.append(MicroAttackStageCondensed(int(stg_val)))
		
		if output_mc_preds: return k_stg, k_conf, mcd_preds
		return k_stg, k_conf
	
	
	'''
	Returns a sorted list of the highest to lowest probable preditions.
	'''
	def predict_prob(self, text) :
		stg_preds, prob_preds = self.predict_k(text, len(MicroAttackStageCondensed), True)
		return stg_preds, prob_preds
		
	'''
	Wrapper for the predict with mc dropout function.  Returns the pred probs for each dropout
	iteration
	'''	
	def predict_mc_prob(self, text, num_iters=10, transform_proba=False) :
		mcd_preds = self.classifier_model.predict_with_mc_dropout(text, n_times=num_iters)
		if transform_proba : mcd_preds = self.transform_mc_probs(mcd_preds)
		return mcd_preds
	
	'''
	Transforms the activation preditions to sum to 1.  
	'''
	def transform_mc_probs(self, mcd_preds) :
		transformed = []
		for cat, stg, pred in mcd_preds :
			new_probs = pred.div(pred.sum())
			transformed.append((cat, stg, new_probs))
		return transformed
		
	'''
	Normalizes the output tensor from predict and returns it.  
	'''
	def transform_probs(self, probs) :
		return probs.div(probs.sum())
			
	'''
	Determines the predicition uncertainty using the Monte Carlo Dropout Uncertainty 
	metric.  High variance/std (close to 1) indicates high model uncertainty and vice
	versa.
	'''
	def predict_mc_uncertainty(self, text, k=3, num_iters=10) :
		k_stg = self.predict_k(text, k)
		mcd_preds = self.predict_mc_prob(text, num_iters=num_iters)
		mcd_uncert,_ = self.monte_carlo_dropout_uncertainty(mcd_preds)
		return k_stg, mcd_uncert
	
	'''
	Method to determine the MC uncertainty. Returns the uncertainty and the top prediction. 
	'''	
	def monte_carlo_dropout_uncertainty(self, mcd_preds) :
		mcd_probs = [probs for _,_,probs in mcd_preds]
		mcd_probs = torch.stack(mcd_probs)
		mcd_means = mcd_probs.mean(dim=0)
		mcd_std = mcd_probs.std(dim=0)
		_,idx = mcd_means.max(0)
		std = mcd_std.data[idx.item()].item()
		return std, idx.item()
		
	'''
	Returns the topk and the uncertainy values using the prediction probs from predict_mc_prob
	'''	
	def mc_dropout_topk_uncertainty(self, mcd_preds, k=3) :
		mcd_probs = [probs for _,_,probs in mcd_preds]
		mcd_probs = torch.stack(mcd_probs)
		mcd_means = mcd_probs.mean(dim=0)
		mcd_std = mcd_probs.std(dim=0)
		topk_pred = torch.topk(mcd_means, k)
		
		k_conf = [float(mcd_std[int(pred)]) for pred in list(topk_pred[1])]
		topk = [int(val) for val in list(topk_pred[1])]
		
		return topk, k_conf  
		
	'''
	Returns the unique predicted classes across all of the MC iterations.
	'''
	def get_predicted_classes(self, mcd_preds) :
		mcd_preds = [int(pred) for _,pred,probs in mcd_preds]
		pred_set = set(mcd_preds)
		return pred_set
		
	
		
		
		
	
		