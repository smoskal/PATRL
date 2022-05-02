import pandas as pd
# from stix2 import FileSystemSource
# from stix2 import Filter
from pymongo import MongoClient

from aif.AttackStages import MacroAttackStage
from aif.AttackStages import MicroAttackStage
from aif.AttackStages import MicroToMacroMapping
from aif.AttackStages import SensorObservability
from aif.AttackStages import SignatureMapping as attkstg_map

attack_location = 'C:/Users/sfm5015/Documents/cti/enterprise-attack'
preattack_location = 'C:/Users/sfm5015/Documents/cti/pre-attack'

database_loc = 'mongodb://localhost:27017'
database_name = 'MITRE_Attack'
collection_name = 'attack_oct2019'
api_username = "api"
api_password = "mongo_api"

def attack_patterns() :
	fs = FileSystemSource(attack_location)
	filt = Filter('type', '=', 'attack-pattern')
	techniques = fs.query([filt])
	return techniques



def get_preattack_patterns() :
	fs = FileSystemSource(preattack_location)
	filt = Filter('type', '=', 'attack-pattern')
	techniques = fs.query([filt])
	return techniques

def attack_stage_class_stats(filter_by_observability=False, observation_type='signature') :
	stats = {}
	for sig, stage in attkstg_map.mapping.items() :
		if stage not in stats :
			stats[stage] = 1
		else:
			stats[stage] += 1

	for stage, sigs in attkstg_map.attack_stage_mapping.items() :
		for sig in sigs :
			if stage in stats :
				stats[stage] += 1
			else:
				stats[stage] = 1

	if not filter_by_observability :
		out_stats = stats
	else :
		if observation_type == 'signature' :
			out_stats = {k : stats[k] for k in SensorObservability.signature_based}
		else :
			print('MitreAttackInterface: Invalid Observation Type -- Ignoring')
			out_stats = stats

	return out_stats

def process_attack_patterns(techniques) :

	out_data = {}

	for technique in techniques :
		inner_dict = {}
		tech_name = technique['name'].replace('.', '')

		inner_dict['name'] = tech_name
		inner_dict['description'] = technique['description']

		phases = []
		for phase in technique['kill_chain_phases'] :
			phases.append(phase['phase_name'])
		inner_dict['kill_chain_phases'] = phases

		if 'x_mitre_data_sources' in technique :
			inner_dict['data_sources'] = technique['x_mitre_data_sources']
		inner_dict['platforms'] = technique['x_mitre_platforms']
		inner_dict['micro_attack_stage'] = ''

		out_data[tech_name] = inner_dict

	return out_data

def insert_to_database(attack_patterns) :
	client = MongoClient('localhost', 27017)
	db = client[database_name]
	collection = db[collection_name]

	for name, data in attack_patterns.items() :
		collection.insert_one(data)

'''
Retrieves all of the attack patterns from the mongo db
'''
def get_attack_patterns() :
	out_data = []
	client = MongoClient('localhost', 27017, username=api_username, password=api_password)
	db = client[database_name]
	collection = db[collection_name]
	attk_patterns = collection.find()

	for pattern in attk_patterns :
		out_data.append(pattern)

	return out_data

def get_patterns_df(observability_list=None, combine_priv_esc=True) :
	attk_patterns = get_attack_patterns()
	out_data = []

	for pattern in attk_patterns :
		attk_stage = MicroAttackStage[pattern['micro_attack_stage']]
		if not observability_list :

			if combine_priv_esc and (attk_stage == MicroAttackStage.USER_PRIV_ESC or attk_stage == MicroAttackStage.ROOT_PRIV_ESC) :
				out_data.append([MicroAttackStage.PRIV_ESC.value, pattern['description']])
			else :
				out_data.append([MicroAttackStage[pattern['micro_attack_stage']].value, pattern['description']])

		elif observability_list and MicroAttackStage[pattern['micro_attack_stage']] in observability_list :
			if combine_priv_esc and (
					attk_stage == MicroAttackStage.USER_PRIV_ESC or attk_stage == MicroAttackStage.ROOT_PRIV_ESC):
				out_data.append([MicroAttackStage.PRIV_ESC.value, pattern['description']])
			else:
				out_data.append([MicroAttackStage[pattern['micro_attack_stage']].value, pattern['description']])
		else :
			pass

	out_df = pd.DataFrame(out_data, columns=['Label', 'Sig'])

	return out_df

'''
A very specific method to convert the mitre attack information into a word cloud
'''
def convert_to_word_cloud(pattern_data, mode='micro') :
	out_words = {}
	overall_words = ''

	# Initializes all of the attack stages to develop the word cloud.
	if mode == 'micro':
		for stage in MicroAttackStage:
			out_words[stage] = ''
	elif mode == 'macro':
		for stage in MacroAttackStage:
			out_words[stage] = ''
	else:
		print('Mode not supported - Use micro or macro for example')
		return None, None

	for pattern in pattern_data :

		micro_stage = MicroAttackStage[pattern['micro_attack_stage']]
		if mode == 'micro' :
			key_stage = micro_stage
		elif mode == 'macro' :
			key_stage = MicroToMacroMapping.mapping[micro_stage]

		overall_words += ' ' + pattern['description']
		out_words[key_stage] += ' ' + pattern['description']

	return out_words, overall_words

def get_attack_word_cloud(mode='micro') :
	db_patterns = get_attack_patterns()
	attk_cloud, overall_cloud = convert_to_word_cloud(db_patterns, mode=mode)
	return attk_cloud, overall_cloud

# techniques = attack_patterns()
# #attack_patterns = process_attack_patterns(techniques)
# db_patterns = get_attack_patterns()
#stage_stats = attack_stage_class_stats(filter_by_observability=True)
#attk_cloud, overall_cloud = convert_to_word_cloud(db_patterns, mode='micro')

