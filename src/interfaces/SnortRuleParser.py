import glob
import json

import pandas as pd
from pymongo import MongoClient

from interfaces.DatabaseInterface import DatabaseInterface
from interfaces.snortrule import SnortRule

dir='../rules/'
database_loc = 'mongodb://localhost:27017'
database_name = 'snort_rules'
collection_name = 'snort3_2020'
suricata_collection = 'rules_2019'
snort_collection = 'snort3_2020'

def obtain_snort_rules(dir) :
	output_rules = []
	for filename in glob.glob(dir+'*.rules') :
		with open(filename, 'r') as f :
			for line_num, line in enumerate(f) :
				if line.startswith('alert') :
					output_rules.append(line)
				elif line.startswith('#alert') :
					output_rules.append(line[1:])
	return output_rules

def rule_to_dict(rules, to_json=True) :
	output = []
	for rule in rules :
		parsed = SnortRule(rule)
		parsed_dict = parsed.__dict__['classdict']

		if to_json :
			rule_json = json.dumps(parsed_dict)
			output.append(rule_json)
		else :
			output.append(parsed_dict)
	return output

def output_to_mongo(json_rules) :
	client = MongoClient('localhost', 27017)
	db = client[database_name]
	collection = db[collection_name]

	for rule in json_rules :
		collection.insert_one(rule)

def collect_signature_msg(as_list=False) :
	data_list = []
	db_int = DatabaseInterface(database_loc, database_name, suricata_collection)
	db = db_int.get_database()
	all_docs = db.find()

	for doc in all_docs:
		data_list.append([doc['msg']])
		
	if as_list :
		return data_list

	out_df = pd.DataFrame(data_list, columns=['Sig'])

	return out_df

def collect_snort_msg(as_list=False) :
	data_list = []
	db_int = DatabaseInterface(database_loc, database_name, snort_collection)
	db = db_int.get_database()
	all_docs = db.find()

	for doc in all_docs:
		data_list.append([doc['msg']])

	if as_list :
		return data_list

	out_df = pd.DataFrame(data_list, columns=['Sig'])

	return out_df

def alerts_by_sid(sort_mode=-1, output_df=False) :
	data_list = []
	db_int = DatabaseInterface(database_loc, database_name, suricata_collection)
	db = db_int.get_database()
	all_docs = db.find()

	for doc in all_docs:
		data_list.append(doc)

	if output_df:
		out_sigs = []

		for alert in data_list :
			out_sigs.append(alert['msg'])

		return pd.DataFrame(out_sigs, columns=['Sig'])

	return data_list

def get_recent_alerts(count, output_df=True) :
	sid_alerts = alerts_by_sid(output_df=output_df)
	return sid_alerts[:count]


#
#rule_list = obtain_snort_rules(dir)
# json_rules = rule_to_dict(rule_list, to_json=False)
# output_to_mongo(json_rules)

sid_alerts = alerts_by_sid()
# recent_alerts = sid_alerts[:1000]
# recent_sigs = []
# for alert in recent_alerts :
# 	recent_sigs.append(alert['msg'])

#recent_alerts = get_recent_alerts(1000)


