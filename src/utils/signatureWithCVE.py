'''
This is to retrieve all of the signatures that contain CVE information
'''

import re

import pandas as pd

from aif.AttackStages import MicroAttackStage
from aif.AttackStages import SignatureMapping as attkstg_map
from interfaces.DatabaseInterface import DatabaseInterface

cve_regex = 'CVE-\d{4}-\d{4,7}'
database_loc = 'mongodb://localhost:27017'
database_name = 'cvedb'
collection_name = 'cves'

cve_keywords = {
	"\"enumerate\"": MicroAttackStage.INFO_DISC,
	"\"local pathnames\"" : MicroAttackStage.INFO_DISC,
	"\"remote attackers to obtain sensitive information\"" : MicroAttackStage.DATA_EXFILTRATION,
	"\"VPN\"" : MicroAttackStage.REMOTE_SERVICE_EXP,
	"\"RDP\"" : MicroAttackStage.REMOTE_SERVICE_EXP,
	"\"elevate privileges\"" : MicroAttackStage.PRIV_ESC,
	"\"admin privileges\"" : MicroAttackStage.ROOT_PRIV_ESC,
	"\"superuser privileges\"" : MicroAttackStage.ROOT_PRIV_ESC,
	"\"gain root privileges\"" : MicroAttackStage.ROOT_PRIV_ESC,
	"\"brute-force\"" : MicroAttackStage.BRUTE_FORCE_CREDS,
	"\"sniffing\"" : MicroAttackStage.NETWORK_SNIFFING,
	"\"denial of service vulnerability\"" : MicroAttackStage.END_POINT_DOS,
	"\"denial of service\" \"network\"" : MicroAttackStage.NETWORK_DOS,
	"\"denial of service\" \"system\"" : MicroAttackStage.END_POINT_DOS,
	"\"resource consumption\"" : MicroAttackStage.END_POINT_DOS
}

def retrieve_cves() :
	mapping = attkstg_map().get_signature_mapping()
	cve_mapping = {}
	cve_list = []
	cve_to_stage = {}

	for sig, stg in mapping.items() :
		result = re.search(cve_regex, sig)

		if result is not None :
			cve_mapping[sig] = (result.group(0), stg)
			if result.group(0) not in cve_list :
				cve_list.append(result.group(0))
			if result.group(0) not in cve_to_stage :
				cve_to_stage[result.group(0)] = stg

	return cve_mapping, cve_list, cve_to_stage

def query_cve_db(query) :
	db_int = DatabaseInterface(database_loc, database_name, collection_name)
	db = db_int.get_database()
	output = []
	docs = db.find(query)

	for val in docs :
		output.append(val)

	return output

'''
Extracts all the summaries from the query using query_cve_db
'''
def summaries_from_query(query_output) :
	output = {}
	for cve in query_output :
		output[cve['id']] = cve['summary']

	return output

def get_cve_data(cve_list) :
	db_int = DatabaseInterface(database_loc, database_name, collection_name)
	db = db_int.get_database()

	output = {}

	for cve in cve_list :
		docs = db.find({'id': cve})

		for val in docs :
			output[cve] = val['summary']

	return output

def signature_to_cve(cve_mapping, cve_summaries) :
	output = {}

	for sig, mapping in cve_mapping.items() :
		output[sig] = cve_summaries[mapping[0]]

	return output

def convert_to_df(cve_summaries, cve_to_stage) :
	data_list = []
	for cve, summary in cve_summaries.items() :
		data_list.append([cve_to_stage[cve].value, summary])

	out_df = pd.DataFrame(data_list, columns=['Label', 'Sig'])
	return out_df

def get_all_cve_summaries() :
	data_list = []
	db_int = DatabaseInterface(database_loc, database_name, collection_name)
	db = db_int.get_database()
	all_docs = db.find()

	for doc in all_docs :
		data_list.append([doc['summary']])

	out_df = pd.DataFrame(data_list, columns=['Sig'])

	return out_df

def cve_summary_by_keywords() :
	data_list = []

	for keyword, stg in cve_keywords.items() :
		query_output = query_cve_db({'$text': {'$search': keyword}})
		query_summary = summaries_from_query(query_output)

		for cve, summary in query_summary.items() :
			data_list.append([stg.value, summary])

	out_df = pd.DataFrame(data_list, columns=['Label', 'Sig'])

	return out_df

cve_mapping, cve_list, cve_to_stage = retrieve_cves()
cve_summaries = get_cve_data(cve_list)
cve_df = convert_to_df(cve_summaries, cve_to_stage)
# sig_summary = signature_to_cve(cve_mapping, cve_summaries)
# query_output = query_cve_db({'$text':{'$search' : "\"sniffing\" "}})
# query_summary = summaries_from_query(query_output)

#cve_attkstg_labels = cve_summary_by_keywords()

