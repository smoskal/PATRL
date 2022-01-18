import sys
sys.path.append('../')

import pandas as pd
import sys
from AttackStagesCondensed import MicroAttackStage
from AttackStagesCondensed import MacroAttackStage
from AttackStagesCondensed import MicroToMacroMapping
from AttackStagesCondensed import MicroToMacroCondensedMapping
from AttackStagesCondensed import MicroAttackStageCondensed
import pickle
from fastai import *
from fastai.text import *
import re
import torch
from AIF_Mappings import RecentAlertsMapping
import random
import LearningUtils
from SignatureAttackStagePredictor import SignatureAttackStagePredictor
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import SnortRuleParser
import statistics

out_path = './'
if not torch.cuda.is_available():
	print("ERROR: CUDA not available.")
else :
	torch.cuda.set_device(0)
	torch.multiprocessing.freeze_support()

def iteration_test(data_set) :
	iter_sched = [3,5,7,10,12,15,20,30,50]
	
	uncert_iters = []
	top1_tracker = []
	
	for iters in tqdm(iter_sched) :
		sig_uncerts = []
		sig_preds = []
		for sig in data_set :
			k_preds, uncert_val = classifier.predict_mc_uncertainty(sig, num_iters=iters)
			top1 = MicroAttackStageCondensed(k_preds[0])
			sig_preds.append(top1)
			sig_uncerts.append(uncert_val)
		uncert_iters.append(sig_uncerts)
		top1_tracker.append(sig_preds)
		
	return uncert_iters, top1_tracker


if __name__ == '__main__' :

	main_data = SnortRuleParser.collect_signature_msg(as_list=True)
	classifier = SignatureAttackStagePredictor() 
	
	code_exe = ['ETPRO WEB_CLIENT Microsoft DirectShow (msvidctl.dll) MPEG-2 Memory Corruption (CVE-2008-0015)',
	'ET EXPLOIT Possible CVE-2016-2209 Symantec PowerPoint Parsing Buffer Overflow M2',
	'ETPRO WEB_CLIENT Microsoft DirectShow (msvidctl.dll) MPEG-2 Memory Corruption (CVE-2008-0015)',
	'ET TROJAN KeyLogger related to FindPOS CnC Beacon',
	'ET EXPLOIT MiCasaVerde VeraLite - Remote Code Execution Outbound (CVE-2016-6255)',
	'ETPRO EXPLOIT HP Data Protector Backup Client Service GET_FILE Buffer Overflow (UTF-16 Little-Endian )',
	'ET EXPLOIT Possible iOS Pegasus Safari Exploit (CVE-2016-4657)',
	'ETPRO EXPLOIT Adobe Shockwave Player Lnam Chunk offset 24 Processing Buffer Overflow Little Endian',
	'ET CURRENT_EVENTS [eSentire] Successful Personalized Phish 2018-06-15']
	
	code_exe_high_conf = ['ET WEB_CLIENT Hex Obfuscation of unescape % Encoding',
	'ET WEB_CLIENT Hex Obfuscation of unescape %u UTF-8 Encoding',
	'ET WEB_SPECIFIC_APPS Pligg check_url.php url parameter SQL Injection',
	'ETPRO EXPLOIT Advantech WebAccess SQL Injection',
	'ET WEB_SPECIFIC_APPS Jetik.net ESA sayfalar.php KayitNo Parameter SQL Injection',
	'ET WEB_SERVER Exploit Suspected PHP Injection Attack (cmd=)'
	'ET WEB_SERVER Exploit Suspected PHP Injection Attack (cmd=)',
	'ET ACTIVEX HP Openview NNM ActiveX AddGroup method Memory corruption Attempt',
	'ET EXPLOIT Possible CVE-2014-3704 Drupal SQLi attempt URLENCODE 21',
	'ET WEB_SPECIFIC_APPS PozScripts Classified Auctions id parameter SQL Injection',
	'ET WEB_SPECIFIC_APPS joomla com_djcatalog component UPDATE SET SQL Injection',
	'ET WEB_CLIENT Hex Obfuscation of replace Javascript Function % Encoding'
	]
	
	info_disc_high = [
	'ET WEB_SERVER Weevely PHP backdoor detected (pcntl_exec() function used)',
	'ETPRO POLICY UltraVnc Session Outbound',
	'GPL FTP LIST buffer overflow attempt',
	'GPL FTP DELE overflow attempt',
	'GPL FTP MDTM overflow attempt',
	'GPL FTP PASS overflow attempt',
	'GPL DNS named version attempt',
	'GPL IMAP status overflow attempt'
	]
	
	info_disc_low = [
	'ET POLICY MOBILE Apple device leaking UDID from SpringBoard',
	'ETPRO CURRENT_EVENTS Weebly Phishing Landing Observed Nov 10',
	'ETPRO WEB_CLIENT MS Edge Out-of-Bounds Vuln (CVE-2017-8618)',
	'ETPRO WEB_CLIENT Xpdf Splash DrawImage Integer Overflow',
	'GPL ICMP Datagram Conversion Error undefined code',
	'GPL SQL og.begin_load ordered gname buffer overflow attempt'
	]
	
	c2_high = [
	'ET TROJAN Observed Buran Ransomware UA (GHOST)',
	'ET TROJAN Backdoor family PCRat/Gh0st CnC traffic (OUTBOUND) 16',
	'ETPRO TROJAN Observed Malicious SSL Cert (Meterpreter CnC)',
	'ETPRO TROJAN Trojan.Win32.Agent.cqr Checkin',
	'ETPRO TROJAN Observed Malicious SSL Cert (Zeus Panda)',
	'ET TROJAN Win32/Aibatook checkin 2',
	'ET TROJAN Observed CDC Ransomware User-Agent',
	'ETPRO TROJAN Observed Meterpreter Communications over TCP DNS',
	'ETPRO TROJAN Win32/Necurs Checkin 2',
	'ETPRO TROJAN Win32/Tiptuf.A Checkin',
	]
	
	c2_low = [
	'ETPRO MOBILE_MALWARE Trojan-Banker.AndroidOS.Asacub.a Checkin 365',
	'ETPRO CURRENT_EVENTS Malicious Redirect Leading to EK Aug 21 2015 T4',
	'ET TROJAN Torpig Reporting User Activity (x25)',
	'ETPRO MOBILE_MALWARE Trojan-Banker.AndroidOS.Asacub.a Checkin 146',
	'ET TROJAN OSX/Flashback.K first execution checkin',
	'ETPRO TROJAN PoisonIvy Keepalive to CnC (youtube.swf actor) 4'
	'ETPRO CURRENT_EVENTS Observed Malicious SSL Cert (MalDoc DL 2019-09-19 2)',
	'ETPRO TROJAN Trojan-Ransomware Radamant Fetch Wallets',
	'ETPRO TROJAN Win32/Remcos RAT Checkin 117'
	]
	
	
	priv_esc_high = [
	'ET ATTACK_RESPONSE Frequent HTTP 401 Unauthorized - Possible Brute Force Attack',
	'ET TELNET busybox MEMES Hackers - Possible Brute Force Attack',
	'ET SCAN Rapid POP3 Connections - Possible Brute Force Attack',
	'ET SCAN Rapid IMAP Connections - Possible Brute Force Attack',
	'ET SCAN Tomcat Auth Brute Force attempt (admin)',
	'ET SCAN LibSSH Based Frequent SSH Connections Likely BruteForce Attack',
	'ETPRO CURRENT_EVENTS Successful Google Docs Phish 2018-10-29',
	'ET SCAN Rapid POP3S Connections - Possible Brute Force Attack',
	'ETPRO EXPLOIT Possible Novidade EK Attempting Intranet Router Compromise M9 (Bruteforce)',
	'ETPRO CURRENT_EVENTS Successful Bank of America Phish 2018-08-23',
	'ET SCAN MYSQL 4.1 brute force root login attempt',
	]
	
	priv_esc_low = [
	'ET CURRENT_EVENTS Evil Redirector Leading to EK Sep 19 2016 (EItest Inject) M2',
	'ET SCAN Tomcat Auth Brute Force attempt (tomcat)',
	'ET MALWARE TargetNetworks.net Spyware Reporting (tn)',
	'ET EXPLOIT TCP Reset from MS Exchange after chunked data, probably crashed it (MS05-021)',
	'ETPRO CURRENT_EVENTS Possible Successful Fedex Phish Jul 28 2015',
	'ETPRO CURRENT_EVENTS Successful FNB First National Bank Phish 2019-09-10',
	'ETPRO MOBILE_MALWARE Trojan-Spy.AndroidOS.SmsThief.dj Exfiltration of SMS via SMTP',
	'ETPRO CURRENT_EVENTS Successful Bank of America Phish 2019-03-19',
	'ET WEB_SPECIFIC_APPS phpBB3 Brute-Force reg attempt (Bad pf_XXXXX)'

	]

	cptc_data_cond, class_labels, class_map = LearningUtils.signature_to_data_frame_condensed(transform_labels=False, output_dict=True)
	ccdc_data_cond = LearningUtils.transform_dataset_condensed(RecentAlertsMapping().ccdc_combined, class_map, output_dict=True)
	
	main_data = []
	main_data.extend(cptc_data_cond)
	main_data.extend(ccdc_data_cond)
	
	datasets = [info_disc_high, info_disc_low, c2_high, c2_low, priv_esc_high,priv_esc_low]
	data_set_names = ['info_disc_high', 'info_disc_low', 'c2_high', 'c2_low', 'priv_esc_high', 'priv_esc_low']
	
	uncerts = {}
	top1_data = {}
	
	# for i, data in enumerate(datasets) :
		# uncert_iters, top1_tracker = iteration_test(data) 
		# uncerts[data_set_names[i]] = uncert_iters
		# top1_data[data_set_names[i]] = top1_tracker
		
	testing_preds = []
	for sig in tqdm(main_data) :
		k_preds, uncert_val = classifier.predict_mc_uncertainty(sig[1], num_iters=10)
		top1 = MicroAttackStageCondensed(k_preds[0])
		is_correct = False
		if sig[0] == k_preds[0] : is_correct = True
		
		topk = []
		for stg in k_preds : topk.append(MicroAttackStageCondensed(stg))
		
		temp_data = [sig[1], uncert_val, is_correct, topk]
		testing_preds.append(temp_data)
		
	sorted_preds = sorted(testing_preds, key=lambda x:x[1], reverse=False)
	top10_testing = sorted_preds[0:10]
	
	sorted_preds = sorted(testing_preds, key=lambda x:x[1], reverse=True)
	bottom10_testing = sorted_preds[0:10]
	
	uncert_iter_top, top1_top = iteration_test(top10_testing)
	uncert_iter_bottom, top1_bottom = iteration_test(bottom10_testing)
	
	
	# iter_sched = [3,5,7,10,12,15,20,30,50]
	
	# uncert_iters = []
	# top1_tracker = []
	
	# for iters in tqdm(iter_sched) :
		# sig_uncerts = []
		# sig_preds = []
		# for sig in code_exe_high_conf :
			# k_preds, uncert_val = classifier.predict_mc_uncertainty(sig, num_iters=iters)
			# top1 = MicroAttackStageCondensed(k_preds[0])
			# sig_preds.append(top1)
			# sig_uncerts.append(uncert_val)
		# uncert_iters.append(sig_uncerts)
		# top1_tracker.append(sig_preds)
	
			