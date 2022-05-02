from enum import Enum

class MicroAttackStage(Enum) :
	INIT = 0

	TARGET_IDEN = 1
	SURFING = 2
	SOCIAL_ENGINEERING = 3
	HOST_DISC = 4
	SERVICE_DISC = 5
	VULN_DISC = 6
	INFO_DISC = 7

	USER_PRIV_ESC = 10
	ROOT_PRIV_ESC = 11
	NETWORK_SNIFFING = 12
	BRUTE_FORCE_CREDS = 13
	ACCT_MANIP = 14
	TRUSTED_ORG_EXP = 15
	PUBLIC_APP_EXP = 16
	REMOTE_SERVICE_EXP = 17
	SPEARPHISHING = 18
	SERVICE_SPECIFIC = 19
	DEFENSE_EVASION = 20
	COMMAND_AND_CONTROL = 21
	LATURAL_MOVEMENT = 22
	ARBITRARY_CODE_EXE = 23
	PRIV_ESC = 99

	END_POINT_DOS = 100
	NETWORK_DOS = 101
	SERVICE_STOP = 102
	RESOURCE_HIJACKING = 103
	DATA_DESTRUCTION = 104
	CONTENT_WIPE = 105
	DATA_ENCRYPTION = 106
	DEFACEMENT = 107
	DATA_MANIPULATION = 108
	DATA_EXFILTRATION = 109
	DATA_DELIVERY = 110
	PHISHING = 111

	NON_MALICIOUS = 999

	#ZD_PRIV_ESC = 1000
	#ZD_TARGETED_EXP = 1001
	#ZD_ENSURE_ACCESS = 1002
	
class MicroAttackStageCondensed(Enum) :
	INIT = 0
	NETWORK_DISC = 1
	VULN_DISC = 2
	INFO_DISC = 3
	
	NETWORK_SNIFFING = 4
	CREDENTIAL_ACCESS = 5
	EXTERNAL_SERVICE_EXP = 6
	SPEARPHISHING = 7
	
	COMMAND_AND_CONTROL = 8
	LATURAL_MOVEMENT = 9
	CODE_EXECUTION = 10
	PRIV_ESC = 11

	DENIAL_OF_SERVICE = 12
	DATA_EXFILTRATION = 13
	DATA_DELIVERY = 14
	PHISHING = 15
	NON_MALICIOUS = 16


class MacroAttackStage(Enum) :
	NONE = 0
	PASSIVE_RECON = 1
	ACTIVE_RECON = 2
	PRIVLEDGE_ESC = 3
	ENSURE_ACCESS = 4
	TARGETED_EXP = 5
	ZERO_DAY = 6
	DISRUPT = 7
	DISTROY = 8
	DISTORT = 9
	DISCLOSURE = 10
	DELIVERY = 11
	
class MitreTactics(Enum) :
	NONE = 0
	INIT_ACCESS = 1
	EXECUTION = 2
	PERSISTENCE = 3
	PRIV_ESC = 4
	DEFENSE_EVASION = 5
	CREDENTIAL_ACCESS = 6
	DISCOVERY = 7
	LATERAL_MOVEMENT = 8
	COMMAND_AND_CONTROL = 9
	EXFILTRATION = 10
	IMPACT = 11
	COLLECTION = 12

class MicroToMitreTactics() :
	def __init__(self) :
		pass
		
	mapping = {
		MicroAttackStage.TARGET_IDEN : MitreTactics.DISCOVERY,
			   MicroAttackStage.SURFING: MitreTactics.DISCOVERY,
			   MicroAttackStage.SOCIAL_ENGINEERING: MitreTactics.DISCOVERY,
			   MicroAttackStage.HOST_DISC: MitreTactics.DISCOVERY,
			   MicroAttackStage.SERVICE_DISC: MitreTactics.DISCOVERY,
			   MicroAttackStage.VULN_DISC: MitreTactics.DISCOVERY,
			   MicroAttackStage.INFO_DISC: MitreTactics.COLLECTION,
			   MicroAttackStage.USER_PRIV_ESC: MitreTactics.PRIV_ESC,
			   MicroAttackStage.ROOT_PRIV_ESC: MitreTactics.PRIV_ESC,
			   MicroAttackStage.NETWORK_SNIFFING: MitreTactics.CREDENTIAL_ACCESS,
			   MicroAttackStage.BRUTE_FORCE_CREDS: MitreTactics.CREDENTIAL_ACCESS,
			   MicroAttackStage.ACCT_MANIP: MitreTactics.CREDENTIAL_ACCESS, #Maybe??
			   MicroAttackStage.TRUSTED_ORG_EXP: MitreTactics.INIT_ACCESS,
			   MicroAttackStage.PUBLIC_APP_EXP: MitreTactics.INIT_ACCESS,
			   MicroAttackStage.REMOTE_SERVICE_EXP: MitreTactics.INIT_ACCESS,
			   MicroAttackStage.SPEARPHISHING: MitreTactics.INIT_ACCESS,
			   MicroAttackStage.SERVICE_SPECIFIC: MitreTactics.EXECUTION,
			   MicroAttackStage.ARBITRARY_CODE_EXE: MitreTactics.EXECUTION,
			   MicroAttackStage.DEFENSE_EVASION: MitreTactics.DEFENSE_EVASION,
			   MicroAttackStage.COMMAND_AND_CONTROL: MitreTactics.COMMAND_AND_CONTROL,
			   MicroAttackStage.LATURAL_MOVEMENT: MitreTactics.LATERAL_MOVEMENT,
			   MicroAttackStage.END_POINT_DOS: MitreTactics.IMPACT,
			   MicroAttackStage.NETWORK_DOS: MitreTactics.IMPACT,
			   MicroAttackStage.SERVICE_STOP: MitreTactics.IMPACT,
			   MicroAttackStage.RESOURCE_HIJACKING: MitreTactics.IMPACT,
			   MicroAttackStage.DATA_DESTRUCTION: MitreTactics.IMPACT,
			   MicroAttackStage.CONTENT_WIPE: MitreTactics.IMPACT,
			   MicroAttackStage.DATA_ENCRYPTION: MitreTactics.IMPACT,
			   MicroAttackStage.DEFACEMENT: MitreTactics.IMPACT,
			   MicroAttackStage.DATA_MANIPULATION: MitreTactics.IMPACT,
			   MicroAttackStage.DATA_EXFILTRATION: MitreTactics.EXFILTRATION,
			   MicroAttackStage.DATA_DELIVERY: MitreTactics.COMMAND_AND_CONTROL,
			   MicroAttackStage.NON_MALICIOUS: MitreTactics.NONE,
			   MicroAttackStage.PRIV_ESC : MitreTactics.PRIV_ESC,
			   MicroAttackStage.INIT : MitreTactics.NONE
	}
		

class MicroToMacroMapping() :
	def __init__(self) :
		pass

	mapping = {MicroAttackStage.TARGET_IDEN : MacroAttackStage.PASSIVE_RECON,
			   MicroAttackStage.SURFING: MacroAttackStage.PASSIVE_RECON,
			   MicroAttackStage.SOCIAL_ENGINEERING: MacroAttackStage.PASSIVE_RECON,
			   MicroAttackStage.HOST_DISC: MacroAttackStage.ACTIVE_RECON,
			   MicroAttackStage.SERVICE_DISC: MacroAttackStage.ACTIVE_RECON,
			   MicroAttackStage.VULN_DISC: MacroAttackStage.ACTIVE_RECON,
			   MicroAttackStage.INFO_DISC: MacroAttackStage.ACTIVE_RECON,
			   MicroAttackStage.USER_PRIV_ESC: MacroAttackStage.PRIVLEDGE_ESC,
			   MicroAttackStage.ROOT_PRIV_ESC: MacroAttackStage.PRIVLEDGE_ESC,
			   MicroAttackStage.NETWORK_SNIFFING: MacroAttackStage.PRIVLEDGE_ESC,
			   MicroAttackStage.BRUTE_FORCE_CREDS: MacroAttackStage.PRIVLEDGE_ESC,
			   MicroAttackStage.ACCT_MANIP: MacroAttackStage.PRIVLEDGE_ESC,
			   MicroAttackStage.TRUSTED_ORG_EXP: MacroAttackStage.TARGETED_EXP,
			   MicroAttackStage.PUBLIC_APP_EXP: MacroAttackStage.TARGETED_EXP,
			   MicroAttackStage.REMOTE_SERVICE_EXP: MacroAttackStage.TARGETED_EXP,
			   MicroAttackStage.SPEARPHISHING: MacroAttackStage.TARGETED_EXP,
			   MicroAttackStage.SERVICE_SPECIFIC: MacroAttackStage.TARGETED_EXP,
			   MicroAttackStage.ARBITRARY_CODE_EXE: MacroAttackStage.TARGETED_EXP,
			   MicroAttackStage.DEFENSE_EVASION: MacroAttackStage.ENSURE_ACCESS,
			   MicroAttackStage.COMMAND_AND_CONTROL: MacroAttackStage.ENSURE_ACCESS,
			   MicroAttackStage.LATURAL_MOVEMENT: MacroAttackStage.ENSURE_ACCESS,
			   MicroAttackStage.END_POINT_DOS: MacroAttackStage.DISRUPT,
			   MicroAttackStage.NETWORK_DOS: MacroAttackStage.DISRUPT,
			   MicroAttackStage.SERVICE_STOP: MacroAttackStage.DISRUPT,
			   MicroAttackStage.RESOURCE_HIJACKING: MacroAttackStage.DISRUPT,
			   MicroAttackStage.DATA_DESTRUCTION: MacroAttackStage.DISTROY,
			   MicroAttackStage.CONTENT_WIPE: MacroAttackStage.DISTROY,
			   MicroAttackStage.DATA_ENCRYPTION: MacroAttackStage.DISTORT,
			   MicroAttackStage.DEFACEMENT: MacroAttackStage.DISTORT,
			   MicroAttackStage.DATA_MANIPULATION: MacroAttackStage.DISTORT,
			   MicroAttackStage.DATA_EXFILTRATION: MacroAttackStage.DISCLOSURE,
			   MicroAttackStage.DATA_DELIVERY: MacroAttackStage.DELIVERY,
			   MicroAttackStage.NON_MALICIOUS: MacroAttackStage.NONE,
			   MicroAttackStage.PRIV_ESC : MacroAttackStage.PRIVLEDGE_ESC,
			   MicroAttackStage.INIT : MacroAttackStage.NONE
			   }
			   
class MicroToMacroCondensedMapping() :
	def __init__(self) :
		pass

	mapping = {MicroAttackStage.TARGET_IDEN : MicroAttackStageCondensed.NETWORK_DISC,
			   MicroAttackStage.SURFING: MicroAttackStageCondensed.NETWORK_DISC,
			   MicroAttackStage.SOCIAL_ENGINEERING: MicroAttackStageCondensed.NETWORK_DISC,
			   MicroAttackStage.HOST_DISC: MicroAttackStageCondensed.NETWORK_DISC,
			   MicroAttackStage.SERVICE_DISC: MicroAttackStageCondensed.NETWORK_DISC,
			   MicroAttackStage.VULN_DISC: MicroAttackStageCondensed.VULN_DISC,
			   MicroAttackStage.INFO_DISC: MicroAttackStageCondensed.INFO_DISC,
			   MicroAttackStage.USER_PRIV_ESC: MicroAttackStageCondensed.PRIV_ESC,
			   MicroAttackStage.ROOT_PRIV_ESC: MicroAttackStageCondensed.PRIV_ESC,
			   MicroAttackStage.NETWORK_SNIFFING: MicroAttackStageCondensed.PRIV_ESC,
			   MicroAttackStage.BRUTE_FORCE_CREDS: MicroAttackStageCondensed.CREDENTIAL_ACCESS,
			   MicroAttackStage.ACCT_MANIP: MicroAttackStageCondensed.PRIV_ESC,
			   MicroAttackStage.TRUSTED_ORG_EXP: MicroAttackStageCondensed.CREDENTIAL_ACCESS,
			   MicroAttackStage.PUBLIC_APP_EXP: MicroAttackStageCondensed.CREDENTIAL_ACCESS,
			   MicroAttackStage.REMOTE_SERVICE_EXP: MicroAttackStageCondensed.CREDENTIAL_ACCESS,
			   MicroAttackStage.SPEARPHISHING: MicroAttackStageCondensed.CREDENTIAL_ACCESS,
			   MicroAttackStage.SERVICE_SPECIFIC: MicroAttackStageCondensed.CODE_EXECUTION,
			   MicroAttackStage.ARBITRARY_CODE_EXE: MicroAttackStageCondensed.CODE_EXECUTION,
			   MicroAttackStage.COMMAND_AND_CONTROL: MicroAttackStageCondensed.COMMAND_AND_CONTROL,
			   MicroAttackStage.END_POINT_DOS: MicroAttackStageCondensed.DENIAL_OF_SERVICE,
			   MicroAttackStage.NETWORK_DOS: MicroAttackStageCondensed.DENIAL_OF_SERVICE,
			   MicroAttackStage.SERVICE_STOP: MicroAttackStageCondensed.DENIAL_OF_SERVICE,
			   MicroAttackStage.RESOURCE_HIJACKING: MicroAttackStageCondensed.DENIAL_OF_SERVICE,
			   MicroAttackStage.DATA_DESTRUCTION: MicroAttackStageCondensed.DENIAL_OF_SERVICE,
			   MicroAttackStage.CONTENT_WIPE: MicroAttackStageCondensed.DENIAL_OF_SERVICE,
			   MicroAttackStage.DATA_ENCRYPTION: MicroAttackStageCondensed.DENIAL_OF_SERVICE,
			   MicroAttackStage.DEFACEMENT: MicroAttackStageCondensed.DENIAL_OF_SERVICE,
			   MicroAttackStage.DATA_MANIPULATION: MicroAttackStageCondensed.DATA_EXFILTRATION,
			   MicroAttackStage.DATA_EXFILTRATION: MicroAttackStageCondensed.DATA_EXFILTRATION,
			   MicroAttackStage.DATA_DELIVERY: MicroAttackStageCondensed.DATA_DELIVERY,
			   MicroAttackStage.NON_MALICIOUS: MicroAttackStageCondensed.NON_MALICIOUS,
			   MicroAttackStage.PRIV_ESC : MicroAttackStageCondensed.PRIV_ESC,
			   MicroAttackStage.PHISHING : MicroAttackStageCondensed.PHISHING,
			   MicroAttackStage.INIT : MicroAttackStageCondensed.INIT
			   }

class SignatureMapping :

	def __init__(self) :
		pass

	mapping = {'ET ATTACK_RESPONSE Net User Command Response': MicroAttackStage.INFO_DISC,#
			   'ET ATTACK_RESPONSE Possible /etc/passwd via HTTP (linux style)': MicroAttackStage.DATA_EXFILTRATION,#
			   'ET ATTACK_RESPONSE Possible BeEF HTTP Headers Inbound': MicroAttackStage.COMMAND_AND_CONTROL,#
			   'ET ATTACK_RESPONSE python shell spawn attempt': MicroAttackStage.COMMAND_AND_CONTROL, #
			   'ET CURRENT_EVENTS Likely Linux/Xorddos DDoS Attack Participation (gggatat456.com)': MicroAttackStage.END_POINT_DOS,
			   'ET CURRENT_EVENTS Likely Linux/Xorddos DDoS Attack Participation (xxxatat456.com)': MicroAttackStage.END_POINT_DOS,
			   'ET CURRENT_EVENTS Malformed HeartBeat Request': MicroAttackStage.VULN_DISC,
			   'ET CURRENT_EVENTS Possible TLS HeartBleed Unencrypted Request Method 3 (Inbound to Common SSL Port)': MicroAttackStage.DATA_EXFILTRATION,
			   'ET CURRENT_EVENTS Possible ZyXELs ZynOS Configuration Download Attempt (Contains Passwords)': MicroAttackStage.DATA_EXFILTRATION,
			   'ET CURRENT_EVENTS QNAP Shellshock CVE-2014-6271': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'ET CURRENT_EVENTS Terse alphanumeric executable downloader high likelihood of being hostile': MicroAttackStage.DATA_DELIVERY, #
			   'ET DNS Query for .su TLD (Soviet Union) Often Malware Related': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET DNS Query to a .tk domain - Likely Hostile': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET DOS Microsoft Remote Desktop (RDP) Syn then Reset 30 Second DoS Attempt': MicroAttackStage.NETWORK_DOS,
			   'ET DOS Possible NTP DDoS Inbound Frequent Un-Authed MON_LIST Requests IMPL 0x03': MicroAttackStage.NETWORK_DOS,
			   'ET DOS Possible SSDP Amplification Scan in Progress': MicroAttackStage.NETWORK_DOS,
			   'ET EXPLOIT Possible GoldenPac Priv Esc in-use': MicroAttackStage.USER_PRIV_ESC,
			   'ET EXPLOIT Possible Postfix CVE-2014-6271 attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'ET EXPLOIT Possible Pure-FTPd CVE-2014-6271 attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'ET EXPLOIT Possible SpamAssassin Milter Plugin Remote Arbitrary Command Injection Attempt': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET EXPLOIT REDIS Attempted SSH Key Upload': MicroAttackStage.USER_PRIV_ESC,
			   'ET FTP Suspicious Quotation Mark Usage in FTP Username': MicroAttackStage.ACCT_MANIP,
			   'ET INFO Executable Download from dotted-quad Host': MicroAttackStage.COMMAND_AND_CONTROL, #Do we need a malware injection stage?
			   'ET INFO Possible Windows executable sent when remote host claims to send a Text File': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET INFO WinHttp AutoProxy Request wpad.dat Possible BadTunnel': MicroAttackStage.DATA_EXFILTRATION, #This is more "man in the middle"
			   'ET MOBILE_MALWARE Android/Code4hk.A Checkin': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET P2P TOR 1.0 Outbound Circuit Traffic': MicroAttackStage.DEFENSE_EVASION,
			   'ET POLICY DNS Update From External net': MicroAttackStage.DATA_MANIPULATION,
			   'ET POLICY Executable and linking format (ELF) file download': MicroAttackStage.COMMAND_AND_CONTROL, #will need to be checked out
			   'ET POLICY Executable and linking format (ELF) file download Over HTTP': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET POLICY Http Client Body contains pass= in cleartext': MicroAttackStage.PRIV_ESC,  #Once again, check this out.
			   'ET POLICY Incoming Basic Auth Base64 HTTP Password detected unencrypted': MicroAttackStage.USER_PRIV_ESC ,
			   'ET POLICY MS Remote Desktop Administrator Login Request': MicroAttackStage.ROOT_PRIV_ESC,
			   'ET POLICY MS Terminal Server Root login': MicroAttackStage.ROOT_PRIV_ESC,
			   'ET POLICY Outgoing Basic Auth Base64 HTTP Password detected unencrypted': MicroAttackStage.USER_PRIV_ESC, #outgoing vs. incoming?
			   'ET POLICY PE EXE or DLL Windows file download HTTP': MicroAttackStage.DATA_DELIVERY,
			   'ET POLICY Python-urllib/ Suspicious User Agent': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET POLICY RDP connection confirm': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET POLICY Suspicious inbound to MSSQL port 1433': MicroAttackStage.VULN_DISC,
			   'ET POLICY Suspicious inbound to Oracle SQL port 1521': MicroAttackStage.VULN_DISC,
			   'ET POLICY Suspicious inbound to PostgreSQL port 5432': MicroAttackStage.VULN_DISC,
			   'ET POLICY Suspicious inbound to mSQL port 4333': MicroAttackStage.VULN_DISC,
			   'ET POLICY Suspicious inbound to mySQL port 3306': MicroAttackStage.VULN_DISC,
			   'ET POLICY curl User-Agent Outbound': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET SCAN Apache mod_deflate DoS via many multiple byte Range values': MicroAttackStage.NETWORK_DOS,
			   'ET SCAN Behavioral Unusual Port 135 traffic Potential Scan or Infection': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Behavioral Unusual Port 139 traffic Potential Scan or Infection': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Behavioral Unusual Port 1433 traffic Potential Scan or Infection': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Behavioral Unusual Port 1434 traffic Potential Scan or Infection': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Behavioral Unusual Port 445 traffic Potential Scan or Infection': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Behavioral Unusually fast Terminal Server Traffic Potential Scan or Infection (Inbound)': MicroAttackStage.HOST_DISC,
			   'ET SCAN DEBUG Method Request with Command': MicroAttackStage.INFO_DISC,
			   'ET SCAN DirBuster Scan in Progress': MicroAttackStage.VULN_DISC, #This is typically trying to see what directories are avabilable
			   'ET SCAN DirBuster Web App Scan in Progress': MicroAttackStage.VULN_DISC,
			   'ET SCAN Hydra User-Agent': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'ET SCAN LibSSH Based Frequent SSH Connections Likely BruteForce Attack': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'ET SCAN Multiple MySQL Login Failures Possible Brute Force Attempt': MicroAttackStage.BRUTE_FORCE_CREDS ,
			   'ET SCAN NMAP OS Detection Probe': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN NMAP SIP Version Detect OPTIONS Scan': MicroAttackStage.VULN_DISC,
			   'ET SCAN Nessus FTP Scan detected (ftp_anonymous.nasl)': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Nessus FTP Scan detected (ftp_writeable_directories.nasl)': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Nessus User Agent': MicroAttackStage.VULN_DISC,
			   'ET SCAN Nikto Web App Scan in Progress': MicroAttackStage.VULN_DISC,
			   'ET SCAN Nmap NSE Heartbleed Request': MicroAttackStage.VULN_DISC,
			   'ET SCAN Nmap Scripting Engine User-Agent Detected (Nmap Scripting Engine)': MicroAttackStage.HOST_DISC,
			   'ET SCAN OpenVAS User-Agent Inbound': MicroAttackStage.VULN_DISC,
			   'ET SCAN Possible Nmap User-Agent Observed': MicroAttackStage.HOST_DISC,
			   'ET SCAN Potential FTP Brute-Force attempt response': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'ET SCAN Potential SSH Scan': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Potential VNC Scan 5800-5820': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Potential VNC Scan 5900-5920': MicroAttackStage.SERVICE_DISC,
			   'ET SCAN Rapid IMAP Connections - Possible Brute Force Attack': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'ET SCAN Rapid IMAPS Connections - Possible Brute Force Attack': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'ET SCAN Rapid POP3 Connections - Possible Brute Force Attack': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'ET SCAN Rapid POP3S Connections - Possible Brute Force Attack': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'ET SCAN Redis SSH Key Overwrite Probing': MicroAttackStage.USER_PRIV_ESC,
			   'ET SNMP Attempted UDP Access Attempt to Cisco IOS 12.1 Hidden Read/Write Community String ILMI': MicroAttackStage.REMOTE_SERVICE_EXP,
			   'ET SNMP Samsung Printer SNMP Hardcode RW Community String': MicroAttackStage.REMOTE_SERVICE_EXP,
			   'ET SNMP missing community string attempt 1': MicroAttackStage.REMOTE_SERVICE_EXP,
			   'ET TROJAN ATTACKER IRCBot - PRIVMSG Response - Directory Listing': MicroAttackStage.DATA_EXFILTRATION,
			   'ET TROJAN ATTACKER IRCBot - PRIVMSG Response - Directory Listing *nix': MicroAttackStage.DATA_EXFILTRATION,
			   'ET TROJAN ATTACKER IRCBot - PRIVMSG Response - ipconfig command output': MicroAttackStage.DATA_EXFILTRATION,
			   'ET TROJAN ATTACKER IRCBot - PRIVMSG Response - net command output': MicroAttackStage.DATA_EXFILTRATION,
			   'ET TROJAN ATTACKER IRCBot - The command completed successfully - PRIVMSG Response': MicroAttackStage.DATA_EXFILTRATION,
			   'ET TROJAN DDoS.XOR Checkin': MicroAttackStage.NETWORK_DOS, #may also be network sniffing
			   'ET TROJAN NgrBot IRC CnC Channel Join': MicroAttackStage.COMMAND_AND_CONTROL, ##
			   'ET TROJAN Windows WMIC COMPUTERSYSTEM get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows WMIC NETLOGIN get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows WMIC NIC get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows WMIC OS get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows WMIC PROCESS get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows WMIC SERVER get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows WMIC SERVICE get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows WMIC SHARE get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows WMIC STARTUP get Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows dir Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows driverquery -si Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows driverquery -v Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows gpresult Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows nbtstat -a Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows nbtstat -n Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows nbtstat -r Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows nbtstat -s Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows netstat Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows quser Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET TROJAN Windows qwinsta Microsoft Windows DOS prompt command exit OUTBOUND': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET WEB_CLIENT BeEF Cookie Outbound': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET WEB_SERVER DD-WRT Information Disclosure Attempt': MicroAttackStage.DATA_EXFILTRATION,
			   'ET WEB_SERVER HTTP 414 Request URI Too Large': MicroAttackStage.COMMAND_AND_CONTROL, #This is if the URL is too long AKA command injection
			   'ET WEB_SERVER PHP Possible file Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			   'ET WEB_SERVER PHP Possible https Local File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			   'ET WEB_SERVER PHP Possible php Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			   'ET WEB_SERVER PHP tags in HTTP POST': MicroAttackStage.DATA_DELIVERY,
			   'ET WEB_SERVER Possible CVE-2014-3120 Elastic Search Remote Code Execution Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'ET WEB_SERVER Possible CVE-2015-1427 Elastic Search Sandbox Escape Remote Code Execution Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'ET WEB_SERVER Possible MySQL SQLi Attempt Information Schema Access': MicroAttackStage.DATA_EXFILTRATION,
			   'ET WEB_SERVER Possible SQL Injection (exec)': MicroAttackStage.COMMAND_AND_CONTROL,
			   'ET WEB_SERVER Tilde in URI - potential .inc source disclosure vulnerability': MicroAttackStage.DATA_EXFILTRATION,
			   'ET WEB_SERVER Tilde in URI - potential .php~ source disclosure vulnerability': MicroAttackStage.DATA_EXFILTRATION,
			   'ET WEB_SERVER WEB-PHP phpinfo access': MicroAttackStage.SURFING, #Changed from exfiltration
			   'ET WEB_SPECIFIC_APPS PHP-CGI query string parameter vulnerability': MicroAttackStage.DATA_EXFILTRATION,
			   'GPL ATTACK_RESPONSE id check returned root': MicroAttackStage.ROOT_PRIV_ESC,
			   'GPL DNS named authors attempt': MicroAttackStage.INFO_DISC,
			   'GPL DNS named version attempt': MicroAttackStage.INFO_DISC,
			   'GPL FTP .forward': MicroAttackStage.INFO_DISC,
			   'GPL FTP CWD ...': MicroAttackStage.INFO_DISC,
			   'GPL FTP CWD .... attempt': MicroAttackStage.INFO_DISC,
			   'GPL FTP CWD Root directory transversal attempt': MicroAttackStage.INFO_DISC,
			   'GPL FTP CWD ~ attempt': MicroAttackStage.INFO_DISC,
			   'GPL FTP CWD ~root attempt': MicroAttackStage.INFO_DISC,
			   'GPL FTP LIST directory traversal attempt': MicroAttackStage.INFO_DISC,
			   'GPL FTP MKD overflow': MicroAttackStage.INFO_DISC,
			   'GPL FTP MKD overflow attempt': MicroAttackStage.INFO_DISC,
			   'GPL FTP PORT bounce attempt': MicroAttackStage.INFO_DISC,
			   'GPL FTP SITE EXEC attempt': MicroAttackStage.INFO_DISC,
			   'GPL ICMP_INFO PING *NIX': MicroAttackStage.SERVICE_DISC,
			   'GPL ICMP_INFO PING BSDtype': MicroAttackStage.SERVICE_DISC,
			   'GPL MISC UPnP malformed advertisement': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'GPL MISC rsh root': MicroAttackStage.ROOT_PRIV_ESC,
			   'GPL NETBIOS DCERPC IActivation little endian bind attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'GPL NETBIOS DCERPC Remote Activation bind attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'GPL NETBIOS SMB-DS ADMIN$ share access': MicroAttackStage.DATA_EXFILTRATION,
			   'GPL NETBIOS SMB-DS C$ share access': MicroAttackStage.DATA_EXFILTRATION,
			   'GPL NETBIOS SMB-DS C$ unicode share access': MicroAttackStage.DATA_EXFILTRATION,
			   'GPL NETBIOS SMB-DS D$ share access': MicroAttackStage.DATA_EXFILTRATION,
			   'GPL NETBIOS SMB-DS IPC$ share access': MicroAttackStage.DATA_EXFILTRATION,
			   'GPL NETBIOS SMB-DS IPC$ unicode share access': MicroAttackStage.DATA_EXFILTRATION,
			   'GPL NETBIOS SMB-DS Session Setup NTMLSSP asn1 overflow attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'GPL NETBIOS SMB-DS Session Setup NTMLSSP unicode asn1 overflow attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'GPL NETBIOS SMB-DS repeated logon failure': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'GPL POLICY Sun JavaServer default password login attempt': MicroAttackStage.BRUTE_FORCE_CREDS,
			   'GPL POP3 POP3 PASS overflow attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'GPL RPC portmap bootparam request TCP': MicroAttackStage.SERVICE_DISC ,
			   'GPL RPC portmap cachefsd request TCP': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'GPL RPC portmap listing TCP 111': MicroAttackStage.SERVICE_DISC,
			   'GPL RPC portmap listing UDP 111': MicroAttackStage.SERVICE_DISC,
			   'GPL RPC portmap mountd request UDP': MicroAttackStage.SERVICE_DISC,
			   'GPL RPC portmap rstatd request TCP': MicroAttackStage.SERVICE_DISC,
			   'GPL RPC portmap rusers request TCP': MicroAttackStage.SERVICE_DISC,
			   'GPL RPC portmap sadmind request TCP': MicroAttackStage.SERVICE_DISC,
			   'GPL RPC portmap ypserv request TCP': MicroAttackStage.SERVICE_DISC,
			   'GPL RPC portmap ypupdated request TCP': MicroAttackStage.ARBITRARY_CODE_EXE,
			   'GPL RPC xdmcp info query': MicroAttackStage.INFO_DISC,
			   'GPL SNMP private access udp': MicroAttackStage.ACCT_MANIP, #check this out, this is a bit unclear
			   'GPL SNMP public access udp': MicroAttackStage.ACCT_MANIP,
			   'GPL WEB_SERVER globals.pl access': MicroAttackStage.INFO_DISC, #Changed from exfiltration
			   'GPL WEB_SERVER mod_gzip_status access': MicroAttackStage.INFO_DISC,#Changed from exfiltration
			   'GPL WEB_SERVER perl post attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET ATTACK_RESPONSE Output of id command from HTTP server': MicroAttackStage.INFO_DISC,  #********  CPTC 2018 STARTS HERE **********
			  'ET CHAT MSN status change': MicroAttackStage.DATA_MANIPULATION,
			  'ET CURRENT_EVENTS Possible TLS HeartBleed Unencrypted Request Method 4 (Inbound to Common SSL Port)': MicroAttackStage.VULN_DISC,
			  'ET EXPLOIT Exim/Dovecot Possible MAIL FROM Command Execution': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET EXPLOIT Possible CVE-2014-3704 Drupal SQLi attempt URLENCODE 1': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET EXPLOIT Possible ZyXELs ZynOS Configuration Download Attempt (Contains Passwords)': MicroAttackStage.DATA_EXFILTRATION,
			  'ET INFO NetSSH SSH Version String Hardcoded in Metasploit': MicroAttackStage.INFO_DISC,
			  'ET INFO SUSPICIOUS Dotted Quad Host MZ Response': MicroAttackStage.DATA_EXFILTRATION,
			  'ET INFO Windows OS Submitting USB Metadata to Microsoft': MicroAttackStage.INFO_DISC,
			  'ET P2P BitTorrent peer sync': MicroAttackStage.DATA_EXFILTRATION,
			  'ET POLICY GNU/Linux APT User-Agent Outbound likely related to package management': MicroAttackStage.NON_MALICIOUS,
			  'ET POLICY Http Client Body contains passwd= in cleartext': MicroAttackStage.INFO_DISC,
			  'ET POLICY IP Check Domain (icanhazip. com in HTTP Host)': MicroAttackStage.INFO_DISC,
			  'ET POLICY Outbound MSSQL Connection to Non-Standard Port - Likely Malware': MicroAttackStage.DATA_EXFILTRATION,
			  'ET POLICY POSSIBLE Web Crawl using Curl': MicroAttackStage.INFO_DISC,
			  'ET POLICY POSSIBLE Web Crawl using Wget': MicroAttackStage.INFO_DISC,
			  'ET POLICY Powershell Activity Over SMB - Likely Lateral Movement': MicroAttackStage.LATURAL_MOVEMENT,
			  'ET POLICY Powershell Command With Hidden Window Argument Over SMB - Likely Lateral Movement': MicroAttackStage.LATURAL_MOVEMENT,
			  'ET POLICY Powershell Command With No Profile Argument Over SMB - Likely Lateral Movement': MicroAttackStage.LATURAL_MOVEMENT,
			  'ET POLICY Powershell Command With NonInteractive Argument Over SMB - Likely Lateral Movement': MicroAttackStage.LATURAL_MOVEMENT,
			  'ET POLICY Proxy TRACE Request - inbound': MicroAttackStage.INFO_DISC,
			  'ET POLICY SMB2 NT Create AndX Request For a .bat File': MicroAttackStage.LATURAL_MOVEMENT,
			  'ET POLICY SMB2 NT Create AndX Request For an Executable File': MicroAttackStage.LATURAL_MOVEMENT,
			  'ET SCAN Apache mod_proxy Reverse Proxy Exposure 1': MicroAttackStage.PUBLIC_APP_EXP,
			  'ET SCAN Grendel-Scan Web Application Security Scan Detected': MicroAttackStage.VULN_DISC,
			  'ET SCAN NMAP SIP Version Detection Script Activity': MicroAttackStage.SERVICE_DISC,
			  'ET SCAN NMAP SQL Spider Scan': MicroAttackStage.VULN_DISC,
			  'ET SCAN Nmap Scripting Engine User-Agent Detected (Nmap NSE)': MicroAttackStage.VULN_DISC,
			  'ET SCAN Potential SSH Scan OUTBOUND': MicroAttackStage.DATA_EXFILTRATION,
			  'ET SCAN SFTP/FTP Password Exposure via sftp-config.json': MicroAttackStage.INFO_DISC,
			  'ET SCAN Sqlmap SQL Injection Scan': MicroAttackStage.VULN_DISC,
			  'ET SCAN Suspicious inbound to MSSQL port 1433': MicroAttackStage.VULN_DISC,
			  'ET SCAN Suspicious inbound to Oracle SQL port 1521': MicroAttackStage.VULN_DISC,
			  'ET SCAN Suspicious inbound to PostgreSQL port 5432': MicroAttackStage.VULN_DISC,
			  'ET SCAN Suspicious inbound to mSQL port 4333': MicroAttackStage.VULN_DISC,
			  'ET SCAN Suspicious inbound to mySQL port 3306': MicroAttackStage.VULN_DISC,
			  'ET TROJAN Backdoor family PCRat/Gh0st CnC traffic (OUTBOUND) 106': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET TROJAN Possible Metasploit Payload Common Construct Bind_API (from server)': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET TROJAN Possible NanoCore C2 64B': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET TROJAN Possible Zendran ELF IRCBot Joining Channel 2': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET USER_AGENTS Go HTTP Client User-Agent': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER /bin/bash In URI, Possible Shell Command Execution Attempt Within Web Exploit': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET WEB_SERVER /bin/sh In URI Possible Shell Command Execution Attempt': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET WEB_SERVER /etc/shadow Detected in URI': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SERVER /system32/ in Uri - Possible Protected Directory Access Attempt': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SERVER Access to /phppath/php Possible Plesk 0-day Exploit June 05 2013': MicroAttackStage.ARBITRARY_CODE_EXE, #Changed from service spec.
			  'ET WEB_SERVER Attempt To Access MSSQL xp_cmdshell Stored Procedure Via URI': MicroAttackStage.ROOT_PRIV_ESC,  #May not be the case
			  'ET WEB_SERVER CRLF Injection - Newline Characters in URL': MicroAttackStage.PUBLIC_APP_EXP,
			  'ET WEB_SERVER ColdFusion adminapi access': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET WEB_SERVER ColdFusion administrator access': MicroAttackStage.ROOT_PRIV_ESC,
			  'ET WEB_SERVER ColdFusion componentutils access': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SERVER ColdFusion password.properties access': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SERVER Coldfusion cfcexplorer Directory Traversal': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SERVER Exploit Suspected PHP Injection Attack (cmd=)': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER IIS 8.3 Filename With Wildcard (Possible File/Dir Bruteforce)': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SERVER Joomla Component SQLi Attempt': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SERVER MYSQL SELECT CONCAT SQL Injection Attempt': MicroAttackStage.DATA_MANIPULATION,
			  'ET WEB_SERVER Onmouseover= in URI - Likely Cross Site Scripting Attempt': MicroAttackStage.TRUSTED_ORG_EXP,
			  'ET WEB_SERVER PHP ENV SuperGlobal in URI': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER PHP Easteregg Information-Disclosure (funny-logo)': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER PHP Easteregg Information-Disclosure (php-logo)': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER PHP Easteregg Information-Disclosure (phpinfo)': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER PHP Easteregg Information-Disclosure (zend-logo)': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER PHP REQUEST SuperGlobal in URI': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER PHP SERVER SuperGlobal in URI': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER PHP SESSION SuperGlobal in URI': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER PHP System Command in HTTP POST': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET WEB_SERVER PHP.//Input in HTTP POST': MicroAttackStage.INFO_DISC, #may be more than this
			  'ET WEB_SERVER Possible Attempt to Get SQL Server Version in URI using SELECT VERSION': MicroAttackStage.INFO_DISC,
			  'ET WEB_SERVER Possible CVE-2013-0156 Ruby On Rails XML YAML tag with !ruby': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER Possible CVE-2014-6271 Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER Possible CVE-2014-6271 Attempt in HTTP Cookie': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER Possible CVE-2014-6271 Attempt in HTTP Version Number': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER Possible CVE-2014-6271 Attempt in Headers': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER Possible Cherokee Web Server GET AUX Request Denial Of Service Attempt': MicroAttackStage.NETWORK_DOS,
			  'ET WEB_SERVER Possible IIS Integer Overflow DoS (CVE-2015-1635)': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER Possible SQL Injection Attempt SELECT FROM': MicroAttackStage.DATA_MANIPULATION,
			  'ET WEB_SERVER Possible SQL Injection Attempt UNION SELECT': MicroAttackStage.DATA_MANIPULATION,
			  'ET WEB_SERVER Possible SQLi xp_cmdshell POST body': MicroAttackStage.ROOT_PRIV_ESC,
			  'ET WEB_SERVER Possible XXE SYSTEM ENTITY in POST BODY.': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SERVER Possible bash shell piped to dev tcp Inbound to WebServer': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET WEB_SERVER SELECT USER SQL Injection Attempt in URI': MicroAttackStage.ACCT_MANIP,
			  'ET WEB_SERVER SQL Injection Local File Access Attempt Using LOAD_FILE': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER Script tag in URI Possible Cross Site Scripting Attempt': MicroAttackStage.ARBITRARY_CODE_EXE, #REMOTE_SERVICE_EXP
			  'ET WEB_SERVER Suspicious Chmod Usage in URI': MicroAttackStage.DATA_MANIPULATION,
			  'ET WEB_SERVER allow_url_include PHP config option in uri': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET WEB_SERVER auto_prepend_file PHP config option in uri': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET WEB_SERVER cmd.exe In URI - Possible Command Execution Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SERVER disable_functions PHP config option in uri': MicroAttackStage.PUBLIC_APP_EXP,
			  'ET WEB_SERVER open_basedir PHP config option in uri': MicroAttackStage.PUBLIC_APP_EXP,
			  'ET WEB_SERVER safe_mode PHP config option in uri': MicroAttackStage.PUBLIC_APP_EXP,
			  'ET WEB_SERVER suhosin.simulation PHP config option in uri': MicroAttackStage.PUBLIC_APP_EXP,
			  'ET WEB_SPECIFIC_APPS Achievo debugger.php config_atkroot parameter Remote File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS AjaxPortal ajaxp_backend.php page Parameter SQL Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS AjaxPortal di.php pathtoserverdata Parameter Remote File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS AlstraSoft AskMe que_id Parameter SELECT FROM SQL Injection Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS BASE base_stat_common.php remote file include': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS BLOG CMS nsextt parameter Cross Site Scripting Vulnerability': MicroAttackStage.TRUSTED_ORG_EXP,
			  'ET WEB_SPECIFIC_APPS BaconMap updatelist.php filepath Local File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Beerwins PHPLinkAdmin edlink.php linkid Parameter SQL Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Community CMS view.php article_id Parameter SQL Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS CultBooking lang parameter Local File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Demium CMS urheber.php name Parameter Local File Inclusion': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS DesktopOnNet don3_requiem.php app_path Parameter Remote File Inclusion': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS DesktopOnNet frontpage.php app_path Parameter Remote File Inclusion': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Enthusiast path parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Fork-CMS js.php module parameter Local File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS FormMailer formmailer.admin.inc.php BASE_DIR Parameter Remote File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Golem Gaming Portal root_path Parameter Remote File inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Horde type Parameter Local File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS IBSng str Parameter Cross Site Scripting Attempt': MicroAttackStage.ARBITRARY_CODE_EXE, #PUBLIC APP!
			  'ET WEB_SPECIFIC_APPS JobHut browse.php pk Parameter SQL Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Joomla 3.7.0 - Sql Injection (CVE-2017-8917)': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Joomla AjaxChat Component ajcuser.php GLOBALS Parameter Remote File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Joomla Dada Mail Manager Component config.dadamail.php GLOBALS Parameter Remote File Inclusion': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Joomla Onguma Time Sheet Component onguma.class.php mosConfig_absolute_path Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Joomla Simple RSS Reader admin.rssreader.php mosConfig_live_site Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Joomla swMenuPro ImageManager.php Remote File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS KR-Web krgourl.php DOCUMENT_ROOT Parameter Remote File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS KingCMS menu.php CONFIG Parameter Remote File Inclusion': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS MAXcms fm_includes_special Parameter Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS MODx CMS snippet.reflect.php reflect_base Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Mambo Component com_smf smf.php Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Microhard Systems 3G/4G Cellular Ethernet and Serial Gateway - Default Credentials': MicroAttackStage.USER_PRIV_ESC,
			  'ET WEB_SPECIFIC_APPS Noname Media Photo Galerie Standard SQL Injection Attempt -- view.php id SELECT': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS OBOphiX fonctions_racine.php chemin_lib parameter Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS OpenX phpAdsNew phpAds_geoPlugin Parameter Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Opencadastre soustab.php script Local File Inclusion Vulnerability': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Oracle JSF2 Path Traversal Attempt': MicroAttackStage.INFO_DISC,
			  'ET WEB_SPECIFIC_APPS OrangeHRM path Parameter Local File Inclusion Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS PHP Aardvark Topsites PHP CONFIG PATH Remote File Include Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS PHP Booking Calendar page_info_message parameter Cross-Site Scripting Vulnerability ': MicroAttackStage.TRUSTED_ORG_EXP,
			  'ET WEB_SPECIFIC_APPS PHP Classifieds class.phpmailer.php lang_path Parameter Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS PHP phpMyAgenda rootagenda Remote File Include Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS PHP-Paid4Mail RFI attempt ': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS PHPOF DB_AdoDB.Class.PHP PHPOF_INCLUDE_PATH parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS PithCMS oldnews_reader.php lang Parameter Local File Inclusion Attempt': MicroAttackStage.DATA_EXFILTRATION, #This is local files only
			  'ET WEB_SPECIFIC_APPS Plone and Zope cmd Parameter Remote Command Execution Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS PointComma pctemplate.php pcConfig Parameter Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Possible JBoss JMX Console Beanshell Deployer WAR Upload and Deployment Exploit Attempt': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ET WEB_SPECIFIC_APPS Possible Mambo/Joomla! com_koesubmit Component \'koesubmit.php\' Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Possible OpenSiteAdmin pageHeader.php Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Possible eFront database.php Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS PozScripts Business Directory Script cid parameter SQL Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS ProdLer prodler.class.php sPath Parameter Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS ProjectButler RFI attempt ': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Request to Wordpress W3TC Plug-in dbcache Directory': MicroAttackStage.INFO_DISC,
			  'ET WEB_SPECIFIC_APPS SAPID get_infochannel.inc.php Remote File inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS SERWeb load_lang.php configdir Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS SERWeb main_prepend.php functionsdir Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS SFS EZ Hotscripts-like Site showcategory.php cid Parameter SQL Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS SFS EZ Hotscripts-like Site software-description.php id Parameter SQL Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Sisplet CMS komentar.php site_path Parameter Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS TECHNOTE shop_this_skin_path Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Turnkeyforms Software Directory showcategory.php cid parameter SQL Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS Ve-EDIT edit_htmlarea.php highlighter Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Vulnerable Magento Adminhtml Access': MicroAttackStage.ROOT_PRIV_ESC,
			  'ET WEB_SPECIFIC_APPS WEB-PHP RCE PHPBB 2004-1315': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ET WEB_SPECIFIC_APPS WHMCompleteSolution templatefile Parameter Local File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS WikyBlog which Parameter Cross Site Scripting Attempt': MicroAttackStage.ARBITRARY_CODE_EXE, #PUBLIC APP!
			  'ET WEB_SPECIFIC_APPS YapBB class_yapbbcooker.php cfgIncludeDirectory Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS Zen Cart loader_file Parameter Local File Inclusion Attempt': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SPECIFIC_APPS axdcms aXconf Parameter Local File Inclusion Attempt': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SPECIFIC_APPS evision cms addplain.php module parameter Local File Inclusion': MicroAttackStage.DATA_EXFILTRATION,
			  'ET WEB_SPECIFIC_APPS p-Table for WordPress wptable-tinymce.php ABSPATH Parameter RFI Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS phPortal gunaysoft.php icerikyolu Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS phPortal gunaysoft.php sayfaid Parameter Remote File Inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS phpSkelSite theme parameter remote file inclusion': MicroAttackStage.DATA_DELIVERY,
			  'ET WEB_SPECIFIC_APPS phptraverse mp3_id.php GLOBALS Parameter Remote File Inclusion Attempt': MicroAttackStage.DATA_DELIVERY,
			  'ETPRO ATTACK_RESPONSE MongoDB Database Enumeration Request': MicroAttackStage.DATA_EXFILTRATION,
			  'ETPRO ATTACK_RESPONSE MongoDB Version Request': MicroAttackStage.INFO_DISC,
			  'ETPRO EXPLOIT SOAP Netgear WNDR Auth Bypass/Info Disclosure': MicroAttackStage.ROOT_PRIV_ESC,
			  'ETPRO SCAN IPMI Get Authentication Request (null seq number - null sessionID)': MicroAttackStage.HOST_DISC,
			  'ETPRO TROJAN Likely Bot Nick in IRC ([country|so_version|computername])': MicroAttackStage.DATA_MANIPULATION,
			  'ETPRO TROJAN Likely Bot Nick in Off Port IRC': MicroAttackStage.DATA_MANIPULATION,
			  'ETPRO TROJAN Win32/Meterpreter Receiving Meterpreter M1': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ETPRO WEB_SERVER JexBoss Common URI struct Observed 2 (INBOUND)': MicroAttackStage.COMMAND_AND_CONTROL,
			  'ETPRO WEB_SERVER Possible Information Leak Vuln CVE-2015-1648': MicroAttackStage.DATA_EXFILTRATION,
			  'ETPRO WEB_SERVER SQLMap Scan Tool User Agent': MicroAttackStage.VULN_DISC,
			  'ETPRO WEB_SPECIFIC_APPS CM Download Manager WP Plugin Code Injection': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ETPRO WEB_SPECIFIC_APPS Majordomo Directory Traversal Attempt': MicroAttackStage.INFO_DISC,
			  'ETPRO WEB_SPECIFIC_APPS PHPMoAdmin RCE Attempt': MicroAttackStage.ARBITRARY_CODE_EXE,
			  'ETPRO WEB_SPECIFIC_APPS ipTIME firmware < 9.58 RCE': MicroAttackStage.ROOT_PRIV_ESC,
			  'GPL ATTACK_RESPONSE directory listing': MicroAttackStage.INFO_DISC,
			  'GPL EXPLOIT .cnf access': MicroAttackStage.INFO_DISC, #Changed from exfiltration
			  'GPL EXPLOIT .htr access': MicroAttackStage.INFO_DISC, #Changed from exfiltration
			  'GPL EXPLOIT /iisadmpwd/aexp2.htr access': MicroAttackStage.ACCT_MANIP,
			  'GPL EXPLOIT /msadc/samples/ access': MicroAttackStage.DATA_EXFILTRATION,
			  'GPL EXPLOIT CodeRed v2 root.exe access': MicroAttackStage.ROOT_PRIV_ESC,
			  'GPL EXPLOIT ISAPI .ida access': MicroAttackStage.DATA_EXFILTRATION,
			  'GPL EXPLOIT ISAPI .idq access': MicroAttackStage.DATA_EXFILTRATION,
			  'GPL EXPLOIT ISAPI .idq attempt': MicroAttackStage.INFO_DISC,
			  'GPL EXPLOIT administrators.pwd access': MicroAttackStage.ROOT_PRIV_ESC,
			  'GPL EXPLOIT fpcount access': MicroAttackStage.INFO_DISC,
			  'GPL EXPLOIT iisadmpwd attempt': MicroAttackStage.INFO_DISC,
			  'GPL EXPLOIT iissamples access': MicroAttackStage.INFO_DISC,
			  'GPL EXPLOIT unicode directory traversal attempt': MicroAttackStage.INFO_DISC,
			  'GPL POLICY PCAnywhere server response': MicroAttackStage.SERVICE_DISC,
			  'GPL SMTP expn root': MicroAttackStage.ROOT_PRIV_ESC,
			  'GPL SMTP vrfy root': MicroAttackStage.ROOT_PRIV_ESC,
			  'GPL WEB_SERVER .htaccess access': MicroAttackStage.INFO_DISC, #Changed from exfiltration
			  'GPL WEB_SERVER .htpasswd access': MicroAttackStage.INFO_DISC, #Changed from exfiltration
			  'GPL WEB_SERVER /~root access': MicroAttackStage.INFO_DISC, #Changed from exfiltration
			  'GPL WEB_SERVER 403 Forbidden': MicroAttackStage.INFO_DISC,
			  'GPL WEB_SERVER DELETE attempt': MicroAttackStage.DATA_DESTRUCTION,
			  'GPL WEB_SERVER Oracle Java Process Manager access': MicroAttackStage.RESOURCE_HIJACKING,
			  'GPL WEB_SERVER Tomcat server snoop access': MicroAttackStage.DATA_EXFILTRATION,
			  'GPL WEB_SERVER author.exe access': MicroAttackStage.INFO_DISC,
			  'GPL WEB_SERVER authors.pwd access': MicroAttackStage.INFO_DISC,
			  'GPL WEB_SERVER global.asa access': MicroAttackStage.INFO_DISC, #Changed from exfiltration
			  'GPL WEB_SERVER iisadmin access': MicroAttackStage.INFO_DISC,#Changed from exfiltration
			  'GPL WEB_SERVER printenv access': MicroAttackStage.INFO_DISC,#Changed from exfiltration
			  'GPL WEB_SERVER python access attempt': MicroAttackStage.COMMAND_AND_CONTROL,
			  'GPL WEB_SERVER service.cnf access': MicroAttackStage.INFO_DISC,#Changed from exfiltration
			  'GPL WEB_SERVER service.pwd': MicroAttackStage.INFO_DISC,
			  'GPL WEB_SERVER services.cnf access': MicroAttackStage.INFO_DISC,
			  'GPL WEB_SERVER viewcode access': MicroAttackStage.INFO_DISC,
			  'GPL WEB_SERVER writeto.cnf access': MicroAttackStage.INFO_DISC,
			  'ETPRO EXPLOIT Possible Wget Arbitrary File Write Exploit Attempt (CVE-2016-4971)' : MicroAttackStage.DATA_DELIVERY,
			   "INDICATOR-SCAN PHP backdoor scan attempt": MicroAttackStage.VULN_DISC,
			   "MALWARE-CNC Win.Trojan.Dorkbot variant outbound connection" : MicroAttackStage.COMMAND_AND_CONTROL,
			   "MALWARE-CNC Win.Trojan.Saeeka variant outbound connection" : MicroAttackStage.COMMAND_AND_CONTROL,
			   "MALWARE-CNC Hacker-Tool sars notifier variant outbound connection php notification" : MicroAttackStage.COMMAND_AND_CONTROL,
			   "MALWARE-CNC Win.Trojan.Alureon.DG runtime traffic detected" : MicroAttackStage.COMMAND_AND_CONTROL,
			   "MALWARE-CNC TT-bot botnet variant outbound connection" : MicroAttackStage.COMMAND_AND_CONTROL,
			   "ETPRO TROJAN CoinMiner Known Malicious Stratum Authline": MicroAttackStage.COMMAND_AND_CONTROL, #WAS RESOURCE_HIJACKING
			   }


	endpointDoS_signatures = [
		"ET DOS Excessive SMTP MAIL-FROM DDoS",
		"ET DOS Possible MYSQL GeomFromWKB() function Denial Of Service Attempt",
		"ET DOS Possible MYSQL SELECT WHERE to User Variable Denial Of Service Attempt"
		"ET DOS Possible MySQL ALTER DATABASE Denial Of Service Attempt",
		"ET DOS Possible SolarWinds TFTP Server Read Request Denial Of Service Attempt",
		"ET DOS SolarWinds TFTP Server Long Write Request Denial Of Service Attempt",
		"ETPRO DOS CA eTrust Intrusion Detection Encryption Key Handling Denial of Service",
		"ETPRO DOS Malformed Email Header Concatination Denial Of Service",
		"ET DOS IBM DB2 kuddb2 Remote Denial of Service Attempt",
		"ET DOS Possible Microsoft SQL Server Remote Denial Of Service Attempt",
		"ETPRO DOS iCal improper resource liberation",
		"ETPRO DOS iCal Null pointer de-reference Count Variable",
		"ETPRO DOS Linux Kernel NetFilter SCTP Unknown Chunk Types Denial of Service 1",
		"ETPRO DOS IBM DB2 Database Server Invalid Data Stream Denial of Service (Published Exploit)",
		"ETPRO DOS Firebird SQL op_connect_request Denial of Service",
		"ET DOS IBM Tivoli Endpoint Buffer Overflow Attempt",
		"ET DOS Microsoft Remote Desktop (RDP) Syn then Reset 30 Second DoS Attempt",
		"ET DOS Microsoft Remote Desktop Protocol (RDP) maxChannelIds Integer indef DoS Attempt",
		"ET DOS Microsoft Remote Desktop Protocol (RDP) maxChannelIds Negative Integer indef DoS Attempt",
		"ET DOS FreeBSD NFS RPC Kernel Panic",
		"ET DOS Trojan.BlackRev V1.Botnet HTTP Login POST Flood Traffic Inbound",
		"ET DOS Possible SSDP Amplification Scan in Progress",
		"ET DOS Outbound Low Orbit Ion Cannon LOIC Tool Internal User May Be Participating in DDOS",
		"ETPRO DOS Possible XMLRPC DoS in Progress",
		"ET DOS HTTP GET AAAAAAAA Likely FireFlood",
		"ET DOS MC-SQLR Response Outbound Possible DDoS Participation",
		"ET DOS Microsoft Windows LSASS Remote Memory Corruption (CVE-2017-0004)",
		"ET DOS Possible SMBLoris NBSS Length Mem Exhaustion Vuln Inbound",
		"ET DOS SMBLoris NBSS Length Mem Exhaustion Attempt (PoC Based)",
		"ET EXPLOIT FortiOS SSL VPN - Pre-Auth Messages Payload Buffer Overflow (CVE-2018-13381)",

	]

	networkDoS_signatures = [
		"ET DOS DNS BIND 9 Dynamic Update DoS attempt",
		"ET DOS Possible Cisco ASA 5500 Series Adaptive Security Appliance Remote SIP Inspection Device Reload Denial of Service Attempt",
		"ET DOS Catalyst memory leak attack",
		"ET DOS Microsoft Streaming Server Malformed Request",
		"ET DOS Potential Inbound NTP denial-of-service attempt (repeated mode 7 request)",
		"ET DOS Potential Inbound NTP denial-of-service attempt (repeated mode 7 reply)",
		"ET DOS Possible VNC ClientCutText Message Denial of Service/Memory Corruption Attempt",
		"ETPRO DOS Oracle Internet Directory Pre-Authentication LDAP Denial of Service Attempt",
		"ET DOS ntop Basic-Auth DOS inbound",
		"ET DOS ntop Basic-Auth DOS outbound",
		"ETPRO DOS Squid Proxy String Processing NULL Pointer Dereference Vulnerability",
		"ET DOS Cisco 514 UDP flood DoS",
		"ETPRO DOS Microsoft Windows Active Directory LDAP SearchRequest Denial of Service Attempt 1",
		"ET DOS ICMP Path MTU lowered below acceptable threshold",
		"ET DOS NetrWkstaUserEnum Request with large Preferred Max Len",
		"ETPRO DOS OpenLDAP Modrdn RDN NULL String Denial of Service Attempt",
		"ETPRO DOS Multiple Vendor ICMP Source Quench Denial of Service",
		"ETPRO DOS ISC DHCP Server Zero Length Client ID Denial of Service",
		"ETPRO DOS Microsoft Windows SMTP Service MX Record Denial Of Service",
		"ETPRO DOS FreeRADIUS RADIUS Server rad_decode Remote Denial of Service",
		"ETPRO DOS Squid Proxy FTP URI Processing Denial of Service",
		"ETPRO DOS Microsoft Host Integration Server snabase.exe Infinite Loop Denial of Service (Exploit Specific)",
		"ETPRO DOS Win32/Whybo.F DDoS Traffic Outbound",
		"ETPRO DOS Microsoft Windows NAT Helper DNS Query Denial of Service",
		"ETPRO DOS Microsoft Host Integration Server snabase.exe Denial of Service 1",
		"ET DOS Cisco Router HTTP DoS",
		"ET DOS Netgear DG632 Web Management Denial Of Service Attempt",
		"ET DOS Cisco 4200 Wireless Lan Controller Long Authorisation Denial of Service Attempt",
		"ETPRO DOS OpenLDAP ber_get_next BER Decoding Denial of Service Attempt",
		"ET DOS Microsoft Windows 7 ICMPv6 Router Advertisement Flood",
		"GPL DOS IGMP dos attack",
		"ET DOS LibuPnP CVE-2012-5963 ST UDN Buffer Overflow",
		"ET DOS Miniupnpd M-SEARCH Buffer Overflow CVE-2013-0229",
		"ETPRO DOS ICMP with truncated IPv6 header CVE-2013-3182",
		"ET DOS Possible NTP DDoS Inbound Frequent Un-Authed MON_LIST Requests IMPL 0x02",
		"ET DOS HOIC with booster inbound",
		"ET DOS Likely NTP DDoS In Progress PEER_LIST_SUM Response to Non-Ephemeral Port IMPL 0x02",
		"ETPRO DOS MS RADIUS DoS Vulnerability CVE-2015-0015",
		"ETPRO DOS Possible mDNS Amplification Scan in Progress",
		"ET DOS Potential Tsunami SYN Flood Denial Of Service Attempt",
		"ET DOS DNS Amplification Attack Possible Outbound Windows Non-Recursive Root Hint Reserved Port",
		"ETPRO DOS MS DNS CHAOS Denial of Service (CVE-2017-0171)",
		"ET DOS Possible Memcached DDoS Amplification Query (set)",

	]

	bruteforce_signatures = [
		"ET SCAN Multiple FTP Root Login Attempts from Single Source - Possible Brute Force Attempt",
		"ET SCAN Multiple FTP Administrator Login Attempts from Single Source - Possible Brute Force Attempt",
		"ET SCAN ICMP PING IPTools",
		"ET SCAN MYSQL 4.1 brute force root login attempt",
		"ET SCAN Medusa User-Agent",
		"ET SCAN ntop-ng Authentication Bypass via Session ID Guessing",
		"ET SCAN Rapid IMAPS Connections - Possible Brute Force Attack",
		"ET SCAN Rapid IMAP Connections - Possible Brute Force Attack",
		"GPL SQL sa brute force failed login attempt",
		"ET ATTACK_RESPONSE Frequent HTTP 401 Unauthorized - Possible Brute Force Attack",
		"ETPRO EXPLOIT Possible Novidade EK Attempting Intranet Router Compromise M7 (Bruteforce)",

	]

	servicedisc_signatures = [
		"ET SCAN Non-Allowed Host Tried to Connect to MySQL Server",
		"GPL SCAN SSH Version map attempt",
		"ET SCAN NMAP OS Detection Probe",
		"ETPRO SCAN Redis INFO Service Probe",


	]

	info_disc_signatures = [
		"ET SCAN PRO Search Crawler Probe",
		"ET SCAN Unusually Fast 400 Error Messages (Bad Request), Possible Web Application Scan",
		"ET SCAN Unusually Fast 404 Error Messages (Page Not Found), Possible Web Application Scan/Directory Guessing Attack",
		"ET SCAN Unusually Fast 403 Error Messages, Possible Web Application Scan",
		"ET SCAN Nessus FTP Scan detected (ftp_anonymous.nasl)",
		"ET SCAN Nessus FTP Scan detected (ftp_writeable_directories.nasl)",
		"ETPRO SCAN Nessus Scanner TFTP Get Attempt",
		"GPL SCAN Finger Version Query",
		"ETPRO SCAN Nessus Scanner TFTP Get Attempt",
		"ET SCAN NMAP SQL Spider Scan",
		"GPL SCAN Finger Account Enumeration Attempt",
		"ET SCAN MySQL Malicious Scanning 1",
		"ET SCAN SFTP/FTP Password Exposure via sftp-config.json",
		"ET SCAN Netsparker Scan in Progress",
		"ET SCAN DEBUG Method Request with Command",
		"ET SCAN DirBuster Scan in Progress",
		"ET SCAN Internet Scanning Project HTTP scan",
		"ET EXPLOIT FortiOS SSL VPN - Information Disclosure (CVE-2018-13379)"
	]

	vuln_disc_signatures = [
		"ET SCAN Havij SQL Injection Tool User-Agent Inbound",
		"ET SCAN Possible SQLMAP Scan",
		"ET SCAN DominoHunter Security Scan in Progress",
		"ET SCAN Potential muieblackcat scanner double-URI and HTTP library",
		"ET SCAN Apache mod_proxy Reverse Proxy Exposure 1",
		"ET SCAN COMMIX Command injection scan attempt",
		"ET SCAN Grendel-Scan Web Application Security Scan Detected",
		"ET SCAN WSFuzzer Web Application Fuzzing",
		"ET SCAN Wikto Scan",
		"ET SCAN Wikto Backend Data Miner Scan",
		"ET SCAN Wapiti Web Server Vulnerability Scan",
		"ET SCAN Suspicious User-Agent - get-minimal - Possible Vuln Scan",
		"ET SCAN SQLBrute SQL Scan Detected",
		"ET SCAN Possible Fast-Track Tool Spidering User-Agent Detected",
		"ET SCAN Metasploit WMAP GET len 0 and type",
		"ET SCAN Behavioral Unusual Port 137 traffic Potential Scan or Infection",
		"ET SCAN Behavioral Unusual Port 135 traffic Potential Scan or Infection",
		"ET SCAN Possible Scanning for Vulnerable JBoss",
		"ET SCAN Nmap NSE Heartbleed Response",
		"ETPRO SCAN NexusTaco Scanning for CVE-2014-3341",

	]

	hostdisc_signatures = [
		"ET SCAN Amap TCP Service Scan Detected",
		"ET SCAN Amap UDP Service Scan Detected",
		"ET SCAN Cisco Torch TFTP Scan",
		"ET SCAN Grim's Ping ftp scanning tool",
		"ET SCAN Modbus Scanning detected",
		"ET SCAN NMAP -sS window 2048",
		"ET SCAN NMAP -sO",
		"ET SCAN NMAP -sA (1)",
		"ET SCAN NMAP -f -sX"
		"ET SCAN Multiple NBTStat Query Responses to External Destination, Possible Automated Windows Network Enumeration",
		"ET SCAN NBTStat Query Response to External Destination, Possible Windows Network Enumeration",
		"ET SCAN Sipvicious Scan",
		"ET SCAN Modified Sipvicious User-Agent Detected (sundayddr)",
		"ET SCAN External to Internal UPnP Request udp port 1900",
		"ET SCAN DCERPC rpcmgmt ifids Unauthenticated BIND",
		"GPL SCAN SolarWinds IP scan attempt",
		"GPL SCAN Broadscan Smurf Scanner",
		"GPL SCAN ISS Pinger",
		"GPL SCAN PING CyberKit 2.2 Windows",
		"GPL SCAN PING NMAP",
		"GPL SCAN Webtrends Scanner UDP Probe",
		"GPL SCAN loopback traffic",
		"ET SCAN Httprecon Web Server Fingerprint Scan",
		"ETPRO SCAN Internal Machine Scanning VNC - Outbound Traffic",
		"ET SCAN ICMP Delphi Likely Precursor to Scan",
		"ETPRO SCAN IPMI Get Authentication Request (null seq number - null sessionID)",
		"ET SCAN ICMP =XXXXXXXX Likely Precursor to Scan",
		"ET SCAN Nmap Scripting Engine User-Agent Detected (Nmap NSE)",
		"ET SCAN Non-Malicious SSH/SSL Scanner on the run",
		"ET SCAN NMAP SIP Version Detection Script Activity"
	]

	def_evasion_signatures = [
		"ET SCAN NNG MS02-039 Exploit False Positive Generator - May Conceal A Genuine Attack",

	]

	remote_serv_signatures = [
		#"ET EXPLOIT Possible Palo Alto SSL VPN sslmgr Format String Vulnerability (Inbound)",
		#"ET EXPLOIT Possible OpenVPN CVE-2014-6271 attempt",

	]

	user_priv_signatures = [


	]

	root_priv_signatures = [
		"ETPRO EXPLOIT Windows Diagnostics Hub Privilege Elevation Vuln Inbound (CVE-2016-3231) 1",
		"ETPRO EXPLOIT ATMFD.DLL Privilege Elevation Vuln (CVE-2016-3220)",
		"ETPRO EXPLOIT Possible CVE-2016-3219 Executable Inbound",
		"ETPRO EXPLOIT Win32k Privilege Elevation Vulnerability (CVE-2016-3254)",
		"ET EXPLOIT Possible MySQL cnf overwrite CVE-2016-6662 Attempt",

	]

	specific_exp_signatures = [
		#"ET EXPLOIT Possible IE Scripting Engine Memory Corruption Vulnerability (CVE-2019-0752)", #moved to arb
		#"ET EXPLOIT FortiOS SSL VPN - Pre-Auth Messages Payload Buffer Overflow (CVE-2018-13381)", #moved to net dos
		"ET EXPLOIT Possible OpenVPN CVE-2014-6271 attempt", #moved to arb
		"ET EXPLOIT Possible Palo Alto SSL VPN sslmgr Format String Vulnerability (Inbound)",
		#"ET EXPLOIT Potential Internet Explorer Use After Free CVE-2013-3163 Exploit URI Struct 1", #moved to arv
		"ET EXPLOIT QNAP Shellshock script retrieval",
		"ET EXPLOIT SolusVM 1.13.03 SQL injection",
		#"ETPRO EXPLOIT Microsoft Edge CSS History Information Disclosure Vulnerability (CVE-2016-7206)", #moved to arb
		"ET EXPLOIT IBM WebSphere - RCE Java Deserialization",
		#"ET EXPLOIT Possible iOS Pegasus Safari Exploit (CVE-2016-4657)",  #moved to arb
		"ET EXPLOIT Possible MySQL cnf overwrite CVE-2016-6662 Attempt",
		"ET EXPLOIT LastPass RCE Attempt",
		#"ETPRO EXPLOIT Possible Wget Arbitrary File Write Exploit Attempt (CVE-2016-4971)", #data deliv
		#"ETPRO EXPLOIT Internet Explorer Memory Corruption Vulnerability (CVE-2016-3211)", #arb
		"ETPRO EXPLOIT Possible HP.SSF.WebService Exploit Attempt",
		"ET EXPLOIT Possible Internet Explorer VBscript failure to handle error case information disclosure CVE-2014-6332 Common Construct M2",
		"ETPRO EXPLOIT Microsoft Office Memory Corruption Vulnerability Pointer Reuse (CVE-2016-0021)",
		"ET EXPLOIT TrendMicro node.js HTTP RCE Exploit Inbound (openUrlInDefaultBrowser)",
		"ET EXPLOIT Possible Postfix CVE-2014-6271 attempt",
		#"ETPRO EXPLOIT Possible HTML Meta Refresh (CVE-2015-6123) Inbound to Server", #arb
		#"ETPRO EXPLOIT Possible HTML Meta Refresh (CVE-2015-6123) via IMAP/POP3", #arb
		"ET EXPLOIT Possible Redirect to SMB exploit attempt - 303",
		#"ETPRO EXPLOIT MSXML3 Same Origin Policy SFB vulnerability 1 (CVE-2015-1646)", #exfil
		"ETPRO EXPLOIT Possible Jetty Web Server Information Leak Attempt",
		#"ETPRO EXPLOIT SChannel Possible Heap Overflow ECDSAWithSHA512 CVE-2014-6321", #arb
		"ETPRO EXPLOIT Netcore Router Backdoor Usage",

	]

	arbitary_exe_signatures = [
		"ET EXPLOIT FortiOS SSL VPN - Remote Code Execution (CVE-2018-13383)",
		"ETPRO EXPLOIT Possible EDGE OOB Access (CVE-2016-0193)",
		"ET EXPLOIT Seagate Business NAS Unauthenticated Remote Command Execution",
		"ET EXPLOIT Possible CVE-2014-6271 exploit attempt via malicious DNS",
		"ET EXPLOIT Possible Pure-FTPd CVE-2014-6271 attempt",
		"ET EXPLOIT Possible Qmail CVE-2014-6271 Mail From attempt",
		"ET EXPLOIT Possible IE Scripting Engine Memory Corruption Vulnerability (CVE-2019-0752)",
		"ET EXPLOIT Possible OpenVPN CVE-2014- 6271 attempt",
		"ET EXPLOIT Potential Internet Explorer Use After Free CVE-2013-3163 Exploit URI Struct 1",
		"ETPRO EXPLOIT Microsoft Edge CSS History Information Disclosure Vulnerability (CVE-2016-7206)",
		"ET EXPLOIT Possible iOS Pegasus Safari Exploit (CVE-2016-4657)",
		"ETPRO EXPLOIT Internet Explorer Memory Corruption Vulnerability (CVE-2016-3211)",
		"ETPRO EXPLOIT Possible HTML Meta Refresh (CVE-2015-6123) Inbound to Server",
		"ETPRO EXPLOIT Possible HTML Meta Refresh (CVE-2015-6123) via IMAP/POP3",
		"ETPRO EXPLOIT SChannel Possible Heap Overflow ECDSAWithSHA512 CVE-2014-6321",
		"ET EXPLOIT Possible OpenVPN CVE-2014-6271 attempt", #Was at Public app
		"ET EXPLOIT Possible Palo Alto SSL VPN sslmgr Format String Vulnerability (Inbound)",

	]

	exfiltration_signatures = [
		"ET EXPLOIT F5 BIG-IP rsync cmi authorized_keys successful exfiltration",
		"ETPRO EXPLOIT MSXML3 Same Origin Policy SFB vulnerability 1 (CVE-2015-1646)",
	]
	
	non_malicious_signatures = [
	
	"SURICATA SMTP invalid reply",
	"SURICATA SMTP invalid pipelined sequence",
	"SURICATA SMTP no server welcome message",
	"SURICATA SMTP tls rejected",
	"SURICATA SMTP data command rejected",
	"SURICATA HTTP gzip decompression failed",
	"SURICATA HTTP request field missing colon",
	"SURICATA HTTP invalid request chunk len",
	"SURICATA HTTP invalid response chunk len",
	"SURICATA HTTP invalid transfer encoding value in request",
	"SURICATA HTTP invalid content length field in request",
	# "SURICATA HTTP status 100-Continue already seen",
	# "SURICATA HTTP unable to match response to request",
	# "SURICATA HTTP request header invalid",
	# "SURICATA HTTP missing Host header",
	# "SURICATA HTTP Host header ambiguous",
	# "SURICATA HTTP invalid response field folding",
	# "SURICATA HTTP response field missing colon",
	# "SURICATA HTTP response header invalid",
	# "SURICATA HTTP multipart generic error",
	# "SURICATA HTTP Host part of URI is invalid",
	# "SURICATA HTTP Host header invalid",
	# "SURICATA HTTP METHOD terminated by non-compliant character",
	# "SURICATA HTTP Request abnormal Content-Encoding header",
	# "SURICATA TLS invalid record type",
	# "SURICATA TLS invalid handshake message",
	# "SURICATA TLS invalid certificate",
	# "SURICATA TLS certificate invalid length",
	# "SURICATA TLS error message encountered",
	# "SURICATA TLS invalid record/traffic",
	# "SURICATA TLS overflow heartbeat encountered, possible exploit attempt (heartbleed)",
	# "SURICATA TLS invalid record version",
	"SURICATA TLS invalid SNI length",
	"SURICATA TLS handshake invalid length",
	"SURICATA DNS Unsolicited response",
	"SURICATA DNS malformed response data",
	"SURICATA DNS Not a response",
	"ET POLICY Vulnerable Java Version 1.7.x Detected",
	"ET POLICY Outdated Flash Version M1",
	"ET POLICY OpenVPN Update Check",
	"ET POLICY DynDNS CheckIp External IP Address Server Response",
	
	]

	attack_stage_mapping = {
		MicroAttackStage.END_POINT_DOS: endpointDoS_signatures,
		MicroAttackStage.NETWORK_DOS: networkDoS_signatures,
		MicroAttackStage.HOST_DISC : hostdisc_signatures,
		MicroAttackStage.VULN_DISC : vuln_disc_signatures,
		MicroAttackStage.INFO_DISC : info_disc_signatures,
		MicroAttackStage.BRUTE_FORCE_CREDS : bruteforce_signatures,
		MicroAttackStage.SERVICE_DISC : servicedisc_signatures,
		MicroAttackStage.SERVICE_SPECIFIC : specific_exp_signatures,
		MicroAttackStage.ARBITRARY_CODE_EXE : arbitary_exe_signatures,
		MicroAttackStage.ROOT_PRIV_ESC : root_priv_signatures,
		MicroAttackStage.USER_PRIV_ESC : user_priv_signatures,
		MicroAttackStage.REMOTE_SERVICE_EXP : remote_serv_signatures,
		MicroAttackStage.DEFENSE_EVASION : def_evasion_signatures,
		MicroAttackStage.DATA_EXFILTRATION : exfiltration_signatures,
		MicroAttackStage.NON_MALICIOUS : non_malicious_signatures,
	}

	unknown_mapping = {
		"SERVER-MYSQL MySQL/MariaDB Server geometry query object integer overflow attempt" : MicroAttackStage.END_POINT_DOS,
		"SERVER-MYSQL Multiple SQL products privilege escalation attempt" : MicroAttackStage.PRIV_ESC,
		"SERVER-MYSQL yaSSL SSL Hello Message buffer overflow attempt" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"INDICATOR-SCAN User-Agent known malicious user-agent Masscan" : MicroAttackStage.HOST_DISC,
		"INDICATOR-SCAN DirBuster brute forcing tool detected" : MicroAttackStage.INFO_DISC,
		"INDICATOR-SCAN inbound probing for IPTUX messenger port" : MicroAttackStage.SERVICE_DISC,
		"INDICATOR-SCAN SSH brute force login attempt" : MicroAttackStage.BRUTE_FORCE_CREDS,
		"SERVER-APACHE Apache server mod_proxy reverse proxy bypass attempt" :  MicroAttackStage.PUBLIC_APP_EXP,
		"SERVER-APACHE Apache header parsing space saturation denial of service attempt" : MicroAttackStage.END_POINT_DOS,
		"SERVER-APACHE Apache malformed ipv6 uri overflow attempt": MicroAttackStage.END_POINT_DOS,
		"SERVER-APACHE Apache Struts remote code execution attempt - POST parameter" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"SERVER-APACHE Apache APR header memory corruption attempt" : MicroAttackStage.END_POINT_DOS,
		"OS-WINDOWS Microsoft Windows DNS client TXT buffer overrun attempt" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"INDICATOR-SCAN PHP backdoor scan attempt": MicroAttackStage.VULN_DISC,
		"INDICATOR-SCAN SSH Version map attempt" : MicroAttackStage.HOST_DISC,
		"INDICATOR-SCAN cybercop os probe" : MicroAttackStage.SERVICE_DISC,
		"INDICATOR-SCAN cybercop udp bomb" : MicroAttackStage.SERVICE_DISC,
		"PROTOCOL-FINGER account enumeration attempt" : MicroAttackStage.INFO_DISC,
		"MALWARE-CNC Win.Trojan.Crisis variant outbound connection" : MicroAttackStage.COMMAND_AND_CONTROL,
		"MALWARE-CNC Gozi trojan checkin" : MicroAttackStage.COMMAND_AND_CONTROL,
		"MALWARE-CNC Flame malware connection - /view.php" : MicroAttackStage.COMMAND_AND_CONTROL,
		"OS-LINUX Linux Kernel keyring object exploit download attempt" : MicroAttackStage.PRIV_ESC,
		"OS-LINUX Linux kernel ARM put_user write outside process address space privilege escalation attempt" : MicroAttackStage.PRIV_ESC,
		"OS-LINUX Linux kernel madvise race condition attempt" : MicroAttackStage.PRIV_ESC,
		"FILE-PDF PDF with click-to-launch executable" : MicroAttackStage.DATA_DELIVERY,
		"FILE-PDF Adobe Acrobat Reader PDF font processing memory corruption attempt" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"OS-OTHER Bash environment variable injection attempt" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"OS-OTHER Intel x86 side-channel analysis information leak attempt" : MicroAttackStage.DATA_EXFILTRATION,
		"OS-OTHER Mac OS X setuid privilege esclatation exploit attempt" : MicroAttackStage.PRIV_ESC,
		"FILE-EXECUTABLE Microsoft Windows Authenticode signature verification bypass attempt" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"FILE-EXECUTABLE Microsoft CLFS.sys information leak attempt" : MicroAttackStage.DATA_EXFILTRATION,
		"FILE-EXECUTABLE Kaspersky Internet Security kl1.sys out of bounds read attempt" : MicroAttackStage.END_POINT_DOS,
		"FILE-EXECUTABLE XOR 0xfe encrypted portable executable file download attempt" : MicroAttackStage.DATA_DELIVERY,
		"FILE-EXECUTABLE Microsoft Windows NTFS privilege escalation attempt" : MicroAttackStage.PRIV_ESC,
		"INDICATOR-SHELLCODE ssh CRC32 overflow /bin/sh" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"INDICATOR-SHELLCODE possible /bin/sh shellcode transfer attempt" : MicroAttackStage.DATA_DELIVERY,
		"FILE-FLASH Adobe Flash Player memory corruption attempt" : MicroAttackStage.PRIV_ESC,
		"FILE-FLASH Adobe Flash Player use after free attempt" : MicroAttackStage.PRIV_ESC,
		"FILE-IMAGE Oracle Java Virtual Machine malformed GIF buffer overflow attempt" : MicroAttackStage.PRIV_ESC,
		"FILE-IMAGE Adobe Acrobat TIFF Software tag heap buffer overflow attempt": MicroAttackStage.ARBITRARY_CODE_EXE,
		"FILE-JAVA Oracle Java privileged protection domain exploitation attempt" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"FILE-JAVA Oracle Java sun.awt.image.ImagingLib.lookupByteBI memory corruption attempt" : MicroAttackStage.SERVICE_SPECIFIC,
		"FILE-JAVA Oracle Java RangeStatisticImpl sandbox breach attempt" : MicroAttackStage.SERVICE_SPECIFIC,
		"FILE-JAVA Oracle Java System.arraycopy race condition attempt" : MicroAttackStage.SERVICE_SPECIFIC,
		"BROWSER-CHROME Apple Safari/Google Chrome Webkit memory corruption attempt" : MicroAttackStage.ARBITRARY_CODE_EXE,
		"BROWSER-CHROME Google Chrome FileReader use after free attempt" : MicroAttackStage.PRIV_ESC,
		"BROWSER-CHROME V8 JavaScript engine Out-of-Memory denial of service attempt" : MicroAttackStage.END_POINT_DOS,


	}


	recent_suricata_alerts = {
		'ETPRO TROJAN TDrop CnC Checkin': MicroAttackStage.COMMAND_AND_CONTROL,

	}


	def get_unknown_mapping(self):
		out_x = []
		out_y = []

		for sig, stage in self.unknown_mapping.items() :
			out_x.append(sig)
			out_y.append(stage.value)

		return out_x, out_y

	def get_signature_mapping(self):
		map = self.mapping

		for stage, sigs in self.attack_stage_mapping.items() :
			for sig in sigs :
				map[sig] = stage

		return map


class SensorObservability :

	def __init__(self) :
		pass

	signature_based = [
		MicroAttackStage.HOST_DISC,
		MicroAttackStage.SERVICE_DISC,
		MicroAttackStage.VULN_DISC,
		MicroAttackStage.INFO_DISC,
		MicroAttackStage.USER_PRIV_ESC,
		MicroAttackStage.ROOT_PRIV_ESC,
		MicroAttackStage.BRUTE_FORCE_CREDS,
		MicroAttackStage.PUBLIC_APP_EXP,
		MicroAttackStage.REMOTE_SERVICE_EXP,
		MicroAttackStage.SERVICE_SPECIFIC,
		MicroAttackStage.ARBITRARY_CODE_EXE,
		MicroAttackStage.COMMAND_AND_CONTROL,
		MicroAttackStage.END_POINT_DOS,
		MicroAttackStage.RESOURCE_HIJACKING,
		MicroAttackStage.NETWORK_DOS,
		MicroAttackStage.SERVICE_STOP,
		MicroAttackStage.DATA_EXFILTRATION,
		MicroAttackStage.DATA_MANIPULATION,
		MicroAttackStage.DATA_DELIVERY,
		MicroAttackStage.PHISHING,
		MicroAttackStage.NON_MALICIOUS,
	]

	binary_testing = [MicroAttackStage.NETWORK_DOS, MicroAttackStage.HOST_DISC]
