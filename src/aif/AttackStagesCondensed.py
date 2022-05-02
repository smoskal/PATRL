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
			   MicroAttackStage.BRUTE_FORCE_CREDS: MicroAttackStageCondensed.PRIV_ESC,
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
			   MicroAttackStage.INIT : MicroAttackStageCondensed.INIT
			   }