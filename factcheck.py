#from transformers import DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
from factcheck.factchecker import MiaFactChecker
#import vaex as vx 

factchecker = MiaFactChecker( type ="combined", embed="mpnet", match_to="claim")
print(factchecker.factcheck_tweet(    """
Dus niet #corona maar het #coronavaccin verantwoordelijk voor #myocarditis en #pericarditis. #vaccinatieschade 👇"""))


