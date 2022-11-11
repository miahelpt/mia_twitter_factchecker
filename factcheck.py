#from transformers import DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP
from factcheck.factchecker import MiaFactChecker
#import vaex as vx 

factchecker = MiaFactChecker( type ="combined", embed="mpnet", match_to="claim")
print(factchecker.factcheck_tweet("""Dus niet #corona maar het #coronavaccin verantwoordelijk voor #myocarditis en #pericarditis. #vaccinatieschade 👇"""))
print(factchecker.factcheck_tweet("""So why were all deaths within 28 days of positive pcr recorded as a covid death regardless of prior health condition. Varadkar himself said they even counted people who had not had a positive test as covid deaths. Inflated "covid" deaths to terrify people."""))
print(factchecker.factcheck_tweet("""Covid “vaccines” cause myocarditis and pericarditis but Covid infections do not. Huge 500k+ participant study. 

We were lied to, again.

How many times have experts told us that post Covid myocarditis is worse? 

This is so disheartening— pun intended.
Show this thread"""))