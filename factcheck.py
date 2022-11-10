from factcheck.factchecker import MiaFactChecker

factchecker = MiaFactChecker( type ="combined", embed="labse")
print(factchecker.factcheck_tweet(    """
Dus niet #corona maar het #coronavaccin verantwoordelijk voor #myocarditis en #pericarditis. #vaccinatieschade 👇"""))