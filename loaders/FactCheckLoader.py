# -*- coding: utf-8 -*-


"""
Interface class for factcheck loading
"""
import helpers.SecretManager
from helpers.Database import get_database
#import pymysql.cursors
import json

class FactCheckLoader:
    def __init__(self, local_db=True):
        self.db=get_database("factcheckloader")
        self.last_run, self.dayssince = self.db.getLastRun()  

    def store_or_retrieve_claim(self, text, claimant, claimdate):
        text = text.replace("\"", "'")
        claimant = claimant.replace("\"", "'")

        if claimdate != "": 
            strQuery = f"""
                select * from claim where text="{text}" and claimant="{claimant}" and claimdate="{claimdate}"
            """
        else: 
            strQuery = f"""
                select * from claim where text="{text}" and claimant="{claimant}"
            """

        claim = self.db.retrieve_one(strQuery)
        if claim is None:
            strQuery = f"""
                insert into claim(`text`, `claimant`, `claimdate`) VALUES("{text}", "{claimant}", "{claimdate}")
            """
            print(strQuery)
            claim_id = self.db.insert(strQuery)
            return claim_id
        else:
            return claim["idclaim"]
        
    def store_or_retrieve_publisher(self, name, site):
        strQuery = f"""
            select * from publisher where name="{name}" and site="{site}"
        """
        
        publisher = self.db.retrieve_one(strQuery)
        if publisher is None:
            strQuery = f"""
                insert into publisher(`name`, `site`) VALUES("{name}", "{site}")
            """
            print(strQuery)


            publisher_id = self.db.insert(strQuery)
            return publisher_id
        else:
            return publisher["idpublisher"]



    #                                url, title, reviewDate, ratnig, languagecode, fullreview
    def store_or_retrieve_factcheck(self, idClaim, idPublisher, url, title, reviewDate, rating, languagecode, fullreview=""):
        title = title.replace("\"", "'")
        rating = rating.replace("\"", "'")

        if reviewDate !="":
            strQuery = f"""
                select * from factcheck where url="{url}" and title="{title}" and reviewDate="{reviewDate}" 
                and rating="{rating}" and languagecode="{languagecode}" and idClaim={idClaim} and idPublisher={idPublisher}
            """
        
        else:
            strQuery = f"""
                select * from factcheck where url="{url}" and title="{title}" 
                and rating="{rating}" and languagecode="{languagecode}" and idClaim={idClaim} and idPublisher={idPublisher}
            """

        review = self.db.retrieve_one(strQuery)

        if review is None:
            #this did not exist yet.
            fullreview = fullreview.replace("\"", "'")
            strQuery = f"""
                insert into factcheck(`url`, `title`, `reviewDate`, `rating`, `languagecode`, `idClaim`, `idPublisher`, `fullreview`) 
                VALUES("{url}", "{title}", "{reviewDate}", "{rating}", "{languagecode}", {idClaim}, {idPublisher}, "{fullreview}")
            """
            print(strQuery)


            review_id = self.db.insert(strQuery)
            return review_id, True
        else:
            return review["idfactcheck"], False


    def store_or_retrieve_paragraph(self, idClaim, idFactcheck, paragraph):
        paragraph = paragraph.replace("\"", "'")

        strQuery = f"""
            insert into paragraph(`idClaim`, `idFactcheck`, `paragraph`) 
            VALUES("{idClaim}", "{idFactcheck}", "{paragraph}")
        """
        review_id = self.db.insert(strQuery)
        return review_id


    def update_last_run(self):
        self.db.updateLastRun()


