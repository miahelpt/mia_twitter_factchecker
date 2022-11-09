# -*- coding: utf-8 -*-


"""
code for loading factchecks from the google api
"""

from regex import P
from loaders.FactCheckLoader import FactCheckLoader
from googleapiclient.discovery import build
import time
from datetime import date
from newspaper import Article
from configparser import ConfigParser

class GoogleFactCheckLoader(FactCheckLoader):
    def __init__(self):
        self.name = "GoogleFactCheckLoader"
        super().__init__()
        print(self.last_run)  
        print(self.dayssince)  
        config = ConfigParser()
        config.read('config.ini')
        self.dev_key = config['google']['developerkey']



    #TODO: this should be dynamic, based on a db of queries, like the hashtags of the tweets.
    def loadFactChecks(self):
        print(self.last_run)
        self.runQuery("covid")
        self.runQuery("vaccine")
        self.runQuery("corona")
        self.runQuery("COVID-19")
        self.runQuery('COVID-19-vaccin')
        super().update_last_run()
        

    def runQuery(self, query): 
        with build('factchecktools', 'v1alpha1', developerKey=self.dev_key) as service:
           # response =.execute()

            if self.dayssince is not None:
                request =  service.claims().search(query=query, maxAgeDays=self.dayssince)

            else: 
                request =  service.claims().search(query=query)

            response = request.execute()
            self.processResponse(response)

            while request is not None:
                print(response)
                try:
                    oldresp = response
                    request = service.claims().search_next(previous_request=request, previous_response=response)
                    
                    if request is None: 
                        break

                    response = request.execute()
                    self.processResponse(response)
                except Exception as e:
                    print(e)
                    #wait 10 seconds
                    time.sleep(10)

                    if request is not None:
                        response = oldresp
                    else: 
                        break


    def processResponse(self,response):
        print(response)
        for c in response["claims"]:
        
            #step 1: store or retrieve the claim
            idClaim = super().store_or_retrieve_claim(c.get("text",""), c.get("claimant", ""), c.get("claimDate", ""))
            
            #step 2: store or retrieve the individual factchecks (might be more than one)
            for review in c["claimReview"]:
                #download the full text, for indexing.
                fulltext = ""
                paragraphs = []

                try:
                    url = review.get("url", "")
                    article = Article(url)
                    article.download()
                    article.parse()
                    fulltext = article.text

                    paragraphs = article.text.split("\n\n")
                    print(paragraphs)

                except Exception as e:
                    print(e)


                publisher = review["publisher"]
                
                #step 2: store or retrieve the publisher
                idPublisher = super().store_or_retrieve_publisher(publisher.get("name",""), publisher.get("site", ""))

                #step 3 store the review
                idFactCheck, _new = super().store_or_retrieve_factcheck(idClaim, idPublisher, review.get("url", ""), review.get("title", ""), review.get("reviewDate",""), review.get("textualRating", ""), review.get("languageCode",""), fullreview=fulltext)

                if _new:
                    #step 4 store the paragraphs
                    for paragraph in paragraphs:
                        #this will be used for paragraph based retrieval
                        if paragraph.strip() != "":
                            super().store_or_retrieve_paragraph(idClaim, idFactCheck, paragraph)
              