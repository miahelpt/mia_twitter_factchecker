
from unicodedata import name
import lookup.embedding
from helpers import Database
from helpers import SecretManager
from helpers import Twitter
import json
import numpy as np
import pandas as pd
import re
from sentence_transformers.cross_encoder import CrossEncoder
from classifiers.factranker import MiaFactRanker #see: https://github.com/lejafar/FactRank
import spacy
from configparser import ConfigParser
import vaex as vx

class MiaFactChecker():
    def __init__(self, type="combined", embed="mpnet", match_to="claim"):
        ###TODO: currently we only check the candidates back to the claim, allow for check on title and individual paragraph (matched) as well.
        print("setting up class to factcheck!")
        self.nlp = spacy.load('nl_core_news_md')
        self.crossEnc = CrossEncoder('./models/cross_encoder')
        self.factranker = MiaFactRanker()
        self.factranker.load_model("factranker")
        #self.db = Database.get_database("embedder") #ensure we have all.
        #self.claims = self.db.retrieve_all(query="select idClaim as id, text from claims.claim")
        #self.factchecks = self.db.retrieve_all(query="select distinct idClaim as id, url from claims.factcheck")
        self.lookup_table = vx.open(f"./models/lookup_{type}.hdf5")
        self.factcheck_lookup = vx.open(f"./models/lookup_factcheck.hdf5")
        self.match_to = match_to
        if embed == "mpnet":
            self.embed = lookup.embedding.SBertEmbeddings() 
        elif embed == "laser":
            self.embed = lookup.embedding.LaserEmbeddings() 
        else:
            self.embed = lookup.embedding.LaBSEEmbeddings() 

        self.index = lookup.embedding.FaissIndex(index_size=self.embed.dimensions, name=type, embedding=self.embed.name) #we use the embeddings index size
        self.index.load_index()
        self.hydrator = Twitter.TwitterHydrator()

    def extract_simil(self, _json):
        return _json["similarity"]

    def factcheck_tweet(self, tweet):
        fr_sentences = self.factranker.extract_from_tweet(tweet)
        factchecks = []
        if(len(fr_sentences)> 0):
            for fr_sentence in fr_sentences:
                fr_to_check = fr_sentence[0]
                tweetEmbedding = self.embed.embed([fr_to_check])
                d,I = self.index.search_index(tweetEmbedding, k=50)
                df_factcheck = []

                it = 0
                itr = 0
                try:
                    for i in I:
                        for c in i:
                            matched_claim = self.lookup_table[self.lookup_table.index== c]
                            claim_type = matched_claim.type.values[0].as_py()
                            if claim_type == "claim":
                                related_factchecks = self.factcheck_lookup[self.factcheck_lookup.idClaim==matched_claim.id.values[0]]
                                self.claim_id=matched_claim.id.values[0]
                                claim_text = matched_claim.text.values[0]
                            else: 
                                #paragraph
                                related_factchecks = self.factcheck_lookup[self.factcheck_lookup.idFactcheck==matched_claim.idFactcheck.values[0]]
                                if(self.match_to=="claim"):
                                    claim_text = related_factchecks.claim.values[0]  
                                elif self.match_to=="title":   
                                    claim_text = related_factchecks.title.values[0]  
                                else:
                                    matched_claim.text.values[0] ##match back to the title?
                                
                                self.claim_id=matched_claim.id.values[0]

                            claim_text = claim_text.as_py() #convert StringScalar to python str
                            simil = self.crossEnc.predict([fr_to_check, claim_text])


                            if  simil>0.3: #d[it][itr]>=0.5:
                                 #dfFactchecks.loc[dfFactchecks['id'] == c]
                               
                                urls = list(set([val.as_py() for val in related_factchecks.url.values]))
                                df_factcheck.append({
                                        "similarity": simil,
                                        "urls": urls,
                                        "type": claim_type,
                                        "tweet_text": fr_to_check,
                                        "matched_claim_id": self.claim_id, #todo: this is the lookup index, we need to fix this
                                        "matched_claim_text": claim_text,
                                        "distance": d[it][itr],
                                        "embedding": self.embed.name,
                                        "metric": "cos",
                                        "index": self.match_to
                                    })
                            itr += 1   
                        it += 1
                except Exception as e:
                    print("Exception!")
                    print(e)

                df_factcheck.sort(key=self.extract_simil, reverse=True)
                unique =  list({ each['matched_claim_id'] : each for each in df_factcheck }.values())

                if(len(unique)>0):
                    factchecks.append(unique)
                    
        print(factchecks)
        return factchecks

