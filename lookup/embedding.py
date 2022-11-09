from laserembeddings import Laser
#import fasttext
import fasttext
import faiss                   # make faiss available
from sentence_transformers import SentenceTransformer
from helpers import Database, SecretManager
import json
import numpy as np
import spacy_universal_sentence_encoder
from helpers.Database import get_database



### Embedder is the superclass from which the different embeddings inherit.
class Embedder():
    def __init__(self): 
        self.db=get_database(self.name)
        self.last_run, self.dayssince = self.db.getLastRun()

class LaserEmbeddings(Embedder):
    #todo: laser embeddings has 1024 dimensions
    def  __init__(self):
        self.laser = Laser()
        path_to_pretrained_model = 'models/lid.176.bin'
        self.fmodel = fasttext.load_model(path_to_pretrained_model)
        self.name="Laser2_pip"
        self.dimensions = 1024

    def embed(self, sentences):
        _sentences = [sentence.replace("\n", "") for sentence in sentences]
        _lang= self.fmodel.predict(_sentences)  # lang is only used for tokenization
        __langs = [lgn[0][9:] for lgn in _lang[0]]
        print(__langs)
       
        embeddings = self.laser.embed_sentences(sentences, lang=__langs)
        print(embeddings.shape)

        return embeddings

class LaBSEEmbeddings(Embedder):
    def  __init__(self):
        #sentences = ["This is an example sentence", "Each sentence is converted"]
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        self.name="LaBSESentTransformers"
        self.dimensions = 768

    def embed(self, sentence):
        embeddings = self.model.encode(sentence, normalize_embeddings=False)
        print(embeddings.shape)
        return embeddings


class SBertEmbeddings(Embedder):
    def  __init__(self):
        #sentences = ["This is an example sentence", "Each sentence is converted"]
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.name="paraphrase-multilingual-mpnet-base-v2"
        self.dimensions = 768

    def embed(self, sentence):
        print(len(sentence))
        embeddings = self.model.encode(sentence, normalize_embeddings=False)
        print(embeddings.shape)
        return embeddings
    
class USEEmbedings(Embedder):
    #TODO: test the USE Embeddings
    def __init__(self):
        self.model = spacy_universal_sentence_encoder.load_model('xx_use_lg')
        self.name="universal_sentence_encoder"
        self.dimensions = 768


    def embed(self, sentence):
        print(len(sentence))
        embeddings = self.model(sentence)
        return embeddings

class LookupIndex():
    def __init__(self, name, embedding):
        print("setting filename")
        self.filename=f"models/{name}-{embedding}.bin"


"""
    Code for the semantic lookup of facts, and embedding them in a Faiss L2 space.
"""
class FaissIndex(LookupIndex):
    def __init__(self, index_size=1024, name="FaissIDMap", embedding="Embedding"):
        print("initializing index")
        super().__init__(name, embedding)
        self.index = faiss.IndexFlatIP(index_size)   # build the index, d=size of vectors 
        self.idmap = faiss.IndexIDMap(self.index)  


    def load_index(self):
        #loads the index from file
        print("loading index")
        self.idmap = faiss.read_index( self.filename)

    def save_index(self):
        #loads the index from file
        print("saving index")
        faiss.write_index(self.idmap, self.filename)


    def index_embed(self, sentence_embeddings, ids):
        print(sentence_embeddings.shape)
        print(ids.shape)
        faiss.normalize_L2(sentence_embeddings)
        self.idmap.add_with_ids(sentence_embeddings, ids)


    def search_index(self, xq, k=5):
        # xq is a n2-by-d matrix with query vectors
        faiss.normalize_L2(xq)
        D, I = self.idmap.search(xq,k)
        return(D,I)

def __main__():
    print("creating lookup indices")
    #TODO: ensure that this runs either through local or remote db --> copy code from test_embed