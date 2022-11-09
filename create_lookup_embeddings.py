from threading import local
from unicodedata import name
import lookup.embedding
from helpers import Database
from helpers import SecretManager
from helpers import Twitter
import json
import numpy as np
import pandas as pd


db = Database.get_database("embedder")

#claims = db.retrieve_all(query="select idClaim as id, text from claims.claim")
#tweets = db.retrieve_all(query="SELECT tweet_id FROM tweets.tweet_has_annotation where relevant=1;")
claims = db.retrieve_all(query="select distinct idClaim as id, text from claim")
paragraphs = db.retrieve_all(query="select distinct  idClaim as id, idFactcheck, paragraph as text from paragraph")

claims_sentences = [d['text'] for d in claims]
claims_ids = [d['id'] for d in claims]

sentences = [d['text'] for d in paragraphs]
ids = [ d['id'] for d in paragraphs]

#paragraphs = list(filter(lambda x: len(x.split()) > 2, paragraphs))
#print(len(sentences))
#sentences = list(filter(lambda x: len(x.split()) > 2, sentences)) #remove very short sentences / headers
#print(len(sentences))

combined = [*claims_sentences, *sentences]
combined_ids = [*claims_ids, *ids]

ids = np.array(ids)
claims_ids = np.array(claims_ids)
combined_ids = np.array(combined_ids)


print("creating LaBSE embeddings")
embed = lookup.embedding.LaBSEEmbeddings()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="paragraphs", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(sentences)
index.index_embed(embeddings,ids)
index.save_index()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="claims", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(claims_sentences)
index.index_embed(embeddings,claims_ids)
index.save_index()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="combined", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(combined)
index.index_embed(embeddings,combined_ids)
index.save_index()

print("creating SBert embeddings")

embed = lookup.embedding.SBertEmbeddings()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="paragraphs", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(sentences)
index.index_embed(embeddings,ids)
index.save_index()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="claims", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(claims_sentences)
index.index_embed(embeddings,claims_ids)
index.save_index()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="combined", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(combined)
index.index_embed(embeddings,combined_ids)
index.save_index()

print("creating LASER embeddings")

embed = lookup.embedding.LaserEmbeddings()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="paragraphs", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(sentences)
index.index_embed(embeddings,ids)
index.save_index()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="claims", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(claims_sentences)
index.index_embed(embeddings,claims_ids)
index.save_index()

index = lookup.embedding.FaissIndex(index_size=embed.dimensions, name="combined", embedding=embed.name) #we use the embeddings index size
embeddings = embed.embed(combined)
index.index_embed(embeddings,combined_ids)
index.save_index()


