from threading import local
from unicodedata import name
import lookup.embedding
from helpers import Database
from helpers import SecretManager
from helpers import Twitter
import json
import numpy as np
import pandas as pd
import vaex as vx

db = Database.get_database("embedder")

#claims = db.retrieve_all(query="select idClaim as id, text from claims.claim")
#tweets = db.retrieve_all(query="SELECT tweet_id FROM tweets.tweet_has_annotation where relevant=1;")
claims = db.retrieve_all(query="select distinct idClaim as id, -1 as idFactcheck, '' as url, text, 'claim' as type from claim")
paragraphs = db.retrieve_all(query="select distinct p.idClaim as id, p.idFactcheck, f.url, p.paragraph as text, 'paragraph' as type from paragraph p inner join factcheck f on f.idfactcheck=p.idfactcheck")
factchecks = db.retrieve_all(query="select idFactcheck, idClaim, url, title from factcheck")
#claims_sentences = [d['text'] for d in claims]
#claims_ids = [d['id'] for d in claims]

#sentences = [d['text'] for d in paragraphs]
#ids = [ d['id'] for d in paragraphs]

#paragraphs = list(filter(lambda x: len(x.split()) > 2, paragraphs))
#print(len(sentences))
#sentences = list(filter(lambda x: len(x.split()) > 2, sentences)) #remove very short sentences / headers
#print(len(sentences))

#combined = [*claims_sentences, *sentences]
#combined_ids = [*claims_ids, *ids]  ## note: might contain duplicate ids!

#ids = np.array(ids)
#claims_ids = np.array(claims_ids)
#combined_ids = np.array(combined_ids)
df_factcheck = pd.DataFrame(factchecks)
df_claims = pd.DataFrame(claims)
df_paragraphs = pd.DataFrame(paragraphs)
df_combined = pd.concat([df_claims, df_paragraphs])

df_claims["index"] = df_claims.index
df_paragraphs["index"] = df_paragraphs.index
df_combined["index"] = df_combined.index
df_factcheck["index"] = df_factcheck.index

vx_combined = vx.from_pandas(df_combined)
vx_claims = vx.from_pandas(df_claims)
vx_paragraphs = vx.from_pandas(df_paragraphs)
vx_factchecks = vx.from_pandas(df_factcheck)


vx_combined.export_hdf5("models/lookup_combined.hdf5")
vx_claims.export_hdf5("models/lookup_claims.hdf5")
vx_paragraphs.export_hdf5("models/lookup_paragraphs.hdf5")
vx_factchecks.export_hdf5("models/lookup_factcheck.hdf5")

print(vx_combined.head(25))
claims_sentences = df_claims["text"].values
sentences = df_paragraphs["text"].values
combined = df_combined["text"].values

ids = df_paragraphs["index"].values
claims_ids = df_claims["index"].values
combined_ids = df_combined["index"].values

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


