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

claims = db.retrieve_all(query="select distinct idClaim as id, -1 as idFactcheck, '' as url, text, 'claim' as type from claim")
paragraphs = db.retrieve_all(query="select distinct p.idClaim as id, p.idFactcheck, f.url, p.paragraph as text, 'paragraph' as type from paragraph p inner join factcheck f on f.idfactcheck=p.idfactcheck")
factchecks = db.retrieve_all(query="select fc.idFactcheck, fc.idClaim, fc.url, fc.title, c.text as claim from factcheck fc inner join claim c on  fc.idClaim=c.idClaim")

##Factchecks allows us to match back to the title and the claim as reported to google for matching.


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


