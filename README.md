# MIA Twitter Factchecker
This repository contains the source code for the MIA Twitter Factchecker. 

The factchecker has been created by [Stichting MIA Helpt](https://www.miahelpt.nl), with financial support from the [SIDN Fund](https://www.sidnfonds.nl)

## Installation (local)
To get started, clone the repo to your local machine, in a new folder. 

### Prerequisites
Running the code depends on: 
1. A local installation of [SQLite](www.sqlite.org) (in case of using a local database) or access to an AWS-hosted MySQL or Aurora database, with credentials stored in AWS SecretManager
2. Access to the Google Factcheck API (requires a developer key)
3. Access to the twitter API, requires developer key and credentials.

### Getting started
1. Rename config_ini to config.ini and fill the required api keys (twitter / google)
2. If using a local database, set use_local to 1, this will create a SQLite database in the main directory; if using AWS, fill the secret manager secret (and if using a tunnel, set tunnel to 1, this will assume a tunnel to the database at localhost:3001)
3. Install the requirements (pip install -r requirements.txt)
4. Download the spacy dutch tokenizer:
    python -m spacy download nl_core_news_sm
4. Download the models:
    ./models/download.sh

### Filling the database and start matching facts.
1. Sync the Google factchecks:
    python load_factchecks.py


