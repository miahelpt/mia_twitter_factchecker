# Code to finetune XLM-Roberta to be able to classify whether a sentence should be factchecked. 
# Based on the datasets from FactRank (Dutch)

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

import numpy as np
from sklearn.metrics import classification_report
from scipy.special import expit
from scipy.special import softmax

import pandas as pd
import re
from helpers.Twitter import TwitterParser

class MiaFactRanker():
    def __init__(self):
        #self.spacy_nlp = spacy.load('nl_core_news_lg')
        self.twitterParser = TwitterParser()

    def train_factrank(self):
        self.df_sentences = pd.read_csv("datasets/factrank_combined.csv")
        self.df_sentences.rename(
            columns={
                "statement": "text",
                "label": "labels"},
            inplace=True)

        self.df_sentences["labels"].replace({"NF": 0, "FNR": 1, "FR": 2}, inplace=True)
        self.id2label = {
            0: "NF",
            1: "FNR",
            2: "FR"
        }
        df = self.df_sentences.copy()
        self.train_model(df, "factranker")

    def train_model(self, df, modelname):
        train, validate, test = \
            np.split(df.sample(frac=1, random_state=42),
                     [int(.6 * len(df)), int(.8 * len(df))])
        LR = 2e-4
        EPOCHS = 15
        BATCH_SIZE = 32
        MODEL = f"xlm-roberta-base"
        MAX_TRAINING_EXAMPLES = -1  # set this to -1 if you want to use the whole training set

        tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

        train_encodings = tokenizer(
            train['text'].tolist(),
            truncation=True,
            padding=True)
        val_encodings = tokenizer(
            validate['text'].tolist(),
            truncation=True,
            padding=True)
        test_encodings = tokenizer(
            test['text'].tolist(),
            truncation=True,
            padding=True)

        class MiaDataSet(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {
                    key: torch.tensor(
                        val[idx]) for key,
                    val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = MiaDataSet(train_encodings, train['labels'].tolist())
        val_dataset = MiaDataSet(val_encodings, validate['labels'].tolist())
        test_dataset = MiaDataSet(test_encodings, test['labels'].tolist())

        training_args = TrainingArguments(
            output_dir='./results',                   # output directory
            num_train_epochs=EPOCHS,                  # total number of training epochs
            # batch size per device during training
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
            # number of warmup steps for learning rate scheduler
            warmup_steps=100,
            weight_decay=0.01,                        # strength of weight decay
            logging_dir='./logs',                     # directory for storing logs
            logging_steps=10,                         # when to print log
            load_best_model_at_end=True,              # load or not best model at the end
            evaluation_strategy = "no",
            save_strategy = 'no'
        )

        print(set(train["labels"].tolist()))

        num_labels = len(set(train["labels"].tolist()))

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=num_labels, id2label=self.id2label)

        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,                       # training arguments, defined above
            train_dataset=train_dataset,              # training dataset
            eval_dataset=val_dataset                  # evaluation dataset
        )

        trainer.train()

        # save best model
        trainer.save_model(f"./models/{modelname}")
        tokenizer.save_pretrained(
            f"./models/{modelname}")

        test_preds_raw, test_labels, _ = trainer.predict(test_dataset)
        test_preds = np.argmax(test_preds_raw, axis=-1)
        print(classification_report(test_labels, test_preds, digits=3))

    def load_model(self, modelname="factranker"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"./models/{modelname}", use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            f"./models/{modelname}")
        self.labels = self.model.config.id2label

    def preprocess(self, text):
        return text 

    def predict(self, raw_text):
        text = self.preprocess(raw_text)
        tokens = self.tokenizer(text, return_tensors='pt')
        output = self.model(**tokens)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        return self.labels[ranking[0]], scores[ranking[0]]

    def clean_tweet(self, tweet):
        tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
        tweet = " ".join(tweet.split())
        tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
        tweet = re.sub("[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+"," ", tweet) #Remove non-alphanumerical characters

        return tweet.strip()


    def extract_from_tweet(self, tweet_text, fr_only=True):
        facts = []
        for sent in self.twitterParser.sentencize(tweet_text):
            sent = self.twitterParser.clean_text(sent) 
            prediction, confidence = self.predict(sent)
            
            if(fr_only==False or (prediction == 'FR' and confidence>0.75)):
                facts.append([sent, prediction, confidence])
        return facts


    def test_set(self):
        for sentence in self.df_sentences.head(10).iterrows():
            prediction, confidence = self.predict(sentence[1]["statement"])
            print(sentence[1]["statement"])
            print(f"predicted: {prediction}, confidence {confidence}")
            print(sentence[1]["label"])

    def test_set_ml(self):
        for tweet in self.df_tweets.head(10).iterrows():
            self.predict_multilabel(tweet[1]["twittertext"])


if __name__ == "__main__":
    ## running directly starts the training process.
    classifier = MiaFactRanker()
    classifier.train_factrank()
    classifier.load_model("factranker")
    classifier.test_set()
