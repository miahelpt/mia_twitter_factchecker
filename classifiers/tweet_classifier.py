# Code to finetune pretrained transformers models to classify different propperties of a tweet.
# 3 types of classification are included: binary (relevant yes/no), multiclass (sentiment) and multilabel (topic)
# To run, it depends on the annotated twitter dataset to be loaded and hydrated.
# MIA Helpt - 2022

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

import numpy as np
from sklearn.metrics import classification_report
from scipy.special import expit
from scipy.special import softmax

import pandas as pd


class MiaTweetClassifier():
    def __init__(self, for_training=False):
        print("setting up classifier base")

        if for_training == True:
            df_tweets = pd.read_csv(
            "./datasets/tweets_hydrated.csv",
            dtype={'tweet_id': object})
            self.df_tweets = df_tweets[df_tweets.twittertext !=
                                    'not_available'].copy()
            self.df_tweets["clean_text"] = self.df_tweets.apply(
                lambda row: self.preprocess(row['twittertext']), axis=1)

    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def train_relevant(self):
        df = self.df_tweets[['clean_text', 'relevant']].copy()
        df.rename(
            columns={
                "clean_text": "text",
                "relevant": "labels"},
            inplace=True)
        self.id2label = {
            0: "no",
            1: "yes"
        }
        self.train_model(df, "relevant")

    def train_sentiment(self):
        self.df_tweets = self.df_tweets[self.df_tweets.relevant == 1]
        df = self.df_tweets[['clean_text', 'sentiment']].copy()
        df.rename(
            columns={
                "clean_text": "text",
                "sentiment": "labels"},
            inplace=True)
        self.id2label = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }
        self.train_model(df, "sentiment")

    def train_topic(self):
        """
            onderwerp_veiligheid,onderwerp_effectiviteit,onderwerp_noodzaak,onderwerp_wantrouwen_complot,onderwerp_anders
        """
        self.df_tweets = self.df_tweets[self.df_tweets.relevant == 1]

        self.df_tweets["onderwerp_veiligheid"] = self.df_tweets["onderwerp_veiligheid"].astype(
            float)
        self.df_tweets["onderwerp_effectiviteit"] = self.df_tweets["onderwerp_effectiviteit"].astype(
            float)
        self.df_tweets["onderwerp_noodzaak"] = self.df_tweets["onderwerp_noodzaak"].astype(
            float)
        self.df_tweets["onderwerp_wantrouwen_complot"] = self.df_tweets["onderwerp_wantrouwen_complot"].astype(
            float)
        self.df_tweets["onderwerp_anders"] = self.df_tweets["onderwerp_anders"].astype(
            float)
        self.df_tweets["topic"] = self.df_tweets[self.df_tweets.columns[6:11]].values.tolist()

        df = self.df_tweets[['twittertext', 'topic']].copy()

        df.rename(
            columns={
                "twittertext": "text",
                "topic": "labels"},
            inplace=True)
        self.labels = [
            "veiligheid",
            "effectiviteit",
            "noodzaak",
            "wantrouwen/complot",
            "anders"]
        self.id2label = {
            0: "safety",
            1: "effictivity",
            2: "neccesity",
            3: "complot",
            4: "other"
        }
        self.train_model_multilabel(df, "topic")

    def train_model(self, df, modelname):
        train, validate, test = \
            np.split(df.sample(frac=1, random_state=42),
                     [int(.6 * len(df)), int(.8 * len(df))])
        LR = 2e-4
        EPOCHS = 15
        BATCH_SIZE = 32
        MODEL = f"cardiffnlp/twitter-roberta-base-jun2022"
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
        )

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
        trainer.save_model(f"./twitter_classification/models/{modelname}")
        tokenizer.save_pretrained(
            f"./twitter_classification/models/{modelname}")

        test_preds_raw, test_labels, _ = trainer.predict(test_dataset)
        test_preds = np.argmax(test_preds_raw, axis=-1)
        print(classification_report(test_labels, test_preds, digits=3))

    def train_model_multilabel(self, df, modelname):
        train, validate, test = \
            np.split(df.sample(frac=1, random_state=42),
                     [int(.6 * len(df)), int(.8 * len(df))])
        LR = 2e-4
        EPOCHS = 15
        BATCH_SIZE = 32
        MODEL = f"cardiffnlp/twitter-roberta-base-jun2022"
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
        )

        num_labels = len(self.labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            id2label=self.id2label)

        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,                       # training arguments, defined above
            train_dataset=train_dataset,              # training dataset
            eval_dataset=val_dataset                  # evaluation dataset
        )

        trainer.train()

        # save best model
        trainer.save_model(f"./twitter_classification/models/{modelname}")
        tokenizer.save_pretrained(
            f"./twitter_classification/models/{modelname}")

        test_preds_raw, test_labels, _ = trainer.predict(test_dataset)
        test_preds = np.argmax(test_preds_raw, axis=-1)
        #import numpy as np
        rounded_labels = np.argmax(test_labels, axis=1)

        print(classification_report(rounded_labels, test_preds, digits=3))

    def load_model(self, modelname):
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"./models/{modelname}", use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            f"./models/{modelname}")
        self.labels = self.model.config.id2label

    def predict(self, raw_text):
        text = self.preprocess(raw_text)
        tokens = self.tokenizer(text, return_tensors='pt')
        output = self.model(**tokens)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        return self.labels[ranking[0]], scores[ranking[0]]

    def predict_multilabel(self, raw_text):
        text = self.preprocess(raw_text)
        tokens = self.tokenizer(text, return_tensors='pt')
        output = self.model(**tokens)
        scores = output[0][0].detach().numpy()
        scores = expit(scores)
        predictions = (scores >= 0.5) * 1
        # Map to classes
        # print(text)
        preds = []
        #conf = []
        for i in range(len(predictions)):
            if predictions[i]:
                preds.append((self.labels[i], scores[i]))
        print(preds)
        return preds

    def test_set(self):
        for tweet in self.df_tweets.head(10).iterrows():
            prediction, confidence = self.predict(tweet[1]["twittertext"])
            print(tweet[1]["twittertext"])
            print(f"predicted: {prediction}, confidence {confidence}")
            print(tweet[1]["annotator_id"])

    def test_set_ml(self):
        for tweet in self.df_tweets.head(10).iterrows():
            self.predict_multilabel(tweet[1]["twittertext"])


if __name__ == "__main__":
    classifier = MiaTweetClassifier()
    classifier.train_relevant()
    classifier.train_sentiment()
    classifier.train_topic()

    # classifier.load_relevant()

    # classifier.train_sentiment()
    classifier.load_model("sentiment")
    classifier.test_set()
