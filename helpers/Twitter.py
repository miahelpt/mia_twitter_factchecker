"""
    Helper for twitter functions
"""
import os
import tweepy as tw
import json 
import csv
import re
import unicodedata
import logging

import spacy
from nltk.tokenize import sent_tokenize
from configparser import ConfigParser



class TwitterHydrator():
    def __init__(self):
        config = ConfigParser()
        config.read('config.ini')


        consumer_key = config.get('tweepy', 'consumer_key')
        consumer_secret = config.get('tweepy', 'consumer_secret')
        access_token = config.get('tweepy', 'access_token')
        access_token_secret = config.get('tweepy', 'access_token_secret')

        auth = tw.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tw.API(auth, wait_on_rate_limit=True)
      
    def hydrate(self, twitterId):
        return self.api.get_status(twitterId, tweet_mode='extended')._json['full_text']

class TwitterParser():
    def __init__(self):
        self.nlp = spacy.load('nl_core_news_sm')
        self.sentence_regex = re.compile(
            r' ?((?:[A-Z@#\d]|[\"][^\n]+?[\"] ?)(?:[\"\(][^\n]{1,30}?[\"\)]|\.{3}|[^?!\.\n]|\.[^ \nA-Z\"]){0,200}(?:!|\?|\n|\.{1})) ?'
        )

    @staticmethod
    def clean_tweet(tweet):
        tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
        tweet = " ".join(tweet.split())
        tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
        tweet = re.sub("[^ \nA-Za-z0-9À-ÖØ-öø-ÿ/]+"," ", tweet) #Remove non-alphanumerical characters
        return tweet.strip()

    @staticmethod
    def clean_text(text):
        # normalize unicode equivalence
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r"\r", "", text)
        # normalize single quotes
        text = re.sub(r"’", "'", text)
        text = re.sub(r"‘", "'", text)
        # normalize double quotes
        text = re.sub(r"”", "\"", text)
        text = re.sub(r"„", "\"", text)
        text = re.sub(r"“", "\"", text)
        # replace linebreak by punctuation when followed by linebreak
        text = re.sub(r"(\"|\')\n", r"\g<1>.", text)
        # normalize dash
        text = re.sub(r"—", "-", text)
        text = re.sub(r"–", "-", text)
        # replace double punctuations
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,", ",", text)
        # different whitespace representations
        text = re.sub(r" ", " ", text)
        text = re.sub(r"­", " ", text)
        # remove unwanted stuff
        text = re.sub(r"^ +", "", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\n$", "", text)
        text = re.sub(r"§", "", text)
        # clean dirt
        text = re.sub(r"…", "...", text)
        text = re.sub(r"[\*\n\\…\=•\[\]\|]", "", text)
        # clean excessive whitespace
        text = re.sub(r" +", " ", text).strip()

        if text:
            # clean leading twitter dirt
            while text.strip().split()[0][0] in {'#', '@'}:
                text = re.sub(r"^[@#][^ ]* *", "", text)
            # clean trailing twitter dirt
            while text.strip().split()[-1][0] in {'#', '@'}:
                text = re.sub(r" *[@#][^ ]* *([.?!]) *$", r"\g<1>", text)

        return text

    def clean_sentence_for_inference(self, sentence):
        sentence = TwitterParser.clean_text(sentence)
        # remove unclosed brackets/quotes
        sentence = re.sub(r"^ *\( ?(?!.*\))", "", sentence)  # remove unclosed brackets
        sentence = re.sub(r"^ *\' ?(?!.*\')", "", sentence)  # remove single quotes
        sentence = re.sub(r"^ *\" ?(?!.*\")", "", sentence)  # remove double quotes
        # remove unwanted stuff
        sentence = re.sub(r"^ +", "", sentence)
        sentence = re.sub(r"[A-Z ]+: <<", "", sentence)
        # remove point at the end of sentence
        sentence = re.sub(r"\.$", "", sentence)
        # remove quotes when apparent at both ends
        sentence = re.sub(r"^\'(.*)\'$", r"\g<1>", sentence)  # single quotes
        sentence = re.sub(r"^\"(.*)\"$", r"\g<1>", sentence)  # double quotes
        # clean twitter dirt
        sentence = re.sub(r"@#", "", sentence)
        # tag digits with NUM
        sentence = self.tag_numeral(sentence)
        return sentence

    def tag_numeral(self, sentence):
        replace_values = {
            token.text: f"{token.text} {token.pos_}" for token in self.nlp(sentence) if token.pos_ in {'NUM'}
        }
        sentence = " ".join([replace_values.get(word, word) for word in sentence.split()])
        return sentence

    def tokenize(self, sentence):
        return [
            tok.text
            for tok in self.nlp.tokenizer(self.clean_sentence_for_inference(sentence).lower())
            if tok.text != " "
        ]

    def sentencize(self, text):
        for sentence in self.sentence_regex.findall(text):
            if len(sentence.split()) < 3 or len(sentence.split()) > 100:
                continue  # too long / too short
            if sentence.count("\"") % 2:
                continue  # unclosed quotes
            if sentence.count("(") and sentence.count("(") - sentence.count(")"):
                continue  # unclosed brackets
            if sentence.count('#') > len(sentence.split()) / 2:
                continue  # probably too many hashtags
            yield sentence.strip()

