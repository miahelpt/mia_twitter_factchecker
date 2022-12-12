import os
import json
from re import T
from tkinter import TOP
import streamlit as st
from classifiers.tweet_classifier import MiaTweetClassifier
from classifiers.factranker import MiaFactRanker
from factcheck.factchecker import MiaFactChecker

st.title('Twitter Factchecker')
txt = st.text_area("Tweet to be analysed")

@st.experimental_singleton
def get_relevance_classifier():
    relevance_classifier = MiaTweetClassifier()
    relevance_classifier.load_model("relevant")
    return relevance_classifier

@st.experimental_singleton
def get_sentiment_classifier():
    sentiment_classifier = MiaTweetClassifier()
    sentiment_classifier.load_model("sentiment")
    return sentiment_classifier

@st.experimental_singleton
def get_topic_classifier():
    topic_classifier = MiaTweetClassifier()
    topic_classifier.load_model("topic")
    return topic_classifier

#factranker = MiaFactRanker()
#factranker.load_model("factranker")
@st.experimental_singleton
def get_factchecker():
    factchecker = MiaFactChecker( type ="combined", embed="mpnet", match_to="claim")
    return factchecker

relevance_classifier = get_relevance_classifier()
sentiment_classifier = get_sentiment_classifier()
topic_classifier = get_topic_classifier()
factchecker = get_factchecker()

def predict(text): 
    print("button click")
    print(txt)
    relevant, confidence = relevance_classifier.predict(text)
    st.write('Relevant:', relevant)

    sentiment, sconfidence = sentiment_classifier.predict(text)
    st.write('Sentiment:', sentiment)

    if(relevant == "yes"):
        topics = topic_classifier.predict_multilabel(text)
        st.write('Topics:', topics)

        factchecks = factchecker.factcheck_tweet(text)
        print(factchecker.factcheck_tweet(text))
        factcheckdisplay = []

        for factcheck in factchecks:
            displayFacts = []            
            for match in factcheck:
                displayFacts.append({
                    "matched claim": match["matched_claim_text"],
                    "urls": match["urls"],
                    "similarity": match["similarity"]
                })
            displayFacts.append({
                "mentioned fact": match["tweet_text"],
                "factchecks": displayFacts
            })
        st.write('Factchecks:', factcheckdisplay)


st.button("Check" , on_click=predict(txt))
