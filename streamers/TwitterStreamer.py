from configparser import ConfigParser

from re import S
from numpy import s_
import tweepy as tw
import json
from classifiers.tweet_classifier import MiaTweetClassifier
from classifiers.factranker import MiaFactRanker
from helpers.Database import get_database

class TwitterListener(tw.Stream):
    def __init__(self, num_tweets_to_grab=-1):
        config = ConfigParser()
        config.read('config.ini')


        consumer_key = config.get('tweepy', 'consumer_key')
        consumer_secret = config.get('tweepy', 'consumer_secret')
        access_token = config.get('tweepy', 'access_token')
        access_token_secret = config.get('tweepy', 'access_token_secret')

        super().__init__(consumer_key, consumer_secret,access_token, access_token_secret)
        self.counter = 0
        self.num_tweets_to_grab = num_tweets_to_grab

        self.relevance_classifier = MiaTweetClassifier()
        self.relevance_classifier.load_model("relevant")

        self.sentiment_classifier = MiaTweetClassifier()
        self.sentiment_classifier.load_model("sentiment")

        self.topic_classifier = MiaTweetClassifier()
        self.topic_classifier.load_model("topic")

        self.factranker = MiaFactRanker()
        self.factranker.load_model("factranker")

        self.db=get_database("streamer")


    def get_hashtags(self): 
        s_query = "select hashtag from hashtags where relevant<>0 order by relevant desc, irrelevant asc limit 15;"
        hashtags = [f"#{h[0]}" for h in self.db.retrieve_all(s_query)]
        if(len(hashtags) == 0): 
            hashtags = ["#covid", "#corona", "#vaccin", "#vaccinatieschade"]
        return hashtags


    def on_data(self,data):
        j = json.loads(data)
        try:
            text = ""
            retweets = 0
            tweet_id = ""
            created_at = ""
            hashtags = [] ## keep a list of relevant / irrelevant per hashtag
            if "retweeted_status" in j:
                try:
                    text = j["retweeted_status"]["extended_tweet"]["full_text"]
                    hashtags = j["retweeted_status"]["extended_tweet"]["entities"]["hashtags"]
                except:
                    text = j["retweeted_status"]["text"]

                retweets = j["retweeted_status"]["retweet_count"]
                tweet_id = j["retweeted_status"]["id"]
                created_at = j["retweeted_status"]["created_at"]

            else:
                try:
                    text = j["extended_tweet"]["full_text"]
                    hashtags = j["extended_tweet"]["entities"]["hashtags"]

                except:
                    text = j["text"]

                retweets = j["retweet_count"]
                tweet_id = j["id"]
                created_at = j["created_at"]
          
            relevant, confidence = self.relevance_classifier.predict(text)
            sentiment, sconfidence = self.sentiment_classifier.predict(text)

            text = text.replace("\"", "'")

            if self.db.local:
                s_query =  f"""
                    INSERT INTO tweets VALUES
                    ({tweet_id}, "{text}", '{relevant}',{confidence}, "{sentiment}", {sconfidence},{retweets}, '{created_at}')
                    ON CONFLICT(tweet_id) DO UPDATE SET retweet_count = {retweets};
                """
            else: 
                s_query =  f"""
                    INSERT INTO tweets VALUES
                    ({tweet_id}, "{text}", '{relevant}',{confidence}, "{sentiment}", {sconfidence},{retweets}, '{created_at}')
                    ON DUPLICATE KEY UPDATE retweet_count = {retweets};
                """
                print(s_query)

            self.db.insert(
                s_query
            )

            if(relevant == 'yes'):
                topics = self.topic_classifier.predict_multilabel(text)

                facts = self.factranker.extract_from_tweet(text)
                for fact in facts:
                    s_query =  f"""
                        INSERT INTO facts VALUES
                        ({tweet_id}, "{fact[0]}", "{fact[1]}", {fact[2]});
                    """
                    self.db.insert(
                        s_query
                    )

        
                for topic in topics:
                    s_query =  f"""
                        INSERT INTO tweet_topics(
                        `tweet_id`,
                        `topic`,
                        `confidence`) VALUES
                        ({tweet_id}, "{topic[0]}",{topic[1]});
                    """
                    self.db.insert(
                        s_query
                    )
                
                for hashtag in hashtags:
                    if self.db.local:
                        s_query =  f"""
                            INSERT INTO hashtags VALUES
                            ("{hashtag["text"]}", 1, 0)
                            ON CONFLICT(hashtag) DO UPDATE SET relevant = relevant+1;
                        """
                    else: 
                        s_query =  f"""
                            INSERT INTO hashtags VALUES
                            ("{hashtag["text"]}", 1, 0)
                            ON DUPLICATE KEY UPDATE relevant = relevant+1;
                        """
                    self.db.insert(
                        s_query
                    )
                    
            else:
                for hashtag in hashtags:
                    if self.db.local:
                        s_query =  f"""
                            INSERT INTO hashtags VALUES
                            ("{hashtag["text"]}", 0, 1)
                            ON CONFLICT(hashtag) DO UPDATE SET irrelevant = irrelevant+1;
                        """
                    else: 
                        s_query =  f"""
                            INSERT INTO hashtags VALUES
                            ("{hashtag["text"]}", 0, 1)
                            ON DUPLICATE KEY UPDATE irrelevant = irrelevant+1;
                        """
                    self.db.insert(
                        s_query
                    )

            #self.con.commit()

        except Exception as e:
            print(e)

    def on_error(self, status):
        print(status)

    def listen(self):
        #start listening
        print("listening in on the twittersphere!")
        super().filter(track=self.get_hashtags(), languages=["nl"])
        super().sample()



