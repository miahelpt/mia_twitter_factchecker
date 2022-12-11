
# -*- coding: utf-8 -*-
import pymysql.cursors
from regex import D
from sqlalchemy import create_engine
import pandas as pd
import helpers.SecretManager
import json
import sqlite3
from configparser import ConfigParser

class MySQL():
    def __init__(self, db_creds):
        self.db_creds = db_creds

    def __enter__(self):
        self.conn = pymysql.connect(host=self.db_creds["host"],
            user=self.db_creds["username"],
            password=self.db_creds["password"],
            port = self.db_creds["port"],
            database="claims",
            cursorclass=pymysql.cursors.DictCursor)
        return self.conn.cursor()

    def __exit__(self, type, value, traceback):
        self.conn.commit()
        self.conn.close()

class SQLite():
    def __init__(self, file='twitterlistener.db'):
        self.file=file

    def __enter__(self):
        self.conn = sqlite3.connect(self.file)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()
        

    def __exit__(self, type, value, traceback):
        self.conn.commit()
        self.conn.close()


##todo: modify this to allow for remote and local dbs
class Database(): 
    def __init__(self, local=True):
        config = ConfigParser()
        config.read('config.ini')
        self.local=local
        if local:
            #open a SQLite connection
            print("opening SQLIte, no creds needed")
        else:
            #open a mysql cloud connection
            print("opening MySQL")
            self.db_creds = json.loads(helpers.SecretManager.get_secret(config["database"]["secret_manager"]))
            if config["database"]["tunnel"] == "1":
                self.db_creds["host"] = "localhost"
                self.db_creds["port"] = 3301

    def get_connection(self):
        if self.local:
            #return connection to local db
            return SQLite()

        else:
            return MySQL(self.db_creds)

    def getLastRun(self):
        with self.get_connection() as cursor:
            # Read a single record
            #ctx = {"loader_name": self.caller_name}
            if(self.local):
                sql = f"""SELECT lastrun, julianday('now') - julianday(lastrun, 'utc') as dayssince FROM management WHERE loader_name='{self.caller_name}'"""
                print(sql)
            else:
                sql = f"SELECT lastrun, DATEDIFF(NOW(), lastrun) as dayssince FROM `management` WHERE `loader_name`='{self.caller_name}'"
            
            cursor.execute(sql)
            result = cursor.fetchone()
            #result["lastrun"]

            if result is not None:
                return result["lastrun"], result["dayssince"]
            else:
                sql = f"INSERT INTO `management` (`loader_name`, lastrun) VALUES ('{self.caller_name}', '2020-1-1')"
                cursor.execute(sql)
        return self.getLastRun()

    
    def updateLastRun(self):
        with self.get_connection() as cursor:
            # Read a single record
            if(self.local):
                sql = f"UPDATE `management` SET lastrun=julianday('now') WHERE `loader_name`='{self.caller_name}'"
            else:
                sql = f"UPDATE `management` SET lastrun=NOW() WHERE `loader_name`='{self.caller_name}'"
            cursor.execute(sql)
               
    def retrieve_one(self,query):
        with self.get_connection() as cursor:
            # Read a single record
            #sql = "SELECT * FROM `management` WHERE `loader_name`=%s"
            cursor.execute(query)
            result = cursor.fetchone()
            return result

        
    def retrieve_all(self,query):
        with self.get_connection() as cursor:
            # Read a single record
            #sql = "SELECT * FROM `management` WHERE `loader_name`=%s"
            cursor.execute(query)
            result = cursor.fetchall()
            return result

    def insert(self,query):
        with self.get_connection() as cursor:
            # Read a single record
            #sql = "SELECT * FROM `management` WHERE `loader_name`=%s"
            cursor.execute(query)
            return cursor.lastrowid

    def execute(self,query):
        with self.get_connection() as cursor:
            # Read a single record
            #sql = "SELECT * FROM `management` WHERE `loader_name`=%s"
            cursor.execute(query)
            return True

    def storePandasDataframe(self, dataframe):
        print(dataframe)
        pdDf = pd.DataFrame(dataframe)
        engine = create_engine(f'mysql+pymysql://{self.db_creds["username"]}:{self.db_creds["password"]}@{self.db_creds["host"]}:{self.db_creds["port"]}/experiments')
        pdDf.to_sql(con=engine, name="TweetMatchedClaim", if_exists='append')


class MySQLDbConnection(Database):
    def __init__(self, name):
        print("connecting to MySQL")
        self.name=name
        print(f"starting {self.name}")
        #setting up database connection
        print("checking lastrun datetime")
        super().__init__(local=False)
        self.caller_name = name

        super().execute("""
            CREATE SCHEMA IF NOT EXISTS `claims` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci ;
        """)

        ###todo: setup tables for clean install
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.facts(idfact INTEGER PRIMARY KEY AUTO_INCREMENT, tweet_id BIGINT, fact text, rating varchar(3), confidence float);
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.tweet_topics(idtweettopic INTEGER PRIMARY KEY AUTO_INCREMENT,  tweet_id BIGINT not null, topic varchar(12) not null, confidence float);
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.tweets(tweet_id BIGINT not null primary key, twitter_text text, relevant varchar(3), confidence real, sentiment varchar(10), sentiment_confidence real, retweet_count int, created_at datetime);
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.hashtags(hashtag varchar(20) not null primary key, relevant int, irrelevant int);
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.management(loader_name varchar(20), lastrun datetime );
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.`claim` (
            `idclaim` INTEGER PRIMARY KEY AUTO_INCREMENT,
            `text` mediumtext,
            `claimant` varchar(45) DEFAULT NULL,
            `claimdate` datetime DEFAULT NULL
            );
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.`publisher` (
            `idpublisher` INTEGER PRIMARY KEY AUTO_INCREMENT,
            `name` varchar(45),
            `site` varchar(45)
            );
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.`factcheck` (
            `idfactcheck` INTEGER PRIMARY KEY AUTO_INCREMENT,
            `url` varchar(255) DEFAULT NULL,
            `title` text,
            `reviewDate` datetime DEFAULT NULL,
            `rating` varchar(45) DEFAULT NULL,
            `languagecode` varchar(45) DEFAULT NULL,
            `fullreview` longtext,
            `idClaim` int(11) DEFAULT NULL,
            `idPublisher` int(11) DEFAULT NULL
            );
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS claims.`paragraph` (
            `idparagraph` INTEGER PRIMARY KEY AUTO_INCREMENT,
            `idClaim` int(11) DEFAULT NULL,
            `idFactcheck` int(11) DEFAULT NULL,
            `paragraph` mediumtext
            );      
        """)

        self.last_run, self.dayssince = self.getLastRun()

        ###todo: setup tables for clean install

class LocalDBConnection(Database):
    def __init__(self, name):
        print("initializing local db")
        print(f"loading database {name}")
        super().__init__(local=True)
        self.caller_name = name

        ###todo: setup tables for clean install
        super().execute("""
            CREATE TABLE IF NOT EXISTS facts(idfact INTEGER PRIMARY KEY AUTOINCREMENT, tweet_id int, fact text, rating varchar(3), confidence float);
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS tweet_topics(idtweettopic INTEGER PRIMARY KEY AUTOINCREMENT,  tweet_id int not null, topic varchar(12) not null, confidence float);
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS tweets(tweet_id int not null primary key, twitter_text text, relevant varchar(3), confidence real, sentiment varchar(10), sentiment_confidence real, retweet_count int, created_at datetime);
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS hashtags(hashtag varchar(20) not null primary key, relevant int, irrelevant int);
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS management(loader_name varchar(20), lastrun datetime );
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS `claim` (
            `idclaim` INTEGER PRIMARY KEY AUTOINCREMENT,
            `text` mediumtext,
            `claimant` varchar(45) DEFAULT NULL,
            `claimdate` datetime DEFAULT NULL
            );
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS `publisher` (
            `idpublisher` INTEGER PRIMARY KEY AUTOINCREMENT,
            `name` varchar(45),
            `site` varchar(45)
            );
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS `factcheck` (
            `idfactcheck` INTEGER PRIMARY KEY AUTOINCREMENT,
            `url` varchar(255) DEFAULT NULL,
            `title` text,
            `reviewDate` datetime DEFAULT NULL,
            `rating` varchar(45) DEFAULT NULL,
            `languagecode` varchar(45) DEFAULT NULL,
            `fullreview` longtext,
            `idClaim` int(11) DEFAULT NULL,
            `idPublisher` int(11) DEFAULT NULL
            );
        """)
        super().execute("""
            CREATE TABLE IF NOT EXISTS `paragraph` (
            `idparagraph` INTEGER PRIMARY KEY AUTOINCREMENT,
            `idClaim` int(11) DEFAULT NULL,
            `idFactcheck` int(11) DEFAULT NULL,
            `paragraph` mediumtext
            );      
        """)
        self.last_run, self.dayssince = self.getLastRun()




def get_database(name):
    config = ConfigParser()
    config.read('config.ini')
    if config["database"]["use_local"] == "1":
        return LocalDBConnection(name)
    else:
        return MySQLDbConnection(name)


    """    
        create table facts(tweet_id int, fact text, rating varchar(3), confidence float);
    """