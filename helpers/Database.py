
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
            sql = "UPDATE `management` SET lastrun=NOW() WHERE `loader_name`=%s"
            cursor.execute(sql, (self.caller_name))
               
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

    def storePandasDataframe(self, dataframe):
        print(dataframe)
        pdDf = pd.DataFrame(dataframe)
        engine = create_engine(f'mysql+pymysql://{self.db_creds["username"]}:{self.db_creds["password"]}@{self.db_creds["host"]}:{self.db_creds["port"]}/experiments')
        pdDf.to_sql(con=engine, name="TweetMatchedClaim", if_exists='append')


class MySQLDbConnection(Database):
    def __init__(self, name):
        self.name=name
        print(f"starting {self.name}")
        #setting up database connection
        print("checking lastrun datetime")
        super().__init__(local=False)
        self.caller_name = name
        self.last_run, self.dayssince = self.getLastRun()

class LocalDBConnection(Database):
    def __init__(self, name):
        print("initializing local db")
        print(f"loading database {name}")
        super().__init__(local=True)
        self.caller_name = name


    """
        create table facts(tweet_id int, fact text, rating varchar(3), confidence float);
    """