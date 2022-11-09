from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import sqlite3
import json as json_lib
from flask import send_from_directory
from flask_cors import CORS


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')

@app.route("/")
def main():
    return render_template('home.html')

@app.route('/scripts/<path>')
def send_report(path):
    return send_from_directory('scripts', path)

@app.route("/listen")
def listen():
    return render_template('listener.html')

@app.route("/annotate")
def annotate():
    return render_template('annotate.html')

@app.route("/factcheck")
def factcheck():
    return render_template('factcheck.html')


@app.route("/trends")
def trends():
    return render_template('trends.html')

@app.route("/top_tweets")
def top_tweets():
    #tweets, datetime_toptweets = get_top_tweets()
    #return render_template('top_tweets.html', tweets = tweets, datetime_toptweets = datetime_toptweets)
    return render_template('top_tweets.html')

#todo modify for getTweets
@socketio.on('my event')
def handle_my_custom_event(json):
    print(json)
    

    s_query =  f"""
            SELECT * FROM tweets 
            WHERE relevant='yes' 
            ORDER BY retweet_count DESC
            LIMIT 20;
        """

    print(s_query)
    #create table tweets(tweet_id int not null primary key, twitter_text text, relevant text, confidence real, retweet_count int, created_at datetime)
    con =  sqlite3.connect("twitterlistener.db")
    con.row_factory = sqlite3.Row
    cursor = con.cursor()

    cursor.execute(
        s_query
    )

    rows = cursor.fetchall()
    json__str = json_lib.dumps( [dict(ix) for ix in rows] )
    print(json__str)
    emit('my response', json__str)
    #if json_str:
    #    json.dumps(rows)
    #    return json.dumps( [dict(ix) for ix in rows] ) #CREATE JSON

    #return rows

    """
        Query for facts that are going round: 
        select inner.* from (select count(tweet_id) as mentions, fact from facts where rating="FR" group by fact) as inner order by inner.mentions desc limit 20;
    """


if __name__ == '__main__':
    socketio.run(app)
