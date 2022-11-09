import socketIOClient from "socket.io-client";
import React, { useState, useEffect } from "react";

const ENDPOINT = "http://127.0.0.1:5000";

const TweetListener = () => {
  const [tweets, setTweets ] = useState([]);

  const socket = socketIOClient(ENDPOINT);  

  socket.emit("my event", {test:'test'}, data => {
  });

  const msgCallback = (data) => {
      const _data = JSON.parse(data);
      console.log(_data);
      setTweets(_data);

      setTimeout(function () {
        socket.emit("my event", {test:'test'}, data => {
        });
      }, 5000);
  };


  socket.on("my response", msgCallback);

  
  return (

    <div>
        <h3>Top 10 tweets by retweet count</h3>
        <ul>
            {tweets?.map(item => {
                return <li>{item.twitter_text}  {item.retweet_count} </li>;
            })}
        </ul>
    </div>
  );



}
export default TweetListener;
