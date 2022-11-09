import React, { useState, useEffect } from "react";
import socketIOClient from "socket.io-client";
const ENDPOINT = "http://127.0.0.1:5000";


function App() {
  const [tweets, setTweets] = useState({});


  useEffect(() => {
    const socket = socketIOClient(ENDPOINT);

    socket.emit("my event", {test:'test'}, data => {
    });

    socket.on("my response", data => {
      const _data = JSON.parse(data);
      console.log(_data);
      setTweets(_data);
      setTimeout(function () {
        socket.emit("my event", {test:'test'}, data => {
        });
        }, 5000);

     
    });
  }, []);

  return (
    tweets, setTweets
  );
}

export default App;
