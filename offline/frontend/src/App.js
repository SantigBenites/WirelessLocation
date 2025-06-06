import React, { useState, useEffect, useCallback } from 'react';
import './App.css'; // Import the CSS file

// Function to get the current time
const getCurrentTime = () => new Date().toISOString();
const url = "127.0.0.1:5050";  // API base URL

const App = () => {
  const [activeButton, setActiveButton] = useState(null);
  const [selectedValue, setSelectedValue] = useState(null); // To store the selected value from the dictionary
  const [showApiButton, setShowApiButton] = useState(false); // To show the new button that triggers the API call
  const [logs, setLogs] = useState([]); // To store log messages
  const [squareStates, setSquareStates] = useState(Array(10).fill(false)); // To store the state of the squares (false = red, true = green)

  // Dictionary of values for each button (just an example, can be replaced with real data)
  const buttonDict = {
    1: 'Value 1',
    2: 'Value 2',
    3: 'Value 3',
    4: 'Value 4',
    5: 'Value 5',
    6: 'Value 6',
    7: 'Value 7',
    8: 'Value 8',
    9: 'Value 9',
    10: 'Value 10',
  };

  const squareDict = {
    1: 'Pico1',
    2: 'Pico2',
    3: 'Pico3',
    4: 'Pico4',
    5: 'Pico5',
    6: 'Pico6',
    7: 'Pico7',
    8: 'Pico8',
    9: 'Pico9',
    10: 'Pico10',
  };

  const reverse_squareDict = {
    'Pico1' : 1,
    'Pico2' : 2,
    'Pico3' : 3,
    'Pico4' : 4,
    'Pico5' : 5,
    'Pico6' : 6,
    'Pico7' : 7,
    'Pico8' : 8,
    'Pico9' : 9,
    'Pico10' : 10,
  };

  // Handler for when a button is pressed
  const handleButtonClick = (buttonId) => {
    setActiveButton(buttonId);
    setSelectedValue(buttonDict[buttonId + 1]); // Set the selected value from the dictionary
    setShowApiButton(true); // Show the new button to make the API call
  };

  // New button to trigger the API call
  const handleApiButtonClick = async () => {
    const currentTime = getCurrentTime();
    const logMessage = `API call executed with value: ${selectedValue} at time: ${currentTime}`;
    setLogs((prevLogs) => [...prevLogs, logMessage]);

    try {
      // First verify the URL is properly formed
      if (!url) {
        throw new Error("Backend URL is not configured");
      }

      const apiUrl = `http://${url}/update-button-id`;
      setLogs((prevLogs) => [...prevLogs, `Attempting to call: ${apiUrl}`]);

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          buttonId: activeButton + 1, 
          timestamp: currentTime 
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! Status: ${response.status}, Response: ${errorText}`);
      }

      const result = await response.json();
      setLogs((prevLogs) => [...prevLogs, `API call succeeded: ${JSON.stringify(result)}`]);

      // Update the square states based on the backend response
      const newSquareStates = [...squareStates];
      newSquareStates[activeButton] = result.output === 'true'; // Assuming the backend returns 'true' or 'false'
      setSquareStates(newSquareStates);

    } catch (error) {
      const errorMessage = `Network Error: ${error.message}. Please verify:
      - Backend is running at ${url}
      - Both devices are on the same network
      - No firewall blocking port 5050`;
      
      setLogs((prevLogs) => [...prevLogs, errorMessage]);
      console.error("API call failed:", error);
    }
  };

  // Function to stop the script
  const handleStopScriptClick = async () => {
    try {
      const response = await fetch(`http://${url}/stop-script`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const result = await response.json();
        setLogs((prevLogs) => [...prevLogs, `Script stopped: ${result.output}`]);
      } else {
        const errorText = await response.text();
        setLogs((prevLogs) => [...prevLogs, `Failed to stop script: ${errorText}`]);
      }
    } catch (error) {
      setLogs((prevLogs) => [...prevLogs, `Error: ${error.message}`]);
    }
  };


  const makeApiCallForSquare = useCallback(async () => {
    try {
      console.log("hello")
      const response = await fetch(`http://${url}/check-status`);
      console.log("hello2")
      const results = await response.json();
      console.log(results)
      setSquareStates((prevStates) => {
        const newStates = [...prevStates];

        // Iterate over the keys in the results object
        Object.keys(results).forEach((key) => {
          // Use reverse_squareDict to get the correct index for the key
          const index = reverse_squareDict[key] - 1; // Subtract 1 because array indices start at 0
          newStates[index] = results[key]; // Update the corresponding square state
        });
    
        return newStates;
      });
    } catch (error) {
      console.error("Error fetching status for squares:", error);
      setLogs((prevLogs) => [...prevLogs, `Error fetching status for squares: ${error.message}`]);
    }
  }, [reverse_squareDict]);

  // Use useEffect to run API calls every n seconds
  useEffect(() => {
    const interval = setInterval(() => {
      makeApiCallForSquare();
    }, 10000); // 10 seconds
  
    return () => clearInterval(interval); // Cleanup interval on component unmount
  }, [makeApiCallForSquare]);

  // Side panel component
  const SidePanel = () => {
    return (
      <div className="side-panel">
        {squareStates.map((isGreen, index) => (
          <div
            key={index}
            className={`square ${isGreen ? 'green' : 'red'}`}
          >
            {squareDict[index + 1]}
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="app-container">
      <div className="main-content">
        {/* Side Panel */}
        <SidePanel />

        <div className="button-container">
          {Object.keys(buttonDict).map((key, index) => (
            <button
              key={index}
              className={`button ${activeButton === index ? 'active' : ''}`}
              onClick={() => handleButtonClick(index)}
            >
              Button {index + 1}
            </button>
          ))}
        </div>

        {showApiButton && (
          <div>
            <button
              className="api-button"
              onClick={handleApiButtonClick}
            >
              Trigger API with {selectedValue}
            </button>
            <button
              className="stop-button"
              onClick={handleStopScriptClick}
            >
              Stop Script
            </button>
          </div>
        )}
      </div>

      {/* Log display area */}
      <div className="log-container">
        <h4>Log:</h4>
        {logs.map((log, index) => (
          <p key={index} className="log-message">
            {log}
          </p>
        ))}
      </div>
    </div>
  );
};

export default App;