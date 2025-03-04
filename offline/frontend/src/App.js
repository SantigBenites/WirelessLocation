import React, { useState, useEffect, useCallback } from 'react';
import './App.css'; // Import the CSS file

// Function to get the current time
const getCurrentTime = () => new Date().toISOString();
const url = "localhost:5050"; // API base URL

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
    setLogs((prevLogs) => [...prevLogs, logMessage]); // Add the log message to the log state

    try {
      const response = await fetch(`http://${url}/update-button-id`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ buttonId: activeButton + 1, timestamp: currentTime }),
      });

      if (response.ok) {
        const result = await response.json();
        setLogs((prevLogs) => [...prevLogs, `API call succeeded: ${result.output}`]);

        // Update the square states based on the backend response
        const newSquareStates = [...squareStates];
        newSquareStates[activeButton] = result.output === 'true'; // Assuming the backend returns 'true' or 'false'
        setSquareStates(newSquareStates);
      } else {
        const errorText = await response.text();
        setLogs((prevLogs) => [...prevLogs, `API call failed: ${errorText}`]);
      }
    } catch (error) {
      setLogs((prevLogs) => [...prevLogs, `Error: ${error.message}`]);
    }
  };

  // Function to make API calls for each square
  const makeApiCallForSquare = useCallback(async () => {
    try {
      const response = await fetch(`http://${url}/check-status`);
      const results = await response.json();
  
      setSquareStates((prevStates) => {
        const newStates = [...prevStates];
        Object.keys(results).forEach((key, index) => {
          newStates[index] = results[key]; // Update each square based on the ping result
        });
        return newStates;
      });
    } catch (error) {
      console.error("Error fetching status for squares:", error);
      setLogs((prevLogs) => [...prevLogs, `Error fetching status for squares: ${error.message}`]);
    }
  }, []);

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
          <button
            className="api-button"
            onClick={handleApiButtonClick}
          >
            Trigger API with {selectedValue}
          </button>
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