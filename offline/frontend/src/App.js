import React, { useState } from 'react';

// Function to get the current time
const getCurrentTime = () => new Date().toISOString();
const image_source = process.env.PUBLIC_URL + '/fcul_c6.png';

const App = () => {
  const [activeButton, setActiveButton] = useState(null);
  const [selectedValue, setSelectedValue] = useState(null); // To store the selected value from the dictionary
  const [showApiButton, setShowApiButton] = useState(false); // To show the new button that triggers the API call
  const [logs, setLogs] = useState([]); // To store log messages

  // Define button size (in pixels)
  const buttonSize = { width: 30, height: 230 };

  // Dictionary of values for each button (just an example, can be replaced with real data)
  const buttonDict = {
    0: 'Value 1',
    1: 'Value 2',
    2: 'Value 3',
    3: 'Value 4',
    4: 'Value 5',
    5: 'Value 6',
    6: 'Value 7',
    7: 'Value 8',
    8: 'Value 9',
    9: 'Value 10',
  };

  // Handler for when a button is pressed
  const handleButtonClick = (buttonId) => {
    setActiveButton(buttonId);
    setSelectedValue(buttonDict[buttonId]); // Set the selected value from the dictionary
    setShowApiButton(true); // Show the new button to make the API call
  };

  // Button positions with offsets
  const buttonPositions = [
    { left: '10%', top: '30%', xOffset: 40, yOffset: 55 },
    { left: '10%', top: '30%', xOffset: 90, yOffset: 55 },
    { left: '10%', top: '30%', xOffset: 145, yOffset: 55 },
    { left: '10%', top: '30%', xOffset: 195, yOffset: 55 },
    { left: '10%', top: '30%', xOffset: 250, yOffset: 55 },
    { left: '10%', top: '30%', xOffset: 305, yOffset: 55 },
    { left: '10%', top: '30%', xOffset: 360, yOffset: 55 },
    { left: '10%', top: '30%', xOffset: 410, yOffset: 55 }
  ];

  // New button to trigger the API call
  const handleApiButtonClick = async () => {
    const currentTime = getCurrentTime();
    const logMessage = `API call executed with value: ${selectedValue} at time: ${currentTime}`;
    setLogs((prevLogs) => [...prevLogs, logMessage]); // Add the log message to the log state

    try {
        const response = await fetch('http://10.10.5.23:5050/update-button-id', { // Use the correct endpoint URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json', // Ensure this matches what Flask expects
            },
            body: JSON.stringify({ buttonId: activeButton + 1, timestamp: currentTime }),
        });

        if (response.ok) {
            const result = await response.json();
            setLogs((prevLogs) => [...prevLogs, `API call succeeded: ${result.output}`]);
        } else {
            const errorText = await response.text();
            setLogs((prevLogs) => [...prevLogs, `API call failed: ${errorText}`]);
        }
    } catch (error) {
        setLogs((prevLogs) => [...prevLogs, `Error: ${error.message}`]);
    }
};


  return (
    <div
      style={{
        position: 'relative',
        width: '60%',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <div
        style={{
          position: 'relative',
          width: '100%',
          height: '80%',
        }}
      >
        <img
          src={image_source}
          alt="Background"
          style={{
            width: '80%',
            height: 'auto',
            objectFit: 'cover',
          }}
        />

        {buttonPositions.map((position, index) => (
          <button
            key={index}
            style={{
              position: 'absolute',
              left: `calc(${position.left} - ${buttonSize.width / 2}px + ${position.xOffset}px)`,
              top: `calc(${position.top} - ${buttonSize.height / 2}px + ${position.yOffset}px)`,
              width: `${buttonSize.width}px`,
              height: `${buttonSize.height}px`,
              backgroundColor: activeButton === index ? 'blue' : 'transparent',
              border: '1px solid black',
              padding: 0,
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              color: activeButton === index ? 'white' : 'black',
            }}
            onClick={() => handleButtonClick(index)}
          >
            Button {index + 1}
          </button>
        ))}

        {showApiButton && (
          <button
            style={{
              position: 'absolute',
              top: '90%',
              left: '50%',
              transform: 'translateX(-50%)',
              padding: '10px',
              backgroundColor: 'green',
              color: 'white',
              cursor: 'pointer',
              border: 'none',
            }}
            onClick={handleApiButtonClick}
          >
            Trigger API with {selectedValue}
          </button>
        )}
      </div>

      {/* Log display area */}
      <div
        style={{
          width: '100%',
          height: '20%',
          overflowY: 'auto',
          border: '1px solid black',
          marginTop: '10px',
          padding: '10px',
          backgroundColor: '#f9f9f9',
        }}
      >
        <h4>Log:</h4>
        {logs.map((log, index) => (
          <p 
            key={index} 
            style={{ 
              margin: 0, 
              fontSize: '12px' // Set a smaller font size here
            }}
          >
            {log}
          </p>
        ))}
      </div>

    </div>
  );
};

export default App;
