import React, { useState } from 'react';

// Function to get the current time
const getCurrentTime = () => new Date().toISOString();
const image_source = process.env.PUBLIC_URL + '/fcul_c6.png';

const App = () => {
  const [activeButton, setActiveButton] = useState(null);
  const [selectedValue, setSelectedValue] = useState(null); // To store the selected value from the dictionary
  const [showApiButton, setShowApiButton] = useState(false); // To show the new button that triggers the API call

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
    { left: '10%', top: '30%', xOffset: 150, yOffset: 155 },
    { left: '10%', top: '30%', xOffset: 205, yOffset: 155 },
    { left: '10%', top: '30%', xOffset: 265, yOffset: 155 },
    { left: '10%', top: '30%', xOffset: 320, yOffset: 155 },
    { left: '10%', top: '30%', xOffset: 380, yOffset: 155 },
    { left: '10%', top: '30%', xOffset: 435, yOffset: 155 },
    { left: '10%', top: '30%', xOffset: 495, yOffset: 155 },
    { left: '10%', top: '30%', xOffset: 550, yOffset: 155 }
  ];

  // New button to trigger the API call
  const handleApiButtonClick = async () => {
    const currentTime = getCurrentTime();
    console.log('API call executed with value:', selectedValue, 'at time ', currentTime);
    const response = await fetch('http://localhost:5050/run-script', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ value: selectedValue, time: currentTime }),
    });
    if (response.ok) {
      console.log('API call executed successfully');
    } else {
      console.error('Error running API call');
    }
  };

  return (
    <div
      style={{
        position: 'relative',
        width: '60%',
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
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
  );
};

export default App;
