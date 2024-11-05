import React, { useState } from "react";
import { runScript } from "./api";

function App() {
  const [selectedButton, setSelectedButton] = useState(null);

  const handleButtonClick = (buttonId) => {
    setSelectedButton(buttonId);
  };

  const handleRunScript = async () => {
    if (selectedButton !== null) {
      const timestamp = new Date().toISOString();
      try {
        const response = await runScript(selectedButton, timestamp);
        console.log(response);
      } catch (error) {
        console.error("Error running script:", error);
      }
    } else {
      alert("Please select a button first.");
    }
  };

  return (
    <div className="App">
      <div style={{ position: "relative", width: "500px", height: "500px" }}>
        <img src="/path/to/your/image.jpg" alt="Overlayed background" style={{ width: "100%", height: "100%" }} />
        {Array.from({ length: 10 }, (_, i) => (
          <button
            key={i}
            onClick={() => handleButtonClick(i + 1)}
            style={{
              position: "absolute",
              top: `${Math.floor(i / 5) * 50 + 50}px`,
              left: `${(i % 5) * 50 + 50}px`,
            }}
          >
            Button {i + 1}
          </button>
        ))}
      </div>
      <button onClick={handleRunScript}>Run Python Script</button>
    </div>
  );
}

export default App;
