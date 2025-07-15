import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [coordinates, setCoordinates] = useState({});
  const [backendCoordinates, setBackendCoordinates] = useState({});
  const [status, setStatus] = useState({});
  const [activePicos, setActivePicos] = useState({});
  const [scriptRunning, setScriptRunning] = useState(false);
  const [triangleShape, setTriangleShape] = useState('');

  const picoIPs = [
    "10.20.0.111", "10.20.0.112", "10.20.0.113", "10.20.0.114",
    "10.20.0.115", "10.20.0.116", "10.20.0.117", "10.20.0.118",
    "10.20.0.119", "10.20.0.110"
  ];

  useEffect(() => {
    const fetchStatus = () => {
      axios.get("http://localhost:5050/check-status").then(res => {
        setStatus(res.data);
      }).catch(() => setStatus({}));

      axios.get("http://localhost:5050/").then(res => {
        const isRunning = res.data?.status === "running";
        setScriptRunning(isRunning);
        if (res.data?.coordinates) {
          setBackendCoordinates(res.data.coordinates);
        }
      }).catch(() => {
        setScriptRunning(false);
        setBackendCoordinates({});
      });
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleCoordinateChange = (ip, axis, value) => {
    setCoordinates(prev => ({
      ...prev,
      [ip]: {
        ...prev[ip],
        [axis]: value === '' ? '' : Number(value)
      }
    }));
  };

  const handleToggleActive = (ip) => {
    setActivePicos(prev => ({
      ...prev,
      [ip]: !prev[ip]
    }));
  };

  const sendCoordinates = () => {
    const filteredCoords = Object.fromEntries(
      Object.entries(coordinates).filter(([ip]) => activePicos[ip])
    );
    const selectedIps = Object.keys(activePicos).filter(ip => activePicos[ip]);

      axios.post("http://localhost:5050/update-coordinates", {
        ...filteredCoords,
        triangle_shape: triangleShape
      }).then(() => {
        console.log("Coordinates and triangle shape updated");
      });

    axios.post("http://localhost:5050/update-active-picos", { ips: selectedIps}).then(() => {
      console.log("Active Pico list updated");
    });
  };

  const startScript = () => {
    axios.post("http://localhost:5050/start-script").then(() => {
      setScriptRunning(true);
      alert("Script started");
    });
  };

  const stopScript = () => {
    axios.post("http://localhost:5050/stop-script").then(() => {
      setScriptRunning(false);
      alert("Script stopped");
    });
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-bold mb-4">Pico Coordinate Mapper</h1>
      <p className="mb-2 font-semibold">
        Script status: <span className={`px-2 py-1 rounded ${scriptRunning ? 'bg-green-200 text-green-800' : 'bg-red-200 text-red-800'}`}>{scriptRunning ? 'Running' : 'Stopped'}</span>
      </p>
      <table className="table-auto border border-collapse">
        <thead>
          <tr>
            <th className="border px-2">IP</th>
            <th className="border px-2">x</th>
            <th className="border px-2">y</th>
            <th className="border px-2">Status</th>
            <th className="border px-2">Active</th>
          </tr>
        </thead>
        <tbody>
          {picoIPs.map(ip => (
            <tr key={ip}>
              <td className="border px-2">{ip}</td>
              <td className="border px-2">
                <input type="number" value={coordinates[ip]?.x !== undefined ? coordinates[ip].x : ''} onChange={e => handleCoordinateChange(ip, 'x', e.target.value)} className="w-16" />
              </td>
              <td className="border px-2">
                <input type="number" value={coordinates[ip]?.y !== undefined ? coordinates[ip].y : ''} onChange={e => handleCoordinateChange(ip, 'y', e.target.value)} className="w-16" />
              </td>
              <td className="border px-2">
                <span style={{
                  width: '10px', height: '10px', borderRadius: '50%', display: 'inline-block',
                  backgroundColor: status[`Pico${picoIPs.indexOf(ip) + 1}`] ? 'green' : 'red'
                }} />
              </td>
              <td className="border px-2 text-center">
                <input type="checkbox" checked={!!activePicos[ip]} onChange={() => handleToggleActive(ip)} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 flex gap-4">
        <button onClick={sendCoordinates} className="p-2 bg-blue-500 text-white rounded">Submit Coordinates</button>
        <button onClick={startScript} className="p-2 bg-green-500 text-white rounded">Start Script</button>
        {scriptRunning && (
          <button onClick={stopScript} className="p-2 bg-red-500 text-white rounded">Stop Script</button>
        )}
      </div>
      <div>
        <label>Triangle Shape: </label>
        <input
          type="text"
          value={triangleShape}
          onChange={(e) => setTriangleShape(e.target.value)}
          placeholder="Enter shape (e.g. equilateral)"
        />
    </div>
      <div className="mt-6">
        <h2 className="text-lg font-semibold mb-2">Backend Stored Coordinates</h2>
        <ul className="list-disc pl-6">
          {Object.entries(backendCoordinates).map(([ip, coord]) => (
            <li key={ip}>{ip}: (x: {coord.x}, y: {coord.y})</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App;
