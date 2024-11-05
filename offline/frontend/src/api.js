import axios from "axios";

export const runScript = async (buttonId, timestamp) => {
  const response = await axios.post("http://localhost:5050/run-script", {
    buttonId,
    timestamp,
  });
  return response.data;
};
