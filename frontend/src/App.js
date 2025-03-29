import React, { useState } from "react";
import { PieChart, Pie, Cell, Tooltip, Legend } from "recharts";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#A28BFE", "#FF6F61", "#6B5B95", "#F7CAC9"];

export default function MLlibVisualization() {
  const [lyrics, setLyrics] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [predictedGenre, setPredictedGenre] = useState([]);
  const [error, setError] = useState("");

  const handleLyricsChange = (e) => {
    setLyrics(e.target.value);
    setError(""); // Clear any previous error when user starts typing
  };

  const handlePredict = async () => {
    if (!lyrics.trim()) {
      setError("Please enter some lyrics before predicting.");
      return;
    }
  
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Cache-Control": "no-cache"
        },
        body: JSON.stringify({ lyrics }),
      });
  
      if (!response.ok) {
        throw new Error("Failed to fetch predictions");
      }
  
      const data = await response.json();

      const predictionArray = await Object.entries(data.predictions).map(([name, value]) => ({ name, value }));
      setPredictions(predictionArray);

      const predicted = await predictionArray.reduce((max, item) => (item.value > max.value ? item : max), predictionArray[0]);
      setPredictedGenre(predicted.name);

    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <div className="min-h-screen p-8 flex flex-col items-center">
      <h1 className="text-2xl font-bold mb-6">MLlib Visualization Assignment: Song Lyrics to Genre Prediction</h1>
      <div className="grid grid-cols-[2fr_0.1fr_2fr] gap-8 w-full max-w-5xl">
        {/* Left Column */}
        <div className="flex flex-col space-y-4">
          <textarea
            className="required:border-red-500 border rounded p-2 w-full h-80"
            placeholder="Paste song lyrics here..."
            value={lyrics}
            onChange={handleLyricsChange}
          />
          {error && <p className="error">{error}</p>}
          <div className="flex space-x-4">
            <button onClick={handlePredict} className="bg-green-500 text-white self-end px-4 font-medium py-2 hover:bg-green-600 rounded ml-auto">Predict</button>
          </div>
        </div>

        <div className="w-px bg-gray-300"></div>

        {/* Right Column */}
        <div className="flex flex-col items-center">
          <PieChart width={480} height={480}>
            <Pie data={predictions} cx={240} cy={200} outerRadius={200} fill="#8884d8" dataKey="value" nameKey="name">
              {predictions.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
          <div className="mt-4 p-2 border rounded w-full text-center">
            <strong className="text-center">Predicted Genre:</strong> {predictedGenre || "N/A"}
          </div>
        </div>
      </div>
    </div>
  );
}