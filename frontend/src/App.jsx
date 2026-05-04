import { useState } from "react";
import axios from "axios";

const API = "http://localhost:5000";

function App() {
  const [resumeText, setResumeText] = useState("");
  const [jobDesc, setJobDesc] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("text");

  const analyzeText = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API}/api/analyze/text`, {
        resume_text: resumeText,
        job_description: jobDesc,
      });
      setResult(res.data);
    } catch (err) {
      alert("Error: " + err.message);
    }
    setLoading(false);
  };
  const analyzePDF = async () => {
    setLoading(true);
    try {
      const form = new FormData();
      form.append("resume", file);
      const res = await axios.post(
        `${API}/api/analyze/pdf?job_description=${encodeURIComponent(jobDesc)}`,
        form,
      );
      setResult(res.data);
    } catch (err) {
      alert("Error: " + err.message);
    }
    setLoading(false);
  };
  return (
    <div
      style={{
        maxWidth: 800,
        margin: "0 auto",
        padding: 32,
        fontFamily: "sans-serif",
      }}
    >
      <h1>AI Resume Analyzer</h1>

      <div style={{ marginBottom: 16 }}>
        <button
          onClick={() => setMode("text")}
          style={{
            marginRight: 8,
            padding: "8px 16px",
            background: mode === "text" ? "#2196F3" : "#eee",
            color: mode === "text" ? "white" : "black",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
          }}
        >
          Paste Text
        </button>
        <button
          onClick={() => setMode("pdf")}
          style={{
            padding: "8px 16px",
            background: mode === "pdf" ? "#2196F3" : "#eee",
            color: mode === "pdf" ? "white" : "black",
            border: "none",
            borderRadius: 4,
            cursor: "pointer",
          }}
        >
          Upload PDF
        </button>
      </div>

      {mode === "text" ? (
        <textarea
          placeholder="Paste your resume here..."
          value={resumeText}
          onChange={(e) => setResumeText(e.target.value)}
          style={{
            width: "100%",
            height: 200,
            padding: 8,
            marginBottom: 16,
            boxSizing: "border-box",
          }}
        />
      ) : (
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
          style={{ marginBottom: 16, display: "block" }}
        />
      )}

      <textarea
        placeholder="Paste job description here..."
        value={jobDesc}
        onChange={(e) => setJobDesc(e.target.value)}
        style={{
          width: "100%",
          height: 150,
          padding: 8,
          marginBottom: 16,
          boxSizing: "border-box",
        }}
      />

      <button
        onClick={mode === "text" ? analyzeText : analyzePDF}
        disabled={loading}
        style={{
          padding: "12px 32px",
          background: "#4CAF50",
          color: "white",
          border: "none",
          borderRadius: 4,
          cursor: "pointer",
          fontSize: 16,
        }}
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {result && (
        <div
          style={{
            marginTop: 32,
            padding: 24,
            background: "#f5f5f5",
            borderRadius: 8,
          }}
        >
          <h2>Results</h2>

          <div style={{ marginBottom: 16 }}>
            <h3>Match Score</h3>
            <div
              style={{
                background: "#ddd",
                borderRadius: 8,
                height: 24,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${result.match_score}%`,
                  background:
                    result.match_score > 70
                      ? "#4CAF50"
                      : result.match_score > 50
                        ? "#FF9800"
                        : "#f44336",
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  paddingLeft: 8,
                  color: "white",
                  fontWeight: "bold",
                }}
              >
                {result.match_score}%
              </div>
            </div>
          </div>

          {result.predicted_category && (
            <p>
              <strong>Predicted Category:</strong> {result.predicted_category}
            </p>
          )}

          <div style={{ display: "flex", gap: 32 }}>
            <div>
              <h3 style={{ color: "#4CAF50" }}>✅ Matched Skills</h3>
              {result.matched_skills?.length > 0 ? (
                result.matched_skills.map((s) => (
                  <span
                    key={s}
                    style={{
                      display: "inline-block",
                      background: "#e8f5e9",
                      padding: "4px 8px",
                      borderRadius: 4,
                      margin: 4,
                    }}
                  >
                    {s}
                  </span>
                ))
              ) : (
                <p>None found</p>
              )}
            </div>
            <div>
              <h3 style={{ color: "#f44336" }}>❌ Missing Skills</h3>
              {result.missing_skills?.length > 0 ? (
                result.missing_skills.map((s) => (
                  <span
                    key={s}
                    style={{
                      display: "inline-block",
                      background: "#ffebee",
                      padding: "4px 8px",
                      borderRadius: 4,
                      margin: 4,
                    }}
                  >
                    {s}
                  </span>
                ))
              ) : (
                <p>None missing</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
