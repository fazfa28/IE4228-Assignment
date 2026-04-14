import { useState, useRef, useEffect } from "react";

const MOCK_GALLERY = [
  { id: 1, name: "P_Fathiah", status: "enrolled", image: "/static/gallery/P_Fathiah.jpg" },
  { id: 2, name: "P_Fazil", status: "enrolled", image: "/static/gallery/P_Fazil.jpg" },
  { id: 3, name: "P_Jasper", status: "enrolled", image: "/static/gallery/P_Jasper.jpg" },
  { id: 4, name: "P_Joe", status: "enrolled", image: "/static/gallery/P_Joe.jpg" },
  { id: 5, name: "P_Mikail", status: "enrolled", image: "/static/gallery/P_Mikail.jpg" },
  { id: 6, name: "P_Roy", status: "enrolled", image: "/static/gallery/P_Roy.jpg" },
  { id: 7, name: "P_Syarah", status: "enrolled", image: "/static/gallery/P_Syarah.jpg" },
];

const MOCK_LOGS = [
  { time: "09:41:02", event: "System initialized", type: "info" },
  { time: "09:41:03", event: "Haar cascade loaded", type: "info" },
  { time: "09:41:04", event: "PCA model loaded — 50 components", type: "info" },
  { time: "09:41:05", event: "Gallery: 7 persons enrolled", type: "success" },
  { time: "09:41:06", event: "Camera stream active", type: "success" },
];

function ScanLine() {
  return (
    <div style={{
      position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
      pointerEvents: "none", overflow: "hidden", borderRadius: "4px",
    }}>
      <div style={{
        position: "absolute", left: 0, right: 0, height: "2px",
        background: "linear-gradient(90deg, transparent, #00ff88, transparent)",
        animation: "scan 3s linear infinite",
        boxShadow: "0 0 12px #00ff88",
      }} />
    </div>
  );
}

function CornerBrackets({ color = "#00ff88", size = 16, thickness = 2 }) {
  const s = { position: "absolute", width: size, height: size };
  const b = `${thickness}px solid ${color}`;
  return (
    <>
      <div style={{ ...s, top: 0, left: 0, borderTop: b, borderLeft: b }} />
      <div style={{ ...s, top: 0, right: 0, borderTop: b, borderRight: b }} />
      <div style={{ ...s, bottom: 0, left: 0, borderBottom: b, borderLeft: b }} />
      <div style={{ ...s, bottom: 0, right: 0, borderBottom: b, borderRight: b }} />
    </>
  );
}

function DetectionBox({ label, confidence, color }) {
  return (
    <div style={{
      position: "absolute",
      top: "28%", left: "22%",
      width: "56%", height: "52%",
      border: `2px solid ${color}`,
      boxShadow: `0 0 0 1px ${color}33, inset 0 0 20px ${color}11`,
      borderRadius: "2px",
      pointerEvents: "none",
    }}>
      <CornerBrackets color={color} size={14} thickness={2} />
      <div style={{
        position: "absolute", bottom: "-28px", left: "0",
        background: color,
        color: "#000",
        fontSize: "11px",
        fontFamily: "'Courier New', monospace",
        fontWeight: "700",
        padding: "2px 8px",
        whiteSpace: "nowrap",
        letterSpacing: "0.05em",
      }}>
        {label} · {confidence}%
      </div>
    </div>
  );
}

export default function FaceRecognitionGUI() {
  const [isRunning, setIsRunning] = useState(false);
  const [activeTab, setActiveTab] = useState("live");
  const [detectionResult, setDetectionResult] = useState(null);
  const [logs, setLogs] = useState(MOCK_LOGS);
  const [frameCount, setFrameCount] = useState(0);
  const [fps, setFps] = useState(0);
  const [algorithm, setAlgorithm] = useState("PCA+LDA");
  const [threshold, setThreshold] = useState(3500);
  const [pcaComponents, setPcaComponents] = useState(50);
  const logsRef = useRef(null);

  const mockDetections = [
    { name: "P_Jasper", confidence: 94, color: "#00ff88" },
    { name: "P_Roy", confidence: 87, color: "#00ff88" },
    { name: "UNKNOWN", confidence: 31, color: "#ff4444" },
    { name: "P_Fazil", confidence: 91, color: "#00ff88" },
  ];

  useEffect(() => {
    if (!isRunning) return;
    let frame = frameCount;
    const fpsInterval = setInterval(() => {
      frame += Math.floor(Math.random() * 5) + 25;
      setFrameCount(frame);
      setFps(Math.floor(Math.random() * 8) + 22);
    }, 1000);

    const detectionInterval = setInterval(() => {
      const pick = mockDetections[Math.floor(Math.random() * mockDetections.length)];
      setDetectionResult(pick);

      const now = new Date();
      const timeStr = now.toTimeString().slice(0, 8);
      const logEntry = {
        time: timeStr,
        event: pick.name === "UNKNOWN"
          ? `Unknown face detected — dist > threshold`
          : `Recognized: ${pick.name} (conf: ${pick.confidence}%)`,
        type: pick.name === "UNKNOWN" ? "warn" : "success",
      };
      setLogs(prev => [...prev.slice(-19), logEntry]);
    }, 2500);

    return () => {
      clearInterval(fpsInterval);
      clearInterval(detectionInterval);
    };
  }, [isRunning]);

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [logs]);

  const toggleCamera = () => {
    if (!isRunning) {
      setIsRunning(true);
      setLogs(prev => [...prev, {
        time: new Date().toTimeString().slice(0, 8),
        event: `Recognition started — Algorithm: ${algorithm}`,
        type: "info",
      }]);
    } else {
      setIsRunning(false);
      setDetectionResult(null);
      setLogs(prev => [...prev, {
        time: new Date().toTimeString().slice(0, 8),
        event: "Recognition stopped",
        type: "warn",
      }]);
    }
  };

  const logColor = { info: "#4af", success: "#0f8", warn: "#fa0", error: "#f44" };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0b0d",
      color: "#c8cdd5",
      fontFamily: "'Courier New', monospace",
      padding: "0",
      display: "flex",
      flexDirection: "column",
    }}>
      <style>{`
        @keyframes scan {
          0% { top: -2px; }
          100% { top: 100%; }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #111; }
        ::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }
        .tab-btn { transition: all 0.15s; }
        .tab-btn:hover { background: #1a1d22 !important; }
        .ctrl-btn { transition: all 0.15s; cursor: pointer; }
        .ctrl-btn:hover { filter: brightness(1.2); }
      `}</style>

      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "12px 24px",
        borderBottom: "1px solid #1e2228",
        background: "#0d0f12",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <div style={{
            width: "32px", height: "32px",
            border: "2px solid #00ff88",
            display: "grid", placeItems: "center",
            position: "relative",
          }}>
            <CornerBrackets color="#00ff88" size={7} thickness={1.5} />
            <div style={{ width: "10px", height: "10px", background: "#00ff88", borderRadius: "50%",
              animation: isRunning ? "blink 1.2s ease-in-out infinite" : "none" }} />
          </div>
          <div>
            <div style={{ fontSize: "14px", fontWeight: "700", letterSpacing: "0.15em", color: "#e8ecf0" }}>
              FACERECOG · SYS
            </div>
            <div style={{ fontSize: "10px", color: "#4a5260", letterSpacing: "0.1em" }}>
              IE4228 — NON-DEEP-LEARNING PIPELINE v1.0
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: "24px", alignItems: "center" }}>
          {[
            { label: "FPS", value: isRunning ? fps : "—" },
            { label: "FRAMES", value: isRunning ? frameCount : "—" },
            { label: "GALLERY", value: MOCK_GALLERY.length },
            { label: "STATUS", value: isRunning ? "LIVE" : "IDLE" },
          ].map(({ label, value }) => (
            <div key={label} style={{ textAlign: "right" }}>
              <div style={{ fontSize: "9px", color: "#4a5260", letterSpacing: "0.12em" }}>{label}</div>
              <div style={{
                fontSize: "14px", fontWeight: "700",
                color: label === "STATUS" ? (isRunning ? "#00ff88" : "#4a5260") : "#c8cdd5",
                animation: label === "STATUS" && isRunning ? "blink 2s ease-in-out infinite" : "none",
              }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Layout */}
      <div style={{ display: "flex", flex: 1, gap: "0", overflow: "hidden" }}>

        {/* Left: Camera Feed */}
        <div style={{
          flex: "1 1 0",
          display: "flex", flexDirection: "column",
          borderRight: "1px solid #1e2228",
          minWidth: 0,
        }}>
          {/* Tab bar */}
          <div style={{
            display: "flex",
            borderBottom: "1px solid #1e2228",
            background: "#0d0f12",
          }}>
            {["live", "gallery", "metrics"].map(tab => (
              <button
                key={tab}
                className="tab-btn"
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: "10px 20px",
                  background: activeTab === tab ? "#13161a" : "transparent",
                  border: "none",
                  borderBottom: activeTab === tab ? "2px solid #00ff88" : "2px solid transparent",
                  color: activeTab === tab ? "#e8ecf0" : "#4a5260",
                  fontSize: "10px",
                  letterSpacing: "0.15em",
                  cursor: "pointer",
                  textTransform: "uppercase",
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Camera view */}
          {activeTab === "live" && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", padding: "16px", gap: "12px" }}>
              {/* Video frame */}
              <div style={{
                position: "relative",
                background: "#070809",
                border: "1px solid #1e2228",
                borderRadius: "4px",
                flex: 1,
                minHeight: "300px",
                overflow: "hidden",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <CornerBrackets color="#1e2228" size={20} thickness={1} />
                {isRunning && <ScanLine />}

                {!isRunning ? (
                  <div style={{ textAlign: "center", color: "#2a2f38" }}>
                    <div style={{ fontSize: "48px", marginBottom: "8px" }}>⬡</div>
                    <div style={{ fontSize: "11px", letterSpacing: "0.15em" }}>CAMERA OFFLINE</div>
                    <div style={{ fontSize: "10px", marginTop: "4px", color: "#1e2228" }}>
                      Press START to begin
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Simulated video noise grid */}
                    <div style={{
                      position: "absolute", inset: 0,
                      backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.015) 2px, rgba(0,255,136,0.015) 4px)`,
                    }} />
                    <div style={{
                      position: "absolute", inset: 0,
                      backgroundImage: `repeating-linear-gradient(90deg, transparent, transparent 2px, rgba(0,255,136,0.008) 2px, rgba(0,255,136,0.008) 4px)`,
                    }} />

                    {/* Live indicator */}
                    <div style={{
                      position: "absolute", top: "10px", left: "10px",
                      display: "flex", alignItems: "center", gap: "6px",
                      fontSize: "10px", letterSpacing: "0.1em",
                      color: "#00ff88",
                    }}>
                      <div style={{
                        width: "6px", height: "6px",
                        background: "#00ff88", borderRadius: "50%",
                        animation: "blink 1s ease-in-out infinite",
                        boxShadow: "0 0 6px #00ff88",
                      }} />
                      LIVE
                    </div>

                    {/* Detection overlay */}
                    {detectionResult && (
                      <DetectionBox
                        label={detectionResult.name}
                        confidence={detectionResult.confidence}
                        color={detectionResult.color}
                      />
                    )}

                    {/* Camera feed placeholder text */}
                    <div style={{
                      color: "#1a1e24",
                      fontSize: "10px",
                      letterSpacing: "0.1em",
                      userSelect: "none",
                    }}>
                      WEBCAM FEED
                    </div>
                  </>
                )}
              </div>

              {/* Detection result panel */}
              <div style={{
                background: "#0d0f12",
                border: "1px solid #1e2228",
                borderRadius: "4px",
                padding: "12px 16px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                minHeight: "56px",
              }}>
                {detectionResult && isRunning ? (
                  <div style={{ display: "flex", alignItems: "center", gap: "16px", animation: "fadeIn 0.3s ease" }}>
                    <div style={{
                      fontSize: "9px", letterSpacing: "0.12em", color: "#4a5260",
                    }}>IDENTITY</div>
                    <div style={{
                      fontSize: "22px", fontWeight: "700", letterSpacing: "0.08em",
                      color: detectionResult.color,
                      textShadow: `0 0 20px ${detectionResult.color}66`,
                    }}>
                      {detectionResult.name}
                    </div>
                    <div style={{
                      padding: "2px 10px",
                      border: `1px solid ${detectionResult.color}`,
                      color: detectionResult.color,
                      fontSize: "11px",
                      borderRadius: "2px",
                    }}>
                      CONF {detectionResult.confidence}%
                    </div>
                    <div style={{
                      padding: "2px 10px",
                      background: detectionResult.name === "UNKNOWN" ? "#ff444422" : "#00ff8822",
                      color: detectionResult.name === "UNKNOWN" ? "#ff4444" : "#00ff88",
                      fontSize: "10px",
                      borderRadius: "2px",
                      letterSpacing: "0.1em",
                    }}>
                      {detectionResult.name === "UNKNOWN" ? "NOT IN GALLERY" : "MATCH FOUND"}
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: "11px", color: "#2a2f38", letterSpacing: "0.1em" }}>
                    — NO DETECTION —
                  </div>
                )}
                <div style={{ fontSize: "10px", color: "#2a2f38", letterSpacing: "0.08em" }}>
                  {isRunning ? `ALGO: ${algorithm}` : ""}
                </div>
              </div>
            </div>
          )}

          {/* Gallery tab */}
          {activeTab === "gallery" && (
            <div style={{ flex: 1, padding: "16px", overflowY: "auto" }}>
              <div style={{ fontSize: "10px", color: "#4a5260", letterSpacing: "0.15em", marginBottom: "12px" }}>
                ENROLLED PERSONS — {MOCK_GALLERY.length} TOTAL
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: "10px" }}>
                {MOCK_GALLERY.map(person => (
                  <div key={person.id} style={{
                    background: "#0d0f12",
                    border: "1px solid #1e2228",
                    borderRadius: "4px",
                    padding: "16px 12px",
                    textAlign: "center",
                    position: "relative",
                  }}>
                    <CornerBrackets color="#1e2228" size={8} thickness={1} />
                    <div style={{
                      width: "48px", height: "48px",
                      background: "#1a1d22",
                      borderRadius: "2px",
                      margin: "0 auto 10px",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: "20px",
                      border: "1px solid #2a2f38",
                      overflow: "hidden",
                    }}>
                      <img src={person.image} alt={person.name} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                    </div>
                    <div style={{ fontSize: "10px", color: "#c8cdd5", letterSpacing: "0.05em", wordBreak: "break-all" }}>
                      {person.name.replace("P_", "")}
                    </div>
                    <div style={{
                      fontSize: "9px", color: "#00ff88", marginTop: "4px",
                      letterSpacing: "0.1em",
                    }}>● ENROLLED</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Metrics tab */}
          {activeTab === "metrics" && (
            <div style={{ flex: 1, padding: "16px", overflowY: "auto" }}>
              <div style={{ fontSize: "10px", color: "#4a5260", letterSpacing: "0.15em", marginBottom: "16px" }}>
                PERFORMANCE METRICS
              </div>
              {[
                { label: "PCA Components (k)", value: "50", sub: "of 65 max" },
                { label: "Train Accuracy", value: "100%", sub: "65 samples" },
                { label: "Test Accuracy", value: "52.2%", sub: "23 samples" },
                { label: "Image Size", value: "90×90 px", sub: "grayscale" },
                { label: "Feature Dim (raw)", value: "8,100", sub: "flattened" },
                { label: "Feature Dim (PCA)", value: "50", sub: "reduced" },
                { label: "Classifier", value: "1-NN", sub: "L2 distance" },
                { label: "Normalization", value: "CLAHE", sub: "clipLimit=2.0" },
                { label: "Detection", value: "Viola-Jones", sub: "Haar Cascade" },
              ].map(({ label, value, sub }) => (
                <div key={label} style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "10px 0",
                  borderBottom: "1px solid #13161a",
                }}>
                  <div style={{ fontSize: "10px", color: "#4a5260", letterSpacing: "0.05em" }}>{label}</div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: "13px", color: "#e8ecf0", fontWeight: "700" }}>{value}</div>
                    <div style={{ fontSize: "9px", color: "#2a2f38" }}>{sub}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right panel */}
        <div style={{
          width: "280px",
          display: "flex", flexDirection: "column",
          borderLeft: "1px solid #1e2228",
          flexShrink: 0,
        }}>
          {/* Controls */}
          <div style={{
            padding: "16px",
            borderBottom: "1px solid #1e2228",
          }}>
            <div style={{ fontSize: "10px", color: "#4a5260", letterSpacing: "0.15em", marginBottom: "12px" }}>
              CONTROLS
            </div>

            {/* Start/Stop */}
            <button
              className="ctrl-btn"
              onClick={toggleCamera}
              style={{
                width: "100%",
                padding: "12px",
                background: isRunning ? "#1a0808" : "#001a0d",
                border: `1px solid ${isRunning ? "#ff4444" : "#00ff88"}`,
                color: isRunning ? "#ff4444" : "#00ff88",
                fontSize: "11px",
                letterSpacing: "0.2em",
                borderRadius: "2px",
                marginBottom: "12px",
                position: "relative",
              }}
            >
              {isRunning ? "◼ STOP RECOGNITION" : "▶ START RECOGNITION"}
            </button>

            {/* Algorithm selector */}
            <div style={{ marginBottom: "12px" }}>
              <div style={{ fontSize: "9px", color: "#4a5260", letterSpacing: "0.1em", marginBottom: "6px" }}>
                ALGORITHM
              </div>
              <div style={{ display: "flex", gap: "6px" }}>
                {["PCA", "PCA+LDA"].map(alg => (
                  <button
                    key={alg}
                    className="ctrl-btn"
                    onClick={() => setAlgorithm(alg)}
                    style={{
                      flex: 1,
                      padding: "7px",
                      background: algorithm === alg ? "#00ff8811" : "transparent",
                      border: `1px solid ${algorithm === alg ? "#00ff88" : "#1e2228"}`,
                      color: algorithm === alg ? "#00ff88" : "#4a5260",
                      fontSize: "10px",
                      letterSpacing: "0.08em",
                      borderRadius: "2px",
                    }}
                  >
                    {alg}
                  </button>
                ))}
              </div>
            </div>

            {/* Threshold slider */}
            <div style={{ marginBottom: "12px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                <div style={{ fontSize: "9px", color: "#4a5260", letterSpacing: "0.1em" }}>
                  UNKNOWN THRESHOLD
                </div>
                <div style={{ fontSize: "10px", color: "#c8cdd5" }}>{threshold.toLocaleString()}</div>
              </div>
              {/* Removed slider */}
            </div>

            {/* PCA Components slider */}
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
                <div style={{ fontSize: "9px", color: "#4a5260", letterSpacing: "0.1em" }}>
                  PCA COMPONENTS
                </div>
                <div style={{ fontSize: "10px", color: "#c8cdd5" }}>k = {pcaComponents}</div>
              </div>
              {/* Removed slider */}
            </div>
          </div>

          {/* Pipeline display */}
          <div style={{ padding: "16px", borderBottom: "1px solid #1e2228" }}>
            <div style={{ fontSize: "10px", color: "#4a5260", letterSpacing: "0.15em", marginBottom: "12px" }}>
              PIPELINE
            </div>
            {[
              { n: "01", label: "VIOLA-JONES DETECTION", active: isRunning },
              { n: "02", label: "FACE ALIGNMENT", active: isRunning },
              { n: "03", label: "CLAHE NORMALIZATION", active: isRunning },
              { n: "04", label: `PCA (k=${pcaComponents})`, active: isRunning },
              { n: "05", label: algorithm === "PCA+LDA" ? "LDA PROJECTION" : "—", active: isRunning && algorithm === "PCA+LDA" },
              { n: "06", label: "1-NN CLASSIFIER", active: isRunning },
            ].map(({ n, label, active }) => (
              <div key={n} style={{
                display: "flex", alignItems: "center", gap: "10px",
                padding: "6px 0",
                opacity: label === "—" ? 0.2 : 1,
              }}>
                <div style={{
                  fontSize: "9px", color: active && label !== "—" ? "#00ff88" : "#2a2f38",
                  animation: active && label !== "—" ? "blink 2s ease-in-out infinite" : "none",
                  minWidth: "8px",
                }}>
                  {active && label !== "—" ? "●" : "○"}
                </div>
                <div style={{ fontSize: "9px", color: "#2a2f38", minWidth: "18px" }}>{n}</div>
                <div style={{
                  fontSize: "10px",
                  color: active && label !== "—" ? "#c8cdd5" : "#2a2f38",
                  letterSpacing: "0.05em",
                }}>
                  {label}
                </div>
              </div>
            ))}
          </div>

          {/* Event log */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{
              padding: "12px 16px 8px",
              fontSize: "10px", color: "#4a5260", letterSpacing: "0.15em",
              borderBottom: "1px solid #1e2228",
            }}>
              EVENT LOG
            </div>
            <div
              ref={logsRef}
              style={{
                flex: 1,
                overflowY: "auto",
                padding: "8px 16px",
              }}
            >
              {logs.map((log, i) => (
                <div key={i} style={{
                  display: "flex", gap: "8px",
                  padding: "4px 0",
                  borderBottom: "1px solid #0d0f12",
                  animation: i === logs.length - 1 ? "fadeIn 0.2s ease" : "none",
                }}>
                  <div style={{ fontSize: "9px", color: "#2a2f38", whiteSpace: "nowrap", minWidth: "52px" }}>
                    {log.time}
                  </div>
                  <div style={{
                    fontSize: "9px",
                    color: logColor[log.type] || "#c8cdd5",
                    lineHeight: "1.5",
                  }}>
                    {log.event}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
