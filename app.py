
from flask import Flask, jsonify, render_template_string, request, Response
import sqlite3
import os
import math
import joblib
import csv
import io
from datetime import datetime, timezone

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "signalbridge.db")

MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_model.joblib")
AI_MODEL = None
AI_FEATURES = []

try:
    saved_model = joblib.load(MODEL_PATH)
    AI_MODEL = saved_model["model"]
    AI_FEATURES = saved_model["features"]
    print("SignalBridge AI model loaded successfully.")
except Exception as exc:
    print(f"WARNING: AI model unavailable: {exc}")



def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                severity INTEGER,
                description TEXT,
                redaction_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                details TEXT,
                created_at TEXT
            )
        """)


def audit_event(action, details):
    try:
        with connect_db() as conn:
            conn.execute("""
                INSERT INTO audit_log (action, details, created_at)
                VALUES (?, ?, ?)
            """, (action, str(details)[:1000], now()))
    except sqlite3.Error as exc:
        app.logger.error("Audit logging failed: %s", exc)


def now():
    return datetime.now(timezone.utc).isoformat()


def severity_number(value):
    if isinstance(value, (int, float)):
        number = int(value)
        if number in (1, 2, 3, 4):
            return number
        raise ValueError("Severity number must be from 1 to 4.")

    mapping = {
        "LOW": 1,
        "MEDIUM": 2,
        "MODERATE": 2,
        "HIGH": 3,
        "CRITICAL": 4
    }
    result = mapping.get(str(value).strip().upper())
    if result is None:
        raise ValueError("Use LOW, MEDIUM, HIGH or CRITICAL.")
    return result


def save_report(data):
    category = str(
        data.get("type") or data.get("category") or "General"
    ).strip()[:100]

    area = str(data.get("area") or "Unspecified").strip()[:150]
    details = str(
        data.get("details") or data.get("description") or ""
    ).strip()[:3000]

    if not details:
        raise ValueError("Report details cannot be empty.")

    severity = severity_number(data.get("severity", "MEDIUM"))
    description = f"Area: {area}\n{details}"
    timestamp = now()

    with connect_db() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(incidents)")
        }

        # Adapt to the existing database without requiring its deletion.
        values = {}
        if "category" in columns:
            values["category"] = category
        if "type" in columns:
            values["type"] = category
        if "area" in columns:
            values["area"] = area
        if "severity" in columns:
            values["severity"] = severity
        if "description" in columns:
            values["description"] = description
        if "details" in columns:
            values["details"] = details
        if "redaction_count" in columns:
            values["redaction_count"] = 0
        if "created_at" in columns:
            values["created_at"] = timestamp
        if "time" in columns:
            values["time"] = timestamp

        if not values:
            raise RuntimeError(
                "The incidents table has no supported columns."
            )

        names = list(values)
        placeholders = ", ".join("?" for _ in names)
        sql = (
            f"INSERT INTO incidents ({', '.join(names)}) "
            f"VALUES ({placeholders})"
        )
        cursor = conn.execute(sql, [values[name] for name in names])
        report_id = cursor.lastrowid

    audit_event("incident_reported", f"Report ID {report_id}")
    return {
        "id": report_id,
        "area": area,
        "type": category,
        "severity": severity,
        "details": details,
        "created_at": timestamp
    }


def get_reports():
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT 500"
        ).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["type"] = item.get("type") or item.get("category") or "General"
        item["details"] = (
            item.get("details") or item.get("description") or ""
        )
        item["area"] = item.get("area") or "Unspecified"
        item["severity_label"] = {
            1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"
        }.get(item.get("severity"), str(item.get("severity", "UNKNOWN")))
        results.append(item)

    return results


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SignalBridge | Disaster Intelligence</title>
<style>
:root {
  --bg:#0b1220; --panel:#141f31; --panel2:#1b2a40;
  --text:#eef4ff; --muted:#9eafc8; --green:#60e6c1;
  --border:#293a51; --red:#ff7777; --yellow:#ffd479;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:15px Arial,sans-serif}
header{padding:24px 5%;background:#101a2a;border-bottom:1px solid var(--border)}
header h1{margin:0;color:var(--green);font-size:28px}
header p{color:var(--muted);margin-bottom:0}
main{max-width:1250px;margin:auto;padding:24px}
nav{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:22px}
button,a.btn{border:1px solid var(--border);background:var(--panel2);
color:var(--text);padding:11px 15px;border-radius:8px;cursor:pointer;
text-decoration:none;font-weight:600}
button:hover,a.btn:hover{border-color:var(--green)}
button.primary{background:var(--green);color:#071721;border:0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:15px}
.card{background:var(--panel);border:1px solid var(--border);
border-radius:13px;padding:19px;margin-bottom:16px}
.card h2{font-size:18px;margin-top:0}
.metric{font-size:30px;font-weight:bold;color:var(--green)}
.muted{color:var(--muted);line-height:1.6}
label{display:block;color:var(--muted);margin:12px 0 6px}
input,select,textarea{width:100%;padding:12px;background:#0b1525;
border:1px solid var(--border);border-radius:8px;color:var(--text)}
textarea{min-height:90px;resize:vertical}
.result{padding:13px;background:#0b1525;border-radius:8px;
margin-top:12px;white-space:pre-wrap;overflow-wrap:anywhere}
.tag{display:inline-block;padding:5px 9px;border-radius:6px;
background:var(--panel2);font-size:12px}
.high{color:var(--red)}.medium{color:var(--yellow)}.low{color:var(--green)}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:11px;border-bottom:1px solid var(--border);
vertical-align:top}
.table-wrap{overflow-x:auto}
section{scroll-margin-top:15px}
footer{text-align:center;padding:24px;color:var(--muted)}
</style>
</head>
<body>
<header>
  <h1>◈ SignalBridge</h1>
  <p>AI-assisted disaster intelligence, community reporting and emergency planning</p>
</header>
<main>
<nav>
  <a class="btn" href="#overview">Overview</a>
  <a class="btn" href="#warning">Early Warning</a>
  <a class="btn" href="#resources">Resource Allocation</a>
  <a class="btn" href="#shelters">Shelter Finder</a>
  <a class="btn" href="#report">Community Report</a>
  <a class="btn" href="#history">Incident History</a>
  <a class="btn" href="#privacy">Privacy Protection</a>
</nav>

<section id="overview" class="grid">
  <div class="card"><h2>Incident Reports</h2>
    <div id="total" class="metric">--</div><p class="muted">Saved reports</p></div>
  <div class="card"><h2>High-Priority Incidents</h2>
    <div id="highCount" class="metric">--</div><p class="muted">High and critical reports</p></div>
  <div class="card"><h2>System Status</h2>
    <div id="status" class="metric">Checking</div>
    <p class="muted">Application and database connection</p></div>
</section>

<section id="warning" class="card">
<h2>1. Early Warning Centre</h2>
<p class="muted">Enter observed conditions to obtain a transparent screening estimate.
This prototype is not an official forecast or validated AI prediction.</p>
<div class="grid">
<div><label>Rainfall (mm)</label><input id="rain" type="number" min="0" value="30"></div>
<div><label>Wind speed (km/h)</label><input id="wind" type="number" min="0" value="20"></div>
<div><label>Humidity (%)</label><input id="humidity" type="number" min="0" max="100" value="65"></div>
<div><label>River rise (cm)</label><input id="river" type="number" min="0" value="5"></div>
<div><label>Historical incidents</label><input id="historyCount" type="number" min="0" value="0"></div>
</div>
<br><button class="primary" onclick="checkRisk()">Assess Conditions</button>
<div id="riskResult" class="result">Enter conditions and run an assessment.</div>
</section>

<section id="resources" class="card">
<h2>2. Resource Allocation Planner</h2>
<p class="muted">Estimate a starting quantity for emergency planning. Confirm all quantities
with local authorities and actual population needs.</p>
<label>People affected</label><input id="people" type="number" min="1" value="100">
<label>Emergency duration (days)</label><input id="days" type="number" min="1" value="3">
<button class="primary" onclick="allocate()">Calculate Requirements</button>
<div id="resourceResult" class="result">Estimated water: --</div>
</section>

<section id="shelters" class="card">
<h2>3. Safe Shelter Finder</h2>
<p class="muted">Search official local emergency management sources for verified shelter
locations. This demo does not contain a verified live shelter directory.</p>
<label>Area or city</label><input id="shelterArea" placeholder="Enter your area or city">
<button onclick="searchShelters()">Find Shelter Information</button>
<div id="shelterResult" class="result">Search for official shelter information.</div>
</section>

<section class="card">
<h2>4. Community Emergency Information</h2>
<p class="muted">For immediate danger, contact local emergency services.
In India, call 112 for emergency assistance.</p>
<label>Emergency type</label>
<select id="emergencyType">
<option>Flooding</option><option>Heavy Rain</option><option>Storm</option>
<option>Fire</option><option>Landslide</option><option>Medical Emergency</option>
<option>Infrastructure Damage</option><option>Other</option>
</select>
<label>Priority</label>
<select id="severity"><option>LOW</option><option>MEDIUM</option>
<option selected>HIGH</option><option>CRITICAL</option></select>
<label>Area</label><input id="area" placeholder="Area or locality">
<label>Report details</label><textarea id="details" maxlength="3000"
placeholder="Describe the observed situation. Do not include personal or sensitive details."></textarea>
<button class="primary" onclick="submitReport()">Submit Community Report</button>
<div id="reportResult" class="result">Reports are saved to the local SQLite database.</div>
</section>

<section id="history" class="card">
<h2>5. Incident History & Analytics</h2>
<button onclick="loadReports()">Refresh Reports</button>
<a class="btn" href="/api/export">Export CSV</a>
<div id="analytics" class="result">Loading report analytics...</div>
<div class="table-wrap"><table>
<thead><tr><th>ID</th><th>Area</th><th>Type</th><th>Severity</th><th>Details</th><th>Time</th></tr></thead>
<tbody id="reportRows"></tbody></table></div>
</section>

<section class="card">
<h2>6. Explainable Risk Assessment</h2>
<p class="muted">The screening result lists which entered conditions contributed to its
rule-based estimate. It does not establish the actual probability of a disaster.</p>
<div id="explanation" class="result">Run an assessment in the Early Warning Centre.</div>
</section>

<section id="privacy" class="card">
<h2>7. Privacy Protection</h2>
<p class="muted">Review text before submitting. This basic demo flags some common
contact details; it is not a complete anonymization or security system.</p>
<label>Text to review</label>
<textarea id="privacyText" placeholder="Paste text without real personal information"></textarea>
<button onclick="scanPrivacy()">Review Text</button>
<div id="privacyResult" class="result">Privacy review has not been run.</div>
</section>
</main>
<footer>SignalBridge · Disaster-response decision support prototype</footer>

<script>
async function api(url, options={}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Request failed");
  return data;
}
function show(id, text) {
  document.getElementById(id).textContent = text;
}
async function checkRisk() {
  const values = {
    rainfall_mm: Number(document.getElementById("rain").value),
    wind_kmh: Number(document.getElementById("wind").value),
    humidity_pct: Number(document.getElementById("humidity").value),
    river_rise_cm: Number(document.getElementById("river").value),
    historical_incidents: Number(document.getElementById("historyCount").value)
  };
  if (Object.values(values).some(v => !Number.isFinite(v) || v < 0) ||
      values.humidity_pct > 100) {
    show("riskResult", "Enter valid non-negative values and humidity from 0 to 100.");
    return;
  }
  try {
    const r = await api("/api/risk", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(values)
    });
    show("riskResult", "Screening level: " + r.level + "\nScore: " + r.score +
      "/100\nReasons:\n- " + r.reasons.join("\n- ") +
      "\n\nThis is a rule-based demo, not an official warning.");
    show("explanation", r.explanation);
  } catch(e) { show("riskResult", e.message); }
}
function allocate() {
  const people = Number(document.getElementById("people").value);
  const days = Number(document.getElementById("days").value);
  if (!Number.isInteger(people) || people < 1 ||
      !Number.isInteger(days) || days < 1) {
    show("resourceResult", "Enter positive whole numbers.");
    return;
  }
  show("resourceResult",
    "Planning estimates for " + people + " people over " + days + " days:\n" +
    "Water at 15 litres/person/day: " +
      (people * days * 15).toLocaleString() + " litres\n" +
    "Food at 2,100 kcal/person/day: " +
      (people * days * 2100).toLocaleString() + " kcal\n\n" +
    "Planning assumptions only. Follow local emergency guidance and actual needs.");
}
function searchShelters() {
  const area = document.getElementById("shelterArea").value.trim();
  if (!area) { show("shelterResult", "Enter an area or city first."); return; }
  const query = encodeURIComponent("official emergency shelters " + area);
  show("shelterResult",
    "No verified shelter listings are built into this demo.\n" +
    "Search official local government and disaster-management information for: " +
    area + "\n\nOpen a web search for official information:");
  const link = document.createElement("a");
  link.href = "https://www.google.com/search?q=" + query;
  link.target = "_blank"; link.rel = "noopener noreferrer";
  link.textContent = "Search official shelter information";
  document.getElementById("shelterResult").appendChild(document.createElement("br"));
  document.getElementById("shelterResult").appendChild(link);
}
async function submitReport() {
  const payload = {
    area:document.getElementById("area").value.trim(),
    type:document.getElementById("emergencyType").value,
    severity:document.getElementById("severity").value,
    details:document.getElementById("details").value.trim()
  };
  if (!payload.area || !payload.details) {
    show("reportResult", "Please enter the area and report details.");
    return;
  }
  try {
    const r = await api("/api/report", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)
    });
    show("reportResult", "Report saved successfully. Report ID: " + r.id);
    document.getElementById("details").value = "";
    await loadReports();
  } catch(e) { show("reportResult", "Could not save report: " + e.message); }
}
async function loadReports() {
  try {
    const r = await api("/api/analytics");
    show("total", r.total);
    show("highCount", r.high_priority);
    show("status", "Online");
    show("analytics", "Total reports: " + r.total +
      "\nLow: " + r.by_severity.LOW +
      "\nMedium: " + r.by_severity.MEDIUM +
      "\nHigh: " + r.by_severity.HIGH +
      "\nCritical: " + r.by_severity.CRITICAL);
    const tbody = document.getElementById("reportRows");
    tbody.replaceChildren();
    r.reports.forEach(item => {
      const tr = document.createElement("tr");
      [item.id, item.area, item.type, item.severity_label,
       item.details, item.created_at || ""].forEach(value => {
        const td = document.createElement("td");
        td.textContent = String(value ?? "");
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  } catch(e) {
    show("status", "Error");
    show("analytics", "Could not load analytics: " + e.message);
  }
}
function scanPrivacy() {
  const text = document.getElementById("privacyText").value;
  const patterns = [
    ["Possible email", /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/ig],
    ["Possible phone number", /(?<!\d)(?:\+?\d[\d ()-]{8,}\d)(?!\d)/g],
    ["Possible Aadhaar-like number", /\b\d{4}\s?\d{4}\s?\d{4}\b/g]
  ];
  const found = patterns.filter(p => p[1].test(text)).map(p => p[0]);
  show("privacyResult", found.length
    ? "Review needed: " + found.join(", ") +
      ". Remove personal details before submitting."
    : "No common contact patterns detected. This does not guarantee anonymity.");
}
loadReports();
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(HTML)


@app.get("/api/health")
def health():
    try:
        with connect_db() as conn:
            conn.execute("SELECT 1").fetchone()
        return jsonify(status="ok", database="connected")
    except sqlite3.Error:
        app.logger.exception("Health check failed")
        return jsonify(status="error", database="unavailable"), 500


@app.post("/api/report")
def create_report():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a valid JSON object."), 400
    try:
        report = save_report(data)
        return jsonify(message="Report saved", **report), 201
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Could not save incident report")
        return jsonify(error="Could not save report. Check the server terminal."), 500


@app.get("/api/analytics")
def analytics():
    try:
        reports = get_reports()
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for item in reports:
            label = item["severity_label"]
            if label in counts:
                counts[label] += 1

        return jsonify(
            total=len(reports),
            high_priority=counts["HIGH"] + counts["CRITICAL"],
            by_severity=counts,
            reports=reports
        )
    except Exception:
        app.logger.exception("Analytics failed")
        return jsonify(error="Could not load analytics."), 500



@app.post("/api/risk")
def risk():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send valid JSON."), 400

    try:
        values = {
            "rainfall_mm": float(data.get("rainfall_mm", 0)),
            "wind_kmh": float(data.get("wind_kmh", 0)),
            "humidity_pct": float(data.get("humidity_pct", 0)),
            "river_rise_cm": float(data.get("river_rise_cm", 0)),
            "historical_incidents": float(data.get("historical_incidents", 0)),
        }

        if not all(math.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError("Values must be finite and non-negative.")
        if values["humidity_pct"] > 100:
            raise ValueError("Humidity must not exceed 100.")
        if not values["historical_incidents"].is_integer():
            raise ValueError("Historical incidents must be a whole number.")

    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify(error=f"Invalid input: {exc}"), 400

    if AI_MODEL is None or not AI_FEATURES:
        return jsonify(error="AI model is unavailable. Check the server logs."), 503

    try:
        features = [values[name] for name in AI_FEATURES]
        prediction = str(AI_MODEL.predict([features])[0]).upper()

        probabilities = {}
        if hasattr(AI_MODEL, "predict_proba"):
            raw_probabilities = AI_MODEL.predict_proba([features])[0]
            probabilities = {
                str(label).upper(): round(float(prob) * 100, 2)
                for label, prob in zip(AI_MODEL.classes_, raw_probabilities)
            }

        confidence = probabilities.get(prediction)

    except (KeyError, ValueError, TypeError, AttributeError):
        app.logger.exception("AI risk prediction failed")
        return jsonify(error="Could not calculate the AI assessment."), 500

    # Retain the existing rule-based screening score for compatibility.
    rain = values["rainfall_mm"]
    wind = values["wind_kmh"]
    humidity = values["humidity_pct"]
    river = values["river_rise_cm"]
    incidents = int(values["historical_incidents"])

    score = 0
    reasons = []

    if rain >= 100:
        score += 35
        reasons.append("Very heavy rainfall entered (+35)")
    elif rain >= 50:
        score += 20
        reasons.append("Heavy rainfall entered (+20)")
    elif rain >= 25:
        score += 10
        reasons.append("Moderate rainfall entered (+10)")

    if wind >= 80:
        score += 25
        reasons.append("Very strong wind entered (+25)")
    elif wind >= 50:
        score += 15
        reasons.append("Strong wind entered (+15)")

    if river >= 30:
        score += 25
        reasons.append("Significant river rise entered (+25)")
    elif river >= 10:
        score += 15
        reasons.append("River rise entered (+15)")

    if humidity >= 90:
        score += 5
        reasons.append("High humidity entered (+5)")

    if incidents >= 3:
        score += 15
        reasons.append("Multiple historical incidents entered (+15)")
    elif incidents > 0:
        score += 5
        reasons.append("Historical incidents entered (+5)")

    score = min(score, 100)
    level = "HIGH" if score >= 50 else "MODERATE" if score >= 25 else "LOW"

    if not reasons:
        reasons.append("No configured rule-based threshold was reached.")

    explanation = (
        f"Rule-based screening score: {score}/100 ({level}). "
        f"Experimental AI prediction: {prediction}. "
        "The AI was trained on synthetic data and is not validated "
        "for real-world disaster forecasting. Follow official warnings."
    )

    audit_event("risk_assessment", explanation)

    return jsonify(
        score=score,
        level=level,
        reasons=reasons,
        explanation=explanation,
        ai_prediction=prediction,
        ai_confidence_pct=confidence,
        ai_probabilities=probabilities,
        model_data_type="synthetic_demo"
    )


@app.get("/api/export")
def export_csv():
    try:
        reports = get_reports()
        output = io.StringIO()
        fields = ["id", "area", "type", "severity_label",
                  "details", "created_at"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow({key: report.get(key, "") for key in fields})

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition":
                     "attachment; filename=signalbridge_reports.csv"}
        )
    except Exception:
        app.logger.exception("CSV export failed")
        return jsonify(error="Could not export reports."), 500


if __name__ == "__main__":
    init_db()
    print("SignalBridge starting...")
    print("Database:", DB_PATH)
    app.run(host="127.0.0.1", port=5000, debug=False)
