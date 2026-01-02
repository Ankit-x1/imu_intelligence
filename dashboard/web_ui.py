from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import json
import os
import uvicorn

app = FastAPI()

templates = Jinja2Templates(directory="templates")

hermes_system = None

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/status")
async def get_status():
    if hermes_system:
        return JSONResponse(hermes_system.get_status())
    return JSONResponse({"error": "System not initialized"})

@app.get("/api/history")
async def get_history():
    try:
        if not os.path.exists("signatures.json"):
            return JSONResponse([])
        with open("signatures.json", "r") as f:
            history = json.load(f)
        return JSONResponse(history[-100:])  # Last 100 samples
    except Exception as e:
        print(f"History API error: {e}")
        return JSONResponse([])

@app.get("/api/anomalies")
async def get_anomalies():
    anomalies = []
    try:
        if not os.path.exists("anomalies.jsonl"):
            return JSONResponse([])
        with open("anomalies.jsonl", "r") as f:
            for line in f:
                try:
                    anomalies.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return JSONResponse(anomalies[-20:])  # Last 20 anomalies
    except Exception as e:
        print(f"Anomalies API error: {e}")
        return JSONResponse([])

@app.get("/api/calibration")
async def get_calibration():
    if hermes_system and getattr(hermes_system, "calibrator", None):
        return JSONResponse(hermes_system.calibrator.get_calibration())
    return JSONResponse({})


def run_dashboard(system):
    global hermes_system
    hermes_system = system
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
