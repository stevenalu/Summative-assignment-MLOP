import os
import io
import sys
import time
import sqlite3
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form

sys.path.append("src")
from preprocessing import train_dir, model_path
from model import load_trained_model, retrain
from prediction import predict

db_path = os.path.join("data", "uploads.db")
start_time = time.time()

app = FastAPI(title="Maize Leaf Disease API")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_trained_model()
    return _model


def init_db():
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS uploads ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "filename TEXT, label TEXT, created TEXT)"
    )
    con.commit()
    con.close()


init_db()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "present" if os.path.exists(model_path) else "missing",
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    return predict(get_model(), io.BytesIO(data))


@app.post("/upload")
async def upload(label: str = Form(...), files: list[UploadFile] = File(...)):
    folder = os.path.join(train_dir, label)
    os.makedirs(folder, exist_ok=True)
    con = sqlite3.connect(db_path)
    saved = 0
    for f in files:
        data = await f.read()
        with open(os.path.join(folder, f.filename), "wb") as out:
            out.write(data)
        con.execute(
            "INSERT INTO uploads (filename, label, created) VALUES (?, ?, ?)",
            (f.filename, label, datetime.now().isoformat(timespec="seconds")),
        )
        saved += 1
    con.commit()
    con.close()
    return {"saved": saved, "label": label}


@app.get("/uploads")
def uploads():
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT filename, label, created FROM uploads ORDER BY id DESC").fetchall()
    con.close()
    return {"count": len(rows), "items": [{"filename": r[0], "label": r[1], "created": r[2]} for r in rows]}


@app.post("/retrain")
def retrain_endpoint(epochs: int = 3):
    history = retrain(epochs=epochs)
    global _model
    _model = load_trained_model()
    return {
        "epochs": epochs,
        "val_accuracy": round(history["val_accuracy"][-1], 4),
        "val_loss": round(history["val_loss"][-1], 4),
    }
