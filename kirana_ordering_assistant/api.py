"""FastAPI entrypoint for Vercel Python Service.
Exposes the Python NLP parser and reorder nudges as an HTTP API.
"""
import sys
import os
from pathlib import Path

# Add current folder to sys.path so local imports work
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from order_parser import deterministic_parse, CATALOG
from reorder_model import get_reorder_nudges_for_store

app = FastAPI(title="Kirana Ordering Assistant Python API", version="1.0.0")

class ParseRequest(BaseModel):
    message: str

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "kirana_ordering_assistant",
        "engine": "Deterministic Hybrid NLP & Predictive Reorder Engine"
    }

@app.get("/catalog")
def get_catalog():
    return CATALOG

@app.post("/parse")
def parse_order(req: ParseRequest):
    result = deterministic_parse(req.message)
    return result

@app.get("/nudges/{store_id}")
def store_nudges(store_id: str):
    nudges = get_reorder_nudges_for_store(store_id, [])
    return {"store_id": store_id, "nudges": nudges}
