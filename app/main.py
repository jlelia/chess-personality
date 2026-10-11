import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

from app.inference import ChessStylometryEngine
from preprocess_games import GRANDMASTERS
from demo import GM_PROFILES

engine: ChessStylometryEngine | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = ChessStylometryEngine()
    yield

app = FastAPI(
    title="Chess Personality Indexer",
    description="Machine-learned stylometry mapping chess games to Grandmaster archetypes",
    version="1.0.0",
    lifespan=lifespan
)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True, parents=True)

class AnalyzeTextRequest(BaseModel):
    pgn_text: str
    player_name: str | None = None

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Chess Personality Indexer Backend Running</h1>")
    return FileResponse(index_file)

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
        "device": str(engine.device) if engine else "none",
        "grandmasters_count": len(GRANDMASTERS)
    }

@app.get("/api/sample")
async def get_sample():
    sample_path = Path("test.pgn")
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample PGN not found")
    content = sample_path.read_text(encoding="utf-8")
    return {
        "player_name": "JamesElia",
        "pgn_text": content
    }

@app.get("/api/grandmasters")
async def get_grandmasters():
    result = []
    for idx, gm in enumerate(GRANDMASTERS):
        meta = GM_PROFILES.get(gm, {"title": "Grandmaster", "description": ""})
        result.append({
            "id": idx,
            "name": gm,
            "title": meta["title"],
            "description": meta["description"]
        })
    return {"grandmasters": result}

@app.post("/api/analyze")
async def analyze_pgn(
    file: UploadFile | None = File(None),
    pgn_text: str | None = Form(None),
    player_name: str | None = Form(None)
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not ready.")

    content = ""
    if file is not None:
        raw_bytes = await file.read()
        content = raw_bytes.decode("utf-8", errors="ignore")
    elif pgn_text:
        content = pgn_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="Please provide a valid PGN file or paste PGN text.")

    try:
        report = engine.analyze_games(content, player_name=player_name)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Mount static files at /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
