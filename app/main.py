import os
import json
import urllib.request
import urllib.error
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

def fetch_online_games(platform: str, username: str, max_games: int = 10) -> str:
    """Fetches recent games in PGN format from public Lichess or Chess.com APIs."""
    clean_user = username.strip()
    if not clean_user:
        raise ValueError("Please provide a valid username.")

    plat = platform.lower().strip()
    if plat == "lichess":
        url = f"https://lichess.org/api/games/user/{clean_user}?max={max_games}&pgnInBody=true"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ChessPersonalityIndexer/1.0 (https://chess.jlelia.net)",
            "Accept": "application/x-chess-pgn"
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                pgn_text = resp.read().decode("utf-8")
                if not pgn_text.strip():
                    raise ValueError(f"No games found on Lichess for username '{clean_user}'.")
                return pgn_text
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"Lichess user '{clean_user}' not found.")
            raise ValueError(f"Lichess API returned status {e.code}: {e.reason}")
        except Exception as e:
            raise ValueError(f"Failed to fetch games from Lichess: {str(e)}")

    elif plat in ("chesscom", "chess.com"):
        url = f"https://api.chess.com/pub/player/{clean_user.lower()}/games/archives"
        req = urllib.request.Request(url, headers={
            "User-Agent": "ChessPersonalityIndexer/1.0 (contact: jlelia@gmail.com)"
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                archives_data = json.loads(resp.read().decode("utf-8"))
                archives = archives_data.get("archives", [])
                if not archives:
                    raise ValueError(f"No game archives found for Chess.com user '{clean_user}'.")
                latest_archive_url = archives[-1]

            req_archive = urllib.request.Request(latest_archive_url, headers={
                "User-Agent": "ChessPersonalityIndexer/1.0 (contact: jlelia@gmail.com)"
            })
            with urllib.request.urlopen(req_archive, timeout=10) as resp_arch:
                month_data = json.loads(resp_arch.read().decode("utf-8"))
                games = month_data.get("games", [])
                if not games:
                    raise ValueError(f"No recent games found for Chess.com user '{clean_user}'.")
                recent_games = games[-max_games:]
                pgns = [g["pgn"] for g in recent_games if "pgn" in g]
                if not pgns:
                    raise ValueError(f"No PGN format games found for '{clean_user}'.")
                return "\n\n".join(pgns)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"Chess.com user '{clean_user}' not found.")
            raise ValueError(f"Chess.com API returned status {e.code}: {e.reason}")
        except Exception as e:
            raise ValueError(f"Failed to fetch games from Chess.com: {str(e)}")
    else:
        raise ValueError(f"Unsupported platform '{platform}'. Please choose 'lichess' or 'chesscom'.")

@app.get("/api/fetch-games")
async def api_fetch_games(platform: str = "lichess", username: str = "", max_games: int = 10):
    try:
        limit = max(1, min(max_games, 25))
        pgn_text = fetch_online_games(platform, username, max_games=limit)
        return {
            "platform": platform,
            "username": username,
            "pgn_text": pgn_text
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch games: {str(e)}")

@app.post("/api/analyze")
async def analyze_pgn(
    file: UploadFile | None = File(None),
    pgn_text: str | None = Form(None),
    player_name: str | None = Form(None),
    side: str = Form("white")
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not ready.")

    content = ""
    if file is not None:
        raw_bytes = await file.read()
        if len(raw_bytes) > 2 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds the 2MB limit.")
        content = raw_bytes.decode("utf-8", errors="ignore")
    elif pgn_text:
        if len(pgn_text.encode("utf-8")) > 2 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="PGN text exceeds the 2MB limit.")
        content = pgn_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="Please provide a valid PGN file or paste PGN text.")

    try:
        report = engine.analyze_games(content, player_name=player_name, side=side)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Mount static files at /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
