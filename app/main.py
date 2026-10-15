import os
import re
import json
import ssl
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

def _compute_static_version() -> str:
    """Content hash of static assets, used as a cache-busting query parameter."""
    digest = hashlib.sha256()
    for name in ("app.js", "style.css"):
        try:
            digest.update((STATIC_DIR / name).read_bytes())
        except OSError:
            pass
    return digest.hexdigest()[:12]

STATIC_VERSION = _compute_static_version()

class AnalyzeTextRequest(BaseModel):
    pgn_text: str
    player_name: str | None = None

@app.middleware("http")
async def cache_headers_middleware(request, call_next):
    """Explicit cache policy: force revalidation of the HTML shell, make
    content-versioned static assets immutable, and never cache API responses."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/"):
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if "v=" in request.url.query else "no-cache"
        )
    elif path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    else:
        response.headers["Cache-Control"] = "no-cache"
    return response

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Chess Personality Indexer Backend Running</h1>")
    # Inject the content-hash version into static asset URLs so browsers fetch
    # fresh JS/CSS after each deploy instead of replaying stale cached files.
    html = index_file.read_text(encoding="utf-8")
    html = html.replace("/static/app.js", f"/static/app.js?v={STATIC_VERSION}")
    html = html.replace("/static/style.css", f"/static/style.css?v={STATIC_VERSION}")
    return HTMLResponse(content=html, headers={"Cache-Control": "no-cache"})

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

class UpstreamFetchError(ValueError):
    """The upstream provider could not be reached (network/DNS/TLS failure).
    Subclasses ValueError so existing handlers keep working, but lets the API
    distinguish upstream outages (502) from invalid user input (400)."""

def _network_error_detail(provider: str, host: str, exc: Exception) -> str:
    """Translates low-level network exceptions into actionable user-facing text."""
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLCertVerificationError):
        proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        hint = ""
        if proxy:
            hint = f" An HTTPS proxy is configured for this server ({proxy}); if it performs TLS interception, exclude {host}."
        return (
            f"TLS certificate verification failed while contacting {provider}: the certificate presented is not valid "
            f"for {host}. Something on the server's network is intercepting HTTPS — most commonly a DNS filter "
            f"(e.g. Pi-hole blocking the domain), a proxy, a VPN, or an ISP redirect. This is a server-side network "
            f"issue, not a browser problem. Check the DNS/proxy configuration of the machine running this app."
            f"{hint}"
        )
    if isinstance(exc, urllib.error.URLError):
        return (
            f"Could not reach {provider} ({reason}). The machine running this app may be offline, firewalled, "
            f"or have DNS problems."
        )
    return f"Failed to fetch games from {provider}: {exc}"

def fetch_online_games(platform: str, username: str, max_games: int = 10) -> str:
    """Fetches recent games in PGN format from public Lichess or Chess.com APIs."""
    clean_user = username.strip()
    if not clean_user:
        raise ValueError("Please provide a valid username.")

    if not re.match(r"^[a-zA-Z0-9_\-\.]{1,40}$", clean_user):
        raise ValueError("Invalid username format. Usernames may only contain alphanumeric characters, underscores, and hyphens.")

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
            raise UpstreamFetchError(f"Lichess API returned status {e.code}: {e.reason}")
        except ValueError:
            raise
        except Exception as e:
            raise UpstreamFetchError(_network_error_detail("Lichess", "lichess.org", e))

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
            raise UpstreamFetchError(f"Chess.com API returned status {e.code}: {e.reason}")
        except ValueError:
            raise
        except Exception as e:
            raise UpstreamFetchError(_network_error_detail("Chess.com", "api.chess.com", e))
    else:
        raise ValueError(f"Unsupported platform '{platform}'. Please choose 'lichess' or 'chesscom'.")

@app.get("/api/fetch-games")
def api_fetch_games(platform: str = "lichess", username: str = "", max_games: int = 10):
    """Runs as a sync endpoint so FastAPI executes the blocking upstream HTTP
    calls in its worker threadpool instead of freezing the event loop."""
    try:
        limit = max(1, min(max_games, 25))
        pgn_text = fetch_online_games(platform, username, max_games=limit)
        return {
            "platform": platform,
            "username": username,
            "pgn_text": pgn_text
        }
    except UpstreamFetchError as e:
        # Upstream unreachable/TLS failure — the client's request was valid.
        raise HTTPException(status_code=502, detail=str(e))
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
        # CPU/GPU-heavy encoding + SVG rendering would otherwise block the
        # event loop (and every concurrent request, including /api/health);
        # push it onto the shared worker threadpool instead.
        report = await run_in_threadpool(
            engine.analyze_games, content, player_name=player_name, side=side
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Mount static files at /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
