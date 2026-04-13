import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import advisor
from game import PlayerSession

# ── session store ─────────────────────────────────────────────
sessions: dict[str, PlayerSession] = {}


# ── startup / shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    advisor.load_model("blackjack_infinite_shoe_model_35m.zip")
    print("Model loaded.")
    yield
    sessions.clear()


app = FastAPI(title="Blackjack API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],    # Nuxt dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request schemas ───────────────────────────────────────────
class CreateSessionRequest(BaseModel):
    starting_balance: float = Field(default=1000.0, gt=0)

class StartRoundRequest(BaseModel):
    bet: float = Field(gt=0)

class ActionRequest(BaseModel):
    action: str = Field(description="hit | stand | double | split")


# ── helpers ───────────────────────────────────────────────────
def _get_session(session_id: str) -> PlayerSession:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


# ── endpoints ─────────────────────────────────────────────────

@app.post("/session", summary="Create a player session")
def create_session(body: CreateSessionRequest):
    """
    Call once when the player opens the game.
    Returns a session_id — store it on the frontend for all subsequent requests.
    """
    session_id            = str(uuid.uuid4())
    sessions[session_id]  = PlayerSession(starting_balance=body.starting_balance)
    return {
        "session_id": session_id,
        "balance":    body.starting_balance,
    }


@app.post("/session/{session_id}/round/start", summary="Deal a new round")
def start_round(session_id: str, body: StartRoundRequest):
    """Start a new round within an existing session."""
    session = _get_session(session_id)
    try:
        state = session.start_round(bet=body.bet)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return state


@app.post("/session/{session_id}/round/action", summary="Send a player action")
def take_action(session_id: str, body: ActionRequest):
    """
    Valid actions: hit | stand | double | split
    Returns updated state + new suggestion.
    When round.round_over is true, balance has already been settled.
    """
    session = _get_session(session_id)
    try:
        state = session.action(body.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return state


@app.get("/session/{session_id}", summary="Get current session state")
def get_state(session_id: str):
    """Reconnect a client that lost state — returns full session snapshot."""
    session = _get_session(session_id)
    return session.get_state()


@app.delete("/session/{session_id}", summary="End the session")
def delete_session(session_id: str):
    """Call when the player closes the game."""
    sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": advisor.model is not None}


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
