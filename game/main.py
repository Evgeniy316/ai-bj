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
# uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

# ── startup / shutdown ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("MODEL NOT LOADED")
    advisor.load_model("blackjack_model_28000000_steps.zip")
    print("MODEL LOADED")
    print("Model loaded.")
    yield
    sessions.clear()


app = FastAPI(title="Blackjack API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request schemas ───────────────────────────────────────────
class SessionRequest(BaseModel):
    session_id: str


class ActionRequest(BaseModel):
    session_id: str
    action: str = Field(description="hit | stand | double | split")


# ── helpers ───────────────────────────────────────────────────
def _get_session(session_id: str) -> PlayerSession:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


# ── endpoints ─────────────────────────────────────────────────

@app.post("/session", summary="Create a player session")
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = PlayerSession()
    return {"session_id": session_id}


@app.post("/round/start", summary="Deal a new round")
def start_round(body: SessionRequest):
    session = _get_session(body.session_id)
    try:
        state = session.start_round()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return state


@app.post("/round/action", summary="Send a player action")
def take_action(body: ActionRequest):
    session = _get_session(body.session_id)
    try:
        state = session.action(body.action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return state


@app.post("/session/state", summary="Get current session state")
def get_state(body: SessionRequest):
    session = _get_session(body.session_id)
    return session.get_state()


@app.post("/session", summary="End the session")
def delete_session(body: SessionRequest):
    sessions.pop(body.session_id, None)
    return {"ok": True}


@app.post("/session/history", summary="Get round history")
def get_history(body: SessionRequest):
    session = _get_session(body.session_id)
    return session.get_history()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": advisor.model is not None}


# ── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
