from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage
import json, uuid

from database import get_db, init_db, User, ChatSession, Message
from auth import hash_password, verify_password, create_token, get_current_user
from agent import agent
from guardrails import check_input, clean_output
from rag import build_vectorstore

app = FastAPI(title="Voxa AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ── Startup ──────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_db()           # create tables
    build_vectorstore() # index docs

# ── Serve frontend ────────────────────────────────────────
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

# ── Auth routes ───────────────────────────────────────────
class RegisterBody(BaseModel):
    username: str
    email: str
    password: str

class LoginBody(BaseModel):
    email: str
    password: str

@app.post("/register")
def register(body: RegisterBody, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    user = User(username=body.username, email=body.email,
                password=hash_password(body.password))
    db.add(user); db.commit()
    return {"token": create_token(user.id), "username": user.username}

@app.post("/login")
def login(body: LoginBody, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password):
        raise HTTPException(401, "Invalid credentials")
    return {"token": create_token(user.id), "username": user.username}

# ── Session routes ────────────────────────────────────────
@app.post("/sessions")
def create_session(user=Depends(get_current_user), db: Session = Depends(get_db)):
    session = ChatSession(user_id=user.id)
    db.add(session); db.commit()
    return {"session_id": session.id, "title": session.title}

@app.get("/sessions")
def get_sessions(user=Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user.id
    ).order_by(ChatSession.created.desc()).all()
    return [{"id": s.id, "title": s.title, "created": s.created} for s in sessions]

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(Message).filter(Message.session_id == session_id).delete()
    db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).delete()
    db.commit()
    return {"deleted": True}

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    msgs = db.query(Message).filter(
        Message.session_id == session_id
    ).order_by(Message.timestamp).all()
    return [{"role": m.role, "content": m.content} for m in msgs]

# ── Chat route ────────────────────────────────────────────
class ChatBody(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
@app.post("/chat")
def chat(body: ChatBody, user=Depends(get_current_user), db: Session = Depends(get_db)):
    safe, reason = check_input(body.message)
    if not safe:
        return {"reply": reason}

    past = db.query(Message).filter(
        Message.session_id == body.session_id
    ).order_by(Message.timestamp).all()

    history = [
        HumanMessage(content=m.content) if m.role == "user"
        else AIMessage(content=m.content)
        for m in past
    ]
    history.append(HumanMessage(content=body.message))

    try:
        result = agent.invoke({"messages": history})
        reply  = result["messages"][-1].content
        reply  = clean_output(reply)
    except Exception as e:
        print(f"Agent error: {e}")
        return {"reply": "I'm a bit busy right now (free server limits) — please try again in a few seconds!"}

    db.add(Message(session_id=body.session_id, role="user",      content=body.message))
    db.add(Message(session_id=body.session_id, role="assistant", content=reply))

    session = db.query(ChatSession).filter(ChatSession.id == body.session_id).first()
    if session and session.title == "New Chat":
        session.title = body.message[:40]
    db.commit()

    return {"reply": reply}