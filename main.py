"""
FastAPI Backend for Document Intelligence Application - MVP
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import shutil
import uuid
from datetime import datetime


#Numpy version: 2.3.4
# Import services
from services.all_services import DocumentService
from services.all_services import SummaryService
from services.all_services import NotesService
from services.all_services import FlashcardService
from services.all_services import QuizService

app = FastAPI(
    title="Document Intelligence API - MVP",
    description="API for document summarization, note-making, flashcards, and quiz generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory session storage
sessions = {}

# Initialize services
document_service = DocumentService()
summary_service = SummaryService()
notes_service = NotesService()
flashcard_service = FlashcardService()
quiz_service = QuizService()


# Pydantic Models
class SessionResponse(BaseModel):
    session_id: str
    filename: str
    chunk_count: int
    message: str


class BaseRequest(BaseModel):
    session_id: str
    query: Optional[str] = None
    num_chunks: Optional[int] = 3


class QuizRequest(BaseRequest):
    num_questions: Optional[int] = 10


# @app.on_event("startup")
# async def startup_event():
#     """Initialize on startup"""
#     print("🚀 Starting Document Intelligence API (MVP)...")
#     os.makedirs("temp_uploads", exist_ok=True)
#     os.makedirs("temp_vector_stores", exist_ok=True)


# @app.on_event("shutdown")
# async def shutdown_event():
#     """Cleanup on shutdown"""
#     print("🛑 Shutting down...")
#     shutil.rmtree("temp_uploads", ignore_errors=True)
#     shutil.rmtree("temp_vector_stores", ignore_errors=True)


@app.get("/")
async def root():
    """Health check"""
    return {
        "message": "Document Intelligence API is running",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "upload": "/api/upload",
            "summarize": "/api/summarize/*",
            "notes": "/api/notes/*",
            "flashcards": "/api/flashcards/*",
            "quiz": "/api/quiz/*"
        }
    }


# ============= UPLOAD =============
@app.post("/api/upload", response_model=SessionResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_method: str = "recursive"
):
    """Upload document and create session"""
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.txt')):
            raise HTTPException(400, "Only PDF and TXT files supported")
        
        # Create session
        session_id = str(uuid.uuid4())
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)  # ✅ ensure directory exists
        temp_file_path = f"{temp_dir}/{session_id}_{file.filename}"
        
        # Save file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        print(f"📄 Processing: {file.filename}")
        chunks, vector_store = await document_service.process_document(
            temp_file_path, session_id, chunking_method
        )
        
        # Store session
        sessions[session_id] = {
            "filename": file.filename,
            "chunks": chunks,
            "vector_store": vector_store,
            "created_at": datetime.now(),
            "file_path": temp_file_path
        }
        
        print(f"✅ Session created: {session_id}")
        return SessionResponse(
            session_id=session_id,
            filename=file.filename,
            chunk_count=len(chunks),
            message="Document processed successfully"
        )
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(500, str(e))


# ============= SUMMARIZE =============
@app.post("/api/summarize/full")
async def summarize_full(session_id: str):
    """Summarize entire document"""
    try:
        session = get_session(session_id)
        print(f"📝 Summarizing full document for session: {session_id}")
        
        summary = await summary_service.summarize_all(session["chunks"])
        
        return {
            "session_id": session_id,
            "summary": summary["output_text"],
            "source_chunks": len(session["chunks"])
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


@app.post("/api/summarize/topic")
async def summarize_topic(request: BaseRequest):
    """Summarize on specific topic"""
    try:
        session = get_session(request.session_id)
        print(f"📝 Summarizing topic '{request.query}' for session: {request.session_id}")
        
        summary, docs = await summary_service.summarize_query(
            session["vector_store"], 
            request.query,
            k=request.num_chunks or 3
        )
        
        return {
            "session_id": request.session_id,
            "summary": summary.content,
            "query": request.query,
            "source_chunks": len(docs)
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


@app.post("/api/summarize/notes")
async def summarize_notes(notes: str):
    """Summarize user notes"""
    try:
        if not notes or len(notes.strip()) < 10:
            raise HTTPException(400, "Notes too short (min 10 characters)")
        
        print(f"📝 Summarizing user notes")
        summary = await summary_service.summarize_notes(notes)
        
        return {
            "session_id": "notes_only",
            "summary": summary.content
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


# ============= NOTES =============
@app.post("/api/notes/topic")
async def create_notes_topic(request: BaseRequest):
    """Create notes on topic"""
    try:
        session = get_session(request.session_id)
        print(f"📓 Creating notes on '{request.query}'")
        
        notes, docs = await notes_service.make_notes_query(
            session["vector_store"],
            request.query,
            k=request.num_chunks or 3
        )
        
        return {
            "session_id": request.session_id,
            "notes": notes.content,
            "query": request.query,
            "source_chunks": len(docs)
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


@app.post("/api/notes/text")
async def create_notes_text(notes_text: str):
    """Create notes from text"""
    try:
        if not notes_text or len(notes_text.strip()) < 10:
            raise HTTPException(400, "Text too short (min 10 characters)")
        
        print(f"📓 Creating notes from text")
        notes = await notes_service.make_notes_notes(notes_text)
        
        return {
            "session_id": "text_only",
            "notes": notes.content
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


# ============= FLASHCARDS =============
@app.post("/api/flashcards/topic")
async def create_flashcards_topic(request: BaseRequest):
    """Generate flashcards on topic"""
    try:
        session = get_session(request.session_id)
        print(f"🎴 Creating flashcards on '{request.query}'")
        
        cards, docs = await flashcard_service.create_flashcards_on_topic(
            session["vector_store"],
            request.query,
            k=request.num_chunks or 3
        )
        
        return {
            "session_id": request.session_id,
            "flashcards": cards.content,
            "query": request.query,
            "source_chunks": len(docs)
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


@app.post("/api/flashcards/notes")
async def create_flashcards_notes(notes: str):
    """Generate flashcards from notes"""
    try:
        if not notes or len(notes.strip()) < 10:
            raise HTTPException(400, "Notes too short (min 10 characters)")
        
        print(f"🎴 Creating flashcards from notes")
        cards = await flashcard_service.create_flashcards_based_notes(notes)
        
        return {
            "session_id": "notes_only",
            "flashcards": cards.content
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


# ============= QUIZ =============
@app.post("/api/quiz/full")
async def create_quiz_full(session_id: str, num_questions: int = 10):
    """Generate quiz from full document"""
    try:
        session = get_session(session_id)
        print(f"📝 Creating quiz ({num_questions} questions)")
        
        quiz = await quiz_service.generate_quiz_all(
            session["chunks"], 
            num_questions
        )
        
        return {
            "session_id": session_id,
            "quiz": quiz.content,
            "num_questions": num_questions,
            "source_chunks": len(session["chunks"])
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


@app.post("/api/quiz/topic")
async def create_quiz_topic(request: QuizRequest):
    """Generate quiz on topic"""
    try:
        session = get_session(request.session_id)
        print(f"📝 Creating quiz on '{request.query}' ({request.num_questions} questions)")
        
        quiz, docs = await quiz_service.generate_quiz_on_topic(
            session["vector_store"],
            request.query,
            num_questions=request.num_questions or 10,
            k=request.num_chunks or 5
        )
        
        return {
            "session_id": request.session_id,
            "quiz": quiz.content,
            "query": request.query,
            "num_questions": request.num_questions or 10,
            "source_chunks": len(docs)
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


@app.post("/api/quiz/notes")
async def create_quiz_notes(notes: str, num_questions: int = 10):
    """Generate quiz from notes"""
    try:
        if not notes or len(notes.strip()) < 10:
            raise HTTPException(400, "Notes too short (min 10 characters)")
        
        print(f"📝 Creating quiz from notes ({num_questions} questions)")
        quiz = await quiz_service.generate_quiz_from_notes(notes, num_questions)
        
        return {
            "session_id": "notes_only",
            "quiz": quiz.content,
            "num_questions": num_questions
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


# ============= UTILITIES =============
@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session and cleanup"""
    try:
        if session_id not in sessions:
            raise HTTPException(404, "Session not found")
        
        session = sessions[session_id]
        
        # Delete files
        if os.path.exists(session["file_path"]):
            os.remove(session["file_path"])
        
        del sessions[session_id]
        print(f"🗑️ Session deleted: {session_id}")
        
        return {"message": "Session deleted successfully"}
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(500, str(e))


@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "count": len(sessions),
        "sessions": [
            {
                "session_id": sid,
                "filename": data["filename"],
                "created_at": data["created_at"].isoformat(),
                "chunk_count": len(data["chunks"])
            }
            for sid, data in sessions.items()
        ]
    }


def get_session(session_id: str):
    """Get session or raise error"""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found. Please upload document first.")
    return sessions[session_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)