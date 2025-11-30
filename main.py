"""
SmartLearn AI - Integrated FastAPI Backend
Supports: PDF, TXT, Video, Audio, Transcripts, Parquet
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
from pathlib import Path
import uuid
from dotenv import load_dotenv

from services.all_services import (
    IntegratedPipeline,
    SessionManager,
    get_supported_extensions
)

load_dotenv()

app = FastAPI(
    title="SmartLearn AI API",
    description="Document Intelligence with Multimodal Support",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session manager
session_manager = SessionManager()

# Directories
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    file_type: str
    doc_id: str
    chunks_count: int
    message: str

class SummaryRequest(BaseModel):
    session_id: str
    topic: Optional[str] = None
    k: int = 3

class NotesRequest(BaseModel):
    session_id: str
    topic: Optional[str] = None
    text: Optional[str] = None
    k: int = 3

class FlashcardsRequest(BaseModel):
    session_id: str
    topic: Optional[str] = None
    notes: Optional[str] = None
    k: int = 3

class QuizRequest(BaseModel):
    session_id: str
    topic: Optional[str] = None
    notes: Optional[str] = None
    num_questions: int = 10
    k: int = 5

class SessionInfo(BaseModel):
    session_id: str
    filename: str
    doc_id: str
    file_type: str
    chunks_count: int

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "message": "SmartLearn AI API",
        "version": "2.0.0",
        "supported_formats": get_supported_extensions(),
        "docs": "/docs"
    }

@app.get("/api/supported-formats")
async def supported_formats():
    """Get all supported file formats"""
    return {
        "formats": get_supported_extensions(),
        "categories": {
            "video": [".mp4", ".mkv", ".mov", ".avi"],
            "audio": [".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"],
            "documents": [".pdf", ".txt"],
            "transcripts": [".srt", ".vtt", ".json", ".parquet"]
        }
    }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload any supported file format
    Supports: PDF, TXT, MP4, MKV, MOV, AVI, WAV, MP3, M4A, FLAC, AAC, OGG, SRT, VTT, JSON, Parquet
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in get_supported_extensions():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(get_supported_extensions())}"
            )

        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process file with integrated pipeline
        print(f"Processing {file.filename} (type: {file_ext})...")
        use_groq = False if os.getenv("OPENAI_API_KEY") else True
        pipeline = IntegratedPipeline(use_groq=use_groq)  # Using OpenAI
        vector_store = pipeline.process_file(str(file_path))

        # Store session
        session_manager.add_session(
            session_id=session_id,
            pipeline=pipeline,
            filename=file.filename,
            file_path=str(file_path)
        )

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            file_type=file_ext,
            doc_id=pipeline.current_doc_id,
            chunks_count=len(vector_store.get_all_chunks()),
            message=f"File processed successfully! {file_ext.upper()} support enabled."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize/full")
async def summarize_full(request: SummaryRequest):
    """Summarize the entire document"""
    try:
        pipeline = session_manager.get_pipeline(request.session_id)
        summary = pipeline.summarize_all()
        
        return {
            "session_id": request.session_id,
            "summary": summary.content
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize/topic")
async def summarize_topic(request: SummaryRequest):
    """Summarize on a specific topic"""
    try:
        if not request.topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        pipeline = session_manager.get_pipeline(request.session_id)
        summary = pipeline.summarize(request.topic, k=request.k)
        
        return {
            "session_id": request.session_id,
            "topic": request.topic,
            "summary": summary.content
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/notes/topic")
async def create_notes_topic(request: NotesRequest):
    """Create structured notes on a topic"""
    try:
        if not request.topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        pipeline = session_manager.get_pipeline(request.session_id)
        notes = pipeline.make_notes(request.topic, k=request.k)
        
        return {
            "session_id": request.session_id,
            "topic": request.topic,
            "notes": notes.content
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/notes/text")
async def create_notes_text(request: NotesRequest):
    """Create notes from provided text"""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        pipeline = session_manager.get_pipeline(request.session_id)
        
        # Use summarizer to process text
        from services.all_services import DocumentSummarizer
        use_groq = False if os.getenv("OPENAI_API_KEY") else True
        ds = DocumentSummarizer(use_groq=use_groq)
        notes = ds.summarize_notes(request.text)
        
        return {
            "session_id": request.session_id,
            "notes": notes.content
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/flashcards/topic")
async def create_flashcards_topic(request: FlashcardsRequest):
    """Generate flashcards on a specific topic"""
    try:
        if not request.topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        pipeline = session_manager.get_pipeline(request.session_id)
        cards = pipeline.create_flashcards(request.topic, k=request.k)
        
        return {
            "session_id": request.session_id,
            "topic": request.topic,
            "flashcards": cards.content
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/flashcards/notes")
async def create_flashcards_notes(request: FlashcardsRequest):
    """Generate flashcards from notes"""
    try:
        if not request.notes:
            raise HTTPException(status_code=400, detail="Notes are required")
        
        pipeline = session_manager.get_pipeline(request.session_id)
        
        from services.all_services import FlashCards
        use_groq = False if os.getenv("OPENAI_API_KEY") else True
        fc = FlashCards(use_groq=use_groq)
        cards = fc.create_flashcards_based_notes(request.notes)
        
        return {
            "session_id": request.session_id,
            "flashcards": cards.content
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quiz/full")
async def generate_quiz_full(request: QuizRequest):
    """Generate quiz from entire document"""
    try:
        pipeline = session_manager.get_pipeline(request.session_id)
        quiz = pipeline.generate_quiz(num_questions=request.num_questions)
        
        return {
            "session_id": request.session_id,
            "quiz": quiz.content,
            "num_questions": request.num_questions
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quiz/topic")
async def generate_quiz_topic(request: QuizRequest):
    """Generate quiz on a specific topic"""
    try:
        if not request.topic:
            raise HTTPException(status_code=400, detail="Topic is required")
        
        pipeline = session_manager.get_pipeline(request.session_id)
        quiz = pipeline.generate_quiz(
            query=request.topic,
            num_questions=request.num_questions,
            k=request.k
        )
        
        return {
            "session_id": request.session_id,
            "topic": request.topic,
            "quiz": quiz.content,
            "num_questions": request.num_questions
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quiz/notes")
async def generate_quiz_notes(request: QuizRequest):
    """Generate quiz from notes"""
    try:
        if not request.notes:
            raise HTTPException(status_code=400, detail="Notes are required")
        
        pipeline = session_manager.get_pipeline(request.session_id)
        
        from services.all_services import QuizGenerator
        use_groq = False if os.getenv("OPENAI_API_KEY") else True
        qg = QuizGenerator(use_groq=use_groq)
        quiz = qg.generate_quiz_from_notes(request.notes, num_questions=request.num_questions)
        
        return {
            "session_id": request.session_id,
            "quiz": quiz.content,
            "num_questions": request.num_questions
        }
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions", response_model=List[SessionInfo])
async def list_sessions():
    """List all active sessions"""
    return session_manager.list_sessions()

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session"""
    try:
        session_manager.delete_session(session_id)
        return {"message": f"Session {session_id} deleted successfully"}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    llm_provider = "openai" if openai_ok else ("groq" if groq_ok else "none")

    return {
        "status": "healthy",
        "active_sessions": len(session_manager.sessions),
        "openai_configured": openai_ok,
        "groq_configured": groq_ok,
        "active_llm_provider": llm_provider
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)