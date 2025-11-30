"""
Integrated Services for SmartLearn AI
Combines your robust pipeline with FastAPI backend
"""

import os
import json
import shutil
import pathlib
import gc
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Whisper and audio
from faster_whisper import WhisperModel
import ffmpeg

# Text processing
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Subtitle parsing
try:
    import pysrt
    HAS_PYSRT = True
except ImportError:
    HAS_PYSRT = False

try:
    import webvtt
    HAS_WEBVTT = True
except ImportError:
    HAS_WEBVTT = False

# ========================================
# CONFIGURATION
# ========================================
WHISPER_MODEL = "base"  # Use "base" for faster processing, "medium" for better accuracy
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}
TRANSCRIPT_EXT = {".txt", ".srt", ".vtt", ".json"}
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200

# Directories
BASE = "."
DATA = f"{BASE}/data"
os.makedirs(f"{DATA}/audio", exist_ok=True)
os.makedirs(f"{DATA}/transcripts", exist_ok=True)
os.makedirs(f"{DATA}/chunks", exist_ok=True)

def get_supported_extensions():
    """Return all supported file extensions"""
    exts = VIDEO_EXT | AUDIO_EXT | TRANSCRIPT_EXT | {".pdf", ".parquet"}
    return sorted(list(exts))

# ========================================
# FILE PROCESSOR
# ========================================
class FileProcessor:
    """Handles file upload and format detection"""

    def __init__(self):
        self.whisper_model = None

    def initialize_whisper(self):
        """Initialize Whisper model once"""
        if self.whisper_model is None:
            print(f"Initializing Whisper model ({WHISPER_MODEL})...")
            self.whisper_model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",  # Change to "cuda" if GPU available
                compute_type="int8"  # Use "float16" for GPU
            )
            gc.collect()

    @staticmethod
    def get_file_stem(fname):
        stem = pathlib.Path(fname).stem
        import re
        sanitized_stem = re.sub(r'[^a-zA-Z0-9._-]', '_', stem)
        sanitized_stem = sanitized_stem.strip('_-.')
        if len(sanitized_stem) < 3:
            sanitized_stem = f"doc_{sanitized_stem}"
        return sanitized_stem

    @staticmethod
    def get_audio_duration(audio_path):
        """Get audio duration in seconds"""
        try:
            probe = ffmpeg.probe(audio_path)
            return float(probe['format']['duration'])
        except:
            return 0.0

    def extract_audio_from_video(self, video_path, out_audio_path, sr=16000):
        """Convert video to audio"""
        if not os.path.exists(out_audio_path):
            print(f"Extracting audio from video...")
            (
                ffmpeg
                .input(video_path)
                .output(out_audio_path, ac=1, ar=sr, format='wav', loglevel="error")
                .overwrite_output()
                .run()
            )
        return out_audio_path

    def transcribe_audio(self, audio_path, doc_id):
        """Transcribe audio using Whisper"""
        self.initialize_whisper()

        print(f"Transcribing audio...")
        segments, info = self.whisper_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            word_timestamps=False
        )

        rows = []
        for i, seg in enumerate(segments):
            rows.append({
                "doc_id": doc_id,
                "segment_idx": i,
                "start_ts": float(seg.start),
                "end_ts": float(seg.end),
                "text": seg.text.strip()
            })

        out_path = f"{DATA}/transcripts/{doc_id}.parquet"
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        return out_path

    def process_video(self, video_path, doc_id):
        """Process video: extract audio, transcribe"""
        audio_path = f"{DATA}/audio/{doc_id}.wav"
        self.extract_audio_from_video(video_path, audio_path)
        return self.transcribe_audio(audio_path, doc_id)

    def process_audio(self, audio_path, doc_id):
        """Process audio file directly"""
        ext = pathlib.Path(audio_path).suffix.lower()
        dst = f"{DATA}/audio/{doc_id}.wav"

        if ext == ".wav":
            shutil.copy(audio_path, dst)
        else:
            print(f"Converting to WAV...")
            (
                ffmpeg
                .input(audio_path)
                .output(dst, ac=1, ar=16000, format='wav', loglevel="error")
                .overwrite_output()
                .run()
            )

        return self.transcribe_audio(dst, doc_id)

    def process_parquet_file(self, parquet_path, doc_id):
        """Process existing parquet transcript file"""
        df = pd.read_parquet(parquet_path)

        if "text" not in df.columns:
            raise ValueError("Parquet file must contain a 'text' column")

        standardized_rows = []
        for idx, row in df.iterrows():
            standardized_row = {
                "doc_id": row.get("doc_id", row.get("video_id", doc_id)),
                "segment_idx": row.get("segment_idx", idx),
                "start_ts": float(row.get("start_ts", 0.0)),
                "end_ts": float(row.get("end_ts", 0.0)),
                "text": str(row["text"]).strip()
            }
            standardized_rows.append(standardized_row)

        std_df = pd.DataFrame(standardized_rows)
        out_path = f"{DATA}/transcripts/{doc_id}.parquet"
        std_df.to_parquet(out_path, index=False)
        return out_path

    def process_transcript_file(self, transcript_path, doc_id):
        """Process transcript files (txt, srt, vtt, json)"""
        ext = pathlib.Path(transcript_path).suffix.lower()

        if ext == ".txt":
            with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            rows = [{"segment_idx": 0, "start_ts": 0.0, "end_ts": 0.0, "text": text}]

        elif ext == ".srt":
            if not HAS_PYSRT:
                raise ValueError("pysrt not installed. Install: pip install pysrt")
            subs = pysrt.open(transcript_path, encoding='utf-8')
            rows = [{
                "segment_idx": i,
                "start_ts": s.start.ordinal/1000.0,
                "end_ts": s.end.ordinal/1000.0,
                "text": s.text.replace('\n', ' ').strip()
            } for i, s in enumerate(subs)]

        elif ext == ".vtt":
            if not HAS_WEBVTT:
                raise ValueError("webvtt-py not installed. Install: pip install webvtt-py")
            
            def hms_to_s(hms):
                parts = hms.split(':')
                if len(parts) == 3:
                    h, m, s = parts
                    return int(h)*3600 + int(m)*60 + float(s)
                return 0.0

            rows = [{
                "segment_idx": i,
                "start_ts": hms_to_s(cap.start),
                "end_ts": hms_to_s(cap.end),
                "text": cap.text.replace('\n', ' ').strip()
            } for i, cap in enumerate(webvtt.read(transcript_path))]

        elif ext == ".json":
            with open(transcript_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            segs = obj.get("segments", [])
            rows = [{
                "segment_idx": i,
                "start_ts": float(s.get("start", 0.0)),
                "end_ts": float(s.get("end", 0.0)),
                "text": str(s.get("text", "")).strip()
            } for i, s in enumerate(segs)]

        else:
            raise ValueError(f"Unsupported transcript format: {ext}")

        df = pd.DataFrame(rows)
        df.insert(0, "doc_id", doc_id)
        out_path = f"{DATA}/transcripts/{doc_id}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path

    def process_pdf(self, pdf_path, doc_id):
        """Process PDF file"""
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text = " ".join([doc.page_content for doc in docs])

        rows = [{"segment_idx": 0, "start_ts": 0.0, "end_ts": 0.0, "text": text}]
        df = pd.DataFrame(rows)
        df.insert(0, "doc_id", doc_id)
        out_path = f"{DATA}/transcripts/{doc_id}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path

    def route_file(self, file_path):
        """Main routing function for any file type"""
        ext = pathlib.Path(file_path).suffix.lower()
        doc_id = self.get_file_stem(file_path)

        if ext in VIDEO_EXT:
            return self.process_video(file_path, doc_id)
        elif ext in AUDIO_EXT:
            return self.process_audio(file_path, doc_id)
        elif ext == ".pdf":
            return self.process_pdf(file_path, doc_id)
        elif ext == ".parquet":
            return self.process_parquet_file(file_path, doc_id)
        elif ext in TRANSCRIPT_EXT:
            return self.process_transcript_file(file_path, doc_id)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

# ========================================
# TEXT CHUNKER
# ========================================
class TextChunker:
    """Smart text chunking with sentence awareness"""

    @staticmethod
    def normalize_whitespace(text):
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def basic_cleanup(text):
        text = text.replace("'", "'").replace(""", "\"").replace(""", "\"")
        return TextChunker.normalize_whitespace(text)

    @staticmethod
    def sentence_time_expand(seg_text, seg_start, seg_end):
        """Split segment into sentences with proportional timestamps"""
        txt = TextChunker.basic_cleanup(seg_text)
        sents = [s for s in sent_tokenize(txt) if s.strip()]

        if not sents:
            return []

        total_chars = sum(len(s) for s in sents)
        if total_chars == 0:
            return []

        dur = max(0.0, float(seg_end) - float(seg_start))
        out = []
        cur = float(seg_start)

        for s in sents:
            frac = len(s) / total_chars
            sdur = frac * dur
            out.append({"text": s, "start_ts": cur, "end_ts": cur + sdur})
            cur += sdur

        if out:
            out[-1]["end_ts"] = float(seg_end)

        return out

    @staticmethod
    def build_sentence_table(transcript_parquet_path):
        """Convert transcript to sentence-level dataframe"""
        df = pd.read_parquet(transcript_parquet_path)
        rows = []

        for _, r in df.iterrows():
            exp = TextChunker.sentence_time_expand(r["text"], r["start_ts"], r["end_ts"])
            rows.extend(exp)

        if not rows:
            rows = [{
                "text": TextChunker.basic_cleanup(" ".join(df["text"].tolist())),
                "start_ts": 0.0,
                "end_ts": 0.0
            }]

        return pd.DataFrame(rows)

    @staticmethod
    def make_chunks_from_sentences(sents_df, max_chars=1800, overlap_chars=200):
        """Create overlapping chunks respecting sentence boundaries"""
        chunks = []
        buf_text = ""
        buf_starts = []
        buf_ends = []

        def flush_buffer():
            if not buf_text.strip():
                return
            chunks.append({
                "text": buf_text.strip(),
                "start_ts": min(buf_starts) if buf_starts else 0.0,
                "end_ts": max(buf_ends) if buf_ends else 0.0
            })

        for _, row in sents_df.iterrows():
            s = str(row["text"]).strip()
            st, et = float(row["start_ts"]), float(row["end_ts"])

            if not s:
                continue

            if len(buf_text) + len(s) + 1 <= max_chars:
                buf_text = (buf_text + " " + s).strip() if buf_text else s
                buf_starts.append(st)
                buf_ends.append(et)
            else:
                flush_buffer()
                buf_text = s
                buf_starts = [st]
                buf_ends = [et]

        flush_buffer()
        return chunks

    @staticmethod
    def build_and_save_chunks(transcript_path, max_chars=1800, overlap_chars=200):
        """Main function to create and save chunks"""
        doc_id = os.path.splitext(os.path.basename(transcript_path))[0]
        sents_df = TextChunker.build_sentence_table(transcript_path)
        chunks = TextChunker.make_chunks_from_sentences(sents_df, max_chars, overlap_chars)

        out_rows = []
        for i, c in enumerate(chunks):
            out_rows.append({
                "doc_id": doc_id,
                "chunk_idx": i,
                "start_ts": float(c["start_ts"]),
                "end_ts": float(c["end_ts"]),
                "text": c["text"]
            })

        cdf = pd.DataFrame(out_rows)
        outp = f"{DATA}/chunks/{doc_id}_chunks.parquet"
        cdf.to_parquet(outp, index=False)
        return outp

# ========================================
# VECTOR STORE
# ========================================
class VectorStore:
    def __init__(self, collection_name="smartlearn", model_name='sentence-transformers/all-mpnet-base-v2'):
        self.collection_name = collection_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        self.persist_directory = f"./temp_vector_stores/{collection_name}"

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    def add_documents(self, docs):
        """Add documents to vector store"""
        self.vector_store.add_documents(docs)

    def similarity_search_with_score(self, query, k=5):
        """Search with similarity scores"""
        results = self.vector_store.similarity_search_with_score(query, k=k)
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results

    def get_all_chunks(self):
        """Returns a list of Document objects"""
        raw_data = self.vector_store.get()
        docs = raw_data["documents"]
        metadatas = raw_data.get("metadatas", [{}] * len(docs))
        return [Document(page_content=doc, metadata=meta) for doc, meta in zip(docs, metadatas)]

# ========================================
# DOCUMENT SUMMARIZER
# ========================================
class DocumentSummarizer:
    def __init__(self, model_name="gpt-3.5-turbo", use_groq=True):
        if use_groq:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model_name="llama-3.1-8b-instant", verbose=False)
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.5)

    def summarize_all(self, chunks):
        """Summarize all chunks"""
        context = "\n\n".join([chunk.page_content for chunk in chunks[:15]])
        context = context[:15000]

        prompt_template = """You are an expert at summarizing documents. Based on the following content, provide a comprehensive summary that captures the main ideas, key points, and important details.

Content:
{context}

Provide a clear, well-structured summary:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
        summary = self.llm.invoke(prompt.format(context=context))
        return summary

    def summarize_query(self, vector_store, query, k=3):
        """Summarize based on query"""
        results = vector_store.similarity_search_with_score(query)
        relevant_docs = [doc for doc, score in results[:k]]

        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        context = context[:15000]

        prompt_template = """Based on the following context, provide a meaningful and insightful summary focusing on: {query}

Context: {context}

If the text does not contain information related to the query, say 'Information not found in documents related to the query'

Summary:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        summary = self.llm.invoke(prompt.format(query=query, context=context))

        return summary, relevant_docs

    def summarize_notes(self, notes):
        """Summarize user-provided notes"""
        notes = notes[:15000]
        prompt_template = """Based on the given notes: {notes}, provide a meaningful and insightful summary.
If the notes are empty, say 'Notes not found!'
Summary:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["notes"])
        summary = self.llm.invoke(prompt.format(notes=notes))
        return summary

# ========================================
# FLASHCARDS
# ========================================
class FlashCards:
    def __init__(self, model_name="gpt-3.5-turbo", use_groq=True):
        if use_groq:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model_name="llama-3.1-8b-instant", verbose=False)
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.5)

    def create_flashcards_on_topic(self, vector_store, query, k=3):
        """Create flashcards on specific topic"""
        results = vector_store.similarity_search_with_score(query)
        relevant_docs = [doc for doc, score in results[:k]]

        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        context = context[:15000]

        prompt_template = """You are an expert in creating flashcards based on the provided information.

Based on the following context, create up to 8 flashcards focusing on the topic: {query}

Context:
{context}

Each flashcard should have:
1. A question or term on one side
2. A short (2-3 lines), memorable answer
3. Simple language that is easy to understand

Format your response as:
Front: [Question]
Back: [Answer]

Flashcards:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        cards = self.llm.invoke(prompt.format(query=query, context=context))

        return cards, relevant_docs

    def create_flashcards_based_notes(self, notes):
        """Create flashcards from notes"""
        notes = notes[:15000]
        prompt_template = """Create up to 8 flashcards based on these notes:
{notes}

Format:
Front: [Question]
Back: [Answer]

Flashcards:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["notes"])
        cards = self.llm.invoke(prompt.format(notes=notes))
        return cards

# ========================================
# QUIZ GENERATOR
# ========================================
class QuizGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", use_groq=True):
        if use_groq:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(model_name="llama-3.1-8b-instant", verbose=False)
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.5)

    def generate_quiz_all(self, chunks, num_questions=10):
        """Generate quiz from all chunks"""
        context = "\n\n".join([chunk.page_content for chunk in chunks[:15]])
        context = context[:15000]

        prompt_template = """Create a quiz with {num_questions} multiple-choice questions.

Content:
{context}

Format:
Question 1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]
Explanation: [Brief explanation]

Quiz:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["num_questions", "context"])
        quiz = self.llm.invoke(prompt.format(num_questions=num_questions, context=context))
        return quiz

    def generate_quiz_on_topic(self, vector_store, query, num_questions=10, k=5):
        """Generate quiz on specific topic"""
        results = vector_store.similarity_search_with_score(query)
        relevant_docs = [doc for doc, score in results[:k]]

        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        context = context[:15000]

        prompt_template = """Create a quiz with {num_questions} questions about: {query}

Context:
{context}

Format:
Question 1: [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct Answer: [A/B/C/D]
Explanation: [Brief explanation]

Quiz:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["num_questions", "query", "context"])
        quiz = self.llm.invoke(prompt.format(num_questions=num_questions, query=query, context=context))

        return quiz, relevant_docs

    def generate_quiz_from_notes(self, notes, num_questions=10):
        """Generate quiz from notes"""
        notes = notes[:15000]
        prompt_template = """Create a quiz with {num_questions} questions from these notes:
{notes}

Format:
Question 1: [Question]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct Answer: [A/B/C/D]

Quiz:"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["num_questions", "notes"])
        quiz = self.llm.invoke(prompt.format(num_questions=num_questions, notes=notes))
        return quiz

# ========================================
# INTEGRATED PIPELINE
# ========================================
class IntegratedPipeline:
    """Main pipeline with multimodal support"""

    def __init__(self, use_groq=True):
        self.file_processor = FileProcessor()
        self.use_groq = use_groq
        self.vector_store = None
        self.current_doc_id = None

    def process_file(self, file_path):
        """Process any supported file type"""
        # Process file
        transcript_path = self.file_processor.route_file(file_path)
        self.current_doc_id = self.file_processor.get_file_stem(file_path)

        # Create chunks
        chunks_path = TextChunker.build_and_save_chunks(
            transcript_path,
            max_chars=CHUNK_SIZE,
            overlap_chars=CHUNK_OVERLAP
        )

        # Load chunks as documents
        df = pd.read_parquet(chunks_path)
        documents = [
            Document(
                page_content=row["text"],
                metadata={
                    "doc_id": row["doc_id"],
                    "chunk_idx": row["chunk_idx"],
                    "start_ts": row["start_ts"],
                    "end_ts": row["end_ts"]
                }
            ) for _, row in df.iterrows()
        ]

        # Create vector store
        self.vector_store = VectorStore(collection_name=f"smartlearn_{self.current_doc_id}")
        self.vector_store.add_documents(documents)

        return self.vector_store

    def summarize(self, query, k=3):
        """Generate summary based on query"""
        if not self.vector_store:
            raise ValueError("No file processed")
        ds = DocumentSummarizer(use_groq=self.use_groq)
        summary, _ = ds.summarize_query(self.vector_store, query, k=k)
        return summary

    def summarize_all(self):
        """Summarize entire document"""
        if not self.vector_store:
            raise ValueError("No file processed")
        ds = DocumentSummarizer(use_groq=self.use_groq)
        all_chunks = self.vector_store.get_all_chunks()
        summary = ds.summarize_all(all_chunks)
        return summary

    def make_notes(self, query, k=3):
        """Create structured notes"""
        if not self.vector_store:
            raise ValueError("No file processed")
        ds = DocumentSummarizer(use_groq=self.use_groq)
        results = self.vector_store.similarity_search_with_score(query)
        relevant_docs = [doc for doc, score in results[:k]]

        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        prompt_template = """Create structured notes focusing on: {query}

Context:
{context}

Notes should include:
1. Key Points
2. Important Details
3. Actionable Insights"""

        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "context"])
        notes = ds.llm.invoke(prompt.format(query=query, context=context))
        return notes

    def create_flashcards(self, query, k=3):
        """Generate flashcards on topic"""
        if not self.vector_store:
            raise ValueError("No file processed")
        fc = FlashCards(use_groq=self.use_groq)
        cards, _ = fc.create_flashcards_on_topic(self.vector_store, query, k=k)
        return cards

    def generate_quiz(self, query=None, num_questions=10, k=5):
        """Generate quiz"""
        if not self.vector_store:
            raise ValueError("No file processed")
        qg = QuizGenerator(use_groq=self.use_groq)

        if query:
            quiz, _ = qg.generate_quiz_on_topic(
                self.vector_store,
                query,
                num_questions=num_questions,
                k=k
            )
        else:
            all_chunks = self.vector_store.get_all_chunks()
            quiz = qg.generate_quiz_all(all_chunks, num_questions=num_questions)

        return quiz

# ========================================
# SESSION MANAGER
# ========================================
class SessionManager:
    """Manage multiple user sessions"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def add_session(self, session_id: str, pipeline: IntegratedPipeline, filename: str, file_path: str):
        """Add a new session"""
        self.sessions[session_id] = {
            "pipeline": pipeline,
            "filename": filename,
            "file_path": file_path,
            "doc_id": pipeline.current_doc_id,
            "chunks_count": len(pipeline.vector_store.get_all_chunks())
        }

    def get_pipeline(self, session_id: str) -> IntegratedPipeline:
        """Get pipeline for session"""
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        return self.sessions[session_id]["pipeline"]

    def list_sessions(self) -> List[Dict]:
        """List all active sessions"""
        return [
            {
                "session_id": sid,
                "filename": data["filename"],
                "doc_id": data["doc_id"],
                "file_type": Path(data["filename"]).suffix,
                "chunks_count": data["chunks_count"]
            }
            for sid, data in self.sessions.items()
        ]

    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        
        # Clean up files
        session = self.sessions[session_id]
        if os.path.exists(session["file_path"]):
            os.remove(session["file_path"])
        
        del self.sessions[session_id]