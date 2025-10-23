# Document Intelligence API - MVP

A FastAPI backend for document summarization, note-making, flashcards, and quiz generation using LLMs and vector stores.

## Features

- Upload PDF or TXT documents
- Automatic chunking and vector storage
- Summarize full documents or by topic
- Generate structured notes
- Create flashcards and quizzes from documents or notes
- REST API endpoints for all features

## Project Structure

```
.
├── main.py
├── requirements_curr.txt
├── .env
├── services/
│   └── all_services.py
├── temp_uploads/
├── temp_vector_stores/
└── ...
```

## Setup

1. **Clone the repository**

2. **Install dependencies**

```bash
pip install -r requirements_curr.txt
```

3. **Set up environment variables**

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

4. **Run the API**

```bash
uvicorn main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000).

## API Endpoints

- `POST /api/upload` — Upload a document
- `POST /api/summarize/full` — Summarize the full document
- `POST /api/summarize/topic` — Summarize on a specific topic
- `POST /api/notes/topic` — Create notes on a topic
- `POST /api/notes/text` — Create notes from text
- `POST /api/flashcards/topic` — Generate flashcards on a topic
- `POST /api/flashcards/notes` — Generate flashcards from notes
- `POST /api/quiz/full` — Generate quiz from full document
- `POST /api/quiz/topic` — Generate quiz on a topic
- `POST /api/quiz/notes` — Generate quiz from notes
- `GET /api/sessions` — List all active sessions

## Notes

- Only PDF and TXT files are supported for upload.
- All data is stored in memory and temporary folders; sessions are lost on restart.
- Requires Python 3.12.

---

**Developed for educational and prototyping purposes.**