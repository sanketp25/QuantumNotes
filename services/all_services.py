"""
Complete Services Module for Document Intelligence API
Save this file in: services/all_services.py

Then in main.py, import as:
from services.all_services import (
    DocumentService, SummaryService, NotesService, 
    FlashcardService, QuizService
)
"""

from pathlib import Path
from typing import List, Tuple
# from langchain. import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_classic.prompts import PromptTemplate
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# DOCUMENT SERVICE - Handles file upload, chunking, vector store creation
# ============================================================================

class DocumentService:
    """Service for processing and chunking documents"""
    
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    async def process_document(
        self, 
        file_path: str, 
        session_id: str,
        chunking_method: str = "recursive"
    ) -> Tuple[List[Document], Chroma]:
        """
        Process document: load, chunk, create vector store
        
        Args:
            file_path: Path to uploaded file
            session_id: Unique session identifier
            chunking_method: 'recursive' or 'semantic'
        
        Returns:
            Tuple of (chunks, vector_store)
        """
        docs = self._load_document(file_path)
        chunks = self._chunk_document(docs, chunking_method)
        vector_store = self._create_vector_store(chunks, session_id)
        return chunks, vector_store
    
    def _load_document(self, file_path: str) -> List[Document]:
        """Load PDF or TXT file"""
        if Path(file_path).suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif Path(file_path).suffix == ".txt":
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type. Use PDF or TXT.")
        return loader.load()

    def _chunk_document(self, docs: List[Document], method: str="recursive") -> List[Document]:
        """Chunk document using specified method"""
        if method == "semantic":
            pass
            # splitter = SemanticChunker(self.embedding_function)
        else:  # recursive (default)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                add_start_index=True
            )
        
        chunks = splitter.split_documents(docs)
        print(f"✂️  Created {len(chunks)} chunks using {method} method")
        return chunks
    
    def _create_vector_store(self, chunks: List[Document], session_id: str) -> Chroma:
        """Create Chroma vector store with persistence"""
        persist_dir = f"temp_vector_stores/{session_id}"
        
        vector_store = Chroma(
            collection_name=f"session_{session_id}",
            embedding_function=self.embedding_function,
            persist_directory=persist_dir
        )
        
        vector_store.add_documents(chunks)
        print(f"💾 Vector store created with {len(chunks)} chunks")
        return vector_store


# ============================================================================
# SUMMARY SERVICE - Handles all summarization operations
# ============================================================================

class SummaryService:
    """Service for document and text summarization"""
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
    
    async def summarize_all(self, chunks: List[Document]) -> dict:
        """
        Summarize all document chunks using map-reduce
        
        Args:
            chunks: List of document chunks
        
        Returns:
            Dictionary with 'output_text' key containing summary
        """
        print(f"📝 Summarizing {len(chunks)} chunks...")
        chain = load_summarize_chain(self.llm, chain_type="map_reduce")
        summary = chain.invoke(chunks)
        print("✅ Summary complete")
        return summary
    
    async def summarize_query(
        self, 
        vector_store: Chroma, 
        query: str, 
        k: int = 3
    ) -> Tuple[any, List[Document]]:
        """
        Summarize document focusing on a specific query
        
        Args:
            vector_store: Chroma vector store
            query: Topic to focus on
            k: Number of relevant chunks to use
        
        Returns:
            Tuple of (summary, relevant_documents)
        """
        # Get relevant documents
        results = vector_store.similarity_search_with_score(query)
        relevant_docs = [doc for doc, score in results[:k]]
        
        # Create prompt
        prompt = PromptTemplate(
            template="""You are an expert at analyzing documents and creating structured notes.

Based on this context, create comprehensive notes on: {query}

Context:
{context}

Format your notes as follows:
1. Key Points: [List the main ideas and concepts]
2. Important Details: [Supporting information and specifics]
3. Actionable Insights: [What can be learned or applied]
4. Additional Information: [Any other relevant details]

Notes:""",
            input_variables=["query", "context"]
        )
        
        # Generate notes
        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        notes = self.llm.invoke(prompt.format(query=query, context=context))
        
        return notes, relevant_docs
    
    async def make_notes_notes(self, notes_text: str) -> any:
        """
        Create structured notes from user-provided text
        
        Args:
            notes_text: User's text/notes
        
        Returns:
            Structured notes
        """
        prompt = PromptTemplate(
            template="""Create structured notes from this text:

{notes}

Format your notes as follows:
1. Key Points: [List the main ideas and concepts]
2. Important Details: [Supporting information and specifics]
3. Actionable Insights: [What can be learned or applied]
4. Additional Information: [Any other relevant details]

Notes:""",
            input_variables=["notes"]
        )
        
        return self.llm.invoke(prompt.format(notes=notes_text))


# ============================================================================
# FLASHCARD SERVICE - Handles flashcard generation
# ============================================================================

class FlashcardService:
    """Service for generating flashcards"""
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
    
    async def create_flashcards_on_topic(
        self, 
        vector_store: Chroma, 
        query: str, 
        k: int = 3
    ) -> Tuple[any, List[Document]]:
        """
        Generate flashcards on a specific topic
        
        Args:
            vector_store: Chroma vector store
            query: Topic to create flashcards on
            k: Number of relevant chunks
        
        Returns:
            Tuple of (flashcards, relevant_documents)
        """
        # Get relevant documents
        results = vector_store.similarity_search_with_score(query)
        relevant_docs = [doc for doc, score in results[:k]]
        
        # Create prompt
        prompt = PromptTemplate(
            template="""You are an expert at creating educational flashcards.

Create up to 8 flashcards on the topic: {query}

Context:
{context}

Guidelines:
- Break down complex concepts into simple, memorable pieces
- Each flashcard should have a clear question and concise answer
- Use simple, easy-to-understand language
- Answers should be 2-3 lines maximum

Format each flashcard as:
Front: [Question or term]
Back: [Answer or definition]

Flashcards:""",
            input_variables=["query", "context"]
        )
        
        # Generate flashcards
        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        cards = self.llm.invoke(prompt.format(query=query, context=context))
        
        return cards, relevant_docs
    
    async def create_flashcards_based_notes(self, notes: str) -> any:
        """
        Generate flashcards from user-provided notes
        
        Args:
            notes: User's notes text
        
        Returns:
            Flashcards
        """
        prompt = PromptTemplate(
            template="""Create up to 8 flashcards from these notes:

{notes}

Guidelines:
- Break down concepts into simple, memorable pieces
- Use simple, easy-to-understand language
- Answers should be 2-3 lines maximum

Format each flashcard as:
Front: [Question or term]
Back: [Answer or definition]

Flashcards:""",
            input_variables=["notes"]
        )
        
        return self.llm.invoke(prompt.format(notes=notes))


# ============================================================================
# QUIZ SERVICE - Handles quiz generation
# ============================================================================

class QuizService:
    """Service for generating multiple-choice quizzes"""
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))
    
    async def generate_quiz_all(
        self, 
        chunks: List[Document], 
        num_questions: int = 10
    ) -> any:
        """
        Generate quiz from all document chunks
        
        Args:
            chunks: List of document chunks
            num_questions: Number of questions to generate
        
        Returns:
            Quiz with questions and answers
        """
        # Use first 15 chunks to avoid token limits
        context = "\n\n".join([chunk.page_content for chunk in chunks[:15]])
        
        prompt = PromptTemplate(
            template="""Create a quiz with {num_questions} multiple-choice questions based on this content.

Content:
{context}

Guidelines:
- Create {num_questions} questions that test key concepts
- Each question should have 4 options (A, B, C, D)
- Clearly mark the correct answer
- Provide a brief explanation for each answer
- Mix difficulty levels (easy, medium, hard)

Format each question exactly as:

Question 1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]
Explanation: [Brief explanation]

Quiz:""",
            input_variables=["num_questions", "context"]
        )
        
        return self.llm.invoke(
            prompt.format(num_questions=num_questions, context=context)
        )
    
    async def generate_quiz_on_topic(
        self, 
        vector_store: Chroma, 
        query: str, 
        num_questions: int = 10, 
        k: int = 5
    ) -> Tuple[any, List[Document]]:
        """
        Generate quiz on a specific topic
        
        Args:
            vector_store: Chroma vector store
            query: Topic to create quiz on
            num_questions: Number of questions
            k: Number of relevant chunks
        
        Returns:
            Tuple of (quiz, relevant_documents)
        """
        # Get relevant documents
        results = vector_store.similarity_search_with_score(query)
        relevant_docs = [doc for doc, score in results[:k]]
        
        # Create prompt
        prompt = PromptTemplate(
            template="""Create {num_questions} multiple-choice questions on the topic: {query}

Context:
{context}

Guidelines:
- Focus questions specifically on: {query}
- Each question should have 4 options (A, B, C, D)
- Clearly mark the correct answer
- Provide a brief explanation
- Mix difficulty levels

DO NOT make up information. If the context lacks sufficient information, create fewer questions and mention this.

Format each question exactly as:

Question 1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]
Explanation: [Brief explanation]

Quiz:""",
            input_variables=["num_questions", "query", "context"]
        )
        
        # Generate quiz
        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        quiz = self.llm.invoke(
            prompt.format(num_questions=num_questions, query=query, context=context)
        )
        
        return quiz, relevant_docs
    
    async def generate_quiz_from_notes(
        self, 
        notes: str, 
        num_questions: int = 10
    ) -> any:
        """
        Generate quiz from user-provided notes
        
        Args:
            notes: User's notes text
            num_questions: Number of questions
        
        Returns:
            Quiz
        """
        prompt = PromptTemplate(
            template="""Create {num_questions} multiple-choice questions from these notes:

{notes}

Guidelines:
- Test understanding of the key concepts
- Each question should have 4 options (A, B, C, D)
- Clearly mark the correct answer
- Provide a brief explanation

DO NOT make up information. If notes are insufficient, say so.

Format each question exactly as:

Question 1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A/B/C/D]
Explanation: [Brief explanation]

Quiz:""",
            input_variables=["num_questions", "notes"]
        )
        
        return self.llm.invoke(
            prompt.format(num_questions=num_questions, notes=notes)
        )



class NotesService:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
    
    async def make_notes_query(
        self, vector_store: Chroma, query: str, k: int = 3
    ) -> Tuple[any, List[Document]]:
        """Create notes on topic"""
        results = vector_store.similarity_search_with_score(query)
        relevant_docs = [doc for doc, score in results[:k]]
        
        prompt = PromptTemplate(
            template="""You are an expert at creating structured notes.

Based on this context, create notes on: {query}

Context:
{context}

Format:
1. Key Points: [main ideas]
2. Important Details: [supporting info]
3. Actionable Insights: [what can be done]
4. Additional Information: [other relevant details]

Notes:""",
            input_variables=["query", "context"]
        )
        
        context = "\n\n".join([chunk.page_content for chunk in relevant_docs])
        notes = self.llm.invoke(prompt.format(query=query, context=context))
        return notes, relevant_docs
    
    async def make_notes_notes(self, notes_text: str) -> any:
        """Create structured notes from text"""
        prompt = PromptTemplate(
            template="""Create structured notes from this text:

{notes}

Format:
1. Key Points: [main ideas]
2. Important Details: [supporting info]
3. Actionable Insights: [what can be done]
4. Additional Information: [other relevant details]

Notes:""",
            input_variables=["notes"]
        )
        return self.llm.invoke(prompt.format(notes=notes_text))