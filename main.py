import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Global variables to store the initialized chain
rag_chain = None
retriever = None

def initialize_rag_chain():
    """Initialize the RAG chain."""
    global rag_chain, retriever
    
    print("Initializing RAG chain...")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Connect to Pinecone index
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Create prompt template (NOT an f-string - these are template variables)
    template = """You are a helpful assistant that answers questions based on provided context documents.

You will receive multiple document chunks that may contain relevant information. Your task is to:
1. Synthesize information from ALL relevant chunks into a single, coherent answer
2. If chunks contain a document title or name, use it exactly as it appears
3. Combine related information from different chunks when they discuss the same topic
4. If information conflicts between chunks, mention the discrepancy
5. If you don't know the answer based on the context, say so clearly

Context from multiple document chunks:
{context}

Question: {question}

Provide a single, well-synthesized answer that combines relevant information from all the context chunks: """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain initialized successfully!")

# Modern lifespan event handler (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG chain
    initialize_rag_chain()
    yield
    # Shutdown: Cleanup (if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Q&A API",
    description="A Retrieval-Augmented Generation API for answering questions based on documents",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS (Fixed typo: allow_credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,  # Fixed typo
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str

class SourceDocument(BaseModel):
    """Model for source document information."""
    content_preview: str
    source: str

class AnswerResponse(BaseModel):  # Fixed typo
    """Response model for the /ask endpoint."""
    answer: str
    sources: list[SourceDocument]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "message": "RAG Q&A API is running",
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)  # Fixed typo
async def ask_question(request: QuestionRequest):
    """
    Main endpoint to ask questions.
    
    Takes a question and returns an answer with source documents.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Retrieve relevant documents
        source_docs = retriever.invoke(request.question)  # Fixed typo
        
        # Get answer from RAG chain
        answer = rag_chain.invoke(request.question)
        
        # Format source documents for response
        sources = []
        for doc in source_docs:
            source_info = SourceDocument(
                content_preview=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                source=doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            )
            sources.append(source_info)
        
        return AnswerResponse(  # Fixed typo
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)