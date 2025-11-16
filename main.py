import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- Global State (Initialized during lifespan) ---
app_state = {}

def initialize_rag_chain():
    """Initializes and returns the RAG chain and retriever."""
    print("Initializing RAG chain...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    template = """You are a helpful assistant that answers questions based on provided context documents... Context: {context} Question: {question} Provide a single, well-synthesized answer..."""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain initialized successfully!")
    return rag_chain, retriever

# This function will now be our dependency
def get_rag_initializer():
    """This function is a dependency that returns the real initializer."""
    return initialize_rag_chain

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG chain and store in app_state
    initializer = get_rag_initializer()
    rag_chain, retriever = initializer()
    app_state["rag_chain"] = rag_chain
    app_state["retriever"] = retriever
    yield
    # Shutdown: Clear state
    app_state.clear()

# --- Dependency Getters ---
def get_rag_chain() -> Runnable:
    return app_state["rag_chain"]

def get_retriever():
    return app_state["retriever"]

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Q&A API",
    description="A Retrieval-Augmented Generation API...",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    content_preview: str
    source: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "RAG Q&A API is running", "status": "healthy", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    rag_chain: Runnable = Depends(get_rag_chain),
    retriever = Depends(get_retriever)
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        source_docs = retriever.invoke(request.question)
        answer = rag_chain.invoke(request.question)
        
        sources = [
            SourceDocument(
                content_preview=doc.page_content[:200] + "...",
                source=doc.metadata.get('source', 'Unknown')
            ) for doc in source_docs
        ]
        
        return AnswerResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)