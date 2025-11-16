import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

def initialize_pinecone():
    """Initialize Pinecone connection and create index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists, create if it doesn't
    if index_name not in pc.list_indexes().names():
        print(f"Creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index '{index_name}' created successfully!")
    else:
        print(f"Index '{index_name}' already exists.")
    
    return pc, index_name

def load_documents(data_dir="data"):
    """Load all PDF documents from the data directory."""
    print(f"Loading documents from {data_dir}...")
    
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    print("Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Characters per chunk
        chunk_overlap=200,  # Overlap between chunks for context
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def ingest_to_pinecone(chunks, pc, index_name):
    """Generate embeddings and store them in Pinecone."""
    print("Generating embeddings and uploading to Pinecone...")
    
    # Initialize embeddings (reads OPENAI_API_KEY from environment automatically)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Cost-effective embedding model
    )
    
    # Create vector store and add documents
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    print("Documents successfully ingested into Pinecone!")

def main():
    """Main function to orchestrate the ingestion process."""
    print("=" * 50)
    print("Starting document ingestion process...")
    print("=" * 50)
    
    # Step 1: Initialize Pinecone
    pc, index_name = initialize_pinecone()
    
    # Step 2: Load documents
    documents = load_documents()
    
    if not documents:
        print("No documents found in the 'data' directory!")
        print("Please add some PDF files to the 'data' folder.")
        return
    
    # Step 3: Split documents into chunks
    chunks = split_documents(documents)
    
    # Step 4: Ingest to Pinecone
    ingest_to_pinecone(chunks, pc, index_name)
    
    print("=" * 50)
    print("Ingestion complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()