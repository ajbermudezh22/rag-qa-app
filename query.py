import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
def initialize_rag_chain():
    """Initialize the RAG (Retrieval-Augmented Generation) chain."""
    print("Initializing RAG chain...")
    
    # Step 1: Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Step 2: Connect to existing Pinecone index
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-index")
    
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    # Step 3: Initialize the LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
    )
    
    # Step 4: Create retriever - INCREASED from 3 to 5 chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Step 5: Create prompt template - ENHANCED for better synthesis
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
    
    # Step 6: Create the chain using LCEL
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

def query(question, rag_chain, retriever):
    """Query the RAG system with a question."""
    print(f"\nQuestion: {question}")
    print("Searching knowledge base and generating answer...\n")
    
    # Retrieve relevant documents
    source_docs = retriever.invoke(question)
    
    # Get answer
    answer = rag_chain.invoke(question)
    
    # Display the answer
    print("=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(answer)
    print("\n" + "=" * 50)
    print("SOURCES (retrieved from these document chunks):")
    print("=" * 50)
    
    # Show which chunks were used
    for i, doc in enumerate(source_docs, 1):
        print(f"\nSource {i}:")
        print(f"Content preview: {doc.page_content[:200]}...")
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            print(f"From: {doc.metadata['source']}")
    
    return answer, source_docs

def main():
    """Main function for interactive querying."""
    print("=" * 50)
    print("RAG Q&A System")
    print("=" * 50)
    
    # Initialize the RAG chain
    rag_chain, retriever = initialize_rag_chain()
    
    # Interactive loop
    print("\nEnter your questions (type 'quit' or 'exit' to stop):\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        try:
            query(question, rag_chain, retriever)
            print("\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    main()