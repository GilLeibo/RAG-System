"""
RAG (Retrieval-Augmented Generation) System Implementation
A comprehensive RAG system that combines document retrieval with LLM generation.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
import json
import pickle
from pathlib import Path

# Core dependencies
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import openai
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.schema import Document
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    exit(1)

# Load environment variables
load_dotenv()

class DocumentProcessor:
    """Handles document loading, processing, and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document from file path."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in ['.txt', '.md']:
            loader = TextLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        return loader.load()
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process multiple documents and return chunks."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                documents = self.load_document(file_path)
                chunks = self.chunk_documents(documents)
                all_chunks.extend(chunks)
                print(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return all_chunks

class VectorStore:
    """Handles vector storage and similarity search using FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not documents:
            return
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index if it doesn't exist
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents."""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return documents
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                doc.metadata['similarity_score'] = float(score)
                results.append(doc)
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk."""
        if self.index is not None:
            faiss.write_index(self.index, f"{filepath}.index")
            with open(f"{filepath}.docs", 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vector store from disk."""
        try:
            self.index = faiss.read_index(f"{filepath}.index")
            with open(f"{filepath}.docs", 'rb') as f:
                self.documents = pickle.load(f)
            print(f"Vector store loaded from {filepath}")
        except Exception as e:
            print(f"Error loading vector store: {e}")

class LLMGenerator:
    """Handles LLM integration for text generation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def generate_response(self, query: str, context: str, max_tokens: int = 500) -> str:
        """Generate response using LLM with retrieved context."""
        prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

class RAGSystem:
    """Main RAG system that combines retrieval and generation."""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "gpt-3.5-turbo",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(embedding_model)
        self.llm_generator = LLMGenerator(llm_model)
    
    def add_documents(self, file_paths: List[str]):
        """Add documents to the RAG system."""
        documents = self.document_processor.process_documents(file_paths)
        self.vector_store.add_documents(documents)
    
    def query(self, question: str, k: int = 5, max_tokens: int = 500) -> Dict[str, Any]:
        """Query the RAG system."""
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(question, k)
        
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "retrieved_docs": []
            }
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate answer
        answer = self.llm_generator.generate_response(question, context, max_tokens)
        
        # Prepare sources
        sources = []
        for doc in retrieved_docs:
            source = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": doc.metadata.get('similarity_score', 0)
            }
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs
        }
    
    def save_system(self, filepath: str):
        """Save the RAG system."""
        self.vector_store.save(filepath)
        print(f"RAG system saved to {filepath}")
    
    def load_system(self, filepath: str):
        """Load the RAG system."""
        self.vector_store.load(filepath)
        print(f"RAG system loaded from {filepath}")

def main():
    """Example usage of the RAG system."""
    # Initialize RAG system
    rag = RAGSystem()
    
    # Example: Add documents (you would replace these with your actual documents)
    print("Setting up RAG system...")
    
    # Create sample documents for demonstration
    sample_docs = [
        "sample_doc1.txt",
        "sample_doc2.txt"
    ]
    
    # Create sample content
    with open("sample_doc1.txt", "w") as f:
        f.write("Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.")
    
    with open("sample_doc2.txt", "w") as f:
        f.write("Machine Learning is a subset of AI that focuses on algorithms that can learn and improve from experience. It includes supervised learning, unsupervised learning, and reinforcement learning techniques.")
    
    try:
        # Add documents to the system
        rag.add_documents(sample_docs)
        
        # Example queries
        queries = [
            "What is artificial intelligence?",
            "What is machine learning?",
            "How does AI relate to machine learning?"
        ]
        
        print("\n" + "="*50)
        print("RAG System Demo")
        print("="*50)
        
        for query in queries:
            print(f"\nQuery: {query}")
            result = rag.query(query)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['sources'])} documents found")
            print("-" * 30)
        
        # Save the system
        rag.save_system("rag_system")
        
    except Exception as e:
        print(f"Error in demo: {e}")
    
    finally:
        # Clean up sample files
        for doc in sample_docs:
            if os.path.exists(doc):
                os.remove(doc)

if __name__ == "__main__":
    main()

