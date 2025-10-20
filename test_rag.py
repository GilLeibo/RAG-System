# Simple test script for the RAG system
# This script tests the basic functionality without requiring OpenAI API

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("+ numpy imported successfully")
    except ImportError as e:
        print(f"- numpy import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("+ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"- sentence-transformers import failed: {e}")
        return False
    
    try:
        import faiss
        print("+ faiss imported successfully")
    except ImportError as e:
        print(f"- faiss import failed: {e}")
        return False
    
    try:
        import openai
        print("+ openai imported successfully")
    except ImportError as e:
        print(f"- openai import failed: {e}")
        return False
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.document_loaders import TextLoader
        from langchain.schema import Document
        print("+ langchain imported successfully")
    except ImportError as e:
        print(f"- langchain import failed: {e}")
        return False
    
    return True

def test_document_processing():
    """Test document processing functionality."""
    print("\nTesting document processing...")
    
    try:
        from rag_system import DocumentProcessor
        
        # Create a test document
        test_doc = "test_document.txt"
        with open(test_doc, "w", encoding="utf-8") as f:
            f.write("This is a test document for the RAG system. " * 50)  # Make it long enough to chunk
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        documents = processor.process_documents([test_doc])
        
        print(f"+ Document processing successful: {len(documents)} chunks created")
        
        # Clean up
        os.remove(test_doc)
        return True
        
    except Exception as e:
        print(f"- Document processing failed: {e}")
        return False

def test_vector_store():
    """Test vector store functionality."""
    print("\nTesting vector store...")
    
    try:
        from rag_system import VectorStore
        from langchain.schema import Document
        
        # Create test documents
        test_docs = [
            Document(page_content="This is about artificial intelligence and machine learning.", metadata={"source": "test1"}),
            Document(page_content="This is about natural language processing and deep learning.", metadata={"source": "test2"}),
            Document(page_content="This is about computer vision and neural networks.", metadata={"source": "test3"})
        ]
        
        vector_store = VectorStore()
        vector_store.add_documents(test_docs)
        
        # Test search
        results = vector_store.search("machine learning", k=2)
        print(f"+ Vector store search successful: {len(results)} results found")
        
        return True
        
    except Exception as e:
        print(f"- Vector store test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("RAG System Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_document_processing,
        test_vector_store
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("+ All tests passed! RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Set up your OpenAI API key in .env file")
        print("2. Run: python example_usage.py")
    else:
        print("- Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python version compatibility")
        print("3. Ensure sufficient disk space and memory")

if __name__ == "__main__":
    main()

