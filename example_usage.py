# Example usage script for the RAG system

from rag_system import RAGSystem
import os

def main():
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGSystem()
    
    # Example documents (replace with your actual documents)
    documents = [
        "sample_ai_doc.txt",
        "sample_ml_doc.txt"
    ]
    
    # Create sample documents
    print("Creating sample documents...")
    
    # AI document
    with open("sample_ai_doc.txt", "w", encoding="utf-8") as f:
        f.write("""
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. 
        AI systems can perform tasks that typically require human intelligence, such as visual perception, 
        speech recognition, decision-making, and language translation. AI can be categorized into narrow AI, 
        which is designed for specific tasks, and general AI, which would have human-like cognitive abilities.
        """)
    
    # Machine Learning document
    with open("sample_ml_doc.txt", "w", encoding="utf-8") as f:
        f.write("""
        Machine Learning is a subset of AI that focuses on algorithms that can learn and improve from experience. 
        It includes supervised learning, where models learn from labeled data, unsupervised learning, where models 
        find patterns in unlabeled data, and reinforcement learning, where models learn through interaction with 
        an environment. Popular ML techniques include neural networks, decision trees, and support vector machines.
        """)
    
    try:
        # Add documents to the system
        print("Adding documents to RAG system...")
        rag.add_documents(documents)
        
        # Example queries
        queries = [
            "What is artificial intelligence?",
            "What is machine learning?",
            "How does AI relate to machine learning?",
            "What are the different types of machine learning?",
            "What is the difference between narrow AI and general AI?"
        ]
        
        print("\n" + "="*60)
        print("RAG System Demo - Question Answering")
        print("="*60)
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: {query}")
            print("-" * 40)
            
            result = rag.query(query, k=3)  # Retrieve top 3 documents
            
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['sources'])} documents found")
            
            # Show source details
            for j, source in enumerate(result['sources'], 1):
                print(f"  Source {j} (similarity: {source['similarity_score']:.3f}):")
                print(f"    {source['content']}")
            
            print("-" * 40)
        
        # Save the system
        print("\nSaving RAG system...")
        rag.save_system("demo_rag_system")
        
        # Demonstrate loading
        print("Loading saved RAG system...")
        new_rag = RAGSystem()
        new_rag.load_system("demo_rag_system")
        
        # Test loaded system
        test_query = "What is the relationship between AI and machine learning?"
        print(f"\nTesting loaded system with: {test_query}")
        result = new_rag.query(test_query)
        print(f"Answer: {result['answer']}")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure you have:")
        print("1. Installed all dependencies: pip install -r requirements.txt")
        print("2. Set up your OpenAI API key in .env file")
        print("3. Sufficient disk space for vector storage")
    
    finally:
        # Clean up sample files
        print("\nCleaning up sample files...")
        for doc in documents:
            if os.path.exists(doc):
                os.remove(doc)
                print(f"Removed {doc}")
        
        # Clean up saved system files
        for ext in [".index", ".docs"]:
            filepath = f"demo_rag_system{ext}"
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Removed {filepath}")

if __name__ == "__main__":
    main()

