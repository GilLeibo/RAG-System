# RAG System - Requirements and Setup

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from rag_system import RAGSystem

# Initialize the RAG system
rag = RAGSystem()

# Add documents to the system
documents = ["document1.pdf", "document2.txt", "document3.md"]
rag.add_documents(documents)

# Query the system
result = rag.query("What is machine learning?")
print(result["answer"])
print(f"Sources: {len(result['sources'])} documents found")

# Save the system for later use
rag.save_system("my_rag_system")

# Load a previously saved system
rag.load_system("my_rag_system")
```

### Advanced Configuration

```python
# Customize the RAG system
rag = RAGSystem(
    embedding_model="all-MiniLM-L6-v2",  # Embedding model
    llm_model="gpt-3.5-turbo",          # LLM model
    chunk_size=1000,                     # Document chunk size
    chunk_overlap=200                    # Overlap between chunks
)

# Query with custom parameters
result = rag.query(
    question="Your question here",
    k=5,                    # Number of documents to retrieve
    max_tokens=500          # Maximum tokens for response
)
```

## Features

- **Document Processing**: Supports PDF, TXT, and Markdown files
- **Vector Search**: Uses FAISS for efficient similarity search
- **Embeddings**: Uses Sentence Transformers for high-quality embeddings
- **LLM Integration**: Compatible with OpenAI GPT models
- **Persistence**: Save and load RAG systems
- **Flexible Configuration**: Customizable chunk sizes, models, and parameters

## File Structure

```
rag_project/
├── rag_system.py          # Main RAG system implementation
├── requirements.txt       # Python dependencies
├── example_usage.py       # Usage examples
├── test_rag.py           # Test suite
├── .env.example          # Environment variables template
└── README.md             # This file
```

## Dependencies

- numpy: Numerical computations
- sentence-transformers: Text embeddings
- faiss-cpu: Vector similarity search
- openai: LLM integration
- langchain: Document processing
- python-dotenv: Environment variable management
- pypdf2: PDF processing
- beautifulsoup4: HTML parsing
- requests: HTTP requests
- tiktoken: Token counting

## Example Output

```
Query: What is artificial intelligence?
Answer: Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

Sources: 2 documents found
```

## Troubleshooting

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **OpenAI API errors**: Check your API key in the `.env` file
3. **Memory issues**: Reduce chunk_size or use a smaller embedding model
4. **Slow performance**: Consider using GPU-accelerated FAISS or smaller models

