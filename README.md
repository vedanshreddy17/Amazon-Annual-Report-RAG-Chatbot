# Amazon Annual Report RAG Chatbot

A locally hosted AI chatbot that answers questions about Amazon's 2024 Annual Report using Retrieval-Augmented Generation (RAG). The application retrieves relevant passages from the report and uses Llama 3.2 to generate responses with PDF page references.

## Technologies Used

- **Python** — Application logic and document processing
- **Llama 3.2 (1B)** — Large Language Model
- **Ollama** — Local LLM inference
- **Sentence Transformers** — Text embeddings
- **ChromaDB** — Vector database for semantic search
- **Streamlit** — Interactive chatbot interface
- **PyPDF** — PDF text extraction

## How It Works

1. Extracts text from Amazon's 2024 Annual Report.
2. Splits document text into smaller passages.
3. Converts passages into vector embeddings using Sentence Transformers.
4. Stores embeddings in ChromaDB for semantic retrieval.
5. Retrieves relevant passages when a user asks a question.
6. Sends the retrieved context to Llama 3.2 through Ollama.
7. Displays the generated answer and retrieved PDF page references.

## Features

- Natural-language questions about Amazon's financial performance
- Semantic search across the annual report
- Local LLM execution without paid API keys
- PDF page references for retrieved information
- Prompt instructions designed to reduce unsupported answers
- Interactive web interface using Streamlit

## Getting Started

### Prerequisites

- Python 3.11
- Ollama
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/vedanshreddy17/Amazon-Annual-Report-RAG-Chatbot.git
cd Amazon-Annual-Report-RAG-Chatbot
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the LLM:

```bash
ollama pull llama3.2:1b
```

Start Ollama in one terminal:

```bash
ollama serve
```

Launch the chatbot in another terminal:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Example Questions

- What was Amazon's total net sales revenue in 2024?
- What were Amazon's three business segments?
- What were the 2024 net sales for North America, International, and AWS?
- Does the report provide an exact revenue forecast for 2030?

## Testing and Improvements

Tested the chatbot using financial questions from Amazon's 2024 Annual Report. Verified its ability to retrieve segment revenue figures and improved the LLM prompt after identifying an incorrect response to a question about future revenue.

## Limitations

- Response quality depends on the retrieved passages and the LLM's ability to interpret them.
- Retrieved page references indicate search results, not independently verified citations.
- The model may still produce incomplete or incorrect answers.
- Initial embedding generation may take additional time.

## Project Scope

A lightweight, locally running RAG application developed to demonstrate practical skills in Python, LLM integration, semantic search, vector databases, prompt engineering, and AI application testing.