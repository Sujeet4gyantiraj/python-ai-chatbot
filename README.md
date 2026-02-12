
# Python AI Chatbot (RAG with FastAPI, Pinecone, LangChain, Redis)

This project is a production-ready AI microservice for building SaaS-based chatbots using Retrieval-Augmented Generation (RAG). It leverages FastAPI, Pinecone, LangChain, and Redis for scalable, modular, and persistent AI chat experiences.

## Features
- **FastAPI**: High-performance API server for chatbot interactions
- **Retrieval-Augmented Generation (RAG)**: Combines LLMs with semantic search over your documents
- **Pinecone**: Vector database for fast, scalable semantic search
- **LangChain**: Modular prompt building, memory, and retriever integration
- **Redis**: Persistent chat memory (last 5 messages per session)
- **Custom Embeddings**: Supports your own embedding API for semantic search
- **Dockerized**: Ready for containerized deployment

## Project Structure
- `main.py` — FastAPI app entrypoint
- `genai_service.py` — Core AI logic, RAG, memory, and prompt orchestration
- `pinecone_client.py` — Pinecone index setup
- `utils/`
  - `chunking.py` — Text chunking utilities
  - `prompt_builder.py` — Prompt formatting, chat memory, Redis helpers
  - `retriver.py` — Pinecone retriever and semantic search logic
- `models.py` — (Optional) SQLAlchemy models for DB integration
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container build instructions
- `docker-compose.yml` — Multi-service orchestration (app + Postgres)

## Setup & Installation
1. **Clone the repository**
2. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```
3. **Configure environment variables**
	- Create a `.env` file with your Pinecone, Redis, and other secrets
4. **Run Redis and Pinecone**
	- Make sure Redis is running locally or update `REDIS_URL`
	- Set up your Pinecone index and API key
5. **Start the FastAPI server**
	```bash
	uvicorn main:app --reload
	```
6. **(Optional) Run with Docker**
	```bash
	docker-compose up --build
	```

## Usage
- Upload documents, ask questions, and get context-aware answers using RAG
- Chat history is persisted per session (last 5 messages) using Redis
- Semantic search is powered by Pinecone and custom embeddings

## Environment Variables
- `PINECONE_API_KEY` — Your Pinecone API key
- `PINECONE_INDEX` — Pinecone index name
- `REDIS_URL` — Redis connection string (default: `redis://localhost:6379/0`)
- (See `.env.example` for more)
4300
615
1385
140
390
300
1400
150
1360
100
120



10260 for five

482 + 241 + 510 + 840 = 2073

pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 7004" --name ai-service-new
