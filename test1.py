


from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List
import os
import io
import csv
import uuid
import datetime
from sqlalchemy.orm import Session
from db import SessionLocal
from models import Bot, KnowledgeSource, KnowledgeChunk
from sentence_transformers import SentenceTransformer
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

# Import the router for AI response
from genai_service import router as genai_router

# Load the .env file
load_dotenv()


app = FastAPI(title="RAG API", version="1.0", root_path="/python-ai-service")

# Mount the AI response router
app.include_router(genai_router, prefix="/ai")

MAX_TOTAL_SIZE = 10 * 1024 * 1024

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

# Helper: extract chunks from file using LangChain's text splitter
def extract_chunks_from_file(file_path, file_type):
	# TODO: Implement real chunk extraction for doc, pdf, csv, etc.
	with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
		text = f.read()
	# Use LangChain's RecursiveCharacterTextSplitter for better chunking
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=500,
		chunk_overlap=50
	)
	return splitter.split_text(text)


model_name = "nomic-ai/nomic-embed-text-v1"
model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': True}

embeddings = HuggingFaceEmbeddings(
	model_name=model_name,
	model_kwargs=model_kwargs
)




import aiohttp



GENAI_API_BASE_URL = 'https://aibot14.studyineurope.xyz/genaiapi' # Change to your actual API base URL

async def embed_content_batch_async(chunks, task_type="search_query"):
	"""Embed each chunk with a separate API call (no batching)."""
	embeddings_list = []
	async with aiohttp.ClientSession() as session:
		for chunk in chunks:
			requestBody = {"text": chunk, "task_type": task_type}
			async with session.post(f"{GENAI_API_BASE_URL}/embed/", json=requestBody) as resp:
				if resp.status != 200:
					raise Exception(f"Embedding API error: {resp.status} {await resp.text()}")
				data = await resp.json()
				print("Received embedding:", data)
				# If the response is a list, use it directly
				if isinstance(data, list):
					embeddings_list.append(data)
				# If the response is a dict with 'embedding', use that
				elif "embedding" in data:
					embeddings_list.append(data["embedding"])
				else:
					raise Exception(f"Unexpected API response: {data}")
	return embeddings_list


@app.post("/bots/{bot_id}/knowledge/upload")
async def upload_knowledge_source(bot_id: str, files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
	# Check total size
	print("Uploading files:", [file.filename for file in files])
	total_size = 0
	temp_file_paths = []
	for file in files:
		contents = await file.read()
		total_size += len(contents)
		if total_size > MAX_TOTAL_SIZE:
			raise HTTPException(status_code=400, detail="Total upload size exceeds 10 MB limit.")
		temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
		with open(temp_path, 'wb') as f:
			f.write(contents)
		temp_file_paths.append((temp_path, file.content_type, file.filename))
	print("Uploading files:", db)
	
	bot = db.query(Bot).filter(Bot.id == bot_id).first()
	if not bot:
		raise HTTPException(status_code=403, detail="Permission denied.")
    
	results = []

	for temp_path, file_type, original_name in temp_file_paths:
		try:
			chunks = extract_chunks_from_file(temp_path, file_type)
			# Filter out small chunks first
			valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 20]
			if not valid_chunks:
				raise Exception(f"Could not extract any valid content from {original_name}.")
			source = KnowledgeSource(
				id=str(uuid.uuid4()),
				fileName=original_name,
				storagePath=temp_path,
				fileType=file_type,
				botId=bot_id,
				createdAt=datetime.datetime.utcnow(),
			)

			db.add(source)
			db.commit()
			db.refresh(source)


			# Batch embed all valid chunks using external API (async)
			embeddings_list = await embed_content_batch_async(valid_chunks)
			try:
				chunk_objs = []
				for chunk, embedding in zip(valid_chunks, embeddings_list):
					# Remove NUL bytes from chunk content
					clean_chunk = chunk.replace('\x00', '')
					# Store embedding as a Python list (for pgvector/ARRAY column)
					chunk_obj = KnowledgeChunk(
						id=str(uuid.uuid4()),
						content=clean_chunk,
						embedding=embedding,  # Store as array/vector
						knowledgeSourceId=source.id
					)
					print("Created chunk:", embedding, "knowledgeSourceId=", source.id)
					chunk_objs.append(chunk_obj)
				db.add_all(chunk_objs)
				db.commit()
			except Exception as e:
				print("Error storing chunks:", e)

			results.append({"file": original_name, "chunksUploaded": len(chunk_objs)})
		except Exception as e:
			results.append({"file": original_name, "error": str(e)})
		finally:
			try:
				os.remove(temp_path)
			except Exception:
				pass

	return {"results": results}

@app.get("/bots/{bot_id}/knowledge")
def get_knowledge_sources(bot_id: str, db: Session = Depends(get_db)):
	sources = db.query(KnowledgeSource).filter(KnowledgeSource.botId == bot_id).order_by(KnowledgeSource.createdAt.desc()).all()
	return sources

@app.delete("/knowledge/{source_id}")
def delete_knowledge_source(source_id: str, db: Session = Depends(get_db)):
	deleted = db.query(KnowledgeSource).filter(KnowledgeSource.id == source_id).delete()
	db.commit()
	if deleted == 0:
		raise HTTPException(status_code=404, detail="File not found or permission denied.")
	return {"message": "File deleted successfully."}
