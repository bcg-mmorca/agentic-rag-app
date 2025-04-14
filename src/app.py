import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.utils import authenticate_vertex_ai


class QuestionRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    source_documents: list

class ProcessRepository(BaseModel):
    path: str

# Load environment variables from a file(default: .env)
load_dotenv("/secrets/.env")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("LOCATION")
CREDENTIALS_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
BUCKET_URI = os.environ.get("BUCKET_URI")

# Define the maximum file size (default: 10 MB)
MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 10)) * 1024 * 1024

authenticate_vertex_ai(PROJECT_ID, LOCATION, CREDENTIALS_FILE, BUCKET_URI)

# Import the agent after authentication
from src.agentic_rag import ask_agent, build_agent, process_repository

agent = build_agent()

app = FastAPI()

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    try:
        user_id = "u-abc123"
        thread_id = "abcd1234"

        result = ask_agent(agent, request.query, thread_id, user_id)

        return AnswerResponse(
            question=request.query,
            answer=result["answer"],
            source_documents=result["context"],
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-repo")
async def process_repo(repo: ProcessRepository):
    try:
        num_chunks = await process_repository(repo.path)
        return {"message": f"Processed {num_chunks} code files"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create full file path
            temp_file_path = Path(temp_dir) / file.filename

            # Save uploaded file
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()

                # Check file size
                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File size exceeds the maximum limit of {MAX_FILE_SIZE // (1024 * 1024)} MB"
                    )

                buffer.write(content)

            # Process the saved file
            num_chunks = await process_repository(str(temp_file_path))
            return {"message": f"Processed {num_chunks} chucks of documents"}

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
