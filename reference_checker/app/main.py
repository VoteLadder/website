import os
import uuid
import json
import asyncio
import logging
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from zipfile import ZipFile
from sqlalchemy.orm import Session
from datetime import datetime

from database import get_db, ProcessingRequest, SessionLocal
from reference_checker import (
    extract_relevant_text, extract_references_section, process_main_content,
    process_references_section, get_all_article_pdfs, process_articles_with_verification,
    save_articles_zip, WORD_LIMIT
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the current directory
current_dir = Path(__file__).parent

# Create necessary directories
os.makedirs(current_dir / "processing", exist_ok=True)
os.makedirs(current_dir / "static", exist_ok=True)
os.makedirs(current_dir / "uploads", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(current_dir / "static")), name="static")

# Global state
processing_queue = asyncio.Queue()
is_processing = False

@app.get("/")
async def read_root():
    """Serve the index.html file"""
    return FileResponse(str(current_dir / "index.html"))

@app.post("/api/submit")
async def submit_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Handle PDF file submission"""
    try:
        # Validate file type
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        request_id = str(uuid.uuid4())
        request_dir = os.path.join(current_dir / "processing", request_id)
        os.makedirs(request_dir, exist_ok=True)

        # Save uploaded PDF
        pdf_path = os.path.join(request_dir, "article.pdf")
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)

        # Create database entry
        db_request = ProcessingRequest(
            request_id=request_id,
            status="pending",
            progress={"stage": "Queued"},
            output_dir=request_dir,
            original_filename=file.filename
        )
        db.add(db_request)
        db.commit()

        # Add to processing queue
        await processing_queue.put((request_id, pdf_path))

        # Return JSON response
        return JSONResponse(content={"request_id": request_id})

    except Exception as e:
        logger.error(f"Error submitting PDF: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"detail": str(e)}
        )

@app.get("/api/status/{request_id}")
async def get_status(request_id: str, db: Session = Depends(get_db)):
    """Get status of a specific request"""
    request = db.query(ProcessingRequest).filter(ProcessingRequest.request_id == request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    return request

@app.get("/api/status")
async def get_all_status(db: Session = Depends(get_db)):
    """Get status of all requests"""
    requests = db.query(ProcessingRequest).order_by(ProcessingRequest.created_at.desc()).all()
    return {"requests": {req.request_id: {
        "status": req.status,
        "progress": req.progress,
        "created_at": req.created_at.isoformat(),
        "updated_at": req.updated_at.isoformat(),
        "original_filename": req.original_filename,
        "article_title": req.article_title
    } for req in requests}}

@app.get("/api/download/{request_id}/{filename}")
async def download_file(request_id: str, filename: str, db: Session = Depends(get_db)):
    """Download processed files"""
    request = db.query(ProcessingRequest).filter(ProcessingRequest.request_id == request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    if request.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")

    file_path = os.path.join(request.output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    return FileResponse(file_path)

async def process_request(request_id: str, pdf_path: str, db: Session):
    """Process a single PDF request"""
    try:
        # Update status to processing
        request = db.query(ProcessingRequest).filter(ProcessingRequest.request_id == request_id).first()
        request.status = "processing"
        request.progress = {"stage": "Extracting text"}
        db.commit()

        # Extract text
        text = await extract_relevant_text(pdf_path)
        if not text:
            raise ValueError("Failed to extract text from PDF")

        # Extract references
        request.progress = {"stage": "Extracting references"}
        db.commit()
        refs_text = await extract_references_section(text)
        if not refs_text:
            raise ValueError("Failed to extract references section")

        # Process main content
        request.progress = {"stage": "Processing main content"}
        db.commit()
        citations = await process_main_content(text)

        # Process references
        request.progress = {"stage": "Processing references"}
        db.commit()
        references = await process_references_section(refs_text)

        # Get PDFs for verification
        request.progress = {"stage": "Retrieving reference PDFs"}
        db.commit()
        article_pdfs = await get_all_article_pdfs(references)

        # Verify references
        request.progress = {"stage": "Verifying references"}
        db.commit()
        verification_results = await process_articles_with_verification(citations, references, article_pdfs)

        # Save results
        request.progress = {"stage": "Saving results"}
        db.commit()

        # Save verification results
        results_path = os.path.join(request.output_dir, "verified.json")
        with open(results_path, "w") as f:
            json.dump(verification_results, f, indent=2)

        # Save detailed report
        report_path = os.path.join(request.output_dir, "confirmations.txt")
        with open(report_path, "w") as f:
            # Write report content
            f.write("Reference Verification Report\n")
            f.write("=========================\n\n")
            # Add report details here

        # Save article PDFs
        zip_path = os.path.join(request.output_dir, "articles.zip")
        await save_articles_zip(article_pdfs, zip_path)

        # Update status to completed
        request.status = "completed"
        request.progress = {"stage": "Completed"}
        db.commit()

    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        request = db.query(ProcessingRequest).filter(ProcessingRequest.request_id == request_id).first()
        request.status = "error"
        request.progress = {"stage": f"Error: {str(e)}"}
        db.commit()

async def process_queue():
    """Background task to process the queue"""
    global is_processing
    db = SessionLocal()

    while True:
        try:
            if not is_processing:
                try:
                    request_id, pdf_path = await processing_queue.get_nowait()
                    is_processing = True

                    try:
                        await process_request(request_id, pdf_path, db)
                    finally:
                        processing_queue.task_done()
                        is_processing = False

                except asyncio.QueueEmpty:
                    await asyncio.sleep(1)
            else:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error in queue processing: {str(e)}")
            is_processing = False
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    """Start the background processing task"""
    asyncio.create_task(process_queue())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.1", port=8001)

