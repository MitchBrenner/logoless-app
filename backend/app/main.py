# Copyright (c) 2025 Mitchell Brenner
# Licensed under the GNU General Public License v3.0 (GPL-3.0-or-later)
# See LICENSE for details.

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import os
import tempfile
import shutil
from .processing.blur_watermark import blur_watermark

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Video Processing API!"}

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    # Validate file extension
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Create a temporary working directory
    tmpdir = tempfile.mkdtemp()
    in_path = os.path.join(tmpdir, file.filename)
    out_name = f"blurred_{file.filename}"
    out_path = os.path.join(tmpdir, out_name)

    # Save uploaded file
    with open(in_path, "wb") as f:
        f.write(await file.read())

    success = blur_watermark(
        input_video_path=in_path,
        output_video_path=out_path,
    )
    if not success:
        # Clean up and return failure
        shutil.rmtree(tmpdir, ignore_errors=True)
        return JSONResponse({"success": False})

    # Schedule temp directory cleanup after response
    if background_tasks:
        background_tasks.add_task(shutil.rmtree, tmpdir, True)

    # Return the processed video file
    return FileResponse(
        path=out_path,
        media_type="video/mp4",
        filename=out_name,
        background=background_tasks
    )
