import logging
import os
import tempfile
import traceback

from fastapi import APIRouter, UploadFile, File, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter(tags=["files"])


@router.post("/extract-file")
async def extract_file(file: UploadFile = File(...)):
    """Extract content from uploaded file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail='No file selected')

        content_bytes = await file.read()

        temp_dir = tempfile.gettempdir()
        # basename 防路径穿越：客户端可能传 "../../evil.py" 之类文件名
        safe_name = os.path.basename(file.filename)
        temp_path = os.path.join(temp_dir, safe_name)
        with open(temp_path, "wb") as f:
            f.write(content_bytes)

        from agent.tools.agent_tools import extract_file_content
        content = extract_file_content.invoke({"file_path": temp_path})

        return {
            'success': True,
            'content': content,
            'filename': file.filename,
            'file_path': temp_path
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
