"""
SAR-Sense API Server (FastAPI)
FastAPI backend with streaming support and thought chain visualization
"""

import os
import sys
import json
import time
import tempfile
import base64
import logging
from io import BytesIO

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAR-Sense API", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/css", StaticFiles(directory="css"), name="css")
app.mount("/js", StaticFiles(directory="js"), name="js")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.on_event("startup")
async def startup_event():
    """Pre-load components on startup to avoid slow first request"""
    import threading

    def preload():
        logger.info("Pre-loading components...")
        try:
            # Pre-load agent (most commonly used)
            get_agent()
            logger.info("Agent loaded")
        except Exception as e:
            logger.warning(f"Failed to pre-load agent: {e}")

        try:
            # Pre-load conversation manager
            get_conv_manager()
            logger.info("Conversation manager loaded")
        except Exception as e:
            logger.warning(f"Failed to pre-load conversation manager: {e}")

        try:
            # Pre-load metrics
            get_metrics()
            logger.info("Metrics loaded")
        except Exception as e:
            logger.warning(f"Failed to pre-load metrics: {e}")

        logger.info("Pre-loading complete")

    # Run in background thread to not block startup
    threading.Thread(target=preload, daemon=True).start()


# Global instances
_yolo_model = None
_agent = None
_vector_store = None
_metrics = None
_conv_manager = None


def get_yolo_model():
    """Lazy load YOLO model"""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        from utils.path_tool import get_abs_path
        model_path = get_abs_path("Detct_prdc/MBE-Net/weights/best.pt")
        _yolo_model = YOLO(model_path)
    return _yolo_model


def get_agent():
    """Lazy load ReactAgent"""
    global _agent
    if _agent is None:
        from agent.react_agent import ReactAgent
        _agent = ReactAgent()
    return _agent


def get_vector_store():
    """Lazy load VectorStoreService"""
    global _vector_store
    if _vector_store is None:
        from rag.vector_store import VectorStoreService
        _vector_store = VectorStoreService()
    return _vector_store


def get_metrics():
    """Lazy load AgentMetrics"""
    global _metrics
    if _metrics is None:
        from agent.metrics_collector import AgentMetrics
        _metrics = AgentMetrics()
    return _metrics


def get_conv_manager():
    """Lazy load ConversationManager"""
    global _conv_manager
    if _conv_manager is None:
        from utils.conversation_manager import ConversationManager
        _conv_manager = ConversationManager()
    return _conv_manager


# ==================== Static Files ====================

@app.get("/")
async def index():
    return FileResponse("index.html")


@app.get("/index.html")
async def index_html():
    return FileResponse("index.html")


@app.get("/detection.html")
async def detection_page():
    return FileResponse("detection.html")


@app.get("/chat.html")
async def chat_page():
    return FileResponse("chat.html")


@app.get("/knowledge.html")
async def knowledge_page():
    return FileResponse("knowledge.html")


@app.get("/metrics.html")
async def metrics_page():
    return FileResponse("metrics.html")


# ==================== Detection API ====================

@app.post("/api/detect")
async def detect_ships(image: UploadFile = File(...)):
    """SAR ship detection endpoint"""
    try:
        logger.info(f"Detection request received: {image.filename}")

        # Read uploaded file
        content = await image.read()

        # Save to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"sar_detect_{image.filename}")
        with open(temp_path, "wb") as f:
            f.write(content)
        logger.info(f"File saved to: {temp_path}")

        # Load image and run detection
        logger.info("Loading image...")
        img = Image.open(BytesIO(content))
        logger.info("Loading YOLO model...")
        model = get_yolo_model()
        logger.info("Running detection...")
        results = model.predict(source=img, imgsz=640)
        logger.info(f"Detection complete, found {len(results[0].boxes)} objects")

        # Get detection results
        ship_count = len(results[0].boxes)

        # Get plotted image
        res_plotted = results[0].plot()
        result_image = Image.fromarray(res_plotted[:, :, ::-1])  # BGR to RGB

        # Convert to base64
        buffered = BytesIO()
        result_image.save(buffered, format="PNG")
        result_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Get original image as base64
        orig_buffered = BytesIO()
        img.save(orig_buffered, format="PNG")
        orig_base64 = base64.b64encode(orig_buffered.getvalue()).decode()

        # Get detection details
        detections = []
        for box in results[0].boxes:
            detections.append({
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0]),
                'bbox': box.xyxy[0].tolist()
            })

        return {
            'success': True,
            'ship_count': ship_count,
            'original_image': orig_base64,
            'result_image': result_base64,
            'detections': detections,
            'temp_path': temp_path,
            'message': f'检测完成，共发现 {ship_count} 个目标'
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Chat API ====================

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """Streaming chat endpoint using SSE"""
    try:
        logger.info("Streaming chat request received")
        data = await request.json()
        message = data.get('message', '')
        display_message = data.get('display_message', message)
        messages_history = data.get('messages', [])
        conversation_id = data.get('conversation_id')

        if not message:
            raise HTTPException(status_code=400, detail='No message provided')

        logger.info(f"Processing streaming message: {message[:50]}...")

        from agent.react_agent import _thought_chains

        logger.info("Loading agent for streaming...")
        agent = get_agent()
        logger.info("Agent loaded for streaming")

        # Build messages list — 优先用对话记忆压缩，降级为硬截断
        conv_mgr = get_conv_manager()
        if conversation_id:
            # 持久化纯文本（不含附件内容），保持对话记录干净
            conv_mgr.append_message(conversation_id, "user", display_message)
            # build_chat_pack 从 JSON 读取历史做压缩
            messages = conv_mgr.build_chat_pack(conversation_id)
            # 替换最后一条为当前完整消息（含附件内容），确保 Agent 能看到附件
            if messages and messages[-1].get("role") == "user":
                messages[-1] = {"role": "user", "content": message}
        else:
            # 无 conversation_id 时降级：硬截断最近 10 轮
            messages = messages_history[-10:] + [{"role": "user", "content": message}]

        async def generate():
            """SSE generator for streaming response"""
            import asyncio
            import threading
            from queue import Queue

            event_queue = Queue()

            # Start conversation metrics
            metrics = get_metrics()
            metrics.start_conversation()

            def run_agent():
                """Run agent in a separate thread"""
                try:
                    for chunk in agent.execute_stream(messages, conversation_id):
                        if chunk and chunk.strip():
                            event_queue.put(('chunk', chunk))
                except Exception as e:
                    event_queue.put(('error', str(e)))
                finally:
                    event_queue.put(('done', None))

            # Start agent in background thread
            agent_thread = threading.Thread(target=run_agent)
            agent_thread.start()

            try:
                logger.info("Starting agent stream...")
                last_step_count = 0

                while True:
                    # Check for new thought chain steps
                    if conversation_id and conversation_id in _thought_chains:
                        current_steps = _thought_chains[conversation_id]["steps"]
                    else:
                        current_steps = []

                    if len(current_steps) > last_step_count:
                        new_steps = current_steps[last_step_count:]
                        for step in new_steps:
                            data = json.dumps({
                                'type': 'thought_step',
                                'step': step
                            }, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                        last_step_count = len(current_steps)

                    # Check queue for events
                    try:
                        event_type, event_data = event_queue.get_nowait()

                        if event_type == 'chunk':
                            data = json.dumps({
                                'type': 'chunk',
                                'content': event_data
                            }, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                        elif event_type == 'error':
                            data = json.dumps({
                                'type': 'error',
                                'message': event_data
                            }, ensure_ascii=False)
                            yield f"data: {data}\n\n"
                        elif event_type == 'done':
                            break
                    except:
                        pass

                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.05)

                # Send any remaining thought steps
                if conversation_id and conversation_id in _thought_chains:
                    current_steps = _thought_chains[conversation_id]["steps"]
                    if len(current_steps) > last_step_count:
                        new_steps = current_steps[last_step_count:]
                        for step in new_steps:
                            data = json.dumps({
                                'type': 'thought_step',
                                'step': step
                            }, ensure_ascii=False)
                            yield f"data: {data}\n\n"

                # Send done signal
                data = json.dumps({'type': 'done'})
                yield f"data: {data}\n\n"

            except Exception as e:
                import traceback
                traceback.print_exc()
                data = json.dumps({
                    'type': 'error',
                    'message': str(e)
                }, ensure_ascii=False)
                yield f"data: {data}\n\n"

            finally:
                metrics.end_conversation()

                # 清理该会话的思考链，防止内存泄漏
                if conversation_id and conversation_id in _thought_chains:
                    del _thought_chains[conversation_id]

                logger.info("Streaming response completed")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Conversation API ====================

@app.get("/api/conversations")
async def list_conversations():
    """List all conversations"""
    try:
        conv_manager = get_conv_manager()
        conversations = conv_manager.list_conversations()
        return {
            'success': True,
            'conversations': conversations
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations")
async def create_conversation(request: Request):
    """Create a new conversation"""
    try:
        data = await request.json()
        first_message = data.get('message', '新对话')
        conv_manager = get_conv_manager()
        conv_id = conv_manager.create_conversation(first_message)
        return {
            'success': True,
            'conversation_id': conv_id
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conv_id}")
async def load_conversation(conv_id: str):
    """Load a conversation"""
    try:
        conv_manager = get_conv_manager()
        conversation = conv_manager.load_conversation(conv_id)
        if conversation:
            return {
                'success': True,
                'conversation': conversation
            }
        else:
            raise HTTPException(status_code=404, detail='Conversation not found')
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation"""
    try:
        conv_manager = get_conv_manager()
        conv_manager.delete_conversation(conv_id)
        return {'success': True}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations/{conv_id}/messages")
async def append_message(conv_id: str, request: Request):
    """Append a message to a conversation"""
    try:
        data = await request.json()
        role = data.get('role', 'user')
        content = data.get('content', '')
        thought_steps = data.get('thought_steps')

        conv_manager = get_conv_manager()
        conv_manager.append_message(conv_id, role, content, thought_steps=thought_steps)

        return {'success': True}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Knowledge Base API ====================

@app.post("/api/knowledge/upload")
async def upload_knowledge(files: list[UploadFile] = File(...)):
    """Upload documents to knowledge base"""
    try:
        if not files:
            raise HTTPException(status_code=400, detail='No files provided')

        from utils.config_handler import chroma_conf
        from utils.path_tool import get_abs_path

        data_dir = get_abs_path(chroma_conf["data_path"])
        uploaded_files = []

        for file in files:
            if file.filename:
                filename = os.path.basename(file.filename)
                file_path = os.path.join(data_dir, filename)
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                uploaded_files.append(filename)

        # Reload documents into vector store
        vector_store = get_vector_store()
        new_count, updated_count, skipped_count, removed_count = vector_store.load_document()

        return {
            'success': True,
            'uploaded_files': uploaded_files,
            'new_count': new_count,
            'updated_count': updated_count,
            'removed_count': removed_count,
            'message': f'成功上传 {len(uploaded_files)} 个文件，新增 {new_count}、更新 {updated_count}、清理 {removed_count}'
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/knowledge/files")
async def list_knowledge_files():
    """List knowledge base files — 从 manifest 读，展示 chunk_count/doc_id/status 等结构化信息"""
    try:
        vector_store = get_vector_store()
        manifest = vector_store.manifest

        files = []
        for filename, entry in manifest.items():
            files.append({
                'name': filename,
                'doc_id': entry.get('doc_id'),
                'file_type': entry.get('file_type'),
                'chunk_count': entry.get('chunk_count'),
                'chunk_method': entry.get('chunk_method'),
                'status': entry.get('status'),
                'ingested_at': entry.get('ingested_at'),
                'file_hash': entry.get('file_hash'),
            })

        return {
            'success': True,
            'files': files,
            'total_files': len(files),
            'total_chunks': sum(e.get('chunk_count', 0) for e in manifest.values())
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/knowledge/files/{doc_id}")
async def delete_knowledge_file(doc_id: str, delete_file: bool = True):
    """Delete one knowledge document by doc_id."""
    try:
        vector_store = get_vector_store()
        target = None
        for filename, entry in vector_store.manifest.items():
            if entry.get("doc_id") == doc_id:
                target = {
                    "filename": filename,
                    "chunk_count": entry.get("chunk_count", 0),
                }
                break

        if target is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        deleted_chunks = vector_store.delete_document_by_doc_id(doc_id, delete_file=delete_file)

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": target["filename"],
            "deleted_chunks": deleted_chunks,
            "deleted_file": delete_file,
            "message": f"已删除 {target['filename']}，清理 {deleted_chunks} 个 chunk",
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Metrics API ====================

@app.get("/api/metrics")
async def get_metrics_data():
    """Get observability metrics"""
    try:
        metrics = get_metrics()

        data = {
            'conversation_rounds': metrics.conversation_rounds,
            'total_tool_calls': metrics.total_tool_calls,
            'overall_success_rate': metrics.overall_success_rate,
            'avg_tool_calls_per_round': metrics.avg_tool_calls_per_round,
            'avg_response_time_s': metrics.avg_response_time_s,
            'llm_call_count': metrics.llm_call_count,
            'tool_stats': metrics.get_tool_stats(),
            'recent_records': metrics.get_recent_records()
        }

        return {
            'success': True,
            'metrics': data
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/metrics/reset")
async def reset_metrics():
    """Reset all metrics"""
    try:
        metrics = get_metrics()
        metrics.reset()
        return {'success': True, 'message': 'Metrics reset successfully'}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== File Extraction API ====================

@app.post("/api/extract-file")
async def extract_file(file: UploadFile = File(...)):
    """Extract content from uploaded file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail='No file selected')

        # Read content
        content_bytes = await file.read()

        # Save to temp (don't delete - agent may need the path)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Health Check ====================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '2.0'
    }


if __name__ == '__main__':
    import uvicorn
    print("=" * 50)
    print("   SAR-Sense API Server (FastAPI)")
    print("=" * 50)
    print()
    print("Access the application at:")
    print("  http://localhost:5000")
    print()
    print("API Documentation at:")
    print("  http://localhost:5000/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=5000)
