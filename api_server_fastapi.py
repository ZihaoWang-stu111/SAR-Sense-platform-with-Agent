"""
SAR-Sense API Server (FastAPI)
FastAPI backend with streaming support and thought chain visualization
"""

import logging
import os
import sys

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

from api.app import app  # noqa: E402

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
