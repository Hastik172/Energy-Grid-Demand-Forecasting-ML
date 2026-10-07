"""
FastAPI Main Application for GridSense.
Serves REST API, Swagger UI at /docs, and the interactive frontend.
Run via: uvicorn backend.app:app --reload --port 8001
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.api_routes import router

app = FastAPI(
    title="GridSense API",
    description="Energy Grid Demand & Renewable Power Forecasting with Uncertainty Quantification (Unit 1 ML)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST routes
app.include_router(router)

# Mount static figures
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figures_dir = os.path.join(base_dir, "figures")
if os.path.exists(figures_dir):
    app.mount("/figures", StaticFiles(directory=figures_dir), name="figures")

# Serve frontend HTML
frontend_path = os.path.join(base_dir, "frontend", "index.html")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_home():
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return "<h1>GridSense API running. Visit <a href='/docs'>/docs</a> for Swagger documentation.</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8001, reload=True)
