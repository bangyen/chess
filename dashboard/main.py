"""Template Dashboard

A simple FastAPI dashboard template. Replace the example endpoints with your data sources.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Analytics Dashboard",
    description="Template dashboard for analytics and data visualization",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


# CORS configuration
cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:5050,http://127.0.0.1:5050"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Mount static files and templates
dashboard_dir = Path(__file__).parent
app.mount(
    "/static", StaticFiles(directory=str(dashboard_dir / "static")), name="static"
)
templates = Jinja2Templates(directory=str(dashboard_dir / "templates"))


# Response models
class OverviewResponse(BaseModel):
    metrics: dict
    chart_data: dict


class TimeSeriesResponse(BaseModel):
    rounds: list[int]
    series_1: list[float]
    series_2: list[float]
    series_3: list[float]


class DetailsResponse(BaseModel):
    items: list[dict]


class HealthResponse(BaseModel):
    status: str
    timestamp: str


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Serve the main dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/overview", response_model=OverviewResponse)
async def overview_endpoint() -> OverviewResponse:
    """Return overview metrics for the dashboard."""
    # Example data - replace with your actual data source
    return OverviewResponse(
        metrics={
            "metric_1": 42,
            "metric_2": 3.14,
            "metric_3": 100,
            "metric_4": 7.5,
        },
        chart_data={
            "labels": ["Category A", "Category B", "Category C"],
            "values": [30, 50, 20],
        },
    )


@app.get("/api/timeseries", response_model=TimeSeriesResponse)
async def timeseries_endpoint() -> TimeSeriesResponse:
    """Return time series data for visualizations."""
    # Example data - replace with your actual data source
    return TimeSeriesResponse(
        rounds=list(range(50)),
        series_1=[i * 1.2 for i in range(50)],
        series_2=[i * 0.8 for i in range(50)],
        series_3=[i * 1.5 for i in range(50)],
    )


@app.get("/api/details", response_model=DetailsResponse)
async def details_endpoint() -> DetailsResponse:
    """Return detailed data for the metrics view."""
    # Example data - replace with your actual data source
    return DetailsResponse(
        items=[
            {"name": "Item 1", "value": 100, "metric": 0.85},
            {"name": "Item 2", "value": 200, "metric": 0.92},
            {"name": "Item 3", "value": 150, "metric": 0.78},
        ]
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", timestamp=datetime.now(timezone.utc).isoformat()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5050,
        log_level="info",
        server_header=False,
        forwarded_allow_ips="*",
    )
