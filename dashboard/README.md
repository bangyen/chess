# Template Dashboard

A modern, production-ready analytics dashboard template built with FastAPI and modern frontend technologies.

## Features

- **FastAPI Backend** - High-performance async API with automatic documentation
- **Modern Frontend** - Responsive design with loading states and toast notifications
- **Interactive Charts** - Chart.js integration with multiple chart types
- **Multi-View Navigation** - Clean navigation between different dashboard views
- **Pydantic Models** - Type-safe request/response validation
- **Error Handling** - Comprehensive error handling with user-friendly messages
- **CORS Support** - Configurable cross-origin resource sharing
- **Health Checks** - Built-in health monitoring endpoints

## Quick Start

```bash
# Install dependencies
pip install -e .

# Start the dashboard
cd dashboard
python main.py
```

That's it! The dashboard will start on `http://localhost:5050`

### Development Mode

For development with auto-reload:

```bash
python run.py
```

## API Endpoints

- `GET /` - Dashboard interface
- `GET /api/overview` - Overview metrics and charts
- `GET /api/timeseries` - Time series data
- `GET /api/details` - Detailed data table
- `GET /api/health` - Health check
- `GET /api/docs` - Interactive API documentation


## Environment Variables

- `CORS_ORIGINS` - Comma-separated list of allowed origins (required for production)
- `ENVIRONMENT` - Set to "production" for production mode

## Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Start in development mode with auto-reload
python dashboard/run.py --dev
```

## Customization

1. **Replace Example Data**: Update the API endpoints in `main.py` with your data sources
2. **Modify UI**: Customize the HTML templates and CSS styles
3. **Add Features**: Extend the JavaScript dashboard class with new functionality

## Production Deployment

1. Set `CORS_ORIGINS` environment variable
2. Set `ENVIRONMENT=production`
3. Use a production ASGI server like Gunicorn with Uvicorn workers

## File Structure

```
template/
├── pyproject.toml        # Project configuration and dependencies
├── dashboard/
│   ├── main.py          # FastAPI application
│   ├── run.py           # Development runner script
│   ├── templates/
│   │   └── dashboard.html # Main dashboard template
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css  # Dashboard styles
│   │   └── js/
│   │       └── dashboard.js # Dashboard JavaScript
│   └── README.md        # This file
```

## License

This template is provided as-is for educational and development purposes.