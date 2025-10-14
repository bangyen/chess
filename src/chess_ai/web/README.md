# Chess AI Web Interface

A modern dashboard-style web interface for the Chess AI explainability engine with sidebar navigation, metric cards, and interactive visualizations.

## Design

The interface features a contemporary dashboard aesthetic:

- **Typography**: Space Grotesk for UI, JetBrains Mono for data
- **Colors**: Light, professional palette with chess green accents
- **Layout**: Sidebar navigation with card-based dashboard
- **Motion**: Smooth transitions (≤ 150ms)
- **Visualizations**: Chart.js integration for move history

## Running

### Quick Start (Makefile)
From the project root:

```bash
make web
```

### Manual Start
```bash
source venv/bin/activate
python -m chess_ai.web.app
```

### Using Launch Script
```bash
./scripts/run_web.sh
```

Then open `http://localhost:5000` in your browser.

## Features

### Play View
- **Interactive Board**: Click-based piece movement with legal move highlighting
- **Engine Control**: Prominent "Request Engine Move" button
- **Move Explanation**: Real-time AI reasoning for engine moves
- **Key Features**: Top 3 position features displayed after each move

### Features View
- **Position Analysis**: Extract and display all 20+ chess features
- **Sortable Features**: Ordered by absolute value/importance
- **Feature Details**: Material, mobility, king safety, center control, and more

## API Endpoints

- `GET /` - Main dashboard application
- `POST /api/game/new` - Start new game
- `GET /api/game/state` - Get current state
- `POST /api/game/move` - Make a move
- `POST /api/engine/move` - Request engine move with explanation
- `POST /api/analysis/features` - Analyze position features
- `GET /api/health` - Health check with engine status

## Requirements

- Flask ≥ 3.0.0
- python-chess ≥ 1.999
- Stockfish (optional, for engine analysis)

## Design Tokens

```css
--color-bg: #F5F7FA
--color-surface: #FFFFFF
--color-text-primary: #1A202C
--color-text-secondary: #4A5568
--color-accent: #2C5F2D
--color-border: #E1E8ED
--color-success: #48BB78
```

## File Structure

Mirrors the template dashboard structure:

```
web/
├── app.py                      # Flask application
├── templates/
│   └── dashboard.html          # Main dashboard template
├── static/
│   ├── css/
│   │   └── style.css          # Design system
│   └── js/
│       ├── chess.js           # Chess board logic
│       └── app.js             # Dashboard application
└── README.md
```

## Architecture

- **Backend**: Flask with RESTful API
- **Frontend**: Vanilla JavaScript (zero dependencies)
- **Styling**: CSS custom properties
- **State**: Server-side game state management
- **Interface**: 2 focused views (Play, Features)
