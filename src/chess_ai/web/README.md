# Chess AI Web Interface

A clean, professional web interface for the Chess AI explainability engine.

## Design

The interface follows a Swiss + Terminal-Modern aesthetic:

- **Typography**: Inter for UI, JetBrains Mono for data
- **Colors**: Dark terminal palette with cyan accents
- **Layout**: Grid-based, generous whitespace
- **Motion**: Minimal transitions (≤ 160ms)

## Running

From the project root:

```bash
source venv/bin/activate
python -m chess_ai.web.app
```

Or use the launch script:

```bash
./run_web.sh
```

Then open `http://localhost:5000` in your browser.

## Features

- **Interactive Board**: Click to select and move pieces
- **Engine Analysis**: Request AI moves with explanations
- **Position Features**: Analyze current board state
- **Engine Metrics**: View explainability performance

## API Endpoints

- `GET /` - Main application
- `POST /api/game/new` - Start new game
- `GET /api/game/state` - Get current state
- `POST /api/game/move` - Make a move
- `POST /api/engine/move` - Request engine move
- `POST /api/analysis/features` - Analyze position

## Requirements

- Flask ≥ 3.0.0
- python-chess ≥ 1.999
- Stockfish (optional, for engine analysis)

## Design Tokens

```css
--bg: #0F1318
--fg: #E5E7EB
--muted: #9AA3AF
--border: #1F2937
--accent: #22D3EE
--radius: 3px
```

