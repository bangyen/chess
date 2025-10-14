# Chess AI Web Application

A modern, dashboard-style web interface for the Chess AI explainability engine with clean, professional design inspired by contemporary analytics dashboards.

## Architecture

### Backend (Flask)
- **Framework**: Flask 3.0+
- **Language**: Python 3.8+
- **Integration**: Direct integration with existing `chess_ai` package
- **Engine**: Optional Stockfish support for AI analysis

### Frontend
- **Design**: Modern dashboard with sidebar navigation
- **CSS**: Custom properties-based design system
- **JavaScript**: Vanilla JS with Chart.js for visualizations
- **Typography**: Space Grotesk (UI), JetBrains Mono (data)

## Design System

### Color Tokens
```css
--color-bg: #F5F7FA           /* Light background */
--color-surface: #FFFFFF      /* Surface/card background */
--color-text-primary: #1A202C /* Primary text */
--color-text-secondary: #4A5568 /* Secondary text */
--color-accent: #2C5F2D       /* Chess green accent */
--color-border: #E1E8ED       /* Subtle borders */
--color-success: #48BB78      /* Success states */
```

### Spacing Scale
```css
--space-xs: 0.25rem    /* 4px */
--space-sm: 0.5rem     /* 8px */
--space-md: 1rem       /* 16px */
--space-lg: 1.5rem     /* 24px */
--space-xl: 2rem       /* 32px */
--space-2xl: 3rem      /* 48px */
```

### Design Principles
- **Dashboard layout**: Sidebar navigation with main content area
- **Card-based UI**: Modular, self-contained components
- **Minimal motion**: ≤ 150ms transitions
- **Clear hierarchy**: Metric cards, charts, and data tables
- **Semantic HTML**: Accessible, meaningful structure
- **Responsive design**: Mobile-friendly with adaptive sidebar

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

### Game Management
- `POST /api/game/new` - Start new game
- `GET /api/game/state` - Get current state
- `POST /api/game/move` - Make player move

### Engine
- `POST /api/engine/move` - Request AI move with explanation

### Analysis
- `POST /api/analysis/features` - Extract position features

### Health
- `GET /api/health` - Health check with engine status

## File Structure

Mirrors the template dashboard structure:

```
src/chess_ai/web/
├── __init__.py
├── app.py                      # Flask application (~250 lines)
├── templates/
│   └── dashboard.html          # Dashboard layout (~200 lines)
├── static/
│   ├── css/
│   │   └── style.css          # Modern design system (~600 lines)
│   └── js/
│       ├── chess.js           # Board rendering (154 lines)
│       └── app.js             # Dashboard logic with Chart.js (~450 lines)
└── README.md                   # Web documentation
```

## Installation

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Install Stockfish** (optional, for AI):
   ```bash
   # macOS
   brew install stockfish
   
   # Ubuntu/Debian
   sudo apt install stockfish
   ```

## Usage

### Quick Start (Makefile)
```bash
make web
```

### Using Launch Script
```bash
./scripts/run_web.sh
```

### Manual Start
```bash
source venv/bin/activate
python -m chess_ai.web.app
```

### Access
Open `http://localhost:5000` in your browser

## Technical Details

### Chess Board Rendering
- 8×8 grid using CSS Grid
- Unicode chess pieces (♔♕♖♗♘♙)
- Click-based move input
- Legal move highlighting
- Classic chess board colors (light/dark squares)

### State Management
- Server-side game state in Flask
- RESTful API communication
- Stateless HTTP requests
- Board synchronization via FEN
- Move history tracking for charts

### Visualizations
- Chart.js for data visualization
- Material balance over time
- Interactive move history
- Responsive chart sizing

### Performance
- Minimal dependencies (Chart.js only)
- Efficient vanilla JavaScript
- CSS-only transitions
- Lazy feature extraction

### Accessibility
- Semantic HTML structure
- High contrast colors (WCAG AA)
- Keyboard navigation support
- Screen reader compatible
- `prefers-reduced-motion` support

## Design Philosophy

### Visual Hierarchy
1. **Navigation**: Sidebar with icon-based menu
2. **Metrics**: Card-based dashboard with key stats
3. **Primary actions**: Accent color buttons (chess green)
4. **Secondary actions**: Bordered buttons with hover states
5. **Status indicators**: Live status badge with pulse animation
6. **Data display**: Cards, charts, and feature grids

### Typography
- **UI text**: Space Grotesk (modern, professional)
- **Numeric data**: JetBrains Mono
- **Base size**: 15px
- **Line height**: 1.5
- **Font weights**: 400 (regular), 500 (medium), 600 (semibold), 700 (bold)

### Spacing
- **Vertical rhythm**: 8px base unit
- **Component padding**: 16-24px
- **Section margins**: 32-48px

### Motion
- **Transitions**: 120ms ease
- **Hover states**: Opacity/border changes
- **No animations**: Unless enhancing UX
- **Reduced motion**: Respects user preference

## Code Quality

### Python
- **Formatter**: Black (88 chars)
- **Linter**: Ruff
- **Type hints**: MyPy compatible
- **Docstrings**: Google style

### CSS
- **Architecture**: Custom properties
- **Naming**: BEM-inspired
- **Size**: 6.5 KB (optimized)
- **Browser support**: Modern evergreen

### JavaScript
- **Style**: ES6+
- **Classes**: Object-oriented
- **Comments**: Minimal, intentional
- **Size**: ~10 KB total

## Implemented Enhancements

Recent additions to the dashboard:
- ✅ Move history visualization with Chart.js
- ✅ Material balance tracking over time
- ✅ Sidebar navigation with multiple views
- ✅ Metric cards for game statistics
- ✅ Modern dashboard layout
- ✅ Health check endpoint

## Future Enhancements

Potential additions (not implemented):
- Opening book integration
- Game export (PGN)
- Multi-game management
- WebSocket real-time updates
- Engine evaluation scores
- Move-by-move position analysis

## Credits

- **Chess Engine**: Stockfish
- **Chess Library**: python-chess
- **Web Framework**: Flask
- **Visualizations**: Chart.js 4.4.0
- **Fonts**: Space Grotesk, JetBrains Mono (Google Fonts)
- **Design Inspiration**: Modern dashboard templates

## License

MIT License - Same as parent project

