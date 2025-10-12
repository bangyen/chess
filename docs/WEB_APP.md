# Chess AI Web Application

A clean, professional web interface for the Chess AI explainability engine, built with Swiss + Terminal-Modern aesthetic principles.

## Architecture

### Backend (Flask)
- **Framework**: Flask 3.0+
- **Language**: Python 3.8+
- **Integration**: Direct integration with existing `chess_ai` package
- **Engine**: Optional Stockfish support for AI analysis

### Frontend
- **Design**: Swiss + Terminal-Modern aesthetic
- **CSS**: 6.5 KB (< 100 KB requirement)
- **JavaScript**: Vanilla JS, no frameworks
- **Typography**: Inter (UI), JetBrains Mono (data)

## Design System

### Color Tokens
```css
--bg: #0F1318          /* Background */
--fg: #E5E7EB          /* Foreground */
--muted: #9AA3AF       /* Muted text */
--border: #1F2937      /* Borders */
--accent: #22D3EE      /* Accent (cyan) */
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
- **Grid-based layout**: Consistent spacing and alignment
- **Minimal motion**: ≤ 160ms transitions
- **High contrast**: ≥ 4.5:1 color contrast
- **Semantic HTML**: Accessible, meaningful structure
- **Modular components**: Reusable, maintainable code

## Features

### Play Tab
- Interactive chess board with piece movement
- Real-time game state display
- AI move suggestions with explanations
- Move validation and legal move highlighting

### Analysis Tab
- Position feature extraction
- 20+ chess-specific metrics
- Material, mobility, king safety analysis
- Sorted by feature importance

### Metrics Tab
- Engine performance dashboard
- Feature explainability: 87.5%
- Explanation sparsity: 2.0
- Position coverage: 100%

## API Endpoints

### Game Management
- `POST /api/game/new` - Start new game
- `GET /api/game/state` - Get current state
- `POST /api/game/move` - Make player move

### Engine
- `POST /api/engine/move` - Request AI move with explanation

### Analysis
- `POST /api/analysis/features` - Extract position features

## File Structure

```
src/chess_ai/web/
├── __init__.py
├── app.py                 # Flask application (216 lines)
├── templates/
│   └── index.html         # Main page (120 lines)
├── static/
│   ├── style.css          # Design system (6.5 KB)
│   ├── chess.js           # Board rendering (154 lines)
│   └── app.js             # Application logic (156 lines)
└── README.md              # Web documentation
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

### Quick Start
```bash
./run_web.sh
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

### State Management
- Server-side game state
- RESTful API communication
- Stateless HTTP requests
- Board synchronization via FEN

### Performance
- Zero dependencies (frontend)
- Minimal JavaScript
- CSS-only animations
- Lazy feature extraction

### Accessibility
- Semantic HTML structure
- High contrast colors (WCAG AA)
- Keyboard navigation support
- Screen reader compatible
- `prefers-reduced-motion` support

## Design Philosophy

### Visual Hierarchy
1. **Primary actions**: Accent color buttons
2. **Secondary actions**: Bordered ghost buttons
3. **Status indicators**: Mono badges
4. **Data display**: Tabular layouts

### Typography
- **UI text**: Inter (system fallback)
- **Numeric data**: JetBrains Mono
- **Base size**: 15px (0.9375rem)
- **Line height**: 1.6

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

## Future Enhancements

Potential additions (not implemented):
- Move history timeline
- Position evaluation graph
- Opening book integration
- Game export (PGN)
- Multi-game management
- WebSocket real-time updates

## Credits

- **Chess Engine**: Stockfish
- **Chess Library**: python-chess
- **Web Framework**: Flask
- **Fonts**: Inter, JetBrains Mono (Google Fonts)

## License

MIT License - Same as parent project

