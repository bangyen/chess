# Web Application Architecture

The Chess AI dashboard is a high-performance web interface built with Rust and vanilla JavaScript.

## Backend (Rust)

- **Framework**: [Axum](https://github.com/tokio-rs/axum)
- **Templating**: [Tera](https://keats.github.io/tera/)
- **Server**: [Tokio](https://tokio.rs/)
- **State Management**: Shared `Arc<RwLock<AppState>>` for thread-safe session handling.

### API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Main dashboard interface |
| `GET` | `/api/state` | Current board FEN and legal moves |
| `POST` | `/api/move` | Submit a player move (UCI format) |
| `POST` | `/api/engine-move` | Request an AI move with explanation |
| `POST` | `/api/new-game` | Reset the board state |
| `POST` | `/api/analyze` | Extract chess features for current position |

## Frontend

- **Vanilla JS**: No heavy frameworks; direct DOM manipulation for performance.
- **Canvas API**: The chessboard is rendered using a reactive Canvas implementation (`chess.js`).
- **CSS3**: Modern layout using CSS Grid and Flexbox with a custom design system.

## File Structure

```plaintext
web/
├── static/
│   ├── css/style.css    # Design system and layout
│   └── js/
│       ├── chess.js     # Board rendering and interaction
│       └── app.js       # Dashboard logic and API integration
└── templates/
    └── dashboard.html   # Main application shell
```

## Development

Run the server locally with:
```bash
just web
```
The server will be available at `http://localhost:5000`.
