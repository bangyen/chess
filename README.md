# Chess AI Tools

A comprehensive chess AI toolkit featuring feature explainability analysis and an interactive explainable chess engine for learning and research.

## Features

- **Feature Explainability Audit**: Evaluate how well chess features explain Stockfish's evaluations
- **Explainable Chess Engine**: Play against Stockfish with move explanations and learning feedback
- **Modern tooling**: Black, Ruff, MyPy, Pytest
- **Pre-commit hooks**: Automated code quality checks
- **Type hints**: Full type checking with MyPy
- **Testing**: Pytest with fixtures and coverage
- **Package structure**: Standard src/ layout

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd chess
   make init
   ```

2. **Run Feature Audit**:
   ```bash
   python -m src.chess_ai.cli.audit
   ```

3. **Play Explainable Chess**:
   ```bash
   # Play against Stockfish with explanations (default: beginner level)
   python -m src.chess_ai.cli.explainable
   
   # Play against stronger Stockfish
   python -m src.chess_ai.cli.explainable --strength intermediate
   ```

4. **Development workflow**:
   ```bash
   make fmt    # Format code
   make lint   # Lint code
   make type   # Type check
   make test   # Run tests
   make all    # Run all checks
   ```

## Project Structure

```
├── src/chess_ai/               # Main package
│   ├── audit.py                # Feature explainability audit
│   ├── features/               # Chess feature extraction
│   ├── explainable_engine.py   # Explainable chess engine
│   └── cli/                    # Command-line interfaces
│       ├── audit.py            # Feature audit CLI
│       └── explainable.py      # Explainable engine CLI
├── tests/                     # Test files
├── scripts/                   # Utility scripts
├── Makefile                   # Development commands
├── pyproject.toml            # Package configuration
└── README.md                 # This file
```

## Explainable Chess Engine

The explainable chess engine provides interactive chess games with AI-powered move explanations:

### Features
- **Move Analysis**: Explains why your moves are good or bad
- **Best Move Recommendations**: Shows the best move and reasoning
- **Educational Explanations**: Covers material, tactics, development, king safety
- **Interactive Commands**: `best`, `reset`, `help`, `quit`

### Usage Examples

```bash
# Start a game against beginner Stockfish
python -m src.chess_ai.cli.explainable

# Play against intermediate Stockfish
python -m src.chess_ai.cli.explainable --strength intermediate

# Game session example:
White to move: e4
✅ Move e4 played

📊 Your Move Analysis:
   Move e4 is reasonable. Positive aspects: Controls central squares, Develops pawn from starting position.

💡 Best move would be: e4
   Move e4 is excellent. Positive aspects: Controls central squares, Develops pawn from starting position.

🤖 Stockfish (beginner) is thinking...
🤖 Stockfish plays: e5

📊 Stockfish's Move Analysis:
   Move e5 is good. Positive aspects: Controls central squares, Develops pawn from starting position.
```

### Commands
- **Moves**: Use standard algebraic notation (e4, Nf3, O-O, etc.)
- **`best`**: Show the best move recommendation
- **`reset`**: Reset the game
- **`help`**: Show available commands
- **`quit`**: Exit the game

## Requirements

- Python 3.8+
- `python-chess` library
- `scikit-learn` for machine learning features
- `tqdm` for progress bars
- Stockfish (recommended, with fallback to chess principles)

## License

MIT License - see LICENSE file for details.
