# Explainable Chess Engine

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://github.com/pytest-dev/pytest)

A comprehensive chess AI toolkit featuring feature explainability analysis, advanced positional metrics, and an interactive explainable chess engine for learning and research.

## Features

### Core Functionality
- **Feature Explainability Audit**: Evaluate how well chess features explain Stockfish's evaluations using machine learning
- **Explainable Chess Engine**: Play against Stockfish with move explanations and learning feedback
- **Advanced Positional Metrics**: Sophisticated chess position analysis including passed pawn momentum, king safety, and piece activity
- **Kendall Tau Correlation**: Statistical analysis of move ranking correlations

### Development & Quality
- **Modern tooling**: Black, Ruff, MyPy, Pytest with comprehensive configuration
- **Pre-commit hooks**: Automated code quality checks and formatting
- **Type hints**: Full type checking with MyPy for robust development
- **Testing**: Pytest with fixtures, coverage, and integration tests
- **Package structure**: Standard src/ layout with proper module organization

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone https://github.com/bangyen/chess.git
   cd chess
   
   # Activate virtual environment (required)
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   
   # Install dependencies and setup development tools
   make init
   ```

2. **Run Feature Audit**:
   ```bash
   chess-ai audit --baseline_features --positions 100
   ```

3. **Play Explainable Chess**:
   ```bash
   chess-ai play --strength intermediate
   ```

4. **Development workflow**:
   ```bash
   make fmt    # Format code
   make lint   # Lint code
   make type   # Type check
   make test   # Run tests
   make all    # Run all checks
   ```

## CLI Commands

The chess AI package provides several command-line interfaces:

### Available Commands
```bash
chess-ai audit [options]    # Run feature explainability audit
chess-ai play [options]     # Play interactive chess with explanations
chess-ai help              # Show help (default)
```

**Alternative commands:**
```bash
chess-ai-audit [options]    # Direct audit command
chess-ai-play [options]     # Direct play command
```

## Project Structure

```
├── src/chess_ai/               # Main package
│   ├── audit.py                # Feature explainability audit core
│   ├── explainable_engine.py   # Explainable chess engine
│   ├── cli/                    # Command-line interfaces
│   │   ├── main.py             # Unified CLI dispatcher
│   │   ├── audit.py            # Feature audit CLI
│   │   └── explainable.py      # Explainable engine CLI
│   ├── engine/                 # Chess engine integration
│   │   ├── config.py           # Engine configuration
│   │   └── interface.py        # Engine interface utilities
│   ├── features/               # Chess feature extraction
│   │   ├── baseline.py         # Baseline feature set
│   │   └── utils.py            # Feature loading utilities
│   ├── metrics/                # Advanced chess metrics
│   │   ├── kendall.py          # Kendall tau correlation
│   │   └── positional.py       # Positional analysis metrics
│   └── utils/                  # General utilities
│       └── sampling.py         # Position sampling utilities
├── tests/                     # Comprehensive test suite
│   ├── unit/                  # Unit tests
│   │   ├── engine/            # Engine tests
│   │   ├── features/          # Feature tests
│   │   ├── metrics/           # Metrics tests
│   │   └── utils/             # Utility tests
│   └── integration/           # Integration tests
├── scripts/                   # Utility scripts
├── venv/                      # Virtual environment
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
# Start a game against beginner Stockfish (recommended)
chess-ai play

# Play against intermediate Stockfish
chess-ai play --strength intermediate

# Legacy syntax still works
python -m src.chess_ai.cli.explainable --strength intermediate

# Game session example:
White to move: e4

Your Move e4:
- Controls central squares
- Develops pawn from starting position

🤖 Stockfish plays: e5
```

### Commands
- **Moves**: Use standard algebraic notation (e4, Nf3, O-O, etc.)
- **`best`**: Show the best move recommendation
- **`reset`**: Reset the game
- **`help`**: Show available commands
- **`quit`**: Exit the game

## Advanced Features

### Feature Analysis
```python
from chess_ai import audit_feature_set, baseline_extract_features
from chess_ai.engine import SFConfig, sf_open

# Run feature explainability audit
config = SFConfig(engine_path="/path/to/stockfish", depth=15, multipv=3)
engine = sf_open(config)
result = audit_feature_set(
    boards=boards, 
    engine=engine, 
    cfg=config, 
    extract_features_fn=baseline_extract_features
)
print(f"R² Score: {result.r2_score}")
```

### Positional Metrics
```python
import chess
from chess_ai.metrics.positional import passed_pawn_momentum_snapshot
from chess_ai.features.baseline import baseline_extract_features

# Analyze position
momentum = passed_pawn_momentum_snapshot(board, chess.WHITE)
features = baseline_extract_features(board)
print(f"Passed pawns: {momentum['pp_count']}, King safety: {features['king_safety_us']}")
```

### Custom Features
Create your own feature extraction and use it:
```python
# my_features.py
def extract_features(board):
    return {'my_feature': 1.0}  # Your custom logic
```
```bash
chess-ai audit --features_module my_features.py --positions 1000
```

## API Reference

### Key Classes
- **`AuditResult`**: Contains R² score, MSE, and top features from feature analysis
- **`SFConfig`**: Stockfish engine configuration (depth, multipv, time limits)
- **`MoveExplanation`**: Move analysis with evaluation and explanations

### Main Functions
- **`baseline_extract_features(board)`**: Extract chess features from a position
- **`audit_feature_set(boards, engine, config, extract_fn)`**: Run feature explainability audit
- **`kendall_tau(rank_a, rank_b)`**: Calculate ranking correlation
- **`passed_pawn_momentum_snapshot(board, color)`**: Analyze passed pawn momentum

For complete API documentation, see the source code docstrings.

## Requirements

### Core Dependencies
- **Python 3.8+** - Modern Python features and type hints
- **python-chess>=1.999** - Chess position representation and move generation
- **numpy>=1.20.0** - Numerical computations and array operations
- **scikit-learn>=1.0.0** - Machine learning for feature analysis
- **tqdm>=4.60.0** - Progress bars for long-running operations

### Development Dependencies
- **pytest>=7.0.0** - Testing framework
- **pytest-cov>=4.0.0** - Test coverage reporting
- **black>=23.0.0** - Code formatting
- **ruff>=0.1.0** - Fast Python linter
- **mypy>=1.0.0** - Static type checking
- **pre-commit>=3.0.0** - Git hooks for code quality

### Optional
- **Stockfish** - Chess engine for analysis (recommended, with fallback to chess principles)

## Development

### Setup
```bash
source venv/bin/activate  # Activate virtual environment
make init                 # Install dependencies and setup tools
```

### Quality Checks
```bash
make all    # Run all checks (format, lint, type, test)
make fmt    # Format with Black
make lint   # Lint with Ruff  
make type   # Type check with MyPy
make test   # Run tests with Pytest
```

### Standards
- Type hints for all public functions
- Docstrings explaining *why* functions exist
- Pytest tests with seeded randomness
- Early input validation with clear exceptions

## License

MIT License - see LICENSE file for details.
