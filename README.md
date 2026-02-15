# Explainable Chess Engine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/chess/blob/main/chess_demo.ipynb)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![License](https://img.shields.io/github/license/bangyen/chess)](LICENSE)

**Chess AI Explainability: 86.7% decisive faithfulness, 2.5 sparsity explanations, 100% position coverage with interactive learning engine**

<p align="center">
  <img src="docs/audit-demo.gif" alt="Demo preview" width="600">
</p>

## Quickstart

### Prerequisites

- Python 3.8+
- Rust 1.70+ (required for building the acceleration module)

### Installation

```bash
git clone https://github.com/bangyen/chess.git
cd chess
pip install -e .
pytest   # optional: run tests
```

### Usage Options

**CLI Tools:**
```bash
# Run feature explainability audit
chess-ai audit --baseline_features --positions 100

# Play interactive chess with explanations
chess-ai play --strength intermediate
```

**Web Interface:**
```bash
# Launch web app
./scripts/run_web.sh

# Or manually
python -m chess_ai.web.app
# Then open http://localhost:5000
```

**Jupyter Notebook:**  
Open in [Colab](https://colab.research.google.com/github/bangyen/chess/blob/main/chess_demo.ipynb) for interactive exploration.

## Results

| Metric | Value | Target |
|--------|-------|--------|
| Feature Explainability | **86.7%** | ≥80% |
| Explanation Sparsity | **2.5** | ≤3.0 |
| Position Coverage | **100%** | ≥95% |

## Features

- **Feature Explainability Audit** — ML-based evaluation of how well chess features explain Stockfish's reasoning with 86.7% decisive faithfulness.
- **Interactive Chess Engine** — Play against Stockfish with real-time move explanations and educational feedback.
- **Rust-Accelerated Analysis** — High-performance feature extraction and evaluation using a custom Rust backend for maximum speed.
- **Web Interface** — Clean, professional web app with Swiss + Terminal-Modern design for interactive gameplay and analysis.
- **Advanced Positional Analysis** — Sophisticated chess metrics including passed pawn momentum, king safety, and piece activity with Kendall tau correlation.

## Repo Structure

```plaintext
chess/
├── src/chess_ai/
│   ├── cli/          # Command-line interface tools
│   ├── engine/       # Chess engine configuration
│   ├── features/     # Feature extraction
│   ├── metrics/      # Evaluation metrics
│   ├── rust_utils/   # Rust extension bindings
│   ├── utils/        # Helper utilities
│   └── web/          # Web interface (Flask app)
├── rust/             # Rust acceleration module source
├── tests/            # Unit and integration tests
├── docs/             # Documentation and design system
├── scripts/          # Example and launch scripts
└── chess_demo.ipynb  # Interactive Colab notebook
```

See [docs/WEB_APP.md](docs/WEB_APP.md) for web interface documentation and [docs/DESIGN_SYSTEM.md](docs/DESIGN_SYSTEM.md) for design specifications.

## Validation

- ✅ Overall test coverage of 96% (`pytest`)
- ✅ Reproducible seeds for experiments
- ✅ Benchmark scripts included

## References

- [Python-Chess Library](https://python-chess.readthedocs.io/) — Chess position representation and move generation.  
- [Stockfish Chess Engine](https://stockfishchess.org/) — Open-source chess engine for analysis and evaluation.  
- [Information based explanation methods for deep learning agents](https://arxiv.org/abs/2309.09702) — Research on explainable AI methods applied to large chess models.

## License

This project is licensed under the [MIT License](LICENSE).
