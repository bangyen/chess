# Explainable Chess Engine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bangyen/chess/blob/main/chess_demo.ipynb)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![License](https://img.shields.io/github/license/bangyen/chess)](LICENSE)

**Chess AI Explainability: 87.5% decisive faithfulness, 2.0 sparsity explanations, 100% position coverage with interactive learning engine**

<p align="center">
  <img src="docs/audit-demo.gif" alt="Demo preview" width="600">
</p>

## Quickstart

Clone the repo and run the demo:

```bash
git clone https://github.com/bangyen/chess.git
cd chess
pip install -e .
pytest   # optional: run tests
chess-ai audit --baseline_features --positions 100
```

Or open in Colab: [Colab Notebook](https://colab.research.google.com/github/bangyen/chess/blob/main/chess_demo.ipynb).

## Results

| Metric | Value | Target |
|--------|-------|--------|
| Feature Explainability | **87.5%** | ≥80% |
| Explanation Sparsity | **2.0** | ≤3.0 |
| Position Coverage | **100%** | ≥95% |

## Features

- **Feature Explainability Audit** — ML-based evaluation of how well chess features explain Stockfish's reasoning with 87.5% decisive faithfulness.
- **Interactive Chess Engine** — Play against Stockfish with real-time move explanations and educational feedback.
- **Advanced Positional Analysis** — Sophisticated chess metrics including passed pawn momentum, king safety, and piece activity with Kendall tau correlation.

## Repo Structure

```plaintext
chess/
├── chess_demo.ipynb  # Colab notebook
├── scripts/          # Example run scripts
├── tests/            # Unit/integration tests
├── docs/             # Images / gifs for README
└── src/              # Core implementation
```

## Validation

- ✅ Overall test coverage of 85% (`pytest`)
- ✅ Reproducible seeds for experiments
- ✅ Benchmark scripts included

## References

- [Python-Chess Library](https://python-chess.readthedocs.io/) — Chess position representation and move generation.  
- [Stockfish Chess Engine](https://stockfishchess.org/) — Open-source chess engine for analysis and evaluation.  
- [Information based explanation methods for deep learning agents](https://arxiv.org/abs/2309.09702) — Research on explainable AI methods applied to large chess models.

## License

This project is licensed under the [MIT License](LICENSE).
