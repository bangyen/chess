# Scripts

Utility scripts for running, benchmarking, and verifying Chess AI tools.

## Available Scripts

### run_web.sh

Launch the web interface with automatic setup and validation.

```bash
./scripts/run_web.sh
```

- Checks for virtual environment
- Installs dependencies if needed
- Validates Stockfish availability
- Starts Flask development server on port 5000

### verify_syzygy_endgames.py

Verify Syzygy tablebase integration on known 3–5 piece endgame positions.

```bash
SYZYGY_PATH=~/syzygy python scripts/verify_syzygy_endgames.py
```

### fetch_syzygy.py

Download Syzygy 3/4/5-piece tablebase files from `tablebase.sesse.net`.

```bash
python scripts/fetch_syzygy.py
```

### benchmark_python_probes.py

Benchmark Python-side feature extraction and Stockfish engine probes.

```bash
python scripts/benchmark_python_probes.py
```

### benchmark_commits.py

Compare audit metrics (R², Kendall τ, faithfulness, sparsity, coverage) across
two git commits.

```bash
python scripts/benchmark_commits.py <commit_a> <commit_b>
```

### profile_probes_rust.py

Profile the Rust `find_best_reply` function at various search depths.

```bash
python scripts/profile_probes_rust.py
```

### test_rust_features.py

Quick sanity check that Rust feature extraction matches the Python baseline.

```bash
python scripts/test_rust_features.py
```

## Adding Scripts

When adding new scripts:

1. Make them executable: `chmod +x scripts/yourscript.sh`
2. Add usage instructions to this README
3. Include error handling and helpful messages
4. Follow the project's code style guidelines
