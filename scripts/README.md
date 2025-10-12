# Scripts

Utility scripts for running Chess AI tools.

## Available Scripts

### run_web.sh

Launch the web interface with automatic setup and validation.

**Usage:**
```bash
./scripts/run_web.sh
```

**Features:**
- Checks for virtual environment
- Installs dependencies if needed
- Validates Stockfish availability
- Starts Flask development server on port 5000

### example.py

Example usage of Chess AI features (legacy).

## Adding Scripts

When adding new scripts:
1. Make them executable: `chmod +x scripts/yourscript.sh`
2. Add usage instructions to this README
3. Include error handling and helpful messages
4. Follow the project's code style guidelines

