#!/bin/bash
# Launch the Chess AI web interface

set -e

echo "🚀 Starting Chess AI Web Interface"
echo ""

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

echo "📦 Activating virtual environment..."
source venv/bin/activate

if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -q -e .
fi

if ! command -v stockfish &> /dev/null; then
    echo "⚠️  Stockfish not found. Engine analysis will be limited."
    echo "   Install: brew install stockfish (macOS) or apt install stockfish (Linux)"
    echo ""
fi

echo "✅ Starting web server..."
echo "🌐 Open http://localhost:5000 in your browser"
echo ""

python -m chess_ai.web.app

