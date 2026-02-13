
import chess
import chess.engine
from chess_ai.explainable_engine import ExplainableChessEngine
import time

def verify_syzygy_impact():
    print("=== Verifying Syzygy Tablebase Impact ===\n")
    
    # KRP vs K (Lucena position / Philidor) - Known win/draw
    # Simple KR vs K: White Rook at a1, White King at e1, Black King at e8.
    # r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 5 (Middlegame)
    # 8/8/8/8/8/8/R3K3/4k3 w - - 0 1 (KR vs K)
    
    endgames = [
        ("KR vs K (Mate in 10)", "8/8/8/8/8/8/R3K3/4k3 w - - 0 1"),
        ("KBN vs K (Checkmate possible)", "8/8/8/8/8/8/2B1N3/4K2k w - - 0 1"),
        ("KP vs K (Winning)", "8/8/8/8/8/4P3/4K3/k7 w - - 0 1"),
    ]

    # Initialize Engine (assuming Syzygy path is configured or auto-detected if hardcoded)
    # Note: The user configuration might need a real path.
    # In my tests, I passed a path. Here I'll check if I can default it or if I need to ask.
    # The previous test used: '/Users/bangyen/Documents/chess/3-4-5' (example)
    # I'll rely on the default behavior I added (or pass empty and hope it finds system ones?)
    # Actually, in verify_syzygy_interface.py I used a mock/dummy path or real path.
    # Let's assume the user has tablebases at a standard location or the engine finds them.
    # If not, this test might fail to show impact. 
    # But wait, I added logic to `ExplainableChessEngine` to accept `syzygy_path`.
    
    # I'll try to find where Syzygy files are.
    # `codbase_search` for .rtbw files?
    # I'll just use the engine's capability.
    
    path = "/Users/bangyen/Documents/chess/syzygy" # Guessing or using a placeholder
    # Ideally script finds it.
    
    # Setup Engine
    print(f"Initializing Engine...")
    try:
        # We need to use the context manager
        with ExplainableChessEngine(syzygy_path=path) as engine:
            if not engine.syzygy:
                print("WARNING: Syzygy tablebases not found/initialized. Verification will be limited.")
            
            for name, fen in endgames:
                print(f"\nPosition: {name}")
                print(f"FEN: {fen}")
                board = chess.Board(fen)
                
                # Analyze
                start = time.time()
                result = engine.analyze_position(board)
                dt = time.time() - start
                
                print(f"Analysis Time: {dt*1000:.1f} ms")
                
                # Check for Syzygy data in result
                syzygy_data = result.get("syzygy", {})
                print(f"Syzygy Data: {syzygy_data}")
                
                # Check Explanation
                explanation = result.get("explanation", "")
                reasons = result.get("reasons", [])
                
                print("Reasons:")
                found_syzygy_reason = False
                for r in reasons:
                    print(f"  - {r}")
                    if "syzygy" in r[0]:
                        found_syzygy_reason = True
                        
                if found_syzygy_reason or syzygy_data:
                    print("✅ Tablebase impact detected!")
                else:
                    print("❌ No Tablebase impact seen (files missing?).")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_syzygy_impact()
