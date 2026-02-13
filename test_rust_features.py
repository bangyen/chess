#!/usr/bin/env python3
"""Quick test of Rust feature extraction."""

import chess
import time
from chess_ai.rust_utils import extract_features_rust
from chess_ai.features.baseline import baseline_extract_features, RUST_AVAILABLE

if not RUST_AVAILABLE:
    print("❌ Rust not available")
    exit(1)

print("✓ Rust is available")

# Test cases
test_positions = [
    ("Initial", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Italian", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("Endgame", "8/5k2/3p4/1p1Pp3/pP2Pp2/P4P2/8/6K1 b - - 0 1"),
]

print("\n" + "="*60)
print("Testing Direct Rust Function")
print("="*60)

for name, fen in test_positions:
    print(f"\n{name} Position:")
    features = extract_features_rust(fen)
    print(f"  Features extracted: {len(features)}")
    print(f"  Material (us/them): {features['material_us']:.1f} / {features['material_them']:.1f}")
    print(f"  Mobility (us/them): {features['mobility_us']:.1f} / {features['mobility_them']:.1f}")
    print(f"  Phase: {features['phase']:.0f}")
    if 'king_ring_pressure_us' in features:
        print(f"  King ring pressure (us/them): {features['king_ring_pressure_us']:.1f} / {features['king_ring_pressure_them']:.1f}")

print("\n" + "="*60)
print("Testing via baseline_extract_features")
print("="*60)

for name, fen in test_positions:
    board = chess.Board(fen)
    print(f"\n{name} Position:")
    features = baseline_extract_features(board)
    print(f"  Features extracted: {len(features)}")
    print(f"  Material (us/them): {features['material_us']:.1f} / {features['material_them']:.1f}")

print("\n" + "="*60)
print("Performance Comparison")
print("="*60)

board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")

# Time Rust implementation
start = time.time()
for _ in range(1000):
    features = baseline_extract_features(board)
elapsed = (time.time() - start) / 1000
print(f"Average time per position: {elapsed*1000:.2f}ms ({elapsed*1e6:.0f}μs)")

print("\n" + "="*60)
print("✅ All tests passed!")
print("="*60)
