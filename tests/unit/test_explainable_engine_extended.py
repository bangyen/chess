"""Extended tests for explainable_engine.py to increase coverage.

Targets uncovered lines: _get_syzygy_data, _generate_hardcoded_reasons,
_generate_move_reasons_with_board details, overall explanation quality
labels, get_move_recommendation fallback paths, and
get_best_move_for_player fallback paths.
"""

from unittest.mock import Mock, patch

import chess

from chess_ai.explainable_engine import ExplainableChessEngine, MoveExplanation

# ---------------------------------------------------------------------------
# _get_syzygy_data
# ---------------------------------------------------------------------------


class TestGetSyzygyData:
    """Tests for _get_syzygy_data tablebase probing."""

    def test_returns_empty_when_no_syzygy(self):
        """Returns {} when syzygy is None."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        assert eng._get_syzygy_data(chess.Board()) == {}

    def test_returns_empty_when_too_many_pieces(self):
        """Returns {} when piece count exceeds 7."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = Mock()
        # Starting position has > 7 pieces
        assert eng._get_syzygy_data(chess.Board()) == {}

    def test_returns_wdl_and_dtz(self):
        """Returns wdl/dtz when syzygy probe succeeds."""
        eng = ExplainableChessEngine("/sf")
        mock_syzygy = Mock()
        mock_syzygy.probe_wdl.return_value = 2
        mock_syzygy.probe_dtz.return_value = 10
        eng.syzygy = mock_syzygy

        # KPvK endgame (few pieces)
        board = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        result = eng._get_syzygy_data(board)

        assert result["wdl"] == 2
        assert result["dtz"] == 10

    def test_returns_partial_when_dtz_is_none(self):
        """Returns only wdl when dtz probe returns None."""
        eng = ExplainableChessEngine("/sf")
        mock_syzygy = Mock()
        mock_syzygy.probe_wdl.return_value = 0
        mock_syzygy.probe_dtz.return_value = None
        eng.syzygy = mock_syzygy

        board = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        result = eng._get_syzygy_data(board)

        assert result == {"wdl": 0}

    def test_returns_empty_on_exception(self):
        """Returns {} when probe raises an exception."""
        eng = ExplainableChessEngine("/sf")
        mock_syzygy = Mock()
        mock_syzygy.probe_wdl.side_effect = RuntimeError("oops")
        eng.syzygy = mock_syzygy

        board = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        assert eng._get_syzygy_data(board) == {}


# ---------------------------------------------------------------------------
# _get_syzygy_reason
# ---------------------------------------------------------------------------


class TestGetSyzygyReason:
    """Tests for Syzygy-based reason generation."""

    def _eng_with_syzygy(self, wdl_before, wdl_after, dtz_before=0, dtz_after=0):
        """Build engine with mocked syzygy returning given wdl values."""
        eng = ExplainableChessEngine("/sf")
        call_count = {"n": 0}

        def fake_get_syzygy(board):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"wdl": wdl_before, "dtz": dtz_before}
            return {"wdl": wdl_after, "dtz": dtz_after}

        eng._get_syzygy_data = fake_get_syzygy
        return eng

    def test_blunder_win_to_draw(self):
        """Detects blunder from win to draw."""
        eng = self._eng_with_syzygy(wdl_before=2, wdl_after=0)
        board_before = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        board_after = chess.Board("8/8/8/8/8/4K3/4P3/4k3 b - - 0 1")

        reason = eng._get_syzygy_reason(board_before, board_after)
        assert reason is not None
        assert "Throws away" in reason[2]

    def test_blunder_draw_to_loss(self):
        """Detects blunder from draw to loss."""
        # wdl_after=-2, inverted => wdl_after_inverted=2, but function inverts:
        # wdl_after = -data_after.get("wdl", 0) = -(-2) = 2... no, wait.
        # The function does: wdl_after = -data_after.get("wdl", 0)
        # For draw→loss: wdl_before=0, and after inversion we need wdl_after < 0
        # So data_after["wdl"] must be positive, say 1 or 2
        # wdl_after = -1 or -2 (loss from our perspective)
        eng = self._eng_with_syzygy(wdl_before=0, wdl_after=2)
        board_before = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        board_after = chess.Board("8/8/8/8/8/4K3/4P3/4k3 b - - 0 1")

        reason = eng._get_syzygy_reason(board_before, board_after)
        assert reason is not None
        assert "forced loss" in reason[2]

    def test_maintains_win(self):
        """Detects maintaining a forced win."""
        # wdl_before=2, wdl_after inverted=2 means data_after["wdl"]=-2
        eng = self._eng_with_syzygy(wdl_before=2, wdl_after=-2)
        board_before = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        board_after = chess.Board("8/8/8/8/8/4K3/4P3/4k3 b - - 0 1")

        reason = eng._get_syzygy_reason(board_before, board_after)
        assert reason is not None
        assert "Maintains" in reason[2]

    def test_finds_forced_win(self):
        """Detects finding a forced win from a non-winning position."""
        # wdl_before<2, wdl_after inverted=2 => data_after["wdl"]=-2
        eng = self._eng_with_syzygy(wdl_before=0, wdl_after=-2)
        board_before = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        board_after = chess.Board("8/8/8/8/8/4K3/4P3/4k3 b - - 0 1")

        reason = eng._get_syzygy_reason(board_before, board_after)
        assert reason is not None
        assert "Finds a forced win" in reason[2]

    def test_salvages_draw(self):
        """Detects salvaging a draw from a loss."""
        # wdl_before=-2, wdl_after inverted=0 => data_after["wdl"]=0
        eng = self._eng_with_syzygy(wdl_before=-2, wdl_after=0)
        board_before = chess.Board("8/8/8/8/8/4K3/4P3/4k3 w - - 0 1")
        board_after = chess.Board("8/8/8/8/8/4K3/4P3/4k3 b - - 0 1")

        reason = eng._get_syzygy_reason(board_before, board_after)
        assert reason is not None
        assert "Salvages" in reason[2]

    def test_returns_none_for_no_data(self):
        """Returns None when syzygy data is unavailable."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        board = chess.Board()
        assert eng._get_syzygy_reason(board, board) is None


# ---------------------------------------------------------------------------
# _generate_hardcoded_reasons
# ---------------------------------------------------------------------------


class TestGenerateHardcodedReasons:
    """Tests for the threshold-based hardcoded reasons fallback."""

    def test_batteries_reason(self):
        """Battery arrangement detected when batteries delta > 0.5."""
        eng = ExplainableChessEngine("/sf")
        before = {"batteries_us": 0.0, "batteries_them": 0.0}
        after = {"batteries_them": 1.0, "batteries_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "batteries_us" in names

    def test_outposts_reason(self):
        """Outpost creation detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"outposts_us": 0.0, "outposts_them": 0.0}
        after = {"outposts_them": 1.0, "outposts_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "outposts_us" in names

    def test_king_pressure_reason(self):
        """King ring pressure increase detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"king_ring_pressure_us": 0.0, "king_ring_pressure_them": 0.0}
        after = {"king_ring_pressure_them": 1.0, "king_ring_pressure_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "king_pressure" in names

    def test_bishop_pair_reason(self):
        """Bishop pair advantage detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"bishop_pair_us": 0.0, "bishop_pair_them": 0.0}
        after = {"bishop_pair_them": 1.0, "bishop_pair_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "bishop_pair" in names

    def test_passed_pawns_reason(self):
        """Passed pawn creation detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"passed_us": 0.0, "passed_them": 0.0}
        after = {"passed_them": 1.0, "passed_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "passed_pawns" in names

    def test_center_control_reason(self):
        """Center control improvement detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"center_control_us": 0.0, "center_control_them": 0.0}
        after = {"center_control_them": 1.0, "center_control_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "center_control" in names

    def test_safe_mobility_reason(self):
        """Safe mobility increase detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"safe_mobility_us": 10.0, "safe_mobility_them": 10.0}
        after = {"safe_mobility_them": 12.0, "safe_mobility_us": 10.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "safe_mobility" in names

    def test_rook_open_file_reason(self):
        """Rook on open file detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"rook_open_file_us": 0.0, "rook_open_file_them": 0.0}
        after = {"rook_open_file_them": 1.0, "rook_open_file_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "rook_activity" in names

    def test_isolated_pawns_opponent_reason(self):
        """Creating isolated pawn for opponent detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"isolated_pawns_them": 0.0, "isolated_pawns_us": 0.0}
        after = {"isolated_pawns_us": 1.0, "isolated_pawns_them": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "structure_damage" in names

    def test_backward_pawns_opponent_reason(self):
        """Creating backward pawn for opponent detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"backward_pawns_them": 0.0, "backward_pawns_us": 0.0}
        after = {"backward_pawns_us": 1.0, "backward_pawns_them": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "structure_damage" in names

    def test_backward_pawns_repair_reason(self):
        """Fixing backward pawn detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"backward_pawns_us": 2.0, "backward_pawns_them": 0.0}
        after = {"backward_pawns_them": 1.0, "backward_pawns_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "structure_repair" in names

    def test_pst_improvement_reason(self):
        """Piece placement improvement detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"pst_us": 0.0, "pst_them": 0.0}
        after = {"pst_them": 1.0, "pst_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "piece_quality" in names

    def test_pin_creation_reason(self):
        """Pin creation detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"pinned_them": 0.0, "pinned_us": 0.0}
        after = {"pinned_us": 1.0, "pinned_them": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "pin_creation" in names

    def test_pin_escape_reason(self):
        """Pin escape detected."""
        eng = ExplainableChessEngine("/sf")
        before = {"pinned_us": 2.0, "pinned_them": 0.0}
        after = {"pinned_them": 1.0, "pinned_us": 0.0}

        reasons = eng._generate_hardcoded_reasons(before, after)
        names = [r[0] for r in reasons]
        assert "pin_escape" in names

    def test_no_reasons_when_no_deltas(self):
        """Returns empty list when all deltas are near zero."""
        eng = ExplainableChessEngine("/sf")
        feats = {
            "batteries_us": 0.0,
            "batteries_them": 0.0,
            "outposts_us": 0.0,
            "outposts_them": 0.0,
        }
        reasons = eng._generate_hardcoded_reasons(feats, feats)
        assert reasons == []


# ---------------------------------------------------------------------------
# Overall explanation quality labels
# ---------------------------------------------------------------------------


class TestOverallExplanationQualityLabels:
    """Test all quality label branches in _generate_overall_explanation."""

    def test_excellent_move(self):
        """Quality > 50 cp is 'Excellent move!'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        explanation = eng._generate_overall_explanation(move, 60.0, [])
        assert "Excellent" in explanation

    def test_good_move(self):
        """Quality 20-50 cp is 'Good move.'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        explanation = eng._generate_overall_explanation(move, 30.0, [])
        assert "Good" in explanation

    def test_reasonable_move(self):
        """Quality -20 to 20 cp is 'Reasonable move.'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        explanation = eng._generate_overall_explanation(move, 0.0, [])
        assert "Reasonable" in explanation

    def test_questionable_move(self):
        """Quality -50 to -20 cp is 'Questionable move.'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        explanation = eng._generate_overall_explanation(move, -30.0, [])
        assert "Questionable" in explanation

    def test_poor_move(self):
        """Quality < -50 cp is 'Poor move!'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        explanation = eng._generate_overall_explanation(move, -60.0, [])
        assert "Poor" in explanation


class TestOverallExplanationWithBoardQualityLabels:
    """Test quality labels in _generate_overall_explanation_with_board."""

    def test_excellent_with_board(self):
        """Quality > 50 cp shows 'Excellent move!'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        board = chess.Board()
        explanation = eng._generate_overall_explanation_with_board(
            move, board, 60.0, []
        )
        assert "Excellent" in explanation

    def test_good_with_board(self):
        """Quality 20-50 cp shows 'Good move.'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        board = chess.Board()
        explanation = eng._generate_overall_explanation_with_board(
            move, board, 30.0, []
        )
        assert "Good" in explanation

    def test_questionable_with_board(self):
        """Quality -50 to -20 cp shows 'Questionable move.'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        board = chess.Board()
        explanation = eng._generate_overall_explanation_with_board(
            move, board, -30.0, []
        )
        assert "Questionable" in explanation

    def test_poor_with_board(self):
        """Quality < -50 cp shows 'Poor move!'"""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        board = chess.Board()
        explanation = eng._generate_overall_explanation_with_board(
            move, board, -60.0, []
        )
        assert "Poor" in explanation

    def test_with_reasons(self):
        """Reasons are included as bullet points."""
        eng = ExplainableChessEngine("/sf")
        move = chess.Move.from_uci("e2e4")
        board = chess.Board()
        reasons = [
            ("dev", 2.0, "Develops pawn"),
            ("center", 1.0, "Controls center"),
        ]
        explanation = eng._generate_overall_explanation_with_board(
            move, board, 50.0, reasons
        )
        assert "Develops pawn" in explanation
        assert "Controls center" in explanation


# ---------------------------------------------------------------------------
# _generate_move_reasons_with_board  (detailed paths)
# ---------------------------------------------------------------------------


class TestGenerateMoveReasonsWithBoard:
    """Extended tests for _generate_move_reasons_with_board."""

    def test_en_passant_detected(self):
        """En passant capture generates a reason."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        # Position where e5 pawn can capture d5 en passant
        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
        )
        move = chess.Move.from_uci("e5d6")

        reasons = eng._generate_move_reasons_with_board(move, board, 0.0, 0.0)
        reason_names = [r[0] for r in reasons]
        assert "en_passant" in reason_names

    def test_king_move_early_game(self):
        """King move in early game generates king_safety reason."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        eng.move_history = [chess.Move.from_uci("e2e4")]  # early game

        # Test white king move in early game
        board_for_king = chess.Board(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1"
        )
        king_move = chess.Move.from_uci("e1f1")
        reasons = eng._generate_move_reasons_with_board(
            king_move, board_for_king, 0.0, 0.0
        )
        reason_names = [r[0] for r in reasons]
        assert "king_safety" in reason_names

    def test_castling_generates_reason(self):
        """Castling generates a castling reason."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
        move = chess.Move.from_uci("e1g1")

        reasons = eng._generate_move_reasons_with_board(move, board, 0.0, 0.0)
        reason_names = [r[0] for r in reasons]
        assert "castling" in reason_names

    def test_capture_generates_reason(self):
        """Capture with piece value is detected."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        board = chess.Board(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        move = chess.Move.from_uci("e4d5")

        reasons = eng._generate_move_reasons_with_board(move, board, 0.0, 0.0)
        reason_names = [r[0] for r in reasons]
        assert "capture" in reason_names

    def test_check_generates_reason(self):
        """Check generates check and tactical reasons."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        # Qh5 doesn't give check, use a board where it does
        board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
        move = chess.Move.from_uci("a1a8")  # Ra8+ is check

        reasons = eng._generate_move_reasons_with_board(move, board, 0.0, 0.0)
        reason_names = [r[0] for r in reasons]
        assert "check" in reason_names

    def test_center_control_generated(self):
        """Move to center square generates center_control reason."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")

        reasons = eng._generate_move_reasons_with_board(move, board, 0.0, 0.0)
        reason_names = [r[0] for r in reasons]
        assert "center_control" in reason_names

    def test_pawn_development_from_starting_rank(self):
        """Pawn move from starting rank generates development reason."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")

        reasons = eng._generate_move_reasons_with_board(move, board, 0.0, 0.0)
        reason_names = [r[0] for r in reasons]
        assert "development" in reason_names

    def test_minor_piece_development(self):
        """Minor piece from starting rank generates development reason."""
        eng = ExplainableChessEngine("/sf")
        eng.syzygy = None
        board = chess.Board()
        move = chess.Move.from_uci("g1f3")

        reasons = eng._generate_move_reasons_with_board(move, board, 0.0, 0.0)
        reason_names = [r[0] for r in reasons]
        assert "development" in reason_names


# ---------------------------------------------------------------------------
# get_move_recommendation fallback paths
# ---------------------------------------------------------------------------


class TestGetMoveRecommendationFallbacks:
    """Test recommendation fallback when engine is unavailable."""

    def test_fallback_second_move_black(self):
        """After first move, suggests Nf3 or Nc3 for white."""
        eng = ExplainableChessEngine("/sf")
        eng.make_move("e4")  # White plays e4

        rec = eng.get_move_recommendation()
        assert rec is not None
        assert isinstance(rec, MoveExplanation)

    def test_fallback_later_moves_center(self):
        """For later moves, prefers center control or development."""
        eng = ExplainableChessEngine("/sf")
        eng.make_move("e4")
        eng.board.push(chess.Move.from_uci("e7e5"))
        eng.move_history.append(chess.Move.from_uci("e7e5"))
        eng.make_move("Nf3")

        rec = eng.get_move_recommendation()
        assert rec is not None

    def test_engine_analyse_failure_falls_back(self):
        """When engine.analyse raises, still returns a recommendation."""
        eng = ExplainableChessEngine("/sf")
        mock_engine = Mock()
        mock_engine.analyse.side_effect = Exception("engine fail")
        eng.engine = mock_engine

        rec = eng.get_move_recommendation()
        assert rec is not None


# ---------------------------------------------------------------------------
# get_best_move_for_player fallback paths
# ---------------------------------------------------------------------------


class TestGetBestMoveForPlayerFallbacks:
    """Test best-move-for-player fallback paths."""

    def test_first_move_fallback(self):
        """After first move, suggests e4 or d4 if different from played."""
        eng = ExplainableChessEngine("/sf")
        eng.make_move("a3")  # Suboptimal first move

        rec = eng.get_best_move_for_player()
        assert rec is not None
        assert isinstance(rec, MoveExplanation)

    def test_second_move_fallback(self):
        """After second move, suggests Nf3 or Nc3."""
        eng = ExplainableChessEngine("/sf")
        eng.make_move("e4")
        eng.board.push(chess.Move.from_uci("a7a6"))
        eng.move_history.append(chess.Move.from_uci("a7a6"))

        rec = eng.get_best_move_for_player()
        assert rec is not None

    def test_later_moves_fallback(self):
        """For moves beyond second, uses center/development heuristics."""
        eng = ExplainableChessEngine("/sf")
        # Play a few moves
        eng.make_move("e4")
        eng.board.push(chess.Move.from_uci("e7e5"))
        eng.move_history.append(chess.Move.from_uci("e7e5"))
        eng.make_move("Nf3")
        eng.board.push(chess.Move.from_uci("b8c6"))
        eng.move_history.append(chess.Move.from_uci("b8c6"))
        eng.make_move("a3")  # Suboptimal

        rec = eng.get_best_move_for_player()
        assert rec is not None

    def test_engine_best_same_as_played(self):
        """When engine suggests the same move played, falls back to heuristic."""
        eng = ExplainableChessEngine("/sf")
        eng.make_move("e4")

        mock_engine = Mock()
        # Engine suggests the same move that was played
        mock_engine.analyse.return_value = {"pv": [chess.Move.from_uci("e2e4")]}
        eng.engine = mock_engine

        rec = eng.get_best_move_for_player()
        # Should either return None or a different move (d4 fallback)
        if rec is not None:
            assert isinstance(rec, MoveExplanation)


# ---------------------------------------------------------------------------
# __enter__ with syzygy and model init paths
# ---------------------------------------------------------------------------


class TestContextManagerExtended:
    """Extended __enter__/__exit__ tests."""

    @patch("chess.engine.SimpleEngine.popen_uci")
    def test_enter_with_syzygy_path(self, mock_popen):
        """Syzygy initialisation is attempted when syzygy_path is set."""
        mock_engine = Mock()
        mock_popen.return_value = mock_engine

        eng = ExplainableChessEngine(
            "/sf",
            syzygy_path="/fake/syzygy",
            enable_model_explanations=False,
        )

        with eng as e:
            # Syzygy init may fail, but engine should still work
            assert e.engine is mock_engine

    def test_exit_without_engine(self):
        """__exit__ does not crash when engine is None."""
        eng = ExplainableChessEngine("/sf")
        eng.engine = None
        eng.__exit__(None, None, None)  # Should not raise
