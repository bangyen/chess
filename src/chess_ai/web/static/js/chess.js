/**
 * Chess board rendering and interaction using Canvas.
 */

const PIECES = {
  'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
  'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
};

const COLORS = {
  light: '#EEEED2',
  dark: '#769656',
  selected: 'rgba(151, 132, 93, 0.8)',
  highlight: 'rgba(44, 95, 45, 0.4)',
  lastMove: 'rgba(255, 255, 0, 0.3)'
};

class ChessBoard {
  constructor(elementId) {
    this.canvas = document.getElementById(elementId);
    this.ctx = this.canvas.getContext('2d');
    this.wrapper = this.canvas.parentElement;

    this.selectedSquare = null;
    this.legalMoves = [];
    this.lastMove = null; // { from, to }

    // Pieces state for animation
    this.pieces = []; // Array of { type, rank, file, currentX, currentY, animating: bool }

    this.onMove = null;

    this.init();
    this.setupEventListeners();
    this.startAnimationLoop();
  }

  init() {
    this.handleResize();
    const initialFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
    this.setPosition(initialFen);
  }

  handleResize() {
    const size = this.wrapper.clientWidth;
    this.canvas.width = size * window.devicePixelRatio;
    this.canvas.height = size * window.devicePixelRatio;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this.squareSize = size / 8;
  }

  setupEventListeners() {
    window.addEventListener('resize', () => this.handleResize());
    this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
  }

  parseFen(fen) {
    const boardFen = fen.split(' ')[0];
    const pieces = [];
    let rank = 0;
    let file = 0;

    for (const char of boardFen) {
      if (char === '/') {
        rank++;
        file = 0;
      } else if (char >= '1' && char <= '8') {
        file += parseInt(char);
      } else {
        pieces.push({
          type: char,
          rank: rank,
          file: file,
          currentX: file,
          currentY: rank,
          animating: false
        });
        file++;
      }
    }
    return pieces;
  }

  setPosition(fen, moveUci = null) {
    const newPieces = this.parseFen(fen);

    if (!moveUci) {
      // Initial setup or reset
      this.pieces = newPieces;
      this.lastMove = null;
    } else {
      // Find what moved
      const from = this.fromSquareId(moveUci.substring(0, 2));
      const to = this.fromSquareId(moveUci.substring(2, 4));

      // Match existing pieces to new pieces to identify animations
      // For a simple implementation, we'll just animate the piece that was at 'from'
      const movingPiece = this.pieces.find(p => p.rank === from[1] && p.file === from[0]);

      if (movingPiece) {
        movingPiece.animating = true;
        movingPiece.targetRank = to[1];
        movingPiece.targetFile = to[0];
        movingPiece.startTime = performance.now();
        movingPiece.duration = 200; // ms
      }

      this.lastMove = { from, to };

      // After animation, we'll sync with newPieces, but for now we keep the moving piece
      setTimeout(() => {
        this.pieces = newPieces;
      }, 210);
    }
  }

  setLegalMoves(moves) {
    this.legalMoves = moves;
  }

  startAnimationLoop() {
    const render = () => {
      this.draw();
      requestAnimationFrame(render);
    };
    requestAnimationFrame(render);
  }

  draw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // 1. Draw Board
    for (let rank = 0; rank < 8; rank++) {
      for (let file = 0; file < 8; file++) {
        this.ctx.fillStyle = (rank + file) % 2 === 0 ? COLORS.light : COLORS.dark;
        this.ctx.fillRect(file * this.squareSize, rank * this.squareSize, this.squareSize, this.squareSize);
      }
    }

    // 2. Draw Last Move Highlight
    if (this.lastMove) {
      this.ctx.fillStyle = COLORS.lastMove;
      this.ctx.fillRect(this.lastMove.from[0] * this.squareSize, this.lastMove.from[1] * this.squareSize, this.squareSize, this.squareSize);
      this.ctx.fillRect(this.lastMove.to[0] * this.squareSize, this.lastMove.to[1] * this.squareSize, this.squareSize, this.squareSize);
    }

    // 3. Draw Selected Square
    if (this.selectedSquare) {
      const [file, rank] = this.fromSquareId(this.selectedSquare);
      this.ctx.fillStyle = COLORS.selected;
      this.ctx.fillRect(file * this.squareSize, rank * this.squareSize, this.squareSize, this.squareSize);

      // 4. Draw Legal Moves
      const relevantMoves = this.legalMoves.filter(m => m.startsWith(this.selectedSquare));
      relevantMoves.forEach(move => {
        const toSquare = move.substring(2, 4);
        const [tFile, tRank] = this.fromSquareId(toSquare);

        this.ctx.fillStyle = COLORS.highlight;
        this.ctx.beginPath();
        this.ctx.arc((tFile + 0.5) * this.squareSize, (tRank + 0.5) * this.squareSize, this.squareSize * 0.15, 0, Math.PI * 2);
        this.ctx.fill();
      });
    }

    // 5. Draw Pieces
    this.pieces.forEach(piece => {
      let x = piece.file;
      let y = piece.rank;

      if (piece.animating) {
        const elapsed = performance.now() - piece.startTime;
        const progress = Math.min(elapsed / piece.duration, 1);

        // Easing (easeInOutQuad)
        const t = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress;

        x = piece.file + (piece.targetFile - piece.file) * t;
        y = piece.rank + (piece.targetRank - piece.rank) * t;

        if (progress === 1) piece.animating = false;
      }

      this.drawPiece(piece.type, x, y);
    });
  }

  drawPiece(type, file, rank) {
    const symbol = PIECES[type];
    this.ctx.font = `${this.squareSize * 0.8}px 'Arial'`;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';

    // Draw shadow for piece
    this.ctx.fillStyle = 'rgba(0,0,0,0.2)';
    this.ctx.fillText(symbol, (file + 0.5) * this.squareSize, (rank + 0.5) * this.squareSize + 2);

    this.ctx.fillStyle = '#000';
    this.ctx.fillText(symbol, (file + 0.5) * this.squareSize, (rank + 0.5) * this.squareSize);
  }

  handleCanvasClick(event) {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const file = Math.floor(x / (rect.width / 8));
    const rank = Math.floor(y / (rect.height / 8));
    const squareId = this.toSquareId(rank, file);

    if (this.selectedSquare) {
      const fromSquare = this.selectedSquare;
      const toSquare = squareId;
      const moveUci = fromSquare + toSquare;

      if (this.legalMoves.includes(moveUci)) {
        if (this.onMove) {
          this.onMove(moveUci);
        }
      }

      this.selectedSquare = null;
    } else {
      const pieceAt = this.pieces.find(p => p.rank === rank && p.file === file);
      if (pieceAt) {
        this.selectedSquare = squareId;
      }
    }
  }

  toSquareId(rank, file) {
    const files = 'abcdefgh';
    return files[file] + (8 - rank);
  }

  fromSquareId(squareId) {
    const files = 'abcdefgh';
    const file = files.indexOf(squareId[0]);
    const rank = 8 - parseInt(squareId[1]);
    return [file, rank];
  }
}

