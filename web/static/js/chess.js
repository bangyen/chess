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

    // Dragging state
    this.draggingPiece = null;
    this.dragX = 0; // Relative to canvas, in squares
    this.dragY = 0;
    this.isMouseDown = false;
    this.dragStartSquare = null;
    this.lastDragMove = null;

    this.onMove = null;
    this.orientation = 'white'; // 'white' or 'black'

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

    this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    window.addEventListener('mouseup', (e) => this.handleMouseUp(e));

    // Touch support
    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.handleMouseDown(e.touches[0]);
    }, { passive: false });
    window.addEventListener('touchmove', (e) => {
      this.handleMouseMove(e.touches[0]);
    }, { passive: false });
    window.addEventListener('touchend', (e) => {
      this.handleMouseUp(e.changedTouches[0]);
    });
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
      this.lastDragMove = null;
    } else {
      if (moveUci === this.lastDragMove) {
        this.pieces = newPieces;
        this.lastMove = this.parseLastMove(moveUci);
        this.lastDragMove = null;
        return;
      }

      // Find what moved
      const from = this.fromSquareId(moveUci.substring(0, 2));
      const to = this.fromSquareId(moveUci.substring(2, 4));

      // Match existing pieces to target the moving piece
      // It might already be at 'to' if updated by drag/click handler
      const movingPiece = this.pieces.find(p => 
        (p.rank === from[1] && p.file === from[0]) || 
        (p.rank === to[1] && p.file === to[0])
      );

      // Remove any OTHER piece at the target square immediately (the captured piece)
      this.pieces = this.pieces.filter(p => p === movingPiece || p.rank !== to[1] || p.file !== to[0]);

      if (movingPiece) {
        movingPiece.animating = true;
        // Ensure starting position is 'from' for the animation
        movingPiece.file = from[0];
        movingPiece.rank = from[1];
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

  setOrientation(orientation) {
    this.orientation = orientation;
  }

  // Maps logical coordinates (0-7) to visual coordinates based on orientation
  toVisual(file, rank) {
    if (this.orientation === 'white') {
      return [file, rank];
    } else {
      return [7 - file, 7 - rank];
    }
  }

  // Maps visual coordinates (click/pixel) to logical coordinates based on orientation
  toLogical(file, rank) {
    if (this.orientation === 'white') {
      return [file, rank];
    } else {
      return [7 - file, 7 - rank];
    }
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
        const [vFile, vRank] = this.toVisual(file, rank);
        this.ctx.fillStyle = (rank + file) % 2 === 0 ? COLORS.light : COLORS.dark;
        this.ctx.fillRect(vFile * this.squareSize, vRank * this.squareSize, this.squareSize, this.squareSize);
      }
    }

    // 2. Draw Last Move Highlight
    if (this.lastMove) {
      const [fFile, fRank] = this.toVisual(this.lastMove.from[0], this.lastMove.from[1]);
      const [tFile, tRank] = this.toVisual(this.lastMove.to[0], this.lastMove.to[1]);
      this.ctx.fillStyle = COLORS.lastMove;
      this.ctx.fillRect(fFile * this.squareSize, fRank * this.squareSize, this.squareSize, this.squareSize);
      this.ctx.fillRect(tFile * this.squareSize, tRank * this.squareSize, this.squareSize, this.squareSize);
    }

    // 3. Draw Selected Square
    if (this.selectedSquare) {
      const [file, rank] = this.fromSquareId(this.selectedSquare);
      const [vFile, vRank] = this.toVisual(file, rank);
      this.ctx.fillStyle = COLORS.selected;
      this.ctx.fillRect(vFile * this.squareSize, vRank * this.squareSize, this.squareSize, this.squareSize);

      // 4. Draw Legal Moves
      const relevantMoves = this.legalMoves.filter(m => m.startsWith(this.selectedSquare));
      relevantMoves.forEach(move => {
        const toSquare = move.substring(2, 4);
        const [tFile, tRank] = this.fromSquareId(toSquare);
        const [vFile, vRank] = this.toVisual(tFile, tRank);

        this.ctx.fillStyle = COLORS.highlight;
        this.ctx.beginPath();
        this.ctx.arc((vFile + 0.5) * this.squareSize, (vRank + 0.5) * this.squareSize, this.squareSize * 0.15, 0, Math.PI * 2);
        this.ctx.fill();
      });
    }

    // 5. Draw Pieces (excluding dragging)
    this.pieces.forEach(piece => {
      if (this.draggingPiece && piece === this.draggingPiece) return;

      let x = piece.file;
      let y = piece.rank;

      if (piece.animating) {
        const elapsed = performance.now() - piece.startTime;
        const progress = Math.min(elapsed / piece.duration, 1);

        // Easing (easeInOutQuad)
        const t = progress < 0.5 ? 2 * progress * progress : -1 + (4 - 2 * progress) * progress;

        const [vtFile, vtRank] = this.toVisual(piece.targetFile, piece.targetRank);
        const [vFile, vRank] = this.toVisual(piece.file, piece.rank);

        x = vFile + (vtFile - vFile) * t;
        y = vRank + (vtRank - vRank) * t;

        if (progress === 1) piece.animating = false;
      } else {
        const [vFile, vRank] = this.toVisual(piece.file, piece.rank);
        x = vFile;
        y = vRank;
      }

      this.drawPiece(piece.type, x, y);
    });

    // 6. Draw Dragging Piece
    if (this.draggingPiece) {
      this.drawPiece(this.draggingPiece.type, this.dragX - 0.5, this.dragY - 0.5, true);
    }
  }

  drawPiece(type, file, rank, isDragging = false) {
    const symbol = PIECES[type];
    const scale = isDragging ? 1.1 : 1.0;
    this.ctx.font = `${this.squareSize * 0.8 * scale}px 'Arial'`;
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';

    // Draw shadow for piece
    this.ctx.fillStyle = 'rgba(0,0,0,0.2)';
    this.ctx.fillText(symbol, (file + 0.5) * this.squareSize, (rank + 0.5) * this.squareSize + (isDragging ? 5 : 2));

    this.ctx.fillStyle = '#000';
    this.ctx.fillText(symbol, (file + 0.5) * this.squareSize, (rank + 0.5) * this.squareSize);
  }

  handleMouseDown(event) {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const vFile = Math.floor(x / (rect.width / 8));
    const vRank = Math.floor(y / (rect.height / 8));
    const [file, rank] = this.toLogical(vFile, vRank);
    const squareId = this.toSquareId(rank, file);

    this.isMouseDown = true;
    this.dragStartSquare = squareId;

    // Check if this is a click-to-move completion (capture or move to occupied square)
    if (this.selectedSquare && this.selectedSquare !== squareId) {
      const moveUci = this.selectedSquare + squareId;
      if (this.legalMoves.includes(moveUci)) {
        // This is a legal move/capture. Do NOT start a new drag.
        // handleMouseUp will pick this up and execute the move.
        return;
      }
    }

    // Toggle selection if clicking the same square
    if (this.selectedSquare === squareId) {
      this.selectedSquare = null;
      return;
    }

    const pieceAt = this.pieces.find(p => p.rank === rank && p.file === file);
    if (pieceAt) {
      this.draggingPiece = pieceAt;
      this.dragX = x / (rect.width / 8);
      this.dragY = y / (rect.height / 8);

      // Also select for click-to-move compatibility
      this.selectedSquare = squareId;
    } else {
      // Clicked on empty square - clear selection unless it's a legal move (handled above)
      this.selectedSquare = null;
    }
  }

  handleMouseMove(event) {
    if (!this.isMouseDown || !this.draggingPiece) return;

    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    this.dragX = x / (rect.width / 8);
    this.dragY = y / (rect.height / 8);
  }

  handleMouseUp(event) {
    if (!this.isMouseDown) return;

    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    const vFile = Math.floor(x / (rect.width / 8));
    const vRank = Math.floor(y / (rect.height / 8));
    const [file, rank] = this.toLogical(vFile, vRank);
    const toSquareId = this.toSquareId(rank, file);

    if (this.draggingPiece) {
      const fromSquare = this.dragStartSquare;
      const moveUci = fromSquare + toSquareId;

      if (fromSquare !== toSquareId && this.legalMoves.includes(moveUci)) {
        if (this.onMove) {
          // Instantly update the piece's rank/file in the pieces array to prevent jump
          this.draggingPiece.rank = rank;
          this.draggingPiece.file = file;
          this.draggingPiece.currentX = file;
          this.draggingPiece.currentY = rank;

          this.lastMove = this.parseLastMove(moveUci);
          this.lastDragMove = moveUci;
          this.onMove(moveUci);
        }
        this.selectedSquare = null;
      } else if (fromSquare === toSquareId) {
        // Keep selected for click-to-move
      } else {
        this.selectedSquare = null;
      }
    } else if (this.selectedSquare) {
      // Click-to-move targeting an empty square (draggingPiece would be null)
      const fromSquare = this.selectedSquare;
      const moveUci = fromSquare + toSquareId;

      if (this.legalMoves.includes(moveUci)) {
        if (this.onMove) {
          // Find and update the moving piece locally
          const [ff, fr] = this.fromSquareId(fromSquare);
          const movingPiece = this.pieces.find(p => p.rank === fr && p.file === ff);
          if (movingPiece) {
            movingPiece.rank = rank;
            movingPiece.file = file;
            movingPiece.currentX = file;
            movingPiece.currentY = rank;
          }

          this.lastMove = this.parseLastMove(moveUci);
          this.onMove(moveUci);
        }
      }
      this.selectedSquare = null;
    }

    this.isMouseDown = false;
    this.draggingPiece = null;
    this.dragStartSquare = null;
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

  parseLastMove(moveUci) {
    const from = this.fromSquareId(moveUci.substring(0, 2));
    const to = this.fromSquareId(moveUci.substring(2, 4));
    return { from, to };
  }
}

