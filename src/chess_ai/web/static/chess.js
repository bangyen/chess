/**
 * Chess board rendering and interaction.
 */

const PIECES = {
  'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',
  'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔'
};

class ChessBoard {
  constructor(elementId) {
    this.element = document.getElementById(elementId);
    this.selectedSquare = null;
    this.legalMoves = [];
    this.position = this.parseInitialPosition();
    this.onMove = null;
    this.render();
  }

  parseInitialPosition() {
    const fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR';
    const board = [];
    let rank = 0;
    let file = 0;
    
    for (const char of fen) {
      if (char === '/') {
        rank++;
        file = 0;
      } else if (char >= '1' && char <= '8') {
        file += parseInt(char);
      } else {
        if (!board[rank]) board[rank] = [];
        board[rank][file] = char;
        file++;
      }
    }
    
    return board;
  }

  setPosition(fen) {
    const parts = fen.split(' ');
    const boardFen = parts[0];
    const board = [];
    let rank = 0;
    let file = 0;
    
    for (const char of boardFen) {
      if (char === '/') {
        rank++;
        file = 0;
      } else if (char >= '1' && char <= '8') {
        file += parseInt(char);
      } else {
        if (!board[rank]) board[rank] = [];
        board[rank][file] = char;
        file++;
      }
    }
    
    this.position = board;
    this.render();
  }

  setLegalMoves(moves) {
    this.legalMoves = moves;
  }

  render() {
    this.element.innerHTML = '';
    
    for (let rank = 0; rank < 8; rank++) {
      for (let file = 0; file < 8; file++) {
        const square = document.createElement('div');
        square.className = 'square';
        square.className += (rank + file) % 2 === 0 ? ' light' : ' dark';
        square.dataset.rank = rank;
        square.dataset.file = file;
        
        if (this.position[rank] && this.position[rank][file]) {
          const piece = this.position[rank][file];
          square.textContent = PIECES[piece];
        }
        
        square.addEventListener('click', (e) => this.handleSquareClick(e));
        this.element.appendChild(square);
      }
    }
  }

  handleSquareClick(event) {
    const square = event.currentTarget;
    const rank = parseInt(square.dataset.rank);
    const file = parseInt(square.dataset.file);
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
      
      this.clearSelection();
    } else {
      if (this.position[rank] && this.position[rank][file]) {
        this.selectedSquare = squareId;
        this.highlightMoves(squareId);
        square.classList.add('selected');
      }
    }
  }

  toSquareId(rank, file) {
    const files = 'abcdefgh';
    return files[file] + (8 - rank);
  }

  highlightMoves(fromSquare) {
    const relevantMoves = this.legalMoves.filter(m => m.startsWith(fromSquare));
    
    relevantMoves.forEach(move => {
      const toSquare = move.substring(2, 4);
      const [file, rank] = this.fromSquareId(toSquare);
      const squareElement = this.element.children[rank * 8 + file];
      if (squareElement) {
        squareElement.classList.add('legal-move');
      }
    });
  }

  fromSquareId(squareId) {
    const files = 'abcdefgh';
    const file = files.indexOf(squareId[0]);
    const rank = 8 - parseInt(squareId[1]);
    return [file, rank];
  }

  clearSelection() {
    this.selectedSquare = null;
    this.element.querySelectorAll('.selected').forEach(el => {
      el.classList.remove('selected');
    });
    this.element.querySelectorAll('.legal-move').forEach(el => {
      el.classList.remove('legal-move');
    });
  }
}

