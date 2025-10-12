/**
 * Main application logic and API integration.
 */

class ChessApp {
  constructor() {
    this.board = new ChessBoard('chess-board');
    this.board.onMove = (move) => this.handlePlayerMove(move);
    this.moveCount = 0;
    this.initializeEventListeners();
    this.initializeGame();
  }

  initializeEventListeners() {
    document.getElementById('btn-new-game').addEventListener('click', () => {
      this.initializeGame();
    });

    document.getElementById('btn-engine-move').addEventListener('click', () => {
      this.requestEngineMove();
    });

    document.getElementById('btn-analyze').addEventListener('click', () => {
      this.analyzePosition();
    });

    document.querySelectorAll('.nav-button').forEach(button => {
      button.addEventListener('click', (e) => {
        this.switchView(e.target.dataset.view);
      });
    });
  }

  switchView(viewName) {
    document.querySelectorAll('.view').forEach(view => {
      view.classList.remove('active');
    });
    document.querySelectorAll('.nav-button').forEach(button => {
      button.classList.remove('active');
    });

    document.getElementById(`view-${viewName}`).classList.add('active');
    document.querySelector(`[data-view="${viewName}"]`).classList.add('active');
  }

  async initializeGame() {
    try {
      const response = await fetch('/api/game/new', {
        method: 'POST'
      });
      
      const data = await response.json();
      this.board.setPosition(data.fen);
      this.board.setLegalMoves(data.legal_moves);
      this.moveCount = 0;
      this.updateGameState();
      this.clearExplanation();
      this.updateStatus('Ready');
    } catch (error) {
      console.error('Failed to initialize game:', error);
      this.updateStatus('Error');
    }
  }

  async handlePlayerMove(move) {
    try {
      const response = await fetch('/api/game/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move })
      });
      
      const data = await response.json();
      
      if (data.success) {
        this.board.setPosition(data.fen);
        this.board.setLegalMoves(data.legal_moves);
        this.moveCount++;
        this.updateGameState();
        
        if (data.is_game_over) {
          this.updateStatus('Game Over');
        }
      }
    } catch (error) {
      console.error('Failed to make move:', error);
    }
  }

  async requestEngineMove() {
    this.updateStatus('Thinking...');
    
    try {
      const response = await fetch('/api/engine/move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ depth: 15 })
      });
      
      const data = await response.json();
      
      if (data.move) {
        await this.handlePlayerMove(data.move);
        this.displayExplanation(data.explanation, data.features);
        this.updateStatus('Ready');
      }
    } catch (error) {
      console.error('Failed to get engine move:', error);
      this.updateStatus('Error');
    }
  }

  async analyzePosition() {
    try {
      const response = await fetch('/api/analysis/features', {
        method: 'POST'
      });
      
      const data = await response.json();
      this.displayFeatures(data.features);
    } catch (error) {
      console.error('Failed to analyze position:', error);
    }
  }

  async updateGameState() {
    try {
      const response = await fetch('/api/game/state');
      const data = await response.json();
      
      document.getElementById('turn-indicator').textContent = 
        data.turn.charAt(0).toUpperCase() + data.turn.slice(1);
      document.getElementById('move-count').textContent = this.moveCount;
      
      // Update board label
      const boardLabel = document.getElementById('board-label');
      if (data.turn === 'white') {
        boardLabel.textContent = 'White to move';
      } else {
        boardLabel.textContent = 'Black to move';
      }
    } catch (error) {
      console.error('Failed to update game state:', error);
    }
  }

  displayExplanation(explanation, features) {
    const container = document.getElementById('explanation-content');
    const topFeatures = Object.entries(features)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 3);
    
    let html = `<p>${explanation}</p>`;
    
    if (topFeatures.length > 0) {
      html += '<p class="text-muted">Key factors:</p>';
      topFeatures.forEach(([name, value]) => {
        const formattedName = this.formatFeatureName(name);
        const formattedValue = value.toFixed(2);
        html += `<p class="feature-name">${formattedName}: <span class="mono">${formattedValue}</span></p>`;
      });
    }
    
    container.innerHTML = html;
  }

  clearExplanation() {
    const container = document.getElementById('explanation-content');
    container.innerHTML = '<p class="text-muted">Make a move or request an engine move to see analysis.</p>';
  }

  formatFeatureName(name) {
    // Replace _us and _them with player indicators
    let formatted = name
      .replace(/_us$/, ' (White)')
      .replace(/_them$/, ' (Black)');
    
    // Convert snake_case to Title Case
    formatted = formatted
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
    
    return formatted;
  }

  displayFeatures(features) {
    const container = document.getElementById('features-table');
    const sortedFeatures = Object.entries(features)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    
    let html = '';
    sortedFeatures.forEach(([name, value]) => {
      const formattedName = this.formatFeatureName(name);
      const formattedValue = value.toFixed(3);
      html += `
        <div class="feature-row">
          <span class="feature-name">${formattedName}</span>
          <span class="feature-value">${formattedValue}</span>
        </div>
      `;
    });
    
    container.innerHTML = html;
  }

  updateStatus(status) {
    document.getElementById('game-status').textContent = status;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new ChessApp();
});

