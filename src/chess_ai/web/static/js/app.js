/**
 * Chess AI Dashboard - Main application logic and API integration.
 */

class ChessApp {
    constructor() {
        this.board = new ChessBoard('chess-board');
        this.board.onMove = (move) => this.handlePlayerMove(move);
        this.moveCount = 0;
        
        this.initializeNavigation();
        this.initializeEventListeners();
        this.initializeGame();
    }

    initializeNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const views = document.querySelectorAll('.view-container');
        
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const viewName = item.getAttribute('data-view');
                
                // Update active states
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');
                
                views.forEach(v => v.classList.add('hidden'));
                document.getElementById(`${viewName}-view`).classList.remove('hidden');
                
                // Update page title
                const titles = {
                    'play': 'Play Chess',
                    'features': 'Position Features'
                };
                document.querySelector('.page-title').textContent = titles[viewName] || 'Chess AI';
            });
        });
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
            this.updateEngineStatus('Ready', true);
        } catch (error) {
            console.error('Failed to initialize game:', error);
            this.updateEngineStatus('Error', false);
        }
    }

    async handlePlayerMove(move, isEngineMove = false, explanation = null) {
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
                
                // Get position features and display
                const features = await this.getPositionFeatures();
                if (features) {
                    // Display features for manual moves too
                    if (!isEngineMove) {
                        this.displayManualMoveInfo(features);
                    } else if (explanation) {
                        this.displayExplanation(explanation, features);
                    }
                }
                
                this.updateGameState();
                
                if (data.is_game_over) {
                    this.handleGameOver();
                }
            }
        } catch (error) {
            console.error('Failed to make move:', error);
        }
    }

    async requestEngineMove() {
        this.updateEngineStatus('Thinking...', true);
        
        try {
            const response = await fetch('/api/engine/move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ depth: 15 })
            });
            
            const data = await response.json();
            
            if (data.move) {
                await this.handlePlayerMove(data.move, true, data.explanation);
                this.updateEngineStatus('Ready', true);
            }
        } catch (error) {
            console.error('Failed to get engine move:', error);
            this.updateEngineStatus('Error', false);
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

    async getPositionFeatures() {
        try {
            const response = await fetch('/api/analysis/features', {
                method: 'POST'
            });
            const data = await response.json();
            return data.features;
        } catch (error) {
            console.error('Failed to get position features:', error);
            return null;
        }
    }

    async updateGameState() {
        try {
            const response = await fetch('/api/game/state');
            const data = await response.json();
            
            // Update board label
            const boardLabel = document.getElementById('board-label');
            if (boardLabel) {
                boardLabel.textContent = data.turn === 'white' ? 'White to move' : 'Black to move';
            }
            
            // Update engine status based on game state
            if (data.is_game_over) {
                this.handleGameOver();
            }
        } catch (error) {
            console.error('Failed to update game state:', error);
        }
    }

    displayManualMoveInfo(features) {
        const container = document.getElementById('explanation-content');
        const keyFeaturesContainer = document.getElementById('key-features');
        
        // Clear explanation for manual moves
        container.innerHTML = '<p class="text-muted">Request an engine move to see analysis.</p>';
        
        // Display top 3 features
        this.displayTopFeatures(features, keyFeaturesContainer);
    }

    displayExplanation(explanation, features) {
        const container = document.getElementById('explanation-content');
        const keyFeaturesContainer = document.getElementById('key-features');
        
        // Display explanation
        let html = `<p>${explanation}</p>`;
        container.innerHTML = html;
        
        // Display top 3 features
        this.displayTopFeatures(features, keyFeaturesContainer);
    }

    displayTopFeatures(features, container) {
        if (!container) return;
        
        const topFeatures = Object.entries(features)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 3);
        
        if (topFeatures.length > 0) {
            let featuresHtml = '';
            topFeatures.forEach(([name, value]) => {
                const formattedName = this.formatFeatureName(name);
                const formattedValue = value.toFixed(2);
                featuresHtml += `
                    <div class="feature-item">
                        <span class="feature-name">${formattedName}</span>
                        <span class="feature-value">${formattedValue}</span>
                    </div>
                `;
            });
            container.innerHTML = featuresHtml;
        }
    }

    clearExplanation() {
        const container = document.getElementById('explanation-content');
        const keyFeaturesContainer = document.getElementById('key-features');
        
        container.innerHTML = '<p class="text-muted">Request an engine move to see analysis.</p>';
        if (keyFeaturesContainer) {
            keyFeaturesContainer.innerHTML = '<p class="text-muted">Key features will appear here after each move.</p>';
        }
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

    updateEngineStatus(status, isActive) {
        const statusText = document.getElementById('engine-status-text');
        const statusDot = document.getElementById('engine-status-dot');
        
        if (statusText) {
            statusText.textContent = status;
        }
        
        if (statusDot) {
            if (isActive) {
                statusDot.classList.remove('error');
            } else {
                statusDot.classList.add('error');
            }
        }
    }

    handleGameOver() {
        this.updateEngineStatus('Game Over', false);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ChessApp();
});
