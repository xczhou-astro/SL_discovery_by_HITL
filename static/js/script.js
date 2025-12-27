class SLDetector {
    constructor() {
        this.hiddenImages = new Set(); // Images marked as non-SL (hidden)
        this.currentBatch = [];
        this.currentScores = {};
        this.isTraining = false;
        this.currentRound = 1;
        this.modelTrained = false;
        this.submitCooldown = false;
        this.cooldownTimer = null;

        this.initializeElements();
        this.bindEvents();
        this.loadInitialData();
    }
    
    initializeElements() {
        // Selection controls
        this.submitSelectionsBtn = document.getElementById('submit-selections-btn');
        this.clearSelectionsBtn = document.getElementById('clear-selections-btn');
        
        // Training overlay
        this.trainingOverlay = document.getElementById('training-overlay');
        this.trainingMessage = document.getElementById('training-message');

        // Display elements
        this.imageGrid = document.getElementById('image-grid');
        
        // Status elements
        this.currentRoundEl = document.getElementById('current-round');
        this.slCountEl = document.getElementById('sl-count');
        this.nonSlCountEl = document.getElementById('non-sl-count');
        this.availableCountEl = document.getElementById('available-count');
        this.totalSubmissionsEl = document.getElementById('total-submissions');
        
        // Visualization elements
        this.visualizationSection = document.getElementById('visualization-section');
        this.visualizationImage = document.getElementById('visualization-image');
        
        // Notification
        this.notification = document.getElementById('notification');
    }
    
    bindEvents() {
        this.submitSelectionsBtn.addEventListener('click', () => this.submitSelections());
        this.clearSelectionsBtn.addEventListener('click', () => this.clearSelections());
    }
    
    async loadInitialData() {
        await this.updateStatus();
        await this.loadImages();
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/get_status');
            const data = await response.json();
            
            if (data.success) {
                this.updateCounts(data);
                this.modelTrained = data.model_trained;
                
                // Show visualization if model is trained
                if (this.modelTrained) {
                    this.showVisualization();
                }
            }
        } catch (error) {
            console.error('Error updating status:', error);
        }
    }

    async loadImages() {
        try {
            console.log('Loading images...');
            this.showNotification('Loading images...', 'info');
            const response = await fetch('/api/get_images');
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Received data:', data);
            
            if (data.error) {
                this.showNotification(data.error, 'error');
                return;
            }
            
            if (!data.success) {
                this.showNotification('Failed to load images', 'error');
                return;
            }
            
            if (!data.galaxy_names || data.galaxy_names.length === 0) {
                this.showNotification('No images available', 'warning');
                console.log('No images returned from server');
                return;
            }
            
            this.currentBatch = data.galaxy_names;
            this.currentScores = {};
            data.galaxy_names.forEach((name, index) => {
                this.currentScores[name] = data.scores[index];
            });
            
            console.log('Rendering', this.currentBatch.length, 'images');
            this.renderImageGrid();
            this.updateCounts(data);
            
            // Start submit cooldown
            this.startSubmitCooldown();
            
            this.showNotification(`Loaded ${this.currentBatch.length} images`, 'success');
        } catch (error) {
            this.showNotification('Error loading images', 'error');
            console.error('Error loading images:', error);
        }
    }

    renderImageGrid() {
        this.imageGrid.innerHTML = '';
        
        this.currentBatch.forEach((galaxyName, index) => {
            const imageItem = document.createElement('div');
            imageItem.className = 'image-item';
            imageItem.dataset.galaxyName = galaxyName;
            
            // Image content container
            const imageContent = document.createElement('div');
            imageContent.className = 'image-content';
            
            const img = document.createElement('img');
            img.src = `/images/${galaxyName}.jpg`;
            img.alt = `Galaxy ${galaxyName}`;
            img.loading = 'lazy';
            
            const overlay = document.createElement('div');
            overlay.className = 'image-overlay';
            overlay.textContent = galaxyName;
            
            const scoreOverlay = document.createElement('div');
            scoreOverlay.className = 'score-overlay';
            const score = this.currentScores[galaxyName] || 0.5;
            scoreOverlay.textContent = `Score: ${score.toFixed(3)}`;
            scoreOverlay.title = `Model confidence: ${score.toFixed(3)}`;
            
            imageContent.appendChild(img);
            imageContent.appendChild(overlay);
            imageContent.appendChild(scoreOverlay);
            
            // Add click event listener - left click to toggle hide/show
            imageItem.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleHidden(galaxyName, imageItem);
            });
            
            // Prevent right-click context menu
            imageItem.addEventListener('contextmenu', (e) => {
                e.preventDefault();
            });
            
            imageItem.appendChild(imageContent);
            this.imageGrid.appendChild(imageItem);
        });
    }

    toggleHidden(galaxyName, imageItem) {
        if (this.hiddenImages.has(galaxyName)) {
            // Unhide the image
            this.hiddenImages.delete(galaxyName);
            imageItem.classList.remove('hidden');
        } else {
            // Hide the image (mark as non-SL)
            this.hiddenImages.add(galaxyName);
            imageItem.classList.add('hidden');
        }
    }
    
    clearSelections() {
        this.hiddenImages.clear();
        
        // Clear visual indicators
        document.querySelectorAll('.image-item').forEach(item => {
            item.classList.remove('hidden');
        });
        
        this.showNotification('Selections cleared', 'info');
    }
    
    async submitSelections() {
        // Check if in cooldown period
        if (this.submitCooldown) {
            this.showNotification('Please wait before submitting again', 'warning');
            return;
        }
        
        // Get SL names (all visible images) and Non-SL names (hidden images)
        const slNames = this.currentBatch.filter(name => !this.hiddenImages.has(name));
        const nonSlNames = Array.from(this.hiddenImages);
        
        if (slNames.length === 0 && nonSlNames.length === 0) {
            this.showNotification('No images to submit', 'error');
            return;
        }
        
        try {
            
            const response = await fetch('/app/submit_selections', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sl_names: slNames,
                    non_sl_names: nonSlNames
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Selections submitted successfully', 'success');
                
                // Update counts immediately after submission
                this.updateCounts(data);
                
                // Clear selections
                this.hiddenImages.clear();
                
                // Check if auto-training is triggered
                if (data.should_train) {
                    this.showNotification('100 selections reached! Starting automatic training...', 'info');
                    await this.autoTrain();
                } else if (data.galaxy_names && data.scores) {
                    // Load new images
                    this.currentBatch = data.galaxy_names;
                    this.currentScores = {};
                    data.galaxy_names.forEach((name, index) => {
                        this.currentScores[name] = data.scores[index];
                    });
                    this.renderImageGrid();
                    this.startSubmitCooldown();
                    this.showNotification(`Loaded ${this.currentBatch.length} new images`, 'success');
                } else {
                    await this.loadImages();
                }
            } else {
                this.showNotification(`Error submitting selections: ${data.error || 'Unknown error'}`, 'error');
            }
        } catch (error) {
            this.showNotification('Error submitting selections', 'error');
            console.error('Error submitting selections:', error);
        }
    }
    
    showTrainingOverlay(message = 'Please wait while the model is being trained.') {
        this.trainingOverlay.style.display = 'flex';
        this.trainingMessage.textContent = message;
    }
    
    hideTrainingOverlay() {
        this.trainingOverlay.style.display = 'none';
    }
    
    async autoTrain() {
        this.isTraining = true;
        this.showTrainingOverlay('Training model automatically...');
        
        const epochs = 300;
        
        try {
            const response = await fetch('/api/run_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ epochs })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showNotification('Model trained successfully!', 'success');
                
                // Update model trained status
                this.modelTrained = true;
                
                // Show visualization
                this.showVisualization();
                
                // Update status
                await this.updateStatus();
                
                // Load new images
                await this.loadImages();
                
            } else {
                this.showNotification('Training failed: ' + (data.error || 'Unknown error'), 'error');
            }
        } catch (error) {
            this.showNotification('Error during training', 'error');
            console.error('Training error:', error);
        } finally {
            this.isTraining = false;
            this.hideTrainingOverlay();
        }
    }
    
    showVisualization() {
        try {
            if (this.modelTrained) {
                this.visualizationSection.style.display = 'block';
                this.visualizationImage.src = '/visualizations/visualizations.png?t=' + new Date().getTime();
            } else {
                console.log('No visualization available yet');
            }
        } catch (error) {
            console.error('Error loading visualization:', error);
        }
    }
    
    
    updateCounts(data) {
        this.currentRoundEl.textContent = data.round || this.currentRound;
        this.slCountEl.textContent = data.sl_count || 0;
        this.nonSlCountEl.textContent = data.non_sl_count || 0;
        this.availableCountEl.textContent = data.available_count || 0;
        this.totalSubmissionsEl.textContent = data.total_submissions || 0;
    }
    
    startSubmitCooldown() {
        // Clear any existing timer first
        if (this.cooldownTimer) {
            clearInterval(this.cooldownTimer);
            this.cooldownTimer = null;
        }
        
        this.submitCooldown = true;
        this.submitSelectionsBtn.disabled = true;
        
        let countdown = 5;
        this.submitSelectionsBtn.textContent = `Submit (${countdown}s)`;
        
        this.cooldownTimer = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                this.submitSelectionsBtn.textContent = `Submit (${countdown}s)`;
            } else {
                clearInterval(this.cooldownTimer);
                this.cooldownTimer = null;
                this.submitCooldown = false;
                this.submitSelectionsBtn.disabled = false;
                this.submitSelectionsBtn.textContent = 'Submit';
            }
        }, 1000);
    }
    
    showNotification(message, type = 'info') {
        this.notification.textContent = message;
        this.notification.className = `notification ${type}`;
        this.notification.classList.add('show');
        
        setTimeout(() => {
            this.notification.classList.remove('show');
        }, 3000);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new SLDetector();
});
