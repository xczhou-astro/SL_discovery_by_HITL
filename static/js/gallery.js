class GalleryManager {
    constructor() {
        this.slItems = [];
        this.nonSlItems = [];
        this.highScoreItems = [];
        this.currentSLPage = 1;
        this.currentNonSLPage = 1;
        this.currentHighScorePage = 1;
        this.itemsPerPage = 10;
        this.slTotalPages = 1;
        this.nonSlTotalPages = 1;
        this.highScoreTotalPages = 1;
        this.currentGallery = 'sl'; // 'sl', 'non_sl', or 'high_score'
        this.minScore = null;
        
        this.initializeElements();
        this.loadGalleryData();
    }
    
    initializeElements() {
        // SL Gallery elements
        this.slGalleryGrid = document.getElementById('sl-gallery-grid');
        this.slCurrentPageEl = document.getElementById('sl-current-page');
        this.slTotalPagesEl = document.getElementById('sl-total-pages');
        this.slTotalCountEl = document.getElementById('sl-total-count');
        this.slConfirmedCountEl = document.getElementById('sl-confirmed-count');
        this.slSelectedCountEl = document.getElementById('sl-selected-count');
        
        // Non-SL Gallery elements
        this.nonSlGalleryGrid = document.getElementById('non-sl-gallery-grid');
        this.nonSlCurrentPageEl = document.getElementById('non-sl-current-page');
        this.nonSlTotalPagesEl = document.getElementById('non-sl-total-pages');
        this.nonSlTotalCountEl = document.getElementById('non-sl-total-count');
        
        // High-Score Gallery elements
        this.highScoreGalleryGrid = document.getElementById('high-score-gallery-grid');
        this.highScoreCurrentPageEl = document.getElementById('high-score-current-page');
        this.highScoreTotalPagesEl = document.getElementById('high-score-total-pages');
        this.highScoreTotalCountEl = document.getElementById('high-score-total-count');
        this.highScoreMinThresholdEl = document.getElementById('high-score-min-threshold');
        
        // Sections
        this.slSection = document.getElementById('sl-gallery-section');
        this.nonSlSection = document.getElementById('non-sl-gallery-section');
        this.highScoreSection = document.getElementById('high-score-gallery-section');
        
        // Tab buttons
        this.slTabBtn = document.getElementById('sl-tab-btn');
        this.nonSlTabBtn = document.getElementById('non-sl-tab-btn');
        this.highScoreTabBtn = document.getElementById('high-score-tab-btn');
        
        this.notification = document.getElementById('notification');
    }
    
    async loadGalleryData() {
        try {
            this.showNotification('Loading gallery...', 'info');
            
            const response = await fetch('/api/get_gallery_data');
            const data = await response.json();
            
            console.log('Gallery data received:', data);
            
            if (data.success) {
                console.log('Total items:', data.items.length);
                
                // Separate SL, non-SL, and high-score items
                this.slItems = data.items.filter(item => item.type === 'sl');
                this.nonSlItems = data.items.filter(item => item.type === 'non_sl');
                this.highScoreItems = data.items.filter(item => item.type === 'high_score');
                this.minScore = data.min_score;
                
                console.log('SL items:', this.slItems.length);
                console.log('Non-SL items:', this.nonSlItems.length);
                console.log('High-score items:', this.highScoreItems.length);
                if (this.nonSlItems.length > 0) {
                    console.log('First non-SL item:', this.nonSlItems[0]);
                }
                
                this.slTotalPages = Math.ceil(this.slItems.length / this.itemsPerPage) || 1;
                this.nonSlTotalPages = Math.ceil(this.nonSlItems.length / this.itemsPerPage) || 1;
                this.highScoreTotalPages = Math.ceil(this.highScoreItems.length / this.itemsPerPage) || 1;
                
                // Update SL counts
                this.slTotalCountEl.textContent = this.slItems.length;
                this.slConfirmedCountEl.textContent = data.confirmed_count;
                this.slSelectedCountEl.textContent = data.selected_sl_count;
                
                // Update Non-SL counts
                this.nonSlTotalCountEl.textContent = this.nonSlItems.length;
                
                // Update High-Score counts
                this.highScoreTotalCountEl.textContent = this.highScoreItems.length;
                this.highScoreMinThresholdEl.textContent = this.minScore !== null ? this.minScore.toFixed(4) : 'N/A';
                
                // Render first page of SL gallery
                this.renderSLPage(1);
                this.showNotification(`Loaded ${this.slItems.length} SL, ${this.nonSlItems.length} non-SL, and ${this.highScoreItems.length} high-score images`, 'success');
            } else {
                this.showNotification('Error loading gallery data', 'error');
            }
        } catch (error) {
            console.error('Error loading gallery:', error);
            this.showNotification('Error loading gallery', 'error');
        }
    }
    
    renderSLPage(pageNumber) {
        if (pageNumber < 1 || pageNumber > this.slTotalPages) {
            return;
        }
        
        this.currentSLPage = pageNumber;
        this.slCurrentPageEl.textContent = this.currentSLPage;
        this.slTotalPagesEl.textContent = this.slTotalPages;
        
        const startIndex = (this.currentSLPage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, this.slItems.length);
        const pageItems = this.slItems.slice(startIndex, endIndex);
        
        this.slGalleryGrid.innerHTML = '';
        
        if (pageItems.length === 0) {
            this.slGalleryGrid.innerHTML = `
                <div class="empty-gallery" style="grid-column: 1 / -1;">
                    <h3>No SL Images Yet</h3>
                    <p>Start selecting strong lensing candidates!</p>
                </div>
            `;
        } else {
            pageItems.forEach(item => {
                this.renderGalleryItem(item, this.slGalleryGrid);
            });
        }
    }
    
    renderNonSLPage(pageNumber) {
        console.log('renderNonSLPage called with page:', pageNumber);
        console.log('Total non-SL items:', this.nonSlItems.length);
        console.log('Total pages:', this.nonSlTotalPages);
        
        if (pageNumber < 1 || pageNumber > this.nonSlTotalPages) {
            console.log('Page number out of range');
            return;
        }
        
        this.currentNonSLPage = pageNumber;
        this.nonSlCurrentPageEl.textContent = this.currentNonSLPage;
        this.nonSlTotalPagesEl.textContent = this.nonSlTotalPages;
        
        const startIndex = (this.currentNonSLPage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, this.nonSlItems.length);
        const pageItems = this.nonSlItems.slice(startIndex, endIndex);
        
        console.log('Rendering items from', startIndex, 'to', endIndex);
        console.log('Page items:', pageItems.length);
        
        this.nonSlGalleryGrid.innerHTML = '';
        
        if (pageItems.length === 0) {
            console.log('No items to render - showing empty message');
            this.nonSlGalleryGrid.innerHTML = `
                <div class="empty-gallery" style="grid-column: 1 / -1;">
                    <h3>No Non-SL Images Yet</h3>
                    <p>Start marking non-SL images!</p>
                </div>
            `;
        } else {
            console.log('Rendering', pageItems.length, 'items');
            pageItems.forEach(item => {
                this.renderGalleryItem(item, this.nonSlGalleryGrid);
            });
        }
    }
    
    renderHighScorePage(pageNumber) {
        if (pageNumber < 1 || pageNumber > this.highScoreTotalPages) {
            return;
        }
        
        this.currentHighScorePage = pageNumber;
        this.highScoreCurrentPageEl.textContent = this.currentHighScorePage;
        this.highScoreTotalPagesEl.textContent = this.highScoreTotalPages;
        
        const startIndex = (this.currentHighScorePage - 1) * this.itemsPerPage;
        const endIndex = Math.min(startIndex + this.itemsPerPage, this.highScoreItems.length);
        const pageItems = this.highScoreItems.slice(startIndex, endIndex);
        
        this.highScoreGalleryGrid.innerHTML = '';
        
        if (pageItems.length === 0) {
            this.highScoreGalleryGrid.innerHTML = `
                <div class="empty-gallery" style="grid-column: 1 / -1;">
                    <h3>No High-Score Candidates</h3>
                    <p>Train the model first to see high-scoring candidates!</p>
                </div>
            `;
        } else {
            pageItems.forEach(item => {
                this.renderGalleryItem(item, this.highScoreGalleryGrid);
            });
        }
    }
    
    renderGalleryItem(item, gridElement) {
        const galleryItem = document.createElement('div');
        galleryItem.className = 'gallery-item';
        
        // Change border color for different item types
        if (item.type === 'non_sl') {
            galleryItem.style.borderColor = '#e67e22';
        } else if (item.type === 'high_score') {
            // Different border colors for high-score items based on selection status
            if (item.is_confirmed) {
                galleryItem.style.borderColor = '#e74c3c'; // Red for COWLS
            } else if (item.is_selected_sl) {
                galleryItem.style.borderColor = '#2ecc71'; // Green for selected SL
            } else if (item.is_selected_non_sl) {
                galleryItem.style.borderColor = '#e67e22'; // Orange for selected non-SL
            } else {
                galleryItem.style.borderColor = '#9b59b6'; // Purple for unselected
            }
        }
        
        const img = document.createElement('img');
        img.src = `/images/${item.name}.jpg`;
        img.alt = `Galaxy ${item.name}`;
        img.loading = 'lazy';
        
        const overlay = document.createElement('div');
        overlay.className = 'gallery-item-overlay';
        overlay.textContent = item.name;
        
        galleryItem.appendChild(img);
        galleryItem.appendChild(overlay);
        
        if (item.is_confirmed) {
            const confirmedBadge = document.createElement('div');
            confirmedBadge.className = 'confirmed-badge';
            confirmedBadge.textContent = 'COWLS';
            galleryItem.appendChild(confirmedBadge);
        } else if (item.is_selected_sl) {
            const selectedBadge = document.createElement('div');
            selectedBadge.className = 'confirmed-badge';
            selectedBadge.style.background = 'rgba(46, 204, 113, 0.9)';
            selectedBadge.textContent = 'SELECTED SL';
            galleryItem.appendChild(selectedBadge);
        } else if (item.is_selected_non_sl) {
            const selectedBadge = document.createElement('div');
            selectedBadge.className = 'confirmed-badge';
            selectedBadge.style.background = 'rgba(230, 126, 34, 0.9)';
            selectedBadge.textContent = 'SELECTED NON-SL';
            galleryItem.appendChild(selectedBadge);
        }
        
        if (item.grade && item.grade !== 'N/A') {
            const gradeBadge = document.createElement('div');
            gradeBadge.className = 'gallery-item-grade';
            gradeBadge.textContent = `Grade: ${item.grade}`;
            galleryItem.appendChild(gradeBadge);
        }
        
        // Show score for high-score items
        if (item.type === 'high_score' && item.score !== undefined) {
            const scoreBadge = document.createElement('div');
            scoreBadge.className = 'gallery-item-grade';
            scoreBadge.style.background = 'rgba(155, 89, 182, 0.9)';
            scoreBadge.textContent = `Score: ${item.score.toFixed(4)}`;
            galleryItem.appendChild(scoreBadge);
        }
        
        gridElement.appendChild(galleryItem);
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

// Global functions for gallery switching and pagination
let galleryManager;

function showSLGallery() {
    if (galleryManager) {
        galleryManager.slSection.style.display = 'block';
        galleryManager.nonSlSection.style.display = 'none';
        galleryManager.highScoreSection.style.display = 'none';
        galleryManager.slTabBtn.className = 'btn btn-primary';
        galleryManager.nonSlTabBtn.className = 'btn btn-secondary';
        galleryManager.highScoreTabBtn.className = 'btn btn-secondary';
        galleryManager.currentGallery = 'sl';
    }
}

function showNonSLGallery() {
    console.log('showNonSLGallery called');
    if (galleryManager) {
        console.log('Gallery manager exists');
        galleryManager.slSection.style.display = 'none';
        galleryManager.nonSlSection.style.display = 'block';
        galleryManager.highScoreSection.style.display = 'none';
        galleryManager.slTabBtn.className = 'btn btn-secondary';
        galleryManager.nonSlTabBtn.className = 'btn btn-primary';
        galleryManager.highScoreTabBtn.className = 'btn btn-secondary';
        galleryManager.currentGallery = 'non_sl';
        console.log('Calling renderNonSLPage(1)');
        // Always render when switching to non-SL tab
        galleryManager.renderNonSLPage(1);
    } else {
        console.log('Gallery manager does not exist!');
    }
}

function showHighScoreGallery() {
    if (galleryManager) {
        galleryManager.slSection.style.display = 'none';
        galleryManager.nonSlSection.style.display = 'none';
        galleryManager.highScoreSection.style.display = 'block';
        galleryManager.slTabBtn.className = 'btn btn-secondary';
        galleryManager.nonSlTabBtn.className = 'btn btn-secondary';
        galleryManager.highScoreTabBtn.className = 'btn btn-primary';
        galleryManager.currentGallery = 'high_score';
        // Always render when switching to high-score tab
        galleryManager.renderHighScorePage(1);
    }
}

// SL Gallery pagination
function nextSLPage() {
    if (galleryManager) {
        galleryManager.renderSLPage(galleryManager.currentSLPage + 1);
    }
}

function previousSLPage() {
    if (galleryManager) {
        galleryManager.renderSLPage(galleryManager.currentSLPage - 1);
    }
}

function goToSLPage(pageNumber) {
    if (galleryManager) {
        galleryManager.renderSLPage(pageNumber);
    }
}

function goToLastSLPage() {
    if (galleryManager) {
        galleryManager.renderSLPage(galleryManager.slTotalPages);
    }
}

// Non-SL Gallery pagination
function nextNonSLPage() {
    if (galleryManager) {
        galleryManager.renderNonSLPage(galleryManager.currentNonSLPage + 1);
    }
}

function previousNonSLPage() {
    if (galleryManager) {
        galleryManager.renderNonSLPage(galleryManager.currentNonSLPage - 1);
    }
}

function goToNonSLPage(pageNumber) {
    if (galleryManager) {
        galleryManager.renderNonSLPage(pageNumber);
    }
}

function goToLastNonSLPage() {
    if (galleryManager) {
        galleryManager.renderNonSLPage(galleryManager.nonSlTotalPages);
    }
}

// High-Score Gallery pagination
function nextHighScorePage() {
    if (galleryManager) {
        galleryManager.renderHighScorePage(galleryManager.currentHighScorePage + 1);
    }
}

function previousHighScorePage() {
    if (galleryManager) {
        galleryManager.renderHighScorePage(galleryManager.currentHighScorePage - 1);
    }
}

function goToHighScorePage(pageNumber) {
    if (galleryManager) {
        galleryManager.renderHighScorePage(pageNumber);
    }
}

function goToLastHighScorePage() {
    if (galleryManager) {
        galleryManager.renderHighScorePage(galleryManager.highScoreTotalPages);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    galleryManager = new GalleryManager();
});
