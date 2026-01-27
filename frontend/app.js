// InspectAI Frontend - JavaScript Application Logic

// Detect if running locally or in production
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : window.location.origin;

// State
let selectedCategory = '';
let selectedFile = null;
let selectedFileDataUrl = null;

// DOM Elements
const categorySelect = document.getElementById('categorySelect');
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const inspectBtn = document.getElementById('inspectBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const systemStatus = document.getElementById('systemStatus');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadCategories();
    setupEventListeners();
    checkBackendHealth();
});

// Check backend health
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateSystemStatus('System Ready', true);
        } else {
            updateSystemStatus('System Warning', false);
        }
    } catch (error) {
        updateSystemStatus('Backend Offline', false);
        showNotification('Cannot connect to backend. Please start the server.', 'error');
    }
}

function updateSystemStatus(message, isHealthy) {
    const statusDot = systemStatus.querySelector('.status-dot');
    const statusText = systemStatus.querySelector('span:last-child');
    
    statusText.textContent = message;
    statusDot.style.background = isHealthy ? 'var(--success-color)' : 'var(--danger-color)';
}

// Load available categories
async function loadCategories() {
    try {
        const response = await fetch(`${API_BASE_URL}/categories`);
        const categories = await response.json();
        
        categorySelect.innerHTML = '<option value="">Select category...</option>';
        
        categories.forEach(cat => {
            const option = document.createElement('option');
            option.value = cat.name;
            option.textContent = `${cat.name} ${cat.model_loaded ? '✓' : '(not trained)'}`;
            option.disabled = !cat.model_loaded;
            categorySelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading categories:', error);
        showNotification('Failed to load categories', 'error');
    }
}

// Setup event listeners
function setupEventListeners() {
    // Category selection
    categorySelect.addEventListener('change', (e) => {
        selectedCategory = e.target.value;
        updateInspectButton();
    });
    
    // Upload area click
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });
    
    // File input change
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('active');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('active');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('active');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Inspect button
    inspectBtn.addEventListener('click', runInspection);
    
    // New inspection button
    document.getElementById('newInspectionBtn').addEventListener('click', resetForm);
    
    // Export button
    document.getElementById('exportBtn').addEventListener('click', exportReport);
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.match('image.*')) {
        showNotification('Please select an image file', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Read file as data URL for preview
    const reader = new FileReader();
    reader.onload = (e) => {
        selectedFileDataUrl = e.target.result;
        updateUploadArea(file.name);
        updateInspectButton();
    };
    reader.readAsDataURL(file);
}

function updateUploadArea(fileName) {
    uploadArea.innerHTML = `
        <div class="upload-prompt">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"></path>
            </svg>
            <p>Image selected</p>
            <div class="file-name">${fileName}</div>
        </div>
    `;
}

function updateInspectButton() {
    inspectBtn.disabled = !(selectedCategory && selectedFile);
}

// Run inspection
async function runInspection() {
    if (!selectedCategory || !selectedFile) {
        return;
    }
    
    // Show loading
    loadingOverlay.style.display = 'flex';
    resultsSection.style.display = 'none';
    
    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('category', selectedCategory);
        formData.append('image', selectedFile);
        formData.append('return_visualizations', 'true');
        
        // Call API
        const response = await fetch(`${API_BASE_URL}/inspect`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Inspection failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Inspection error:', error);
        showNotification('Inspection failed. Please try again.', 'error');
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

// Display inspection results
function displayResults(result) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Update decision banner
    const decisionBanner = document.getElementById('decisionBanner');
    const isPass = result.decision === 'PASS';
    
    decisionBanner.className = `decision-banner ${isPass ? 'pass' : 'fail'}`;
    
    // Update decision icon
    const decisionIcon = document.getElementById('decisionIcon');
    decisionIcon.textContent = isPass ? '✓' : '✗';
    
    // Update decision text
    document.getElementById('decisionStatus').textContent = result.decision;
    document.getElementById('decisionMessage').textContent = isPass 
        ? 'No anomalies detected. Product passes inspection.'
        : 'Anomalies detected. Product failed inspection.';
    
    // Update score
    document.getElementById('scoreValue').textContent = result.anomaly_score.toFixed(2);
    
    // Update threshold info
    document.getElementById('thresholdValue').textContent = result.threshold.toFixed(2);
    document.getElementById('resultCategory').textContent = result.category;
    document.getElementById('resultCategoryNote').textContent = result.category;
    document.getElementById('timestamp').textContent = formatTimestamp(result.timestamp);
    
    // Update visualizations
    document.getElementById('originalImage').src = selectedFileDataUrl;
    document.getElementById('heatmapImage').src = `data:image/png;base64,${result.heatmap_base64}`;
    document.getElementById('overlayImage').src = `data:image/png;base64,${result.overlay_base64}`;
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Format timestamp
function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
}

// Reset form for new inspection
function resetForm() {
    selectedFile = null;
    selectedFileDataUrl = null;
    selectedCategory = '';
    
    categorySelect.value = '';
    imageInput.value = '';
    
    uploadArea.innerHTML = `
        <div class="upload-prompt">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
            <p>Drop image here or click to browse</p>
            <p class="file-info">Supported: JPG, PNG</p>
        </div>
    `;
    
    resultsSection.style.display = 'none';
    updateInspectButton();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Export report
function exportReport() {
    // Get current results
    const category = document.getElementById('resultCategory').textContent;
    const decision = document.getElementById('decisionStatus').textContent;
    const score = document.getElementById('scoreValue').textContent;
    const threshold = document.getElementById('thresholdValue').textContent;
    const timestamp = document.getElementById('timestamp').textContent;
    
    // Create report content
    const report = `
InspectAI - Inspection Report
==============================

Category: ${category}
Decision: ${decision}
Anomaly Score: ${score}
Threshold: ${threshold}
Timestamp: ${timestamp}

Status: ${decision === 'PASS' ? 'Product approved for shipment' : 'Product requires further inspection'}

Generated by InspectAI v1.0
    `.trim();
    
    // Download as text file
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `inspection_report_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showNotification('Report exported successfully', 'success');
}

// Show notification
function showNotification(message, type = 'info') {
    // Simple alert for now - can be enhanced with a proper notification system
    alert(message);
}
