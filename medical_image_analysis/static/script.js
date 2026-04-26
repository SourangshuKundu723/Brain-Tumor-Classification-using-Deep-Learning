const backendURL = "http://127.0.0.1:5000";
let analysisHistory = [];
let startTime = 0;
let currentPatient = {};
let selectedType = null;
// Initialize

function selectType(type) {
  selectedType = type;

  // Remove highlight
  document.getElementById('ctCard').classList.remove('selected-card');
  document.getElementById('lungCard').classList.remove('selected-card');

  // Hide all
  document.getElementById('ctForm').classList.add('hidden');
  document.getElementById('lungForm').classList.add('hidden');
  document.getElementById('ctDropArea').classList.add('hidden');
  document.getElementById('lungDropArea').classList.add('hidden');

  // Show selected
  document.getElementById(`${type}Card`).classList.add('selected-card');
  document.getElementById(`${type}Form`).classList.remove('hidden');
}


document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  setupDragAndDrop();
  updateStatistics();
  setupSearch();

  // 🔒 Disable buttons initially
  document.getElementById("ctBtn").disabled = true;
  document.getElementById("lungBtn").disabled = true;
});


// Drag and Drop functionality
function setupDragAndDrop() {
  const types = ['ct', 'lung'];

  types.forEach(type => {
    const dropArea = document.getElementById(`${type}DropArea`);
    const fileInput = document.getElementById(`${type}Image`);

    // 🔥 SAFETY CHECK (THIS WAS MISSING)
    if (!dropArea || !fileInput) {
      console.log(`Missing element for ${type}`);
      return;
    }

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
      dropArea.addEventListener(eventName, () => {
        dropArea.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, () => {
        dropArea.classList.remove('drag-over');
      });
    });

    dropArea.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length) {
        fileInput.files = files;
        previewImage(type);
      }
    });

    dropArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => previewImage(type));
  });
}

// Preview Image
function previewImage(type) {
  const input = document.getElementById(`${type}Image`);
  const preview = document.getElementById(`${type}Preview`);
  const file = input.files[0];

  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
  } else {
    alert('Please select a valid image file');
    input.value = '';
  }
}

function submitPatient(type) {

  //if (type !== selectedType) {
   // alert("Please select correct section first");
   // return;
  //}
   // relaxed condition for better UX
  selectedType = type;
  const name = document.getElementById(`${type}Name`).value;
  const age = document.getElementById(`${type}Age`).value;
  const gender = document.getElementById(`${type}Gender`).value;

  if (!name || !age || !gender) {
    alert("Fill required fields");
    return;
  }

  currentPatient[type] = {
    name,
    age,
    gender,
    phone: document.getElementById(`${type}Phone`).value,
    address: document.getElementById(`${type}Address`).value
  };

  // SHOW upload
  document.getElementById(`${type}DropArea`).classList.remove('hidden');

  // ENABLE button
  document.getElementById(`${type}Btn`).disabled = false;

  alert("Now upload image");
  console.log("Submit working", type);
}

// Upload and Analyze
function uploadImage(type) {
  const input = document.getElementById(`${type}Image`);
  const file = input.files[0];

  if (!file) {
    alert('Please select an image!');
    return;
  }

  const btn = document.getElementById(`${type}Btn`);
  const loader = document.getElementById(`${type}Loader`);
  const content = document.getElementById(`${type}Content`);

  btn.disabled = true;
  loader.classList.remove('hidden');
  content.classList.add('hidden');

  startTime = Date.now();

  const formData = new FormData();
  formData.append('image', file);
  formData.append('type', type);

  fetch(`${backendURL}/predict-${type}`, {
    method: 'POST',
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);
      
      // Update UI
      document.getElementById(`${type}Prediction`).textContent = data.prediction;
      document.getElementById(`${type}ConfidenceText`).textContent = `${data.confidence}%`;
      document.getElementById(`${type}Time`).textContent = new Date().toLocaleString();

      // Update confidence badge
      const confidence = parseInt(data.confidence);
      const badge = document.getElementById(`${type}Confidence`);
      badge.textContent = `${data.confidence}%`;
      badge.classList.remove('confidence-high', 'confidence-medium', 'confidence-low');
      
      if (confidence >= 80) {
        badge.classList.add('confidence-high');
      } else if (confidence >= 60) {
        badge.classList.add('confidence-medium');
      } else {
        badge.classList.add('confidence-low');
      }

      // Save to history
      addToHistory({
  type: type.toUpperCase(),
  prediction: data.prediction,
  confidence: data.confidence,
  timestamp: new Date().toISOString(),
  duration: duration,
  fileName: file.name,

  // 🔥 IMPORTANT
  patient: currentPatient[type] || {}
});

      content.classList.remove('hidden');
    })
    .catch(error => {
      console.error('Error:', error);
      alert('Error processing image. Make sure the backend server is running!');
    })
    .finally(() => {
      loader.classList.add('hidden');
      btn.disabled = false;
    });
}

// History Management
function addToHistory(entry) {
  analysisHistory.unshift(entry);
  saveHistory();
  updateStatistics();
  updateHistoryTable();
}

function saveHistory() {
  localStorage.setItem('medaiHistory', JSON.stringify(analysisHistory));
}

function loadHistory() {
  const saved = localStorage.getItem('medaiHistory');
  if (saved) {
    analysisHistory = JSON.parse(saved);
    updateHistoryTable();
    updateStatistics();
  }
}

function updateHistoryTable() {
  const tbody = document.getElementById('historyTableBody');
  const searchValue = document.getElementById('searchInput').value.toLowerCase();

 const filtered = analysisHistory.filter(entry => 
  entry.type.toLowerCase().includes(searchValue) ||
  entry.prediction.toLowerCase().includes(searchValue) ||
  (entry.patient?.name && entry.patient.name.toLowerCase().includes(searchValue))
);

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr class="empty-state">
        <td colspan="9">No analysis found matching your search.</td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = filtered.map((entry, index) => `
    <tr>
  <td>${index + 1}</td>
  <td>
    <span class="type-badge ${entry.type === 'CT' ? 'type-ct' : 'type-lung'}">
      ${entry.type}
    </span>
  </td>
  <td>${entry.patient?.name || '—'}</td>
  <td>${entry.patient?.age || '—'}</td>
  <td><strong>${entry.prediction}</strong></td>
  <td>
    <span class="confidence-badge ${getConfidenceClass(entry.confidence)}">
      ${entry.confidence}%
    </span>
  </td>
  <td>${new Date(entry.timestamp).toLocaleString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })}</td>
  <td>${entry.duration}s</td>
  <td>
    <button class="btn btn-small btn-primary" onclick="downloadHistoryPDF(${analysisHistory.indexOf(entry)})">
      <i class="fas fa-file-pdf"></i>
    </button>
    <button class="btn btn-small btn-secondary" onclick="downloadHistoryCSV(${analysisHistory.indexOf(entry)})">
      <i class="fas fa-download"></i>
    </button>
  </td>
</tr>
  `).join('');
}

function getConfidenceClass(confidence) {
  const conf = parseInt(confidence);
  if (conf >= 80) return 'confidence-high';
  if (conf >= 60) return 'confidence-medium';
  return 'confidence-low';
}

// Statistics
function updateStatistics() {
  const total = analysisHistory.length;
  const ctCount = analysisHistory.filter(e => e.type === 'CT').length;
  const lungCount = analysisHistory.filter(e => e.type === 'LUNG').length;
  const avgTime = total > 0 ? 
    (analysisHistory.reduce((sum, e) => sum + parseFloat(e.duration || 0), 0) / total).toFixed(1) : 
    '0';

  document.getElementById('totalAnalysis').textContent = total;
  document.getElementById('ctCount').textContent = ctCount;
  document.getElementById('lungCount').textContent = lungCount;
  document.getElementById('avgTime').textContent = avgTime + 's';
}

// Search
function setupSearch() {
  const searchInput = document.getElementById('searchInput');
  if (searchInput) {
    searchInput.addEventListener('input', updateHistoryTable);
  }
}

// Reset Analysis
function resetAnalysis(type) {
  const elements = {
    preview: document.getElementById(`${type}Preview`),
    input: document.getElementById(`${type}Image`),
    btn: document.getElementById(`${type}Btn`),
    content: document.getElementById(`${type}Content`)
  };

  elements.preview.style.display = 'none';
  elements.preview.src = '';
  elements.input.value = '';
  elements.btn.disabled = false;
  elements.content.classList.add('hidden');
}

// PDF Export with advanced formatting
function downloadPDF(type) {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

  // Header
  doc.setFillColor(37, 99, 235);
  doc.rect(0, 0, 210, 40, 'F');
  
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(24);
  doc.text('MediScan AI', 105, 15, { align: 'center' });
  doc.setFontSize(10);
  doc.text('Medical Imaging Analysis Report', 105, 25, { align: 'center' });

  // Reset colors
  doc.setTextColor(0, 0, 0);

  // Report Details
let yPos = 50;
const lineHeight = 8;

// Patient Info
const p = analysisHistory[0]?.patient || currentPatient[type] || {};
if (p.name) {
  doc.setFontSize(13);
  doc.setFont(undefined, 'bold');
  doc.text('Patient Information', 20, yPos); yPos += lineHeight;
  doc.setFont(undefined, 'normal');
  doc.setFontSize(11);
  doc.text(`Name    : ${p.name}`, 20, yPos); yPos += lineHeight;
  doc.text(`Age     : ${p.age}`, 20, yPos); yPos += lineHeight;
  doc.text(`Gender  : ${p.gender}`, 20, yPos); yPos += lineHeight;
  doc.text(`Phone   : ${p.phone}`, 20, yPos); yPos += lineHeight;
  doc.text(`Address : ${p.address}`, 20, yPos); yPos += lineHeight + 4;
}

doc.setFontSize(14);
doc.setFont(undefined, 'bold');
doc.text('Analysis Results', 20, yPos);
yPos += lineHeight;

doc.setFont(undefined, 'normal');
doc.setFontSize(11);

const analysisType = type === 'ct' ? 'Brain MRI Analysis' : 'Lung Disease Detection';
const prediction   = document.getElementById(`${type}Prediction`)?.textContent || analysisHistory[0]?.prediction || 'N/A';
const confidence   = document.getElementById(`${type}ConfidenceText`)?.textContent || (analysisHistory[0]?.confidence + '%') || 'N/A';
const timeAnalyzed = document.getElementById(`${type}Time`)?.textContent || new Date(analysisHistory[0]?.timestamp).toLocaleString() || 'N/A';

doc.text(`Type             : ${analysisType}`, 20, yPos); yPos += lineHeight;
doc.text(`Prediction       : ${prediction}`,   20, yPos); yPos += lineHeight;
doc.text(`Confidence Score : ${confidence}`,   20, yPos); yPos += lineHeight;
doc.text(`Analyzed         : ${timeAnalyzed}`, 20, yPos); yPos += lineHeight + 5;

  // Image preview
  const preview = document.getElementById(`${type}Preview`);
  if (preview.src) {
    try {
      doc.addImage(preview.src, 'JPEG', 20, yPos, 170, 100);
      yPos += 105;
    } catch (e) {
      console.log('Could not add image to PDF');
    }
  }

  yPos += 10;

  // Disclaimer
  doc.setFontSize(9);
  doc.setFont(undefined, 'italic');
  doc.setTextColor(100, 100, 100);
  doc.text('Disclaimer: This report is for diagnostic support only.', 20, yPos);
  yPos += 5;
  doc.text('Always consult qualified medical professionals for final diagnosis.', 20, yPos);

  // Footer
  doc.setFontSize(8);
  doc.setTextColor(150, 150, 150);
  doc.text(`Generated: ${new Date().toLocaleString()}`, 105, 280, { align: 'center' });

  doc.save(`${type}_analysis_report_${new Date().getTime()}.pdf`);
}

// CSV Export
function downloadCSV(type) {
  const entry = analysisHistory[0];
  if (!entry) {
    alert('No analysis to export');
    return;
  }

  const headers = ['Patient Name', 'Age', 'Gender', 'Phone', 'Address', 'Analysis Type', 'Prediction', 'Confidence (%)', 'Date & Time', 'Duration (Seconds)', 'File Name'];
  const values = [
    entry.patient?.name || 'N/A',
    entry.patient?.age || 'N/A',
    entry.patient?.gender || 'N/A',
    entry.patient?.phone || 'N/A',
    entry.patient?.address || 'N/A',
    entry.type,
    entry.prediction,
    entry.confidence,
    new Date(entry.timestamp).toLocaleString(),
    entry.duration,
    entry.fileName || 'N/A'
  ];

  const csv = [headers.join(','), values.join(',')].join('\n');
  downloadAsFile(csv, `${type}_analysis_${new Date().getTime()}.csv`, 'text/csv');
}

// Export all data as CSV
function exportAllData() {
  if (analysisHistory.length === 0) {
    alert('No data to export');
    return;
  }

  const headers = ['#', 'Type', 'Patient Name', 'Age', 'Gender', 'Phone', 'Address', 'Prediction', 'Confidence (%)', 'Date & Time', 'Duration (Seconds)', 'File Name'];
  const rows = analysisHistory.map((entry, index) => [
    index + 1,
    entry.type,
    entry.patient?.name || 'N/A',
    entry.patient?.age || 'N/A',
    entry.patient?.gender || 'N/A',
    entry.patient?.phone || 'N/A',
    entry.patient?.address || 'N/A',
    entry.prediction,
    entry.confidence,
    new Date(entry.timestamp).toLocaleString(),
    entry.duration,
    entry.fileName || 'N/A'
  ]);

  const csv = [
    headers.join(','),
    ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
  ].join('\n');

  downloadAsFile(csv, `medai_pro_analysis_history_${new Date().getTime()}.csv`, 'text/csv');
}

// Download History PDF
function downloadHistoryPDF(index) {
  if (index < 0 || index >= analysisHistory.length) {
    alert('Analysis not found');
    return;
  }

  const entry = analysisHistory[index];
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();

  doc.setFillColor(37, 99, 235);
  doc.rect(0, 0, 210, 35, 'F');
  
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(20);
  doc.text('MediScan AI Analysis', 105, 15, { align: 'center' });
  doc.setFontSize(10);
  doc.text('Detailed Analysis Report', 105, 25, { align: 'center' });

  doc.setTextColor(0, 0, 0);
  doc.setFontSize(12);
  doc.setFont(undefined, 'bold');
  doc.text('Analysis Details', 20, 50);

  doc.setFont(undefined, 'normal');
  doc.setFontSize(11);
  let yPos = 60;

  const details = [
    `Type: ${entry.type}`,
    `Prediction: ${entry.prediction}`,
    `Confidence: ${entry.confidence}%`,
    `Date & Time: ${new Date(entry.timestamp).toLocaleString()}`,
    `Analysis Duration: ${entry.duration} seconds`,
    `File: ${entry.fileName || 'N/A'}`
  ];

  details.forEach(detail => {
    doc.text(detail, 20, yPos);
    yPos += 8;
  });

  yPos += 10;
  doc.setFontSize(9);
  doc.setFont(undefined, 'italic');
  doc.setTextColor(100, 100, 100);
  doc.text('For medical decision-making, always consult qualified professionals.', 20, yPos);

  doc.setFontSize(8);
  doc.setTextColor(150, 150, 150);
  doc.text(`Report Generated: ${new Date().toLocaleString()}`, 105, 270, { align: 'center' });

  doc.save(`analysis_${entry.type}_${new Date().getTime()}.pdf`);
}

// Download History CSV
function downloadHistoryCSV(index) {
  if (index < 0 || index >= analysisHistory.length) {
    alert('Analysis not found');
    return;
  }

  const entry = analysisHistory[index];
  const headers = ['Patient Name', 'Age', 'Gender', 'Phone', 'Address', 'Analysis Type', 'Prediction', 'Confidence (%)', 'Date & Time', 'Duration (Seconds)', 'File Name'];
  const values = [
  entry.patient?.name || 'N/A',
  entry.patient?.age || 'N/A',
  entry.patient?.gender || 'N/A',
  entry.patient?.phone || 'N/A',
  entry.patient?.address || 'N/A',
  entry.type,
  entry.prediction,
  entry.confidence,
  new Date(entry.timestamp).toLocaleString(),
  entry.duration,
  entry.fileName || 'N/A'
  ];

  const csv = [headers.join(','), values.join(',')].join('\n');
  downloadAsFile(csv, `${entry.type}_analysis_${new Date().getTime()}.csv`, 'text/csv');
}

// Helper: Download file
function downloadAsFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

// Clear all data
function clearAllData() {
  if (confirm('Are you sure you want to clear all analysis history? This action cannot be undone.')) {
    analysisHistory = [];
    saveHistory();
    updateStatistics();
    updateHistoryTable();
    ['ct', 'lung'].forEach(type => resetAnalysis(type));
    alert('All data cleared successfully!');
  }
}

function clearSection(type) {
  analysisHistory = analysisHistory.filter(e => e.type !== type.toUpperCase());
  saveHistory();
  updateHistoryTable();
  updateStatistics();

  alert(`${type.toUpperCase()} data cleared`);
}
