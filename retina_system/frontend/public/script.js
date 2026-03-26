const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const dropZoneContent = document.getElementById('dropZoneContent');
const analyzeBtn = document.getElementById('analyzeBtn');
const errorMsg = document.getElementById('errorMsg');
const resultSection = document.getElementById('resultSection');

// Result elements
const resOriginal = document.getElementById('resOriginal');
const resMask = document.getElementById('resMask');
const statPx = document.getElementById('statPx');
const statRatio = document.getElementById('statRatio');
const statDensity = document.getElementById('statDensity');

let currentFile = null;

// Handle Drag & Drop
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError("Lütfen geçerli bir görüntü dosyası yükleyin.");
        return;
    }
    
    currentFile = file;
    errorMsg.hidden = true;
    analyzeBtn.disabled = false;
    resultSection.hidden = true;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.hidden = false;
        dropZoneContent.hidden = true;
    };
    reader.readAsDataURL(file);
}

function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.hidden = false;
}

// Handle API Call
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Setup UI for loading
    analyzeBtn.disabled = true;
    document.querySelector('.btn-text').textContent = "Analiz Ediliyor...";
    document.querySelector('.spinner').hidden = false;
    errorMsg.hidden = true;
    resultSection.hidden = true;

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        const response = await fetch("http://localhost:8000/api/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || "Sunucu hatası oluştu.");
        }

        const data = await response.json();
        
        // Show results
        resOriginal.src = previewImage.src;
        resMask.src = "data:image/png;base64," + data.mask_base64;
        
        statPx.textContent = data.vessel_pixel_count.toLocaleString();
        statRatio.textContent = (data.vessel_area_ratio * 100).toFixed(2) + "%";
        statDensity.textContent = (data.vessel_density * 100).toFixed(2) + "%";
        
        resultSection.hidden = false;
        resultSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        showError(err.message);
    } finally {
        analyzeBtn.disabled = false;
        document.querySelector('.btn-text').textContent = "Analiz Et (Attention U-Net)";
        document.querySelector('.spinner').hidden = true;
    }
});
