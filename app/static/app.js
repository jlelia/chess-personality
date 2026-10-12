document.addEventListener('DOMContentLoaded', () => {
  // Tabs
  const tabUpload = document.getElementById('tab-upload');
  const tabPaste = document.getElementById('tab-paste');
  const viewUpload = document.getElementById('view-upload');
  const viewPaste = document.getElementById('view-paste');

  // File Upload
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('pgn-file-input');
  const browseBtn = document.getElementById('browse-btn');
  const fileNameDisplay = document.getElementById('file-name-display');

  // Inputs
  const pgnTextInput = document.getElementById('pgn-text-input');
  const playerNameInput = document.getElementById('player-name-input');
  const sampleBtn = document.getElementById('sample-btn');
  const analyzeBtn = document.getElementById('analyze-btn');

  // Side Selector
  const sideBtns = document.querySelectorAll('.side-btn');
  const usernameWrapper = document.getElementById('username-wrapper');
  let selectedSide = 'white';

  // Error Banner
  const errorBanner = document.getElementById('error-banner');
  const errorMessage = document.getElementById('error-message');

  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove('hidden');
  }

  function hideError() {
    errorBanner.classList.add('hidden');
  }

  sideBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sideBtns.forEach(b => {
        b.classList.remove('active');
        const check = b.querySelector('.side-check');
        if (check) check.classList.add('hidden');
      });

      btn.classList.add('active');
      const activeCheck = btn.querySelector('.side-check');
      if (activeCheck) activeCheck.classList.remove('hidden');

      selectedSide = btn.getAttribute('data-side');

      if (selectedSide === 'auto') {
        usernameWrapper.classList.remove('hidden');
        playerNameInput.focus();
      } else {
        usernameWrapper.classList.add('hidden');
      }
      hideError();
    });
  });

  pgnTextInput.addEventListener('input', hideError);

  // Sections
  const loadingCard = document.getElementById('loading-card');
  const resultsSection = document.getElementById('results-section');

  // Hero Card Elements
  const heroName = document.getElementById('hero-name');
  const heroArchetype = document.getElementById('hero-archetype');
  const heroDescription = document.getElementById('hero-description');
  const heroSimilarity = document.getElementById('hero-similarity');
  const heroGamesBadge = document.getElementById('hero-games-badge');
  const heroAvatar = document.getElementById('hero-avatar');

  // Traits
  const traitTactical = document.getElementById('trait-tactical');
  const barTactical = document.getElementById('bar-tactical');
  const traitPositional = document.getElementById('trait-positional');
  const barPositional = document.getElementById('bar-positional');
  const traitInitiative = document.getElementById('trait-initiative');
  const barInitiative = document.getElementById('bar-initiative');
  const traitEndgame = document.getElementById('trait-endgame');
  const barEndgame = document.getElementById('bar-endgame');

  // Lists & Tables
  const podiumList = document.getElementById('podium-list');
  const gmTableBody = document.getElementById('gm-table-body');
  const filterInput = document.getElementById('filter-input');

  let selectedFile = null;
  let currentRankings = [];

  // Tab switching
  tabUpload.addEventListener('click', () => {
    tabUpload.classList.add('active');
    tabPaste.classList.remove('active');
    viewUpload.classList.remove('hidden');
    viewPaste.classList.add('hidden');
    hideError();
  });

  tabPaste.addEventListener('click', () => {
    tabPaste.classList.add('active');
    tabUpload.classList.remove('active');
    viewPaste.classList.remove('hidden');
    viewUpload.classList.add('hidden');
    hideError();
  });

  // Browse file
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and drop
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
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    hideError();
  }

  // Load sample game
  sampleBtn.addEventListener('click', async () => {
    try {
      sampleBtn.disabled = true;
      sampleBtn.textContent = 'Loading...';
      const res = await fetch('/api/sample');
      const data = await res.json();

      tabPaste.click();
      pgnTextInput.value = data.pgn_text;
      document.getElementById('side-white-btn').click();
      hideError();
    } catch (err) {
      showError('Failed to load sample game.');
    } finally {
      sampleBtn.disabled = false;
      sampleBtn.textContent = 'Try Sample Game';
    }
  });

  // Analyze Action
  analyzeBtn.addEventListener('click', async () => {
    hideError();
    const isPasteActive = tabPaste.classList.contains('active');
    const pgnText = pgnTextInput.value.trim();
    const playerName = playerNameInput.value.trim();

    if (isPasteActive && !pgnText) {
      showError('Please paste PGN notation first, or click "Try Sample Game".');
      return;
    }

    if (!isPasteActive && !selectedFile) {
      showError('Please select or drop a PGN file first.');
      return;
    }

    // UI state
    analyzeBtn.disabled = true;
    loadingCard.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('side', selectedSide);

    if (selectedSide === 'auto' && playerName) {
      formData.append('player_name', playerName);
    }

    if (isPasteActive) {
      formData.append('pgn_text', pgnText);
    } else {
      formData.append('file', selectedFile);
    }

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Analysis failed. Please ensure your input is valid PGN notation.');
      }

      const report = await response.json();
      renderResults(report);
    } catch (err) {
      showError(err.message);
    } finally {
      loadingCard.classList.add('hidden');
      analyzeBtn.disabled = false;
    }
  });

  function renderResults(report) {
    const top = report.top_match;
    currentRankings = report.all_rankings;

    // Hero
    heroName.textContent = top.name;
    heroArchetype.textContent = top.title;
    heroDescription.textContent = top.description;
    heroSimilarity.textContent = `${top.similarity_pct}%`;
    heroGamesBadge.textContent = `${report.games_analyzed} ${report.games_analyzed === 1 ? 'Game' : 'Games'} Analyzed (${report.player_name})`;

    // Avatar icon based on archetype
    const avatars = ['♚', '♛', '♜', '♝', '♞'];
    heroAvatar.textContent = avatars[Math.floor(Math.random() * avatars.length)];

    // Traits
    const radar = report.style_radar;
    setTrait(traitTactical, barTactical, radar.tactical_sharpness);
    setTrait(traitPositional, barPositional, radar.positional_solidity);
    setTrait(traitInitiative, barInitiative, radar.dynamic_initiative);
    setTrait(traitEndgame, barEndgame, radar.endgame_technique);

    // Podium
    podiumList.innerHTML = '';
    report.top_5.forEach((item, idx) => {
      const div = document.createElement('div');
      div.className = 'podium-item';
      div.innerHTML = `
        <span class="podium-rank">#${idx + 1}</span>
        <div class="podium-name-group">
          <div class="podium-name">${item.name}</div>
          <div class="podium-title">${item.title}</div>
        </div>
        <span class="podium-score">${item.similarity_pct}%</span>
      `;
      podiumList.appendChild(div);
    });

    // Full Table
    renderTable(currentRankings);

    // Reveal and scroll smoothly
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function setTrait(labelEl, barEl, value) {
    labelEl.textContent = `${value}%`;
    barEl.style.width = '0%';
    setTimeout(() => {
      barEl.style.width = `${value}%`;
    }, 100);
  }

  function renderTable(rankings) {
    gmTableBody.innerHTML = '';
    rankings.forEach((item) => {
      const tr = document.createElement('tr');
      if (item.rank <= 3) tr.classList.add('top-row');

      const cosVal = item.cosine_sim >= 0 ? `+${item.cosine_sim.toFixed(4)}` : `${item.cosine_sim.toFixed(4)}`;
      tr.innerHTML = `
        <td><strong>#${item.rank}</strong></td>
        <td><strong>${item.name}</strong></td>
        <td>${item.title}</td>
        <td><strong>${item.similarity_pct}%</strong></td>
        <td style="color: var(--text-muted); font-family: monospace;">${cosVal}</td>
      `;
      gmTableBody.appendChild(tr);
    });
  }

  // Filter input for table
  filterInput.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    if (!q) {
      renderTable(currentRankings);
      return;
    }

    const filtered = currentRankings.filter(r => 
      r.name.toLowerCase().includes(q) || r.title.toLowerCase().includes(q)
    );
    renderTable(filtered);
  });
});
