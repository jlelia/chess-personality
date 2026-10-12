document.addEventListener('DOMContentLoaded', () => {
  // Stale-asset guard: if the served HTML predates this script version, some
  // element lookups below return null and binding throws mid-way, silently
  // disabling every control registered after the crash point. Detect the
  // mismatch and prompt a reload instead of half-initializing the page.
  const requiredIds = [
    'tab-upload', 'tab-paste', 'tab-online',
    'view-upload', 'view-paste', 'view-online',
    'fetch-online-btn', 'analyze-btn', 'share-card-btn'
  ];
  const missingIds = requiredIds.filter(id => !document.getElementById(id));
  if (missingIds.length > 0) {
    console.warn('[ChessPersonality] Stale page assets detected (missing: ' + missingIds.join(', ') + ').');
    const reloadBanner = document.createElement('div');
    reloadBanner.style.cssText =
      'position:fixed;top:0;left:0;right:0;z-index:9999;padding:12px;text-align:center;' +
      'background:#d4af37;color:#0b0f17;font:600 14px Inter,system-ui,sans-serif;cursor:pointer;';
    reloadBanner.textContent = 'This page was updated in the background — click here to reload and restore full functionality.';
    reloadBanner.addEventListener('click', () => window.location.reload());
    document.body.prepend(reloadBanner);
    return;
  }

  // Tabs
  const tabUpload = document.getElementById('tab-upload');
  const tabPaste = document.getElementById('tab-paste');
  const tabOnline = document.getElementById('tab-online');
  const viewUpload = document.getElementById('view-upload');
  const viewPaste = document.getElementById('view-paste');
  const viewOnline = document.getElementById('view-online');

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

  // Online Import Elements
  const platLichessBtn = document.getElementById('plat-lichess-btn');
  const platChesscomBtn = document.getElementById('plat-chesscom-btn');
  const onlineUsernameInput = document.getElementById('online-username-input');
  const onlineCountSelect = document.getElementById('online-count-select');
  const fetchOnlineBtn = document.getElementById('fetch-online-btn');
  const onlineFeedback = document.getElementById('online-feedback');
  let selectedPlatform = 'lichess';

  // Side Selector
  const sideBtns = document.querySelectorAll('.side-btn');
  const usernameWrapper = document.getElementById('username-wrapper');
  let selectedSide = 'white';

  // Error Banner
  const errorBanner = document.getElementById('error-banner');
  const errorMessage = document.getElementById('error-message');

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

  // Mini Board Elements
  const miniBoardFrame = document.getElementById('mini-board-frame');
  const boardPanelTitle = document.getElementById('board-panel-title');
  const boardPovBadge = document.getElementById('board-pov-badge');
  const shareCardBtn = document.getElementById('share-card-btn');
  const shareCanvas = document.getElementById('share-canvas');

  // Traits
  const traitTactical = document.getElementById('trait-tactical');
  const barTactical = document.getElementById('bar-tactical');
  const traitPositional = document.getElementById('trait-positional');
  const barPositional = document.getElementById('bar-positional');
  const traitInitiative = document.getElementById('trait-initiative');
  const barInitiative = document.getElementById('bar-initiative');
  const traitEndgame = document.getElementById('trait-endgame');
  const barEndgame = document.getElementById('bar-endgame');

  // Podium, Ledger & Tables
  const podiumList = document.getElementById('podium-list');
  const gmTableBody = document.getElementById('gm-table-body');
  const filterInput = document.getElementById('filter-input');
  const gamesLedgerCard = document.getElementById('games-ledger-card');
  const gamesLedgerBody = document.getElementById('games-ledger-body');

  let selectedFile = null;
  let currentRankings = [];
  let currentReport = null;

  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove('hidden');
  }

  // Escape untrusted strings (PGN header values, usernames, etc.) before they
  // are interpolated into innerHTML templates. PGN headers are attacker-
  // controllable, e.g. [White "<img src=x onerror=...>"].
  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  function hideError() {
    errorBanner.classList.add('hidden');
  }

  function showOnlineFeedback(msg, type = 'info') {
    onlineFeedback.textContent = msg;
    onlineFeedback.className = `online-feedback ${type}`;
    onlineFeedback.classList.remove('hidden');
  }

  function hideOnlineFeedback() {
    onlineFeedback.classList.add('hidden');
  }

  // Side Selector logic
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

  // Tab switching
  function switchTab(activeTab, activeView) {
    [tabUpload, tabPaste, tabOnline].forEach(t => t.classList.remove('active'));
    [viewUpload, viewPaste, viewOnline].forEach(v => v.classList.add('hidden'));

    activeTab.classList.add('active');
    activeView.classList.remove('hidden');
    hideError();
    hideOnlineFeedback();
  }

  tabUpload.addEventListener('click', () => switchTab(tabUpload, viewUpload));
  tabPaste.addEventListener('click', () => switchTab(tabPaste, viewPaste));
  tabOnline.addEventListener('click', () => switchTab(tabOnline, viewOnline));

  // Platform selection (Lichess vs Chess.com)
  platLichessBtn.addEventListener('click', () => {
    platLichessBtn.classList.add('active');
    platChesscomBtn.classList.remove('active');
    selectedPlatform = 'lichess';
    onlineUsernameInput.placeholder = 'e.g. MagnusCarlsen';
    hideOnlineFeedback();
  });

  platChesscomBtn.addEventListener('click', () => {
    platChesscomBtn.classList.add('active');
    platLichessBtn.classList.remove('active');
    selectedPlatform = 'chesscom';
    onlineUsernameInput.placeholder = 'e.g. Hikaru';
    hideOnlineFeedback();
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

  // Fetch games online
  fetchOnlineBtn.addEventListener('click', async () => {
    const username = onlineUsernameInput.value.trim();
    const count = onlineCountSelect.value;

    if (!username) {
      showOnlineFeedback('Please enter a username.', 'error');
      onlineUsernameInput.focus();
      return;
    }

    try {
      fetchOnlineBtn.disabled = true;
      fetchOnlineBtn.textContent = 'Fetching...';
      showOnlineFeedback(`Connecting to ${selectedPlatform === 'lichess' ? 'Lichess' : 'Chess.com'}...`, 'info');

      const url = `/api/fetch-games?platform=${selectedPlatform}&username=${encodeURIComponent(username)}&max_games=${count}`;
      const res = await fetch(url);
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || 'Failed to fetch games.');
      }

      pgnTextInput.value = data.pgn_text;
      playerNameInput.value = username;

      // Automatically switch to Auto perspective with this username
      document.getElementById('side-auto-btn').click();

      showOnlineFeedback(`Fetched ${count} games successfully! Running neural stylometry...`, 'success');

      // Trigger analysis immediately for a seamless UX
      triggerAnalysis(data.pgn_text, username);
    } catch (err) {
      showOnlineFeedback(err.message, 'error');
    } finally {
      fetchOnlineBtn.disabled = false;
      fetchOnlineBtn.textContent = 'Fetch & Load';
    }
  });

  // Analyze Action
  analyzeBtn.addEventListener('click', () => {
    const isPasteActive = tabPaste.classList.contains('active');
    const isOnlineActive = tabOnline.classList.contains('active');
    const pgnText = pgnTextInput.value.trim();
    const playerName = playerNameInput.value.trim();

    if ((isPasteActive || isOnlineActive) && !pgnText) {
      showError('Please paste PGN notation or fetch games from Lichess/Chess.com first.');
      return;
    }

    if (!isPasteActive && !isOnlineActive && !selectedFile) {
      showError('Please select or drop a PGN file first.');
      return;
    }

    triggerAnalysis(pgnText, playerName, !isPasteActive && !isOnlineActive ? selectedFile : null);
  });

  async function triggerAnalysis(pgnText, playerName, file = null) {
    hideError();

    // UI state
    analyzeBtn.disabled = true;
    loadingCard.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('side', selectedSide);

    if (selectedSide === 'auto' && playerName) {
      formData.append('player_name', playerName);
    }

    if (file) {
      formData.append('file', file);
    } else {
      formData.append('pgn_text', pgnText);
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
      currentReport = report;
      renderResults(report);
    } catch (err) {
      showError(err.message);
    } finally {
      loadingCard.classList.add('hidden');
      analyzeBtn.disabled = false;
    }
  }

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

    // Final Position Mini Board
    if (report.primary_final_svg) {
      miniBoardFrame.innerHTML = report.primary_final_svg;
      const initialPov = (report.games && report.games[0]) ? report.games[0].played_as : 'White';
      boardPanelTitle.textContent = 'Final Position (Game #1)';
      boardPovBadge.textContent = `POV: ${initialPov}`;
    }

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
          <div class="podium-name">${escapeHtml(item.name)}</div>
          <div class="podium-title">${escapeHtml(item.title)}</div>
        </div>
        <span class="podium-score">${escapeHtml(item.similarity_pct)}%</span>
      `;
      podiumList.appendChild(div);
    });

    // Games Ledger Table
    if (report.games && report.games.length > 0) {
      gamesLedgerCard.classList.remove('hidden');
      gamesLedgerBody.innerHTML = '';

      report.games.forEach((game, idx) => {
        const tr = document.createElement('tr');

        // Determine result styling
        let resClass = 'res-draw';
        let resText = game.result;
        if (game.result === '1-0') {
          resClass = (game.played_as === 'White') ? 'res-win' : 'res-loss';
        } else if (game.result === '0-1') {
          resClass = (game.played_as === 'Black') ? 'res-win' : 'res-loss';
        } else if (game.result === '1/2-1/2') {
          resClass = 'res-draw';
        }

        tr.innerHTML = `
          <td><strong>#${escapeHtml(game.game_num)}</strong></td>
          <td>${escapeHtml(game.white) || 'White'}</td>
          <td>${escapeHtml(game.black) || 'Black'}</td>
          <td><strong class="${resClass}">${escapeHtml(resText)}</strong></td>
          <td>${escapeHtml(game.played_as)}</td>
          <td>${escapeHtml(game.plies)} plies</td>
          <td>
            <button type="button" class="inspect-btn" data-game-idx="${idx}">Inspect</button>
          </td>
        `;

        const inspectBtn = tr.querySelector('.inspect-btn');
        inspectBtn.addEventListener('click', () => {
          miniBoardFrame.innerHTML = game.final_board_svg;
          boardPanelTitle.textContent = `Final Position (Game #${game.game_num})`;
          boardPovBadge.textContent = `POV: ${game.played_as}`;
          document.querySelector('.hero-card').scrollIntoView({ behavior: 'smooth', block: 'center' });
        });

        gamesLedgerBody.appendChild(tr);
      });
    } else {
      gamesLedgerCard.classList.add('hidden');
    }

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
        <td><strong>#${escapeHtml(item.rank)}</strong></td>
        <td><strong>${escapeHtml(item.name)}</strong></td>
        <td>${escapeHtml(item.title)}</td>
        <td><strong>${escapeHtml(item.similarity_pct)}%</strong></td>
        <td style="color: var(--text-muted); font-family: monospace;">${escapeHtml(cosVal)}</td>
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

  // High-Resolution Social Card Generator
  shareCardBtn.addEventListener('click', async () => {
    if (!currentReport) return;

    const originalText = shareCardBtn.innerHTML;
    shareCardBtn.innerHTML = '<span>⏳ Generating...</span>';
    shareCardBtn.disabled = true;

    try {
      await generateSocialCard(currentReport);
    } catch (err) {
      console.error('Card generation failed:', err);
      showError('Failed to generate match card.');
    } finally {
      shareCardBtn.innerHTML = originalText;
      shareCardBtn.disabled = false;
    }
  });

  async function generateSocialCard(report) {
    const canvas = shareCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;  // 1200
    const height = canvas.height; // 630

    // 1. Background Luxury Gradient
    const bgGrad = ctx.createLinearGradient(0, 0, width, height);
    bgGrad.addColorStop(0, '#0b0f17');
    bgGrad.addColorStop(0.5, '#121826');
    bgGrad.addColorStop(1, '#080c13');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // 2. Card Outer Border
    ctx.save();
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.45)';
    ctx.lineWidth = 2;
    roundRect(ctx, 24, 24, width - 48, height - 48, 16);
    ctx.stroke();
    ctx.restore();

    // 3. Top Header Bar
    ctx.save();
    ctx.font = 'bold 16px "Cinzel", Georgia, serif';
    ctx.fillStyle = '#d4af37';
    ctx.fillText('CHESS PERSONALITY INDEXER', 60, 68);

    ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.fillStyle = '#9ca3af';
    ctx.textAlign = 'right';
    const subHeader = `${report.games_analyzed} ${report.games_analyzed === 1 ? 'Game' : 'Games'} Analyzed • ${report.player_name}`;
    ctx.fillText(subHeader, width - 60, 68);

    // Header divider line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(60, 90);
    ctx.lineTo(width - 60, 90);
    ctx.stroke();
    ctx.restore();

    const top = report.top_match;

    // 4. Primary Match Left Column
    // Avatar circle
    ctx.save();
    const avatarX = 100;
    const avatarY = 160;
    const avatarR = 40;
    ctx.beginPath();
    ctx.arc(avatarX, avatarY, avatarR, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(212, 175, 55, 0.12)';
    ctx.fill();
    ctx.strokeStyle = '#d4af37';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.font = '38px serif';
    ctx.fillStyle = '#f5d77f';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('♚', avatarX, avatarY + 2);
    ctx.restore();

    // GM Name & Archetype
    ctx.save();
    ctx.textAlign = 'left';
    ctx.font = 'bold 36px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.fillText(top.name, 160, 152);

    ctx.font = '600 18px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.fillStyle = '#38bdf8';
    ctx.fillText(top.title, 160, 182);

    // Stat Alignment Callout
    ctx.font = 'bold 50px "Cinzel", Georgia, serif';
    ctx.fillStyle = '#d4af37';
    ctx.fillText(`${top.similarity_pct}%`, 60, 260);

    ctx.font = 'bold 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.fillStyle = '#9ca3af';
    ctx.fillText('STYLISTIC ALIGNMENT INDEX', 60, 282);

    // Archetype Description
    ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    wrapText(ctx, top.description, 60, 315, 460, 22);

    // 5. Stylistic Radar Bars
    const traits = [
      { name: 'Tactical Sharpness', val: report.style_radar.tactical_sharpness, color: '#ef4444' },
      { name: 'Positional Solidity', val: report.style_radar.positional_solidity, color: '#0284c7' },
      { name: 'Dynamic Initiative', val: report.style_radar.dynamic_initiative, color: '#eab308' },
      { name: 'Endgame Technique', val: report.style_radar.endgame_technique, color: '#059669' }
    ];

    let startBarY = 380;
    const barW = 340;
    const barH = 7;

    traits.forEach(t => {
      // Label & Value
      ctx.font = '500 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(t.name, 60, startBarY);

      ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillStyle = '#f5d77f';
      ctx.textAlign = 'right';
      ctx.fillText(`${t.val}%`, 60 + barW, startBarY);
      ctx.textAlign = 'left';

      // Bar BG
      ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
      roundRect(ctx, 60, startBarY + 6, barW, barH, 3);
      ctx.fill();

      // Bar Fill
      ctx.fillStyle = t.color;
      const fillW = Math.max(4, (t.val / 100) * barW);
      roundRect(ctx, 60, startBarY + 6, fillW, barH, 3);
      ctx.fill();

      startBarY += 36;
    });
    ctx.restore();

    // 6. Right Column: Final Board Position SVG
    const svgString = report.primary_final_svg;
    if (svgString) {
      await drawSvgToCanvas(ctx, svgString, 720, 120, 420, 420);

      ctx.save();
      ctx.font = '600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
      ctx.fillStyle = '#d4af37';
      ctx.textAlign = 'center';
      const playedAs = (report.games && report.games[0]) ? report.games[0].played_as : 'White';
      ctx.fillText(`Final Position (Game #1) • POV: ${playedAs}`, 930, 560);
      ctx.restore();
    }

    // 7. Footer Watermark
    ctx.save();
    ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('chess.jlelia.net • Machine-Learned Stylometry via ResNet & Transformer JEPA', width / 2, 602);
    ctx.restore();

    // 8. Trigger PNG Download
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const cleanName = top.name.toLowerCase().replace(/[^a-z0-9]/g, '-');
      a.href = url;
      a.download = `chess-personality-${cleanName}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 'image/png');
  }

  function drawSvgToCanvas(ctx, svgStr, x, y, w, h) {
    return new Promise((resolve) => {
      const img = new Image();
      // Ensure SVG has proper dimensions and namespace
      let cleanSvg = svgStr;
      if (!cleanSvg.includes('xmlns=')) {
        cleanSvg = cleanSvg.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
      }

      img.onload = () => {
        // Draw dark frame backdrop
        ctx.save();
        ctx.fillStyle = '#262626';
        roundRect(ctx, x - 10, y - 10, w + 20, h + 20, 10);
        ctx.fill();
        ctx.strokeStyle = 'rgba(212, 175, 55, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.drawImage(img, x, y, w, h);
        ctx.restore();
        resolve();
      };

      img.onerror = () => {
        console.warn('SVG image drawing failed, proceeding without board.');
        resolve();
      };

      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(cleanSvg);
    });
  }

  function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = text.split(' ');
    let line = '';
    let currY = y;

    for (let n = 0; n < words.length; n++) {
      const testLine = line + words[n] + ' ';
      const metrics = ctx.measureText(testLine);
      const testWidth = metrics.width;
      if (testWidth > maxWidth && n > 0) {
        ctx.fillText(line, x, currY);
        line = words[n] + ' ';
        currY += lineHeight;
      } else {
        line = testLine;
      }
    }
    ctx.fillText(line, x, currY);
  }
});
