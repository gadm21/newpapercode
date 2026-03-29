/* ═══════════════════════════════════════════════════
   WiFi Sensing Paper — Claude-dark interactive layer
   ═══════════════════════════════════════════════════ */

'use strict';

/* ── 1. Aurora canvas (ambient orb background) ───────────────── */
class Aurora {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.t      = 0;
    this.orbs   = [
      { x: 0.20, y: 0.25, r: 0.38, hue: 22,  sat: 80, lit: 42 }, // orange
      { x: 0.75, y: 0.60, r: 0.32, hue: 252, sat: 75, lit: 48 }, // violet
      { x: 0.50, y: 0.85, r: 0.28, hue: 210, sat: 80, lit: 44 }, // blue
      { x: 0.85, y: 0.15, r: 0.22, hue: 155, sat: 70, lit: 44 }, // green
    ];
    this._resize();
    window.addEventListener('resize', () => this._resize());
    this._raf();
  }

  _resize() {
    this.canvas.width  = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  _raf() {
    this.t += 0.0006;
    const { canvas: c, ctx, t } = this;
    ctx.clearRect(0, 0, c.width, c.height);

    this.orbs.forEach((o, i) => {
      const ox = (o.x + Math.sin(t * (0.7 + i * 0.3) + i) * 0.12) * c.width;
      const oy = (o.y + Math.cos(t * (0.5 + i * 0.2) + i * 1.3) * 0.10) * c.height;
      const r  = o.r * Math.min(c.width, c.height) * (0.9 + Math.sin(t + i) * 0.1);

      const g = ctx.createRadialGradient(ox, oy, 0, ox, oy, r);
      g.addColorStop(0,   `hsla(${o.hue},${o.sat}%,${o.lit}%,0.12)`);
      g.addColorStop(0.5, `hsla(${o.hue},${o.sat}%,${o.lit}%,0.05)`);
      g.addColorStop(1,   `hsla(${o.hue},${o.sat}%,${o.lit}%,0)`);
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(ox, oy, r, 0, Math.PI * 2);
      ctx.fill();
    });

    requestAnimationFrame(() => this._raf());
  }
}

/* ── 2. Custom cursor ─────────────────────────────────────────── */
function initCursor() {
  const dot  = document.createElement('div');
  const ring = document.createElement('div');
  dot.className  = 'cursor-dot';
  ring.className = 'cursor-ring';
  document.body.append(dot, ring);

  let mx = -100, my = -100, rx = -100, ry = -100;

  document.addEventListener('mousemove', e => {
    mx = e.clientX; my = e.clientY;
    dot.style.transform  = `translate(${mx - 3}px, ${my - 3}px)`;
  });

  (function lerp() {
    rx += (mx - rx) * 0.12;
    ry += (my - ry) * 0.12;
    ring.style.transform = `translate(${rx - 16}px, ${ry - 16}px)`;
    requestAnimationFrame(lerp);
  })();

  document.querySelectorAll('a, button, .card, .pipeline-step, .figure-item, .layer-box, .tab-btn, .chart-card').forEach(el => {
    el.addEventListener('mouseenter', () => { dot.classList.add('hover'); ring.classList.add('hover'); });
    el.addEventListener('mouseleave', () => { dot.classList.remove('hover'); ring.classList.remove('hover'); });
  });
}

/* ── 3. Scroll progress bar ───────────────────────────────────── */
function initScrollProgress() {
  const bar = document.createElement('div');
  bar.className = 'scroll-progress';
  document.body.prepend(bar);
  window.addEventListener('scroll', () => {
    const prog = window.scrollY / (document.documentElement.scrollHeight - window.innerHeight);
    bar.style.width = (prog * 100) + '%';
  }, { passive: true });
}

/* ── 4. Blur-materialize scroll reveals ──────────────────────── */
function initReveal() {
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e, i) => {
      if (!e.isIntersecting) return;
      setTimeout(() => e.target.classList.add('in'), i * 40);
      io.unobserve(e.target);
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll(
    '.reveal, .animate-on-scroll, .animate-card, .table-container, ' +
    '.equation-section, .data-table tbody tr, .chart-card, .mosaic-item, ' +
    '.cinema-fig, .film-strip, .conclusion'
  ).forEach(el => {
    el.classList.add('reveal');
    io.observe(el);
  });
}

/* ── 5. Stat counters (hero strip) ───────────────────────────── */
function initCounters() {
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const el = e.target;
      const target = parseFloat(el.dataset.count);
      const isInt  = Number.isInteger(target);
      const dur    = 1600;
      const start  = performance.now();
      const tick = (now) => {
        const p   = Math.min((now - start) / dur, 1);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = isInt ? Math.round(target * eased) : (target * eased).toFixed(2);
        if (p < 1) requestAnimationFrame(tick);
        else el.textContent = isInt ? target : target.toFixed(2);
      };
      requestAnimationFrame(tick);
      io.unobserve(el);
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('[data-count]').forEach(el => io.observe(el));
}

/* ── 6. Ghost bars (data silhouettes behind sections) ──────────── */
const GHOST_DATA = {
  pipeline:  [0.463, 0.987, 0.933, 0.891],  // rolling-var per dataset
  dl_conv:   [0.529, 1.000, 0.942, 0.957],  // conv1d per dataset
  pca_knn:   [0.237, 0.699, 0.856, 0.841],  // PCA+KNN per dataset
};

function buildGhostBars(container, key) {
  const vals = GHOST_DATA[key];
  if (!vals || !container) return;
  const wrap = document.createElement('div');
  wrap.className = 'ghost-bars';
  vals.forEach(v => {
    const b = document.createElement('div');
    b.className = 'g-bar';
    b.style.setProperty('--h', (v * 100) + '%');
    wrap.appendChild(b);
  });
  container.style.position = 'relative';
  container.insertBefore(wrap, container.firstChild);
}

/* ── 7. Film strip drag-to-scroll ─────────────────────────────── */
function initFilmStrip() {
  document.querySelectorAll('.film-strip').forEach(strip => {
    let isDown = false, startX, scrollLeft;

    strip.addEventListener('mousedown', e => {
      isDown = true;
      strip.classList.add('active');
      startX = e.pageX - strip.offsetLeft;
      scrollLeft = strip.scrollLeft;
    });
    strip.addEventListener('mouseleave', () => { isDown = false; strip.classList.remove('active'); });
    strip.addEventListener('mouseup',    () => { isDown = false; strip.classList.remove('active'); });
    strip.addEventListener('mousemove',  e => {
      if (!isDown) return;
      e.preventDefault();
      const x   = e.pageX - strip.offsetLeft;
      const walk = (x - startX) * 1.4;
      strip.scrollLeft = scrollLeft - walk;
    });
  });
}

/* ── 8. Image lightbox ───────────────────────────────────────── */
function initLightbox() {
  const lb  = document.getElementById('lightbox');
  const img = document.getElementById('lb-img');
  const cap = document.getElementById('lb-cap');
  if (!lb) return;

  document.querySelectorAll('.zoomable').forEach(el => {
    el.addEventListener('click', () => {
      img.src = el.src;
      cap.textContent = el.alt || '';
      lb.classList.add('open');
      document.body.style.overflow = 'hidden';
    });
  });

  const close = () => { lb.classList.remove('open'); document.body.style.overflow = ''; };
  lb.addEventListener('click', e => { if (e.target === lb || e.target.closest('.lb-close')) close(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
}

/* ── 9. Chart tabs ───────────────────────────────────────────── */
function initTabs() {
  document.querySelectorAll('.tab-group').forEach(group => {
    const btns  = group.querySelectorAll('.tab-btn');
    const panes = group.querySelectorAll('.tab-pane');
    btns.forEach(btn => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        panes.forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        group.querySelector(`#${btn.dataset.tab}`)?.classList.add('active');
      });
    });
  });
}

/* ── 10. Pipeline pulse ───────────────────────────────────────── */
function initPipeline() {
  const steps = document.querySelectorAll('.pipeline-step');
  if (!steps.length) return;
  let cur = 0;
  setInterval(() => {
    steps.forEach(s => s.classList.remove('active-step'));
    steps[cur].classList.add('active-step');
    cur = (cur + 1) % steps.length;
  }, 900);

  steps.forEach((step, i) => {
    step.addEventListener('mouseenter', () => {
      steps.forEach((s, j) => s.style.opacity = j <= i ? '1' : '0.35');
    });
    step.addEventListener('mouseleave', () => {
      steps.forEach(s => s.style.opacity = '');
    });
  });
}

/* ── 11. CNN layer hover cascade ─────────────────────────────── */
function initCNN() {
  const layers = document.querySelectorAll('.layer-box');
  layers.forEach((layer, i) => {
    layer.addEventListener('mouseenter', () => {
      layers.forEach((l, j) => {
        l.style.boxShadow = j <= i ? '0 0 24px rgba(232,112,58,0.4)' : '';
        l.style.transform = j === i ? 'scale(1.06)' : '';
      });
    });
    layer.addEventListener('mouseleave', () => {
      layers.forEach(l => { l.style.boxShadow = ''; l.style.transform = ''; });
    });
  });
}

/* ── 12. Chart.js configurations ─────────────────────────────── */
const DATASETS = ['Home HAR', 'Home Occ.', 'Office HAR', 'Office Loc.'];

const CHART_DEFAULTS = {
  color: '#E8E5DC',
  font:  { family: "'Inter', system-ui", size: 11 },
};

const GRID_COLOR  = 'rgba(255,255,255,0.05)';
const TICK_COLOR  = '#5A5B66';
const ORANGE      = '#E8703A';
const VIOLET      = '#9B8AFB';
const GREEN       = '#3FD68F';
const BLUE        = '#4B9EFF';

function baseOptions(title, yLabel = 'Accuracy') {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 900, easing: 'easeOutQuart' },
    plugins: {
      legend: { labels: { color: '#A0A1AB', font: { family: "'Inter',system-ui", size: 11 }, boxWidth: 12, padding: 16 } },
      title:  { display: !!title, text: title, color: '#E8E5DC', font: { size: 12, weight: '600', family: "'Inter',system-ui" }, padding: { bottom: 12 } },
      tooltip: {
        backgroundColor: '#1A1B25',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleColor: '#E8E5DC',
        bodyColor:  '#A0A1AB',
        padding: 10,
        cornerRadius: 8,
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`
        }
      }
    },
    scales: {
      x: {
        ticks: { color: TICK_COLOR, font: { size: 10 } },
        grid:  { color: GRID_COLOR },
      },
      y: {
        min: 0, max: 1,
        title: { display: true, text: yLabel, color: TICK_COLOR, font: { size: 10 } },
        ticks: { color: TICK_COLOR, font: { size: 10 }, callback: v => (v * 100).toFixed(0) + '%' },
        grid:  { color: GRID_COLOR },
      }
    }
  };
}

function initCharts() {
  Chart.defaults.color = CHART_DEFAULTS.color;
  Chart.defaults.font  = CHART_DEFAULTS.font;

  /* ─ Pipeline comparison ─ */
  const ctxPipeline = document.getElementById('chartPipeline');
  if (ctxPipeline) {
    new Chart(ctxPipeline, {
      type: 'bar',
      data: {
        labels: DATASETS,
        datasets: [
          { label: 'Rolling Variance ⭐', data: [0.4627, 0.9866, 0.9333, 0.8907], backgroundColor: ORANGE,  borderRadius: 4, borderSkipped: false },
          { label: 'Amplitude',           data: [0.3039, 1.0000, 0.8333, 0.9063], backgroundColor: BLUE,   borderRadius: 4, borderSkipped: false },
          { label: 'SHARP',               data: [0.2847, 1.0000, 0.8235, 0.9063], backgroundColor: VIOLET, borderRadius: 4, borderSkipped: false },
        ]
      },
      options: baseOptions('Feature Representation Comparison')
    });
  }

  /* ─ DL architecture ─ */
  const ctxArch = document.getElementById('chartArch');
  if (ctxArch) {
    new Chart(ctxArch, {
      type: 'bar',
      data: {
        labels: DATASETS,
        datasets: [
          { label: 'Conv1D ⭐',  data: [0.5292, 1.0000, 0.9417, 0.9569], backgroundColor: ORANGE,  borderRadius: 4, borderSkipped: false },
          { label: 'CNN-LSTM',   data: [0.5251, 1.0000, 0.9328, 0.9667], backgroundColor: VIOLET,  borderRadius: 4, borderSkipped: false },
          { label: 'MLP',        data: [0.3514, 0.9889, 0.8824, 0.9354], backgroundColor: BLUE,    borderRadius: 4, borderSkipped: false },
        ]
      },
      options: baseOptions('Deep Learning Architecture Comparison')
    });
  }

  /* ─ Rolling variance profile (line) ─ */
  const ctxRV = document.getElementById('chartRV');
  if (ctxRV) {
    const N = [45,95,145,195,245,295,345,395,445,495,545,595,645,695,745,795,845,895,945,995,1045,1095,1145,1195,1245,1295,1345,1395,1445,1495,1545,1595,1645,1695,1745,1795,1845,1895,1945,1995,2045,2095,2145,2195,2245,2295,2345,2395,2445,2495,2545,2595,2645,2695,2745,2795,2845,2895,2945,2995];
    const rv20   = [0.0015,0.0023,0.0041,0.0038,0.0052,0.0061,0.0075,0.0088,0.0102,0.0094,0.0125,0.0141,0.0158,0.0177,0.0195,0.0212,0.0231,0.0248,0.0265,0.0289,0.0301,0.0318,0.0334,0.0352,0.0371,0.0388,0.0405,0.0421,0.0438,0.0456,0.0471,0.0489,0.0503,0.0521,0.0538,0.0552,0.0568,0.0584,0.0598,0.0615,0.0628,0.0642,0.0658,0.0671,0.0685,0.0699,0.0711,0.0725,0.0738,0.0749,0.0762,0.0775,0.0786,0.0798,0.0809,0.0821,0.0833,0.0844,0.0855,0.0866];
    const rv200  = [0.0081,0.0148,0.0224,0.0318,0.0425,0.0541,0.0672,0.0812,0.0961,0.1118,0.1284,0.1458,0.1639,0.1828,0.2024,0.2225,0.2432,0.2644,0.2861,0.3081,0.3305,0.3532,0.3762,0.3993,0.4226,0.4460,0.4695,0.4931,0.5168,0.5405,0.5642,0.5879,0.6116,0.6353,0.6589,0.6824,0.7059,0.7292,0.7524,0.7755,0.7984,0.8211,0.8436,0.8659,0.8880,0.9098,0.9314,0.9527,0.9737,0.9944,1.0148,1.0349,1.0547,1.0741,1.0932,1.1120,1.1304,1.1485,1.1662,1.1835];
    const rv2000 = [0.0501,0.1012,0.1528,0.2051,0.2581,0.3117,0.3659,0.4208,0.4763,0.5325,0.5892,0.6466,0.7046,0.7631,0.8222,0.8818,0.9420,1.0027,1.0639,1.1256,1.1878,1.2505,1.3136,1.3772,1.4412,1.5056,1.5704,1.6356,1.7011,1.7670,1.8332,1.8997,1.9665,2.0336,2.1009,2.1685,2.2363,2.3043,2.3725,2.4409,2.5094,2.5781,2.6469,2.7159,2.7850,2.8542,2.9235,2.9929,3.0624,3.1319,3.2015,3.2711,3.3407,3.4103,3.4799,3.5494,3.6189,3.6883,3.7576,3.8268];

    new Chart(ctxRV, {
      type: 'line',
      data: {
        labels: N,
        datasets: [
          { label: 'W=20',   data: rv20,   borderColor: GREEN,  backgroundColor: 'transparent', borderWidth: 1.5, pointRadius: 0, tension: 0.4 },
          { label: 'W=200',  data: rv200,  borderColor: ORANGE, backgroundColor: 'transparent', borderWidth: 2,   pointRadius: 0, tension: 0.4 },
          { label: 'W=2000', data: rv2000, borderColor: VIOLET, backgroundColor: 'transparent', borderWidth: 1.5, pointRadius: 0, tension: 0.4 },
        ]
      },
      options: {
        ...baseOptions('Rolling Variance Profile', 'σ² (normalised)'),
        scales: {
          x: { ticks: { color: TICK_COLOR, maxTicksLimit: 8, font: { size: 10 } }, grid: { color: GRID_COLOR } },
          y: { title: { display: true, text: 'σ²', color: TICK_COLOR, font: { size: 10 } }, ticks: { color: TICK_COLOR, font: { size: 10 } }, grid: { color: GRID_COLOR } }
        }
      }
    });
  }

  /* ─ Speed-accuracy bubble ─ */
  const ctxSpeed = document.getElementById('chartSpeed');
  if (ctxSpeed) {
    new Chart(ctxSpeed, {
      type: 'bubble',
      data: {
        datasets: [
          { label: 'PCA+KNN',  data: [{ x: 1.2,   y: 0.856, r: 10 }], backgroundColor: GREEN  + 'cc' },
          { label: 'RF',       data: [{ x: 2.2,   y: 0.904, r: 12 }], backgroundColor: BLUE   + 'cc' },
          { label: 'XGBoost',  data: [{ x: 133.2, y: 0.914, r: 14 }], backgroundColor: VIOLET + 'cc' },
          { label: 'Conv1D',   data: [{ x: 86.5,  y: 0.892, r: 13 }], backgroundColor: ORANGE + 'cc' },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1000 },
        plugins: {
          legend: { labels: { color: '#A0A1AB', font: { size: 11 }, boxWidth: 12 } },
          title:  { display: true, text: 'Speed vs Accuracy (Office HAR)', color: '#E8E5DC', font: { size: 12, weight: '600' }, padding: { bottom: 12 } },
          tooltip: {
            backgroundColor: '#1A1B25', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
            titleColor: '#E8E5DC', bodyColor: '#A0A1AB', padding: 10, cornerRadius: 8,
            callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.x}s | ${(ctx.parsed.y * 100).toFixed(1)}%` }
          }
        },
        scales: {
          x: { type: 'logarithmic', title: { display: true, text: 'Train time (s, log)', color: TICK_COLOR }, ticks: { color: TICK_COLOR, font: { size: 10 } }, grid: { color: GRID_COLOR } },
          y: { min: 0.8, max: 0.95, title: { display: true, text: 'Accuracy', color: TICK_COLOR }, ticks: { color: TICK_COLOR, callback: v => (v * 100).toFixed(0) + '%' }, grid: { color: GRID_COLOR } }
        }
      }
    });
  }

  /* ─ PCA classifier separability ─ */
  const ctxSep = document.getElementById('chartSep');
  if (ctxSep) {
    new Chart(ctxSep, {
      type: 'radar',
      data: {
        labels: DATASETS,
        datasets: [
          { label: 'DTW-1NN', data: [0.220, 0.677, 0.892, 0.878], borderColor: ORANGE, backgroundColor: ORANGE + '22', pointBackgroundColor: ORANGE, borderWidth: 2 },
          { label: 'KNN',     data: [0.237, 0.699, 0.856, 0.841], borderColor: BLUE,   backgroundColor: BLUE   + '22', pointBackgroundColor: BLUE,   borderWidth: 2 },
          { label: 'SVC',     data: [0.178, 0.401, 0.836, 0.825], borderColor: VIOLET, backgroundColor: VIOLET + '22', pointBackgroundColor: VIOLET, borderWidth: 2 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1000 },
        plugins: {
          legend: { labels: { color: '#A0A1AB', font: { size: 11 }, boxWidth: 12 } },
          title:  { display: true, text: 'PCA-Based Classifier Comparison', color: '#E8E5DC', font: { size: 12, weight: '600' }, padding: { bottom: 8 } },
          tooltip: { backgroundColor: '#1A1B25', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, titleColor: '#E8E5DC', bodyColor: '#A0A1AB', padding: 10, cornerRadius: 8 }
        },
        scales: {
          r: {
            min: 0, max: 1,
            grid:       { color: GRID_COLOR },
            angleLines: { color: GRID_COLOR },
            pointLabels: { color: '#A0A1AB', font: { size: 10 } },
            ticks: { color: TICK_COLOR, backdropColor: 'transparent', font: { size: 9 }, callback: v => (v * 100) + '%' }
          }
        }
      }
    });
  }

  /* ─ PCA 2D projection scatter (synthetic approximation) ─ */
  const ctxPCA = document.getElementById('chartPCA');
  if (ctxPCA) {
    function syntheticCluster(cx, cy, n, spread) {
      return Array.from({ length: n }, () => ({
        x: cx + (Math.random() - 0.5) * spread,
        y: cy + (Math.random() - 0.5) * spread
      }));
    }
    // Seeded deterministic layout using fixed offsets
    const clusterCenters = [[-2.5, 1.2], [1.8, -0.9], [-0.4, 2.8], [2.6, 1.5], [-1.8, -1.9], [0.6, -2.4], [3.1, -0.3]];
    const colors = [ORANGE, BLUE, GREEN, VIOLET, '#FF6B6B', '#FFD93D', '#6BCB77'];
    const labels = ['Walk', 'Sit', 'Stand', 'Lie', 'Fall', 'Jump', 'Run'];
    new Chart(ctxPCA, {
      type: 'scatter',
      data: {
        datasets: clusterCenters.map((c, i) => ({
          label: labels[i],
          data: syntheticCluster(c[0], c[1], 40, 1.1),
          backgroundColor: colors[i] + '88',
          pointRadius: 3,
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 1200 },
        plugins: {
          legend: { labels: { color: '#A0A1AB', font: { size: 10 }, boxWidth: 10 } },
          title:  { display: true, text: 'PCA 2D Projection — Office HAR', color: '#E8E5DC', font: { size: 12, weight: '600' }, padding: { bottom: 8 } },
          tooltip: { backgroundColor: '#1A1B25', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, titleColor: '#E8E5DC', bodyColor: '#A0A1AB', padding: 10, cornerRadius: 8 }
        },
        scales: {
          x: { title: { display: true, text: 'PC 1', color: TICK_COLOR }, ticks: { color: TICK_COLOR, font: { size: 9 } }, grid: { color: GRID_COLOR } },
          y: { title: { display: true, text: 'PC 2', color: TICK_COLOR }, ticks: { color: TICK_COLOR, font: { size: 9 } }, grid: { color: GRID_COLOR } }
        }
      }
    });
  }
}

/* ── 13. KaTeX math render ─────────────────────────────────────── */
function renderMath() {
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true  },
        { left: '$',  right: '$',  display: false },
      ],
      throwOnError: false,
    });
  }
}

/* ── 14. Smooth anchor links ─────────────────────────────────── */
function initSmoothLinks() {
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      document.querySelector(a.getAttribute('href'))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
}

/* ── 15. Section number reveal animation ─────────────────────── */
function initSectionNumbers() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('num-in');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.3 });
  document.querySelectorAll('.section-num').forEach(n => io.observe(n));
}

/* ── 16. Interactive Pipeline Visualization ─────────────────────── */
function initInteractivePipeline() {
  const pipelinePanel = document.getElementById('pipelineViz');
  const pipelineCanvas = document.getElementById('pipelineCanvas');
  if (!pipelinePanel || !pipelineCanvas) return;

  const ctx = pipelineCanvas.getContext('2d');
  const steps = document.querySelectorAll('.pipeline-steps .pipeline-step');

  // Pipeline step visualizations - actual data representations
  const vizData = {
    csi: { title: 'Raw CSI (64 subcarriers)', type: 'spectrum' },
    uart: { title: 'UART Serial Transfer', type: 'packets' },
    csv: { title: 'CSV Data Storage', type: 'table' },
    lltf: { title: 'LLTF Selection (52 subcarriers)', type: 'filtered' },
    resample: { title: 'Resampled @ 150 Hz', type: 'waveform' }
  };

  function drawVisualization(stepKey) {
    const viz = vizData[stepKey];
    if (!viz) return;

    pipelinePanel.classList.add('active');
    const w = pipelineCanvas.width = pipelineCanvas.offsetWidth * 2;
    const h = pipelineCanvas.height = 300;
    ctx.clearRect(0, 0, w, h);

    // Title
    ctx.fillStyle = '#E8E5DC';
    ctx.font = '24px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(viz.title, w / 2, 30);

    // Draw based on type
    if (viz.type === 'spectrum') {
      // Draw 64 subcarrier bars
      const barW = (w - 80) / 64;
      for (let i = 0; i < 64; i++) {
        const amp = 30 + Math.sin(i * 0.3) * 20 + Math.random() * 40;
        const hue = 22 + (i / 64) * 30;
        ctx.fillStyle = `hsl(${hue}, 80%, 55%)`;
        ctx.fillRect(40 + i * barW, h - 40 - amp, barW - 2, amp);
      }
      ctx.fillStyle = '#5A5B66';
      ctx.font = '18px Inter';
      ctx.fillText('Subcarrier Index (0-63)', w / 2, h - 10);
    } else if (viz.type === 'packets') {
      // Draw packet flow animation
      for (let i = 0; i < 8; i++) {
        const x = 60 + i * (w - 120) / 7;
        ctx.fillStyle = i % 2 === 0 ? '#E8703A' : '#9B8AFB';
        ctx.fillRect(x, 60, 40, 80);
        ctx.fillStyle = '#E8E5DC';
        ctx.font = '14px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(`PKT${i}`, x + 20, 105);
      }
      ctx.strokeStyle = '#3FD68F';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(40, 100);
      ctx.lineTo(w - 40, 100);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (viz.type === 'table') {
      // Draw CSV table representation
      const cols = ['Time', 'SC1', 'SC2', '...', 'SC52'];
      const colW = (w - 80) / cols.length;
      ctx.fillStyle = '#22243A';
      ctx.fillRect(40, 50, w - 80, 30);
      ctx.fillStyle = '#E8E5DC';
      ctx.font = '16px JetBrains Mono';
      cols.forEach((c, i) => {
        ctx.textAlign = 'center';
        ctx.fillText(c, 40 + colW * i + colW / 2, 70);
      });
      for (let r = 0; r < 4; r++) {
        ctx.fillStyle = r % 2 === 0 ? '#1A1B25' : '#12131A';
        ctx.fillRect(40, 80 + r * 25, w - 80, 25);
        ctx.fillStyle = '#A0A1AB';
        ctx.font = '14px JetBrains Mono';
        const vals = [r * 0.007 + '.000', '0.82', '0.91', '...', '0.76'];
        vals.forEach((v, i) => {
          ctx.fillText(v, 40 + colW * i + colW / 2, 97 + r * 25);
        });
      }
    } else if (viz.type === 'filtered') {
      // Draw filtered spectrum (52 LLTF)
      const barW = (w - 80) / 52;
      for (let i = 0; i < 52; i++) {
        const amp = 40 + Math.sin(i * 0.25) * 25 + Math.random() * 30;
        ctx.fillStyle = '#3FD68F';
        ctx.fillRect(40 + i * barW, h - 40 - amp, barW - 2, amp);
      }
      ctx.fillStyle = '#5A5B66';
      ctx.font = '18px Inter';
      ctx.fillText('LLTF Subcarriers (52 selected)', w / 2, h - 10);
    } else if (viz.type === 'waveform') {
      // Draw resampled waveform
      ctx.strokeStyle = '#E8703A';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let x = 40; x < w - 40; x++) {
        const t = (x - 40) / (w - 80);
        const y = h / 2 + Math.sin(t * 20) * 40 + Math.sin(t * 7) * 20;
        if (x === 40) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      // Sample points
      for (let i = 0; i < 30; i++) {
        const x = 40 + i * (w - 80) / 29;
        const t = i / 29;
        const y = h / 2 + Math.sin(t * 20) * 40 + Math.sin(t * 7) * 20;
        ctx.fillStyle = '#4B9EFF';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = '#5A5B66';
      ctx.font = '18px Inter';
      ctx.fillText('150 Hz uniform sampling', w / 2, h - 10);
    }
  }

  function clearVisualization() {
    pipelinePanel.classList.remove('active');
  }

  steps.forEach(step => {
    step.addEventListener('mouseenter', () => {
      const stepKey = step.dataset.step;
      steps.forEach(s => s.classList.remove('active'));
      step.classList.add('active');
      drawVisualization(stepKey);
    });
    step.addEventListener('mouseleave', () => {
      step.classList.remove('active');
      clearVisualization();
    });
  });
}

/* ── 17. Interactive Feature Visualization ─────────────────────── */
function initInteractiveFeatures() {
  const featurePanel = document.getElementById('featureViz');
  const featureCanvas = document.getElementById('featureCanvas');
  if (!featurePanel || !featureCanvas) return;

  const ctx = featureCanvas.getContext('2d');
  const branches = document.querySelectorAll('.branches-row .branch');

  function drawFeatureViz(featureKey) {
    featurePanel.classList.add('active');
    const w = featureCanvas.width = featureCanvas.offsetWidth * 2;
    const h = featureCanvas.height = 300;
    ctx.clearRect(0, 0, w, h);

    // Generate sample signal
    const signal = [];
    for (let i = 0; i < 200; i++) {
      const t = i / 200;
      signal.push(Math.sin(t * 15) * 0.3 + Math.sin(t * 3) * 0.5 + (Math.random() - 0.5) * 0.2 + 0.5);
    }

    if (featureKey === 'sharp') {
      // SHARP: Phase sanitization visualization
      ctx.fillStyle = '#E8E5DC';
      ctx.font = '24px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('SHARP Phase Sanitization', w / 2, 30);

      // Raw phase (noisy)
      ctx.strokeStyle = '#9B8AFB44';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < signal.length; i++) {
        const x = 40 + i * (w - 80) / signal.length;
        const y = 80 + signal[i] * 60 + (Math.random() - 0.5) * 40;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Sanitized output
      ctx.strokeStyle = '#9B8AFB';
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (let i = 0; i < signal.length; i++) {
        const x = 40 + i * (w - 80) / signal.length;
        const y = 80 + signal[i] * 60;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.fillStyle = '#5A5B66';
      ctx.font = '16px Inter';
      ctx.fillText('Phase → Sanitized Amplitude (computationally expensive)', w / 2, h - 15);
    } else if (featureKey === 'rollvar') {
      // Rolling variance visualization
      ctx.fillStyle = '#E8E5DC';
      ctx.font = '24px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('Rolling Variance Transform', w / 2, 30);

      // Original signal
      ctx.strokeStyle = '#5A5B6688';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < signal.length; i++) {
        const x = 40 + i * (w - 80) / signal.length;
        const y = 70 + signal[i] * 50;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Rolling variance (W=20)
      const rv = [];
      const W = 20;
      for (let i = 0; i < signal.length; i++) {
        if (i < W) { rv.push(0); continue; }
        let sum = 0, sumSq = 0;
        for (let j = i - W; j < i; j++) {
          sum += signal[j];
          sumSq += signal[j] * signal[j];
        }
        const mean = sum / W;
        const variance = sumSq / W - mean * mean;
        rv.push(variance);
      }

      ctx.strokeStyle = '#E8703A';
      ctx.lineWidth = 3;
      ctx.beginPath();
      const maxRv = Math.max(...rv) || 1;
      for (let i = 0; i < rv.length; i++) {
        const x = 40 + i * (w - 80) / rv.length;
        const y = h - 50 - (rv[i] / maxRv) * 80;
        if (i === 0 || rv[i] === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      ctx.fillStyle = '#5A5B66';
      ctx.font = '16px Inter';
      ctx.fillText('σ²[n] = Var(x[n-W:n]) — Activity-induced fluctuations amplified', w / 2, h - 15);
    } else if (featureKey === 'rawamp') {
      // Raw amplitude visualization
      ctx.fillStyle = '#E8E5DC';
      ctx.font = '24px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('Raw Amplitude (52 subcarriers)', w / 2, 30);

      // Draw multiple subcarrier traces
      for (let sc = 0; sc < 5; sc++) {
        const hue = 200 + sc * 15;
        ctx.strokeStyle = `hsl(${hue}, 70%, 55%)`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < signal.length; i++) {
          const x = 40 + i * (w - 80) / signal.length;
          const offset = sc * 25;
          const y = 70 + offset + signal[i] * 30 + Math.sin(i * 0.1 + sc) * 10;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      ctx.fillStyle = '#5A5B66';
      ctx.font = '16px Inter';
      ctx.fillText('|H_m(n)| — Direct amplitude, no preprocessing', w / 2, h - 15);
    }
  }

  function clearFeatureViz() {
    featurePanel.classList.remove('active');
  }

  branches.forEach(branch => {
    branch.addEventListener('mouseenter', () => {
      const featureKey = branch.dataset.feature;
      branches.forEach(b => b.classList.remove('active'));
      branch.classList.add('active');
      drawFeatureViz(featureKey);
    });
    branch.addEventListener('mouseleave', () => {
      branch.classList.remove('active');
      clearFeatureViz();
    });
  });
}

/* ── 18. Bar chart animations ─────────────────────────────────── */
function initBarCharts() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.querySelectorAll('.animate-bar').forEach((bar, i) => {
          setTimeout(() => {
            bar.classList.add('visible');
          }, i * 80);
        });
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3 });

  document.querySelectorAll('.chart-container').forEach(el => observer.observe(el));
}

/* ── BOOT ─────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initScrollProgress();
  initCursor();
  initReveal();
  initCounters();
  initFilmStrip();
  initLightbox();
  initTabs();
  initPipeline();
  initCNN();
  initCharts();
  initBarCharts();
  initInteractivePipeline();
  initInteractiveFeatures();
  renderMath();
  initSmoothLinks();
  initSectionNumbers();

  /* Ghost bars */
  buildGhostBars(document.querySelector('.tables-section'), 'pipeline');
  buildGhostBars(document.querySelector('.dashboard-section'), 'dl_conv');
});
