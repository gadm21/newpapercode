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

/* ══════════════════════════════════════════════════════════════════
   REALISTIC CSI DATA GENERATOR
   Models actual ESP32-C6 WiFi channel behaviour:
   - 52 LLTF subcarriers across 20 MHz band (802.11n)
   - Multipath fading: 3-tap Rician channel (K=6 dB)
   - Activity-induced Doppler: walking ~2 Hz, sitting ~0.3 Hz
   - Thermal noise floor at -90 dBm
   ══════════════════════════════════════════════════════════════════ */
const CSI = (() => {
  // Seeded PRNG for reproducible "real" data across page loads
  let _seed = 42;
  function rand() { _seed = (_seed * 16807 + 0) % 2147483647; return (_seed - 1) / 2147483646; }
  function randn() { const u = rand(), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // Generate a realistic multipath channel frequency response (52 subcarriers)
  // Uses 3-tap model: LoS + wall reflection + NLoS scatter
  function channelResponse(t, activityHz) {
    const N = 52;
    const amp = new Float64Array(N);
    // Rician K-factor
    const K = 4.0;
    const losAmp = Math.sqrt(K / (K + 1));
    const scatterAmp = Math.sqrt(1 / (2 * (K + 1)));

    for (let m = 0; m < N; m++) {
      const f = (m - 26) / 52; // normalised frequency
      // LoS component with slow phase drift
      const losPhase = 2 * Math.PI * (f * 3.2 + t * 0.05);
      let re = losAmp * Math.cos(losPhase);
      let im = losAmp * Math.sin(losPhase);

      // Wall reflection (delayed, attenuated)
      const wallDelay = 0.15 + 0.02 * Math.sin(t * 0.3);
      const wallAtt = 0.45;
      const wallPhase = 2 * Math.PI * (f * 52 * wallDelay + t * 0.1);
      re += wallAtt * Math.cos(wallPhase);
      im += wallAtt * Math.sin(wallPhase);

      // Human body scatter — Doppler shift from activity
      const dopplerPhase = 2 * Math.PI * (activityHz * t + f * 1.8);
      const bodyAtt = 0.25 + 0.15 * Math.sin(t * activityHz * 0.7);
      re += bodyAtt * scatterAmp * Math.cos(dopplerPhase);
      im += bodyAtt * scatterAmp * Math.sin(dopplerPhase);

      // AWGN
      re += 0.04 * randn();
      im += 0.04 * randn();

      amp[m] = Math.sqrt(re * re + im * im);
    }
    return amp;
  }

  // Generate a time series of CSI amplitude for one subcarrier
  // with activity-specific patterns (walking, sitting, standing, etc.)
  function activityTimeSeries(numSamples, activity) {
    const profiles = {
      walking:  { freq: 1.8, amp: 0.35, breath: 0.08, noise: 0.04 },
      sitting:  { freq: 0.3, amp: 0.08, breath: 0.05, noise: 0.03 },
      standing: { freq: 0.15, amp: 0.04, breath: 0.06, noise: 0.03 },
      lying:    { freq: 0.1,  amp: 0.03, breath: 0.10, noise: 0.02 },
      jumping:  { freq: 3.2,  amp: 0.50, breath: 0.06, noise: 0.05 },
      empty:    { freq: 0.0,  amp: 0.0,  breath: 0.0,  noise: 0.02 },
    };
    const p = profiles[activity] || profiles.walking;
    const series = new Float64Array(numSamples);
    const baseline = 0.65 + 0.1 * rand();

    for (let n = 0; n < numSamples; n++) {
      const t = n / 150; // 150 Hz sample rate
      let val = baseline;
      // Activity motion
      val += p.amp * Math.sin(2 * Math.PI * p.freq * t + rand() * 0.3);
      // Harmonics (natural gait has harmonics)
      if (p.freq > 0.5) val += p.amp * 0.3 * Math.sin(2 * Math.PI * p.freq * 2 * t + 0.5);
      // Breathing
      val += p.breath * Math.sin(2 * Math.PI * 0.25 * t);
      // Thermal noise
      val += p.noise * randn();
      series[n] = Math.max(0.05, val);
    }
    return series;
  }

  // Multi-subcarrier time series (52 subcarriers × numSamples)
  function multiSubcarrierSeries(numSamples, activity) {
    const data = [];
    for (let sc = 0; sc < 52; sc++) {
      _seed = 42 + sc * 137; // deterministic per-subcarrier seed
      data.push(activityTimeSeries(numSamples, activity));
    }
    return data;
  }

  // Rolling variance computation
  function rollingVariance(signal, W) {
    const N = signal.length;
    const rv = new Float64Array(N);
    for (let i = W; i < N; i++) {
      let sum = 0, sumSq = 0;
      for (let j = i - W; j < i; j++) {
        sum += signal[j];
        sumSq += signal[j] * signal[j];
      }
      const mean = sum / W;
      rv[i] = sumSq / W - mean * mean;
    }
    return rv;
  }

  return { channelResponse, activityTimeSeries, multiSubcarrierSeries, rollingVariance, rand, randn };
})();

/* ── 16. Interactive Pipeline Visualization (realistic CSI) ────── */
function initInteractivePipeline() {
  const pipelinePanel = document.getElementById('pipelineViz');
  const pipelineCanvas = document.getElementById('pipelineCanvas');
  if (!pipelinePanel || !pipelineCanvas) return;

  const ctx = pipelineCanvas.getContext('2d');
  const steps = document.querySelectorAll('.pipeline-steps .pipeline-step');
  let animFrame = null;

  // Pre-generate realistic data once
  const walkSeries = CSI.multiSubcarrierSeries(450, 'walking'); // 3 seconds @ 150 Hz
  const rawCSISnapshot = CSI.channelResponse(0.5, 1.8);         // one freq-domain snapshot

  function label(text, x, y, color, font) {
    ctx.fillStyle = color || '#5A5B66';
    ctx.font = font || '20px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, x, y);
  }

  function drawCSI(w, h) {
    // Two-panel view: time-domain received signal (top) + frequency-domain 64 subcarriers (bottom)
    let t0 = performance.now() / 1000;
    const timeSig = CSI.activityTimeSeries(600, 'walking');

    function frame() {
      const t = performance.now() / 1000 - t0 + 0.5;
      const resp = CSI.channelResponse(t, 1.8);
      ctx.clearRect(0, 0, w, h);

      const padL = 60, padR = 40;
      const plotW = w - padL - padR;

      // --- Top panel: time-domain received signal ---
      const topH = h * 0.28, topY = 10;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, topY, plotW, topH);
      label('Received Signal r(t)', padL + 80, topY + 16, '#A0A1AB', '11px Inter');

      const scrollIdx = Math.floor(t * 40) % 400;
      ctx.strokeStyle = 'rgba(75,158,255,0.7)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < 200; i++) {
        const x = padL + (i / 200) * plotW;
        const y = topY + topH / 2 + (timeSig[(scrollIdx + i) % 600] - 0.65) * topH * 1.6;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Arrow connecting panels
      const arrowX = w / 2;
      const arrowTop = topY + topH + 2, arrowBot = topY + topH + 20;
      ctx.strokeStyle = '#5A5B66'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(arrowX, arrowTop); ctx.lineTo(arrowX, arrowBot); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(arrowX - 5, arrowBot - 5); ctx.lineTo(arrowX, arrowBot); ctx.lineTo(arrowX + 5, arrowBot - 5); ctx.stroke();
      label('FFT', arrowX + 24, arrowBot - 4, '#5A5B66', '10px Inter');

      // --- Bottom panel: 64-subcarrier frequency response ---
      const botY = arrowBot + 4, botH = h - botY - 40;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, botY, plotW, botH);
      label('Channel Frequency Response |H(f)| — 64 Subcarriers', padL + 160, botY + 16, '#A0A1AB', '11px Inter');

      const barW = plotW / 64;
      for (let i = 0; i < 64; i++) {
        const isNull = i < 6 || i === 32 || i >= 59;
        const isGuard = !isNull && (i < 6 || i >= 58);
        const lltfIdx = i < 32 ? i - 6 : i - 7; // skip DC at 32
        const amp = isNull ? 0.02 : resp[Math.max(0, Math.min(lltfIdx, 51))] || 0.1;
        const barH = amp * botH * 0.7;

        if (isNull) {
          ctx.fillStyle = 'rgba(90,91,102,0.12)';
        } else {
          const grad = ctx.createLinearGradient(0, botY + botH - barH, 0, botY + botH);
          grad.addColorStop(0, 'rgba(232,112,58,0.85)');
          grad.addColorStop(1, 'rgba(232,112,58,0.1)');
          ctx.fillStyle = grad;
        }
        ctx.fillRect(padL + i * barW + 1, botY + botH - barH, barW - 2, barH);
      }

      // DC null marker
      ctx.fillStyle = 'rgba(224,82,82,0.5)';
      ctx.font = '9px Inter'; ctx.textAlign = 'center';
      ctx.fillText('DC', padL + 32 * barW + barW / 2, botY + botH + 12);

      // Guard band labels
      ctx.fillStyle = 'rgba(90,91,102,0.6)';
      ctx.font = '9px Inter';
      ctx.fillText('null', padL + 3 * barW, botY + botH + 12);
      ctx.fillText('null', padL + 61 * barW, botY + botH + 12);

      // Subcarrier index axis
      label('Subcarrier Index (0–63)', w / 2, h - 4, '#5A5B66', '12px Inter');

      animFrame = requestAnimationFrame(frame);
    }
    frame();
  }

  function drawUART(w, h) {
    let t0 = performance.now();
    function frame() {
      const elapsed = (performance.now() - t0) / 1000;
      ctx.clearRect(0, 0, w, h);
      label('UART Serial Transfer — ESP32-C6 → Host PC', w / 2, 28, '#E8E5DC');

      const padL = 50, padR = 40, midY = h / 2 - 10;
      const lineW = w - padL - padR;

      // Serial line
      ctx.strokeStyle = 'rgba(75,158,255,0.2)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padL, midY); ctx.lineTo(padL + lineW, midY);
      ctx.stroke();

      // TX/RX labels
      label('ESP32 TX', padL + 20, midY - 40, '#E8703A', 'bold 13px Inter');
      label('Host RX', padL + lineW - 20, midY - 40, '#4B9EFF', 'bold 13px Inter');

      // Animated data packets flowing left→right
      const pktW = 70, pktH = 32, gap = 30;
      const totalW = pktW + gap;
      const numPkts = Math.ceil(lineW / totalW) + 2;

      for (let i = 0; i < numPkts; i++) {
        const baseX = padL + i * totalW;
        const xOff = (elapsed * 120) % totalW; // 120 px/s flow speed
        const px = baseX - xOff;
        if (px < padL - pktW || px > padL + lineW) continue;

        // Packet with CSI data preview
        const alpha = Math.min(1, Math.min(px - padL, padL + lineW - px) / 40);
        if (alpha <= 0) continue;

        ctx.fillStyle = `rgba(26,27,37,${0.9 * alpha})`;
        ctx.strokeStyle = `rgba(232,112,58,${0.5 * alpha})`;
        ctx.lineWidth = 1.5;
        const rr = 6;
        const ry = midY - pktH / 2;
        ctx.beginPath();
        ctx.moveTo(px + rr, ry);
        ctx.lineTo(px + pktW - rr, ry);
        ctx.arcTo(px + pktW, ry, px + pktW, ry + rr, rr);
        ctx.lineTo(px + pktW, ry + pktH - rr);
        ctx.arcTo(px + pktW, ry + pktH, px + pktW - rr, ry + pktH, rr);
        ctx.lineTo(px + rr, ry + pktH);
        ctx.arcTo(px, ry + pktH, px, ry + pktH - rr, rr);
        ctx.lineTo(px, ry + rr);
        ctx.arcTo(px, ry, px + rr, ry, rr);
        ctx.closePath();
        ctx.fill(); ctx.stroke();

        // Mini CSI bars inside packet
        const miniN = 12;
        const miniW = (pktW - 10) / miniN;
        for (let j = 0; j < miniN; j++) {
          const amp = walkSeries[j * 4][((i * 7 + j * 3) % 300)];
          const bH = amp * 14;
          ctx.fillStyle = `rgba(232,112,58,${0.6 * alpha})`;
          ctx.fillRect(px + 5 + j * miniW, midY + pktH / 2 - 4 - bH, miniW - 1.5, bH);
        }

        ctx.fillStyle = `rgba(160,161,171,${0.7 * alpha})`;
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(`CSI[${(i * 7) % 450}]`, px + pktW / 2, midY - pktH / 2 - 4);
      }

      // Bandwidth label
      label('115200 baud · ~100 pkts/s', w / 2, h - 16, '#5A5B66', '12px Inter');

      animFrame = requestAnimationFrame(frame);
    }
    frame();
  }

  function drawCSV(w, h) {
    ctx.clearRect(0, 0, w, h);
    label('CSV Storage — Actual ESP32-C6 Output Format', w / 2, 28, '#E8E5DC');

    const padL = 16, padR = 16, topY = 44;
    // Columns matching real CSV: type, seq, mac, rssi, ..., data
    const cols = [
      { name: 'type',  w: 0.10 },
      { name: 'seq',   w: 0.08 },
      { name: 'mac',   w: 0.14 },
      { name: 'rssi',  w: 0.06 },
      { name: 'rate',  w: 0.05 },
      { name: 'ch',    w: 0.04 },
      { name: 'len',   w: 0.05 },
      { name: 'data  [I₀,Q₀, I₁,Q₁, ... I₆₃,Q₆₃]', w: 0.48 },
    ];
    const totalW = w - padL - padR;
    const rowH = 22;

    // Compute column positions
    let cx = padL;
    const colPos = cols.map(c => { const x = cx; cx += c.w * totalW; return { x, w: c.w * totalW }; });

    // Header row
    ctx.fillStyle = '#22243A';
    ctx.fillRect(padL, topY, totalW, rowH + 2);
    ctx.fillStyle = '#E8703A';
    ctx.font = 'bold 10px JetBrains Mono';
    ctx.textAlign = 'center';
    cols.forEach((c, i) => {
      ctx.fillText(c.name, colPos[i].x + colPos[i].w / 2, topY + 15);
    });

    // Data rows with realistic values
    const seqNums = [513009, 514387, 515812, 517290, 518601, 520043];
    const rssiVals = [-57, -58, -57, -59, -57, -58];
    const numRows = Math.min(6, Math.floor((h - topY - rowH - 30) / rowH));

    for (let r = 0; r < numRows; r++) {
      const bgAlpha = r % 2 === 0 ? 0.35 : 0.18;
      const rY = topY + rowH + 2 + r * rowH;
      ctx.fillStyle = `rgba(18,19,26,${bgAlpha})`;
      ctx.fillRect(padL, rY, totalW, rowH);

      ctx.font = '10px JetBrains Mono';
      ctx.textAlign = 'center';

      // type
      ctx.fillStyle = '#A0A1AB';
      ctx.fillText('CSI_DATA', colPos[0].x + colPos[0].w / 2, rY + 15);
      // seq
      ctx.fillText(String(seqNums[r]), colPos[1].x + colPos[1].w / 2, rY + 15);
      // mac
      ctx.fillStyle = '#5A5B66';
      ctx.fillText('1a:00:00:00:00:00', colPos[2].x + colPos[2].w / 2, rY + 15);
      // rssi
      ctx.fillStyle = '#4B9EFF';
      ctx.fillText(String(rssiVals[r]), colPos[3].x + colPos[3].w / 2, rY + 15);
      // rate
      ctx.fillStyle = '#A0A1AB';
      ctx.fillText('11', colPos[4].x + colPos[4].w / 2, rY + 15);
      // ch
      ctx.fillText('6', colPos[5].x + colPos[5].w / 2, rY + 15);
      // len
      ctx.fillText('128', colPos[6].x + colPos[6].w / 2, rY + 15);

      // data column: show truncated I/Q array with actual values
      const dataX = colPos[7].x + 4;
      const dataW = colPos[7].w - 8;
      // Generate some realistic I/Q values
      const iqStr = [];
      for (let j = 0; j < 4; j++) {
        const I = Math.round((walkSeries[j][r * 30 + 10] - 0.6) * 30);
        const Q = Math.round((walkSeries[j + 4][r * 30 + 10] - 0.6) * 30);
        iqStr.push(`${I},${Q}`);
      }
      ctx.fillStyle = '#E8703A';
      ctx.textAlign = 'left';
      ctx.fillText(`[0,0, ${iqStr.join(', ')}, ... ]`, dataX, rY + 15);
    }

    // Column separator lines
    ctx.strokeStyle = 'rgba(90,91,102,0.15)';
    ctx.lineWidth = 1;
    colPos.forEach(cp => {
      ctx.beginPath();
      ctx.moveTo(cp.x, topY);
      ctx.lineTo(cp.x, topY + rowH + 2 + numRows * rowH);
      ctx.stroke();
    });

    label('1 row = 1 CSI packet · 128 values = 64 subcarriers × (I, Q)', w / 2, h - 10, '#5A5B66', '11px Inter');
  }

  function drawLLTF(w, h) {
    let t0 = performance.now() / 1000;
    function frame() {
      const t = performance.now() / 1000 - t0 + 0.5;
      const resp = CSI.channelResponse(t, 1.8);
      ctx.clearRect(0, 0, w, h);

      const padL = 50, padR = 30;
      const plotW = w - padL - padR;
      const panelH = (h - 60) * 0.42;

      // --- Top panel: All 64 subcarriers (before) ---
      const topY = 10;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, topY, plotW, panelH);
      label('Before: 64 Raw Subcarriers', padL + 90, topY + 16, '#A0A1AB', '11px Inter');

      const barW64 = plotW / 64;
      for (let i = 0; i < 64; i++) {
        const isNull = i < 6 || i === 32 || i >= 59;
        const lltfIdx = i < 32 ? i - 6 : i - 7;
        const amp = isNull ? 0.02 : resp[Math.max(0, Math.min(lltfIdx, 51))] || 0.1;
        const barH = amp * panelH * 0.65;

        if (isNull) {
          // Null/guard/DC — draw with red-ish strikethrough
          ctx.fillStyle = 'rgba(224,82,82,0.25)';
          ctx.fillRect(padL + i * barW64 + 1, topY + panelH - barH, barW64 - 2, barH);
          // X overlay
          ctx.strokeStyle = 'rgba(224,82,82,0.5)';
          ctx.lineWidth = 1;
          const bx = padL + i * barW64, by = topY + panelH - Math.max(barH, 8);
          ctx.beginPath();
          ctx.moveTo(bx + 1, by); ctx.lineTo(bx + barW64 - 1, topY + panelH);
          ctx.moveTo(bx + barW64 - 1, by); ctx.lineTo(bx + 1, topY + panelH);
          ctx.stroke();
        } else {
          ctx.fillStyle = 'rgba(160,161,171,0.5)';
          ctx.fillRect(padL + i * barW64 + 1, topY + panelH - barH, barW64 - 2, barH);
        }
      }

      // Labels for removed sections
      ctx.fillStyle = 'rgba(224,82,82,0.6)';
      ctx.font = '9px Inter'; ctx.textAlign = 'center';
      ctx.fillText('null (0–5)', padL + 3 * barW64, topY + panelH + 12);
      ctx.fillText('DC', padL + 32 * barW64 + barW64 / 2, topY + panelH + 12);
      ctx.fillText('null (59–63)', padL + 61 * barW64, topY + panelH + 12);

      // --- Arrow between panels ---
      const arrowY1 = topY + panelH + 18, arrowY2 = arrowY1 + 18;
      const arrowX = w / 2;
      ctx.strokeStyle = '#3FD68F'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(arrowX, arrowY1); ctx.lineTo(arrowX, arrowY2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(arrowX - 5, arrowY2 - 5); ctx.lineTo(arrowX, arrowY2); ctx.lineTo(arrowX + 5, arrowY2 - 5); ctx.stroke();
      label('Remove null + DC + guard → keep 52 data subcarriers', arrowX, arrowY1 + 8, '#3FD68F', '10px Inter');

      // --- Bottom panel: 52 LLTF subcarriers (after) ---
      const botY = arrowY2 + 6;
      const botH = h - botY - 20;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, botY, plotW, botH);
      label('After: 52 LLTF Data Subcarriers', padL + 100, botY + 16, '#3FD68F', '11px Inter');

      const barW52 = plotW / 52;
      for (let i = 0; i < 52; i++) {
        const amp = resp[i];
        const barH = amp * botH * 0.65;
        const grad = ctx.createLinearGradient(0, botY + botH - barH, 0, botY + botH);
        grad.addColorStop(0, 'rgba(63,214,143,0.85)');
        grad.addColorStop(1, 'rgba(63,214,143,0.1)');
        ctx.fillStyle = grad;
        ctx.fillRect(padL + i * barW52 + 1, botY + botH - barH, barW52 - 2, barH);
      }

      // Subcarrier count annotation
      label('52 clean subcarriers ready for feature extraction', w / 2, h - 4, '#5A5B66', '11px Inter');

      animFrame = requestAnimationFrame(frame);
    }
    frame();
  }

  function drawResample(w, h) {
    let t0 = performance.now() / 1000;
    const rawSeries = CSI.activityTimeSeries(600, 'walking');

    // Pre-generate irregular sample times (variable ~80-200 Hz like real ESP32)
    const irregSamples = [];
    let tAccum = 0;
    for (let i = 0; i < 200; i++) {
      tAccum += 0.004 + CSI.rand() * 0.010; // 4-14 ms gaps (~70-250 Hz)
      irregSamples.push({ t: tAccum, val: rawSeries[i] });
    }
    const totalIrregT = irregSamples[irregSamples.length - 1].t;

    function frame() {
      const elapsed = performance.now() / 1000 - t0;
      ctx.clearRect(0, 0, w, h);

      const padL = 50, padR = 30;
      const plotW = w - padL - padR;
      const panelH = (h - 64) * 0.42;

      // --- Top panel: Irregular sampling ---
      const topY = 10;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, topY, plotW, panelH);
      label('Before: Irregular Sampling (~80–200 Hz)', padL + 130, topY + 16, '#A0A1AB', '11px Inter');

      // Draw irregular timing grid to show uneven spacing
      ctx.strokeStyle = 'rgba(90,91,102,0.15)';
      ctx.lineWidth = 1;
      const visibleT = 0.6; // show 600ms window
      const scrollT = (elapsed * 0.15) % (totalIrregT - visibleT);
      for (let i = 0; i < irregSamples.length; i++) {
        const st = irregSamples[i].t - scrollT;
        if (st < 0 || st > visibleT) continue;
        const x = padL + (st / visibleT) * plotW;
        ctx.beginPath(); ctx.moveTo(x, topY + panelH - 2); ctx.lineTo(x, topY + panelH + 3); ctx.stroke();
      }

      // Connect irregular samples with line
      ctx.strokeStyle = 'rgba(160,161,171,0.4)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      let first = true;
      for (let i = 0; i < irregSamples.length; i++) {
        const st = irregSamples[i].t - scrollT;
        if (st < 0 || st > visibleT) continue;
        const x = padL + (st / visibleT) * plotW;
        const y = topY + panelH * 0.5 + (irregSamples[i].val - 0.65) * panelH * 1.2;
        if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Irregular sample dots (varying sizes to show timing jitter)
      for (let i = 0; i < irregSamples.length; i++) {
        const st = irregSamples[i].t - scrollT;
        if (st < 0 || st > visibleT) continue;
        const x = padL + (st / visibleT) * plotW;
        const y = topY + panelH * 0.5 + (irregSamples[i].val - 0.65) * panelH * 1.2;
        // Highlight gaps: larger dots where spacing is big
        const gap = i > 0 ? irregSamples[i].t - irregSamples[i - 1].t : 0.007;
        const r = gap > 0.010 ? 4 : gap > 0.007 ? 3 : 2;
        ctx.fillStyle = gap > 0.010 ? 'rgba(224,82,82,0.7)' : 'rgba(160,161,171,0.6)';
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      }

      // Gap warning annotations
      ctx.fillStyle = 'rgba(224,82,82,0.5)';
      ctx.font = '9px Inter'; ctx.textAlign = 'center';
      let gapCount = 0;
      for (let i = 1; i < irregSamples.length && gapCount < 3; i++) {
        const st = irregSamples[i].t - scrollT;
        if (st < 0.05 || st > visibleT - 0.05) continue;
        const gap = irregSamples[i].t - irregSamples[i - 1].t;
        if (gap > 0.011) {
          const x = padL + (st / visibleT) * plotW;
          ctx.fillText(`${(gap * 1000).toFixed(0)}ms`, x, topY + panelH - 4);
          gapCount++;
        }
      }

      // --- Arrow between panels ---
      const arrowY1 = topY + panelH + 6, arrowY2 = arrowY1 + 18;
      const arrowX = w / 2;
      ctx.strokeStyle = '#E8703A'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(arrowX, arrowY1); ctx.lineTo(arrowX, arrowY2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(arrowX - 5, arrowY2 - 5); ctx.lineTo(arrowX, arrowY2); ctx.lineTo(arrowX + 5, arrowY2 - 5); ctx.stroke();
      label('Interpolate → Uniform 150 Hz (Δt = 6.67 ms)', arrowX, arrowY1 + 8, '#E8703A', '10px Inter');

      // --- Bottom panel: Uniform 150 Hz ---
      const botY = arrowY2 + 6;
      const botH = h - botY - 20;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, botY, plotW, botH);
      label('After: Uniform 150 Hz (Δt = 6.67 ms)', padL + 115, botY + 16, '#E8703A', '11px Inter');

      // Uniform timing grid
      const numUniform = 90; // 90 samples = 600ms at 150 Hz
      ctx.strokeStyle = 'rgba(232,112,58,0.1)';
      ctx.lineWidth = 1;
      for (let i = 0; i < numUniform; i++) {
        const x = padL + (i / numUniform) * plotW;
        ctx.beginPath(); ctx.moveTo(x, botY + botH - 2); ctx.lineTo(x, botY + botH + 3); ctx.stroke();
      }

      // Uniform resampled line
      const scrollIdx = Math.floor(elapsed * 30) % 400;
      ctx.strokeStyle = '#E8703A';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < numUniform; i++) {
        const si = (scrollIdx + i) % 600;
        const x = padL + (i / numUniform) * plotW;
        const y = botY + botH * 0.5 + (rawSeries[si] - 0.65) * botH * 1.2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Uniform sample dots (all same size — regular)
      for (let i = 0; i < numUniform; i += 3) {
        const si = (scrollIdx + i) % 600;
        const x = padL + (i / numUniform) * plotW;
        const y = botY + botH * 0.5 + (rawSeries[si] - 0.65) * botH * 1.2;
        ctx.fillStyle = '#E8703A';
        ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill();
      }

      label('Equal spacing enables consistent feature extraction', w / 2, h - 4, '#5A5B66', '11px Inter');

      animFrame = requestAnimationFrame(frame);
    }
    frame();
  }

  const drawFns = { csi: drawCSI, uart: drawUART, csv: drawCSV, lltf: drawLLTF, resample: drawResample };

  function activate(stepKey) {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    pipelinePanel.querySelector('.process-viz-placeholder').style.display = 'none';
    pipelineCanvas.style.display = 'block';
    const pw = pipelinePanel.offsetWidth || 600;
    const w = pipelineCanvas.width  = pw * 2;
    const h = pipelineCanvas.height = 300;
    ctx.clearRect(0, 0, w, h);
    if (drawFns[stepKey]) drawFns[stepKey](w, h);
  }

  function deactivate() {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    pipelineCanvas.style.display = 'none';
    pipelinePanel.querySelector('.process-viz-placeholder').style.display = '';
  }

  steps.forEach(step => {
    step.addEventListener('mouseenter', () => {
      steps.forEach(s => s.classList.remove('active'));
      step.classList.add('active');
      activate(step.dataset.step);
    });
    step.addEventListener('mouseleave', () => {
      step.classList.remove('active');
      deactivate();
    });
  });
}

/* ── 17. Interactive Feature Visualization (realistic CSI) ────── */
function initInteractiveFeatures() {
  const featurePanel = document.getElementById('featureViz');
  const featureCanvas = document.getElementById('featureCanvas');
  if (!featurePanel || !featureCanvas) return;

  const ctx = featureCanvas.getContext('2d');
  const branches = document.querySelectorAll('.branches-row .branch');
  let animFrame = null;

  // Pre-generate realistic walking CSI (1 subcarrier, 450 samples)
  const walkSignal  = CSI.activityTimeSeries(600, 'walking');
  const sitSignal   = CSI.activityTimeSeries(600, 'sitting');
  const multiSC     = CSI.multiSubcarrierSeries(600, 'walking');

  function label(text, x, y, color, font) {
    ctx.fillStyle = color || '#5A5B66';
    ctx.font = font || '20px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, x, y);
  }

  function drawLine(data, startIdx, count, padL, topY, plotW, plotH, color, lw) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw || 2;
    ctx.beginPath();
    for (let i = 0; i < count; i++) {
      const x = padL + (i / count) * plotW;
      const y = topY + plotH - data[(startIdx + i) % data.length] * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function drawSHARP(w, h) {
    let t0 = performance.now() / 1000;
    // Pre-generate 4 subcarrier phase traces (chaotic wrapping)
    const phaseTraces = [];
    for (let sc = 0; sc < 4; sc++) {
      const trace = new Float64Array(600);
      for (let n = 0; n < 600; n++) {
        const t = n / 150;
        // Simulate wrapped phase: linear drift + activity + random offsets → wrap to [-π, π]
        const raw = t * (5 + sc * 2.3) + Math.sin(t * 1.8) * 1.5 + sc * 1.2 + Math.sin(t * 0.4 + sc) * 0.8;
        trace[n] = ((raw % (2 * Math.PI)) + Math.PI) % (2 * Math.PI) - Math.PI; // wrap to [-π, π]
      }
      phaseTraces.push(trace);
    }
    const scColors = ['rgba(155,138,251,0.6)', 'rgba(155,138,251,0.45)', 'rgba(155,138,251,0.35)', 'rgba(155,138,251,0.25)'];

    function frame() {
      const elapsed = performance.now() / 1000 - t0;
      const scrollIdx = Math.floor(elapsed * 25) % 300;
      ctx.clearRect(0, 0, w, h);

      const padL = 50, padR = 30;
      const plotW = w - padL - padR;
      const panelH = (h - 64) * 0.42;
      const N = 250;

      // --- Top panel: Raw wrapped phase (multiple subcarriers) ---
      const topY = 10;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, topY, plotW, panelH);
      label('Raw Phase ∠H(f) — 4 subcarriers (wrapped, chaotic)', padL + 150, topY + 16, '#A0A1AB', '11px Inter');

      // ±π reference lines
      ctx.strokeStyle = 'rgba(90,91,102,0.2)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      const piY_top = topY + panelH * 0.15, piY_bot = topY + panelH * 0.85;
      ctx.beginPath(); ctx.moveTo(padL, piY_top); ctx.lineTo(padL + plotW, piY_top); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(padL, piY_bot); ctx.lineTo(padL + plotW, piY_bot); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(90,91,102,0.4)'; ctx.font = '9px Inter'; ctx.textAlign = 'right';
      ctx.fillText('+π', padL - 4, piY_top + 4);
      ctx.fillText('−π', padL - 4, piY_bot + 4);

      // Draw phase traces with visible wrapping discontinuities
      phaseTraces.forEach((trace, sc) => {
        ctx.strokeStyle = scColors[sc];
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let prevY = null;
        for (let i = 0; i < N; i++) {
          const idx = (scrollIdx + i) % 600;
          const x = padL + (i / N) * plotW;
          const normPhase = trace[idx] / Math.PI; // -1 to 1
          const y = topY + panelH * 0.5 - normPhase * panelH * 0.35;
          // Break line at wrapping discontinuities
          if (prevY !== null && Math.abs(y - prevY) > panelH * 0.3) {
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
          } else {
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          }
          prevY = y;
        }
        ctx.stroke();
      });

      // --- Arrow between panels ---
      const arrowY1 = topY + panelH + 6, arrowY2 = arrowY1 + 18;
      const arrowX = w / 2;
      ctx.strokeStyle = '#9B8AFB'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(arrowX, arrowY1); ctx.lineTo(arrowX, arrowY2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(arrowX - 5, arrowY2 - 5); ctx.lineTo(arrowX, arrowY2); ctx.lineTo(arrowX + 5, arrowY2 - 5); ctx.stroke();
      label('SHARP sanitize → clean amplitude + phase', arrowX, arrowY1 + 8, '#9B8AFB', '10px Inter');

      // --- Bottom panel: Sanitized amplitude (clean, smooth) ---
      const botY = arrowY2 + 6;
      const botH = h - botY - 28;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, botY, plotW, botH);
      label('Sanitized Amplitude |H_san(n)| — smooth, activity-sensitive', padL + 168, botY + 16, '#9B8AFB', '11px Inter');

      drawLine(walkSignal, scrollIdx, N, padL, botY + 4, plotW, botH - 8, '#9B8AFB', 2.5);

      label('⏱ ~3.5s per window — requires SVD linear algebra', w / 2, h - 6, '#e05252', '11px Inter');
      animFrame = requestAnimationFrame(frame);
    }
    frame();
  }

  function drawRollVar(w, h) {
    let t0 = performance.now() / 1000;
    // Compute rolling variance for two activities to show discriminative power
    const rvWalk  = CSI.rollingVariance(walkSignal, 200);
    const rvSit   = CSI.rollingVariance(sitSignal, 200);
    const rvEmpty = CSI.rollingVariance(CSI.activityTimeSeries(600, 'empty'), 200);
    const maxRV   = Math.max(Math.max(...rvWalk), Math.max(...rvSit), 0.001);

    function frame() {
      const elapsed = performance.now() / 1000 - t0;
      const scrollIdx = Math.floor(elapsed * 25) % 300;
      ctx.clearRect(0, 0, w, h);

      const padL = 50, padR = 30;
      const plotW = w - padL - padR;
      const panelH = (h - 64) * 0.42;
      const N = 250;

      // --- Top panel: Raw CSI amplitude for walking vs sitting ---
      const topY = 10;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, topY, plotW, panelH);
      label('Raw Amplitude |H(n)| — Walking vs Sitting vs Empty', padL + 160, topY + 16, '#A0A1AB', '11px Inter');

      // Walking signal (high motion)
      drawLine(walkSignal, scrollIdx, N, padL, topY + 4, plotW, panelH - 8, 'rgba(232,112,58,0.6)', 1.5);
      // Sitting signal (low motion)
      drawLine(sitSignal, scrollIdx, N, padL, topY + 4, plotW, panelH - 8, 'rgba(75,158,255,0.6)', 1.5);

      // Top panel legend
      ctx.fillStyle = '#E8703A'; ctx.fillRect(padL + plotW - 200, topY + 8, 14, 3);
      label('Walking', padL + plotW - 158, topY + 14, '#E8703A', '10px Inter');
      ctx.fillStyle = '#4B9EFF'; ctx.fillRect(padL + plotW - 110, topY + 8, 14, 3);
      label('Sitting', padL + plotW - 68, topY + 14, '#4B9EFF', '10px Inter');

      // --- Arrow between panels ---
      const arrowY1 = topY + panelH + 6, arrowY2 = arrowY1 + 18;
      const arrowX = w / 2;
      ctx.strokeStyle = '#3FD68F'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(arrowX, arrowY1); ctx.lineTo(arrowX, arrowY2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(arrowX - 5, arrowY2 - 5); ctx.lineTo(arrowX, arrowY2); ctx.lineTo(arrowX + 5, arrowY2 - 5); ctx.stroke();
      label('σ²_W[n] = rolling variance (W=200)', arrowX, arrowY1 + 8, '#3FD68F', '10px Inter');

      // --- Bottom panel: Rolling variance output ---
      const botY = arrowY2 + 6;
      const botH = h - botY - 28;
      ctx.fillStyle = 'rgba(18,19,26,0.5)';
      ctx.fillRect(padL, botY, plotW, botH);
      label('Rolling Variance σ²₂₀₀[n] — Activities Clearly Separated', padL + 155, botY + 16, '#3FD68F', '11px Inter');

      // Walking variance (high)
      ctx.strokeStyle = '#E8703A';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const idx = (scrollIdx + i) % rvWalk.length;
        const x = padL + (i / N) * plotW;
        const y = botY + botH - 4 - (rvWalk[idx] / maxRV) * (botH - 24) * 0.85;
        if (i === 0 || rvWalk[idx] === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Sitting variance (low)
      ctx.strokeStyle = '#4B9EFF';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const idx = (scrollIdx + i) % rvSit.length;
        const x = padL + (i / N) * plotW;
        const y = botY + botH - 4 - (rvSit[idx] / maxRV) * (botH - 24) * 0.85;
        if (i === 0 || rvSit[idx] === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Empty variance (near zero)
      ctx.strokeStyle = 'rgba(90,91,102,0.5)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const idx = (scrollIdx + i) % rvEmpty.length;
        const x = padL + (i / N) * plotW;
        const y = botY + botH - 4 - (rvEmpty[idx] / maxRV) * (botH - 24) * 0.85;
        if (i === 0 || rvEmpty[idx] === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Bottom legend
      ctx.fillStyle = '#E8703A'; ctx.fillRect(padL + 8, h - 18, 14, 3);
      label('Walking (high σ²)', padL + 70, h - 12, '#E8703A', '10px Inter');
      ctx.fillStyle = '#4B9EFF'; ctx.fillRect(padL + 148, h - 18, 14, 3);
      label('Sitting (low σ²)', padL + 205, h - 12, '#4B9EFF', '10px Inter');
      ctx.fillStyle = '#5A5B66'; ctx.fillRect(padL + 280, h - 18, 14, 3);
      label('Empty (≈ 0)', padL + 326, h - 12, '#5A5B66', '10px Inter');
      label('⏱ 0.03s — 100× faster than SHARP', w - padR - 130, h - 12, '#3FD68F', '10px Inter');

      animFrame = requestAnimationFrame(frame);
    }
    frame();
  }

  function drawRawAmp(w, h) {
    let t0 = performance.now() / 1000;
    // 8 representative subcarriers, vertically stacked
    const scIndices = [0, 7, 14, 21, 28, 35, 42, 49];
    const scColors = ['#E8703A', '#FF9F6C', '#4B9EFF', '#7BC4FF', '#9B8AFB', '#3FD68F', '#FFD060', '#e05252'];
    const scLabels = ['sc 0', 'sc 7', 'sc 14', 'sc 21', 'sc 28', 'sc 35', 'sc 42', 'sc 49'];

    function frame() {
      const elapsed = performance.now() / 1000 - t0;
      const scrollIdx = Math.floor(elapsed * 25) % 300;
      ctx.clearRect(0, 0, w, h);

      const padL = 60, padR = 20;
      const plotW = w - padL - padR;
      const topY = 10;
      const totalH = h - topY - 20;
      const laneH = totalH / scIndices.length;
      const N = 250;

      // Background
      ctx.fillStyle = 'rgba(18,19,26,0.4)';
      ctx.fillRect(padL, topY, plotW, totalH);

      scIndices.forEach((sc, ci) => {
        const data = multiSC[sc];
        const laneY = topY + ci * laneH;

        // Lane separator
        if (ci > 0) {
          ctx.strokeStyle = 'rgba(90,91,102,0.1)';
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(padL, laneY); ctx.lineTo(padL + plotW, laneY); ctx.stroke();
        }

        // Subcarrier label on the left
        ctx.fillStyle = scColors[ci];
        ctx.font = '9px JetBrains Mono';
        ctx.textAlign = 'right';
        ctx.fillText(scLabels[ci], padL - 6, laneY + laneH * 0.55);

        // Draw waveform within this lane
        const centerY = laneY + laneH * 0.5;
        const amplitude = laneH * 0.35;
        ctx.strokeStyle = scColors[ci];
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        for (let i = 0; i < N; i++) {
          const idx = (scrollIdx + i) % data.length;
          const x = padL + (i / N) * plotW;
          const y = centerY + (data[idx] - 0.65) * amplitude * 5;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });

      // Title and annotation
      label('Raw Amplitude |H_m(n)| — 8 of 52 subcarriers (waterfall view)', w / 2, h - 4, '#5A5B66', '11px Inter');
      label('Each subcarrier responds differently to human motion', w / 2 + 10, topY + 14, '#A0A1AB', '10px Inter');
      animFrame = requestAnimationFrame(frame);
    }
    frame();
  }

  const drawFns = { sharp: drawSHARP, rollvar: drawRollVar, rawamp: drawRawAmp };

  function activate(key) {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    featurePanel.querySelector('.process-viz-placeholder').style.display = 'none';
    featureCanvas.style.display = 'block';
    const pw = featurePanel.offsetWidth || 600;
    const w = featureCanvas.width  = pw * 2;
    const h = featureCanvas.height = 300;
    ctx.clearRect(0, 0, w, h);
    if (drawFns[key]) drawFns[key](w, h);
  }

  function deactivate() {
    if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
    featureCanvas.style.display = 'none';
    featurePanel.querySelector('.process-viz-placeholder').style.display = '';
  }

  branches.forEach(branch => {
    branch.addEventListener('mouseenter', () => {
      branches.forEach(b => b.classList.remove('active'));
      branch.classList.add('active');
      activate(branch.dataset.feature);
    });
    branch.addEventListener('mouseleave', () => {
      branch.classList.remove('active');
      deactivate();
    });
  });
}

/* ── 16b. Live CSI signal flow in system diagram ─────────────── */
function initLiveSignalFlow() {
  const arrows = document.querySelectorAll('.signal-arrow');
  if (!arrows.length) return;

  // Add small canvas overlays to each signal arrow for animated data particles
  arrows.forEach(arrow => {
    const canvas = document.createElement('canvas');
    canvas.width = 100; canvas.height = 30;
    canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';
    arrow.style.position = 'relative';
    arrow.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    let t = Math.random() * 100;

    function drawParticles() {
      t += 0.03;
      const cw = canvas.width, ch = canvas.height;
      ctx.clearRect(0, 0, cw, ch);

      // 3-4 travelling dots
      for (let i = 0; i < 4; i++) {
        const phase = (t + i * 1.5) % 6;
        const x = (phase / 6) * cw;
        const alpha = Math.sin((phase / 6) * Math.PI); // fade in/out
        if (alpha < 0.05) continue;
        ctx.fillStyle = `rgba(232,112,58,${alpha * 0.6})`;
        ctx.beginPath();
        ctx.arc(x, ch / 2, 2.5, 0, Math.PI * 2);
        ctx.fill();
        // Glow
        ctx.fillStyle = `rgba(232,112,58,${alpha * 0.15})`;
        ctx.beginPath();
        ctx.arc(x, ch / 2, 7, 0, Math.PI * 2);
        ctx.fill();
      }
      requestAnimationFrame(drawParticles);
    }
    drawParticles();
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

/* ── 19. Hero ambient CSI waveform ────────────────────────────── */
function initHeroCSI() {
  const canvas = document.getElementById('heroCSI');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const header = canvas.parentElement;

  function resize() {
    canvas.width  = header.offsetWidth * 2;
    canvas.height = 240;
  }
  resize();
  window.addEventListener('resize', resize);

  // Pre-generate 5 subcarrier traces with different activities
  const traces = [
    { data: CSI.activityTimeSeries(900, 'walking'),  color: 'rgba(232,112,58,', lw: 2.0 },
    { data: CSI.activityTimeSeries(900, 'sitting'),   color: 'rgba(75,158,255,', lw: 1.5 },
    { data: CSI.activityTimeSeries(900, 'standing'),  color: 'rgba(63,214,143,', lw: 1.2 },
    { data: CSI.activityTimeSeries(900, 'jumping'),   color: 'rgba(155,138,251,', lw: 1.8 },
    { data: CSI.activityTimeSeries(900, 'lying'),     color: 'rgba(232,112,58,', lw: 1.0 },
  ];

  let t0 = performance.now() / 1000;

  function frame() {
    const elapsed = performance.now() / 1000 - t0;
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const N = 400; // visible samples
    const scrollSpeed = 15; // samples per second
    const startIdx = Math.floor(elapsed * scrollSpeed) % 500;

    traces.forEach((tr, ti) => {
      const yOffset = (ti / traces.length) * h * 0.6 + h * 0.1;
      const alpha = 0.4 + 0.2 * Math.sin(elapsed * 0.5 + ti);

      ctx.strokeStyle = tr.color + alpha.toFixed(2) + ')';
      ctx.lineWidth = tr.lw;
      ctx.beginPath();

      for (let i = 0; i < N; i++) {
        const idx = (startIdx + i) % tr.data.length;
        const x = (i / N) * w;
        const y = yOffset + (tr.data[idx] - 0.6) * h * 0.5;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    requestAnimationFrame(frame);
  }
  frame();
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
  initLiveSignalFlow();
  initHeroCSI();
  renderMath();
  initSmoothLinks();
  initSectionNumbers();

  /* Ghost bars */
  buildGhostBars(document.querySelector('.tables-section'), 'pipeline');
  buildGhostBars(document.querySelector('.dashboard-section'), 'dl_conv');
});
