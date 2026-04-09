/* ============================================================
   InstaSHAP Presentation – Improvements / Phase 2 Animations
   ============================================================
   Visualizations for InstaSHAP 2.0 proposed improvements:
   - Hybrid SHAP pipeline
   - Adaptive sampling
   - Side-by-side comparison
   - Performance improvement graphs
   ============================================================ */

(function () {
  'use strict';

  const COLORS = {
    v1:       '#F59E0B',
    v2:       '#10B981',
    text:     '#F8FAFC',
    textDim:  '#94A3B8',
    grid:     'rgba(71,85,105,0.3)',
    accent:   '#06B6D4',
    input:    '#3B82F6',
    model:    '#8B5CF6',
    error:    '#EF4444',
  };

  /* ---------- Side-by-Side Comparison ---------- */
  class SideBySideViz {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.leftTitle = opts.leftTitle || 'InstaSHAP v1';
      this.rightTitle = opts.rightTitle || 'InstaSHAP v2';
      this.leftBars = opts.leftBars || [
        { label: 'Accuracy',  value: 0.65 },
        { label: 'Speed',     value: 0.95 },
        { label: 'Interact.', value: 0.20 },
        { label: 'Robust.',   value: 0.45 },
      ];
      this.rightBars = opts.rightBars || [
        { label: 'Accuracy',  value: 0.88 },
        { label: 'Speed',     value: 0.82 },
        { label: 'Interact.', value: 0.72 },
        { label: 'Robust.',   value: 0.78 },
      ];
      this.animProgress = 0;
      this.running = false;
      this._resize();
    }

    _resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.W = rect.width;
      this.H = rect.height;
      this.canvas.width = this.W * dpr;
      this.canvas.height = this.H * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    start() {
      this.animProgress = 0;
      this.running = true;
      this._loop();
    }

    stop() { this.running = false; }

    _loop() {
      if (!this.running) return;
      this.animProgress = Math.min(1, this.animProgress + 0.015);
      this._draw();
      if (this.animProgress < 1) requestAnimationFrame(() => this._loop());
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const midX = this.W / 2;
      const margin = { top: 60, bottom: 30, side: 40 };
      const barH = 22;
      const gap = 50;
      const ease = t => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      const prog = ease(this.animProgress);

      // Titles
      ctx.font = 'bold 15px Inter';
      ctx.textAlign = 'center';
      ctx.fillStyle = COLORS.v1;
      ctx.fillText(this.leftTitle, midX / 2, 30);
      ctx.fillStyle = COLORS.v2;
      ctx.fillText(this.rightTitle, midX + midX / 2, 30);

      // VS divider
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(midX, margin.top - 10);
      ctx.lineTo(midX, this.H - margin.bottom);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = 'bold 16px Inter';
      ctx.fillStyle = COLORS.accent;
      ctx.fillText('VS', midX, margin.top + 5);

      // Draw bars
      const numBars = this.leftBars.length;
      const totalH = this.H - margin.top - margin.bottom - 20;
      const barSpacing = totalH / numBars;

      for (let i = 0; i < numBars; i++) {
        const y = margin.top + 20 + barSpacing * i;
        const lb = this.leftBars[i];
        const rb = this.rightBars[i];
        const maxBarW = midX - margin.side - 50;

        // Labels (center)
        ctx.font = '11px Inter';
        ctx.fillStyle = COLORS.textDim;
        ctx.textAlign = 'center';
        ctx.fillText(lb.label, midX, y + barH / 2 + 4);

        // Left bar (grows left from center)
        const lw = lb.value * maxBarW * prog;
        const lx = midX - 40 - lw;
        ctx.fillStyle = COLORS.v1;
        this._roundRect(ctx, lx, y, lw, barH, 3);
        ctx.fill();
        if (prog > 0.5) {
          ctx.font = 'bold 10px JetBrains Mono';
          ctx.fillStyle = COLORS.text;
          ctx.textAlign = 'right';
          ctx.fillText(`${(lb.value * 100).toFixed(0)}%`, lx - 5, y + barH / 2 + 4);
        }

        // Right bar (grows right from center)
        const rw = rb.value * maxBarW * prog;
        const rx = midX + 40;
        ctx.fillStyle = COLORS.v2;
        this._roundRect(ctx, rx, y, rw, barH, 3);
        ctx.fill();
        if (prog > 0.5) {
          ctx.font = 'bold 10px JetBrains Mono';
          ctx.fillStyle = COLORS.text;
          ctx.textAlign = 'left';
          ctx.fillText(`${(rb.value * 100).toFixed(0)}%`, rx + rw + 5, y + barH / 2 + 4);
        }

        // Improvement indicator
        if (prog > 0.8 && rb.value > lb.value) {
          const improv = ((rb.value - lb.value) / lb.value * 100).toFixed(0);
          ctx.font = 'bold 9px Inter';
          ctx.fillStyle = COLORS.v2;
          ctx.textAlign = 'left';
          ctx.fillText(`+${improv}%`, rx + rw + 35, y + barH / 2 + 4);
        }
      }
    }

    _roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r);
      ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r);
      ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
    }

    destroy() { this.stop(); }
  }

  /* ---------- Adaptive Sampling Visualization ---------- */
  class AdaptiveSamplingViz {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.animProgress = 0;
      this.running = false;
      this.time = 0;
      this.points = [];
      this.importantRegions = [];

      this._generateData();
      this._resize();
    }

    _resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.W = rect.width;
      this.H = rect.height;
      this.canvas.width = this.W * dpr;
      this.canvas.height = this.H * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    _generateData() {
      // Generate sample points in feature space
      this.points = [];
      for (let i = 0; i < 80; i++) {
        this.points.push({
          x: Math.random(),
          y: Math.random(),
          importance: Math.random(),
          sampled: false,
          adaptiveSampled: false,
        });
      }

      // Define important regions (high impact)
      this.importantRegions = [
        { cx: 0.3, cy: 0.3, r: 0.15 },
        { cx: 0.7, cy: 0.6, r: 0.12 },
      ];

      // Mark adaptive sampling
      for (const p of this.points) {
        for (const region of this.importantRegions) {
          const dx = p.x - region.cx;
          const dy = p.y - region.cy;
          if (Math.sqrt(dx*dx + dy*dy) < region.r) {
            p.importance = 0.7 + Math.random() * 0.3;
            p.adaptiveSampled = true;
          }
        }
        // Random uniform sampling
        if (Math.random() < 0.2) p.sampled = true;
      }
    }

    start() {
      this.animProgress = 0;
      this.running = true;
      this._loop();
    }

    stop() { this.running = false; }

    _loop() {
      if (!this.running) return;
      this.time += 0.016;
      this.animProgress = Math.min(1, this.animProgress + 0.008);
      this._draw();
      requestAnimationFrame(() => this._loop());
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const halfW = this.W / 2 - 20;
      const margin = { top: 55, bottom: 40, left: 30, right: 20 };

      // Titles
      ctx.font = 'bold 13px Inter';
      ctx.textAlign = 'center';
      ctx.fillStyle = COLORS.v1;
      ctx.fillText('Uniform Sampling (v1)', halfW / 2 + margin.left, 25);
      ctx.fillStyle = COLORS.v2;
      ctx.fillText('Adaptive Sampling (v2)', this.W / 2 + halfW / 2 + 10, 25);

      ctx.font = '10px Inter';
      ctx.fillStyle = COLORS.textDim;
      ctx.fillText('Random sample selection', halfW / 2 + margin.left, 42);
      ctx.fillText('Focus on high-impact regions', this.W / 2 + halfW / 2 + 10, 42);

      // Draw left panel (uniform)
      this._drawPanel(ctx, margin.left, margin.top, halfW - margin.left, this.H - margin.top - margin.bottom,
                       false);

      // Divider
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(this.W / 2, margin.top);
      ctx.lineTo(this.W / 2, this.H - margin.bottom);
      ctx.strokeStyle = COLORS.grid;
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw right panel (adaptive)
      this._drawPanel(ctx, this.W / 2 + 10, margin.top, halfW - margin.right, this.H - margin.top - margin.bottom,
                       true);
    }

    _drawPanel(ctx, ox, oy, pw, ph, isAdaptive) {
      const revealedCount = Math.floor(this.points.length * this.animProgress);

      // Important regions highlight (adaptive only)
      if (isAdaptive) {
        for (const region of this.importantRegions) {
          const cx = ox + region.cx * pw;
          const cy = oy + region.cy * ph;
          const r = region.r * Math.min(pw, ph);

          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(16,185,129,0.08)';
          ctx.fill();
          ctx.strokeStyle = 'rgba(16,185,129,0.3)';
          ctx.setLineDash([3,3]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // Points
      for (let i = 0; i < Math.min(this.points.length, revealedCount); i++) {
        const p = this.points[i];
        const x = ox + p.x * pw;
        const y = oy + p.y * ph;

        const isSelected = isAdaptive ? p.adaptiveSampled : p.sampled;
        const r = isSelected ? 5 : 3;
        const alpha = isSelected ? 0.9 : 0.25;
        const color = isSelected ? (isAdaptive ? COLORS.v2 : COLORS.v1) : COLORS.grid;

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = this._hexToRgba(color, alpha);
        ctx.fill();

        if (isSelected) {
          ctx.shadowColor = color;
          ctx.shadowBlur = 6;
          ctx.fill();
          ctx.shadowBlur = 0;
        }
      }
    }

    _hexToRgba(hex, a) {
      if (hex.startsWith('rgba')) return hex;
      const r = parseInt(hex.slice(1,3),16);
      const g = parseInt(hex.slice(3,5),16);
      const b = parseInt(hex.slice(5,7),16);
      return `rgba(${r},${g},${b},${a})`;
    }

    destroy() { this.stop(); }
  }

  /* ---------- Improvement Radar Chart ---------- */
  class RadarChart {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.axes = opts.axes || ['Accuracy', 'Speed', 'Interactions', 'Robustness', 'Scalability'];
      this.v1Values = opts.v1Values || [0.6, 0.95, 0.2, 0.4, 0.7];
      this.v2Values = opts.v2Values || [0.85, 0.8, 0.7, 0.75, 0.85];
      this.animProgress = 0;
      this.running = false;
      this._resize();
    }

    _resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.W = rect.width;
      this.H = rect.height;
      this.canvas.width = this.W * dpr;
      this.canvas.height = this.H * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    start() {
      this.animProgress = 0;
      this.running = true;
      this._loop();
    }

    stop() { this.running = false; }

    _loop() {
      if (!this.running) return;
      this.animProgress = Math.min(1, this.animProgress + 0.012);
      this._draw();
      if (this.animProgress < 1) requestAnimationFrame(() => this._loop());
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const cx = this.W / 2;
      const cy = this.H / 2 + 10;
      const R = Math.min(this.W, this.H) * 0.35;
      const n = this.axes.length;
      const ease = t => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      const prog = ease(this.animProgress);

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('InstaSHAP v1 vs v2 Capability Radar', cx, 22);

      // Grid rings
      for (let ring = 1; ring <= 5; ring++) {
        const r = R * ring / 5;
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
          const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
          const x = cx + Math.cos(angle) * r;
          const y = cy + Math.sin(angle) * r;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }

      // Axis lines & labels
      for (let i = 0; i < n; i++) {
        const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
        const x = cx + Math.cos(angle) * R;
        const y = cy + Math.sin(angle) * R;

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 0.5;
        ctx.stroke();

        // Label
        const lx = cx + Math.cos(angle) * (R + 20);
        const ly = cy + Math.sin(angle) * (R + 20);
        ctx.font = '11px Inter';
        ctx.fillStyle = COLORS.textDim;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.axes[i], lx, ly);
      }

      // v1 polygon
      ctx.beginPath();
      for (let i = 0; i <= n; i++) {
        const idx = i % n;
        const angle = (idx / n) * Math.PI * 2 - Math.PI / 2;
        const r = R * this.v1Values[idx] * prog;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(245,158,11,0.15)';
      ctx.fill();
      ctx.strokeStyle = COLORS.v1;
      ctx.lineWidth = 2;
      ctx.stroke();

      // v2 polygon
      ctx.beginPath();
      for (let i = 0; i <= n; i++) {
        const idx = i % n;
        const angle = (idx / n) * Math.PI * 2 - Math.PI / 2;
        const r = R * this.v2Values[idx] * prog;
        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(16,185,129,0.15)';
      ctx.fill();
      ctx.strokeStyle = COLORS.v2;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Dots
      for (let i = 0; i < n; i++) {
        const angle = (i / n) * Math.PI * 2 - Math.PI / 2;

        // v1
        const r1 = R * this.v1Values[i] * prog;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(angle)*r1, cy + Math.sin(angle)*r1, 4, 0, Math.PI*2);
        ctx.fillStyle = COLORS.v1;
        ctx.fill();

        // v2
        const r2 = R * this.v2Values[i] * prog;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(angle)*r2, cy + Math.sin(angle)*r2, 4, 0, Math.PI*2);
        ctx.fillStyle = COLORS.v2;
        ctx.fill();
      }

      // Legend
      ctx.textBaseline = 'alphabetic';
      const legY = this.H - 15;
      ctx.font = '11px Inter';
      ctx.textAlign = 'left';
      ctx.fillStyle = COLORS.v1;
      ctx.fillRect(cx - 120, legY - 8, 12, 12);
      ctx.fillText('v1 (Current)', cx - 104, legY + 2);
      ctx.fillStyle = COLORS.v2;
      ctx.fillRect(cx + 20, legY - 8, 12, 12);
      ctx.fillText('v2 (Proposed)', cx + 36, legY + 2);
    }

    destroy() { this.stop(); }
  }

  /* ---------- Expose ---------- */
  window.SideBySideViz = SideBySideViz;
  window.AdaptiveSamplingViz = AdaptiveSamplingViz;
  window.RadarChart = RadarChart;

})();
