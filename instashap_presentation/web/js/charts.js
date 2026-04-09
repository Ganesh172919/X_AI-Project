/* ============================================================
   InstaSHAP Presentation – Charts & Graphs
   ============================================================
   Animated charts for complexity comparison, speed benchmarks,
   accuracy trade-offs, and performance metrics.
   ============================================================ */

(function () {
  'use strict';

  const COLORS = {
    shap:     '#EF4444',
    instashap:'#10B981',
    kernel:   '#F59E0B',
    tree:     '#3B82F6',
    text:     '#F8FAFC',
    textDim:  '#94A3B8',
    grid:     'rgba(71,85,105,0.3)',
    accent:   '#06B6D4',
  };

  /* ---------- Complexity Comparison Chart ---------- */
  class ComplexityChart {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
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

      const margin = { top: 55, right: 30, bottom: 55, left: 70 };
      const cw = this.W - margin.left - margin.right;
      const ch = this.H - margin.top - margin.bottom;
      const ease = t => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      const prog = ease(this.animProgress);

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('Computational Complexity: O(2ⁿ) vs O(k·n)', this.W / 2, 22);

      // Axes
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, this.H - margin.bottom);
      ctx.lineTo(this.W - margin.right, this.H - margin.bottom);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Axis labels
      ctx.font = '11px Inter';
      ctx.fillStyle = COLORS.textDim;
      ctx.textAlign = 'center';
      ctx.fillText('Number of Features (n)', this.W / 2, this.H - 10);

      ctx.save();
      ctx.translate(15, this.H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Computation Time', 0, 0);
      ctx.restore();

      // X-axis ticks
      const features = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];
      for (let i = 0; i < features.length; i++) {
        const x = margin.left + (i / (features.length - 1)) * cw;
        ctx.font = '9px JetBrains Mono';
        ctx.fillStyle = COLORS.textDim;
        ctx.textAlign = 'center';
        ctx.fillText(features[i], x, this.H - margin.bottom + 18);

        // Grid line
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, this.H - margin.bottom);
        ctx.strokeStyle = 'rgba(71,85,105,0.15)';
        ctx.stroke();
      }

      // Exact SHAP: O(2^n)
      const maxExp = Math.pow(2, 20);
      ctx.beginPath();
      ctx.strokeStyle = COLORS.shap;
      ctx.lineWidth = 2.5;
      const pointsDrawn = Math.floor(features.length * prog);
      for (let i = 0; i <= pointsDrawn; i++) {
        const n = features[i];
        const x = margin.left + (i / (features.length - 1)) * cw;
        const val = Math.pow(2, n);
        const normVal = Math.min(1, Math.log(val) / Math.log(maxExp));
        const y = this.H - margin.bottom - normVal * ch;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // InstaSHAP: O(k*n) — nearly linear
      ctx.beginPath();
      ctx.strokeStyle = COLORS.instashap;
      ctx.lineWidth = 2.5;
      const k = 50;
      for (let i = 0; i <= pointsDrawn; i++) {
        const n = features[i];
        const x = margin.left + (i / (features.length - 1)) * cw;
        const val = k * n;
        const normVal = Math.min(1, Math.log(val + 1) / Math.log(maxExp));
        const y = this.H - margin.bottom - normVal * ch;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // KernelSHAP: O(k^2*n)
      ctx.beginPath();
      ctx.strokeStyle = COLORS.kernel;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      for (let i = 0; i <= pointsDrawn; i++) {
        const n = features[i];
        const x = margin.left + (i / (features.length - 1)) * cw;
        const val = k * k * n;
        const normVal = Math.min(1, Math.log(val + 1) / Math.log(maxExp));
        const y = this.H - margin.bottom - normVal * ch;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Legend
      const legX = margin.left + 20;
      let legY = margin.top + 10;
      const legendItems = [
        { label: 'Exact SHAP O(2ⁿ)', color: COLORS.shap },
        { label: 'KernelSHAP O(k²·n)', color: COLORS.kernel },
        { label: 'InstaSHAP O(k·n)', color: COLORS.instashap },
      ];
      for (const item of legendItems) {
        ctx.beginPath();
        ctx.moveTo(legX, legY);
        ctx.lineTo(legX + 25, legY);
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.font = '10px Inter';
        ctx.fillStyle = item.color;
        ctx.textAlign = 'left';
        ctx.fillText(item.label, legX + 30, legY + 4);
        legY += 18;
      }

      // Annotation
      if (prog > 0.7) {
        ctx.font = 'bold 11px Inter';
        ctx.fillStyle = COLORS.shap;
        ctx.textAlign = 'right';
        ctx.fillText('🔥 Explodes!', this.W - margin.right - 5, margin.top + 25);

        ctx.fillStyle = COLORS.instashap;
        ctx.fillText('✓ Manageable', this.W - margin.right - 5, this.H - margin.bottom - 30);
      }
    }

    destroy() { this.stop(); }
  }

  /* ---------- Speed Comparison Bar Chart ---------- */
  class SpeedChart {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.data = opts.data || [
        { label: 'Exact SHAP',   time: 4200, color: COLORS.shap },
        { label: 'KernelSHAP',   time: 850,  color: COLORS.kernel },
        { label: 'TreeSHAP',     time: 120,  color: COLORS.tree },
        { label: 'InstaSHAP',    time: 35,   color: COLORS.instashap },
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

      const margin = { top: 55, right: 60, bottom: 40, left: 110 };
      const cw = this.W - margin.left - margin.right;
      const ch = this.H - margin.top - margin.bottom;
      const ease = t => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      const prog = ease(this.animProgress);

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('Execution Time Comparison (ms) – 10 Features', this.W / 2, 22);

      const maxTime = Math.max(...this.data.map(d => d.time));
      const barH = Math.min(40, ch / this.data.length - 15);

      for (let i = 0; i < this.data.length; i++) {
        const d = this.data[i];
        const y = margin.top + (ch / this.data.length) * i + (ch / this.data.length - barH) / 2;
        const barW = (d.time / maxTime) * cw * prog;

        // Bar
        const grad = ctx.createLinearGradient(margin.left, 0, margin.left + barW, 0);
        grad.addColorStop(0, this._hexToRgba(d.color, 0.4));
        grad.addColorStop(1, d.color);
        ctx.fillStyle = grad;
        this._roundRect(ctx, margin.left, y, barW, barH, 4);
        ctx.fill();

        // Label
        ctx.font = '12px Inter';
        ctx.fillStyle = COLORS.text;
        ctx.textAlign = 'right';
        ctx.fillText(d.label, margin.left - 10, y + barH / 2 + 4);

        // Value
        if (prog > 0.3) {
          ctx.font = 'bold 11px JetBrains Mono';
          ctx.fillStyle = COLORS.text;
          ctx.textAlign = 'left';
          ctx.fillText(`${d.time} ms`, margin.left + barW + 8, y + barH / 2 + 4);
        }

        // Speedup label
        if (prog > 0.8 && i === this.data.length - 1) {
          ctx.font = 'bold 12px Inter';
          ctx.fillStyle = COLORS.instashap;
          ctx.fillText(`${(maxTime / d.time).toFixed(0)}× faster!`, margin.left + barW + 60, y + barH / 2 + 4);
        }
      }
    }

    _roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r);
      ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r);
      ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
    }

    _hexToRgba(hex, a) {
      const r = parseInt(hex.slice(1,3),16);
      const g = parseInt(hex.slice(3,5),16);
      const b = parseInt(hex.slice(5,7),16);
      return `rgba(${r},${g},${b},${a})`;
    }

    destroy() { this.stop(); }
  }

  /* ---------- Accuracy vs Speed Trade-off Curve ---------- */
  class TradeoffChart {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.animProgress = 0;
      this.running = false;
      this.methods = opts.methods || [
        { label: 'Exact SHAP',  speed: 0.05, accuracy: 1.0,  color: COLORS.shap,      r: 8 },
        { label: 'KernelSHAP',  speed: 0.4,  accuracy: 0.92, color: COLORS.kernel,     r: 8 },
        { label: 'TreeSHAP',    speed: 0.75, accuracy: 0.98, color: COLORS.tree,       r: 8 },
        { label: 'InstaSHAP',   speed: 0.95, accuracy: 0.78, color: COLORS.instashap,  r: 10 },
      ];
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

      const margin = { top: 55, right: 30, bottom: 55, left: 60 };
      const cw = this.W - margin.left - margin.right;
      const ch = this.H - margin.top - margin.bottom;
      const ease = t => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      const prog = ease(this.animProgress);

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('Accuracy vs Speed Trade-off', this.W / 2, 22);

      // Axes
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, this.H - margin.bottom);
      ctx.lineTo(this.W - margin.right, this.H - margin.bottom);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Axis labels
      ctx.font = '11px Inter';
      ctx.fillStyle = COLORS.textDim;
      ctx.textAlign = 'center';
      ctx.fillText('Speed →', this.W / 2, this.H - 12);
      ctx.save();
      ctx.translate(15, this.H / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText('Accuracy →', 0, 0);
      ctx.restore();

      // Grid
      for (let i = 0; i <= 4; i++) {
        const frac = i / 4;
        // Horizontal
        const y = this.H - margin.bottom - frac * ch;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(this.W - margin.right, y);
        ctx.strokeStyle = 'rgba(71,85,105,0.15)';
        ctx.stroke();
        ctx.font = '9px JetBrains Mono';
        ctx.fillStyle = COLORS.textDim;
        ctx.textAlign = 'right';
        ctx.fillText((frac * 100).toFixed(0) + '%', margin.left - 8, y + 4);

        // Vertical
        const x = margin.left + frac * cw;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, this.H - margin.bottom);
        ctx.stroke();
      }

      // Ideal zone
      ctx.fillStyle = 'rgba(16,185,129,0.05)';
      ctx.fillRect(margin.left + cw * 0.6, margin.top, cw * 0.4, ch * 0.4);
      if (prog > 0.5) {
        ctx.font = '9px Inter';
        ctx.fillStyle = 'rgba(16,185,129,0.4)';
        ctx.textAlign = 'center';
        ctx.fillText('Ideal Zone', margin.left + cw * 0.8, margin.top + 15);
      }

      // Pareto front curve
      if (prog > 0.3) {
        ctx.beginPath();
        ctx.moveTo(margin.left + 0.05 * cw, this.H - margin.bottom - 1.0 * ch);
        ctx.quadraticCurveTo(
          margin.left + 0.5 * cw, this.H - margin.bottom - 0.95 * ch,
          margin.left + 0.95 * cw, this.H - margin.bottom - 0.78 * ch
        );
        ctx.strokeStyle = 'rgba(6,182,212,0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Method dots
      for (const m of this.methods) {
        const x = margin.left + m.speed * cw;
        const y = this.H - margin.bottom - m.accuracy * ch;
        const scale = prog;

        // Glow
        ctx.beginPath();
        ctx.arc(x, y, m.r * scale * 2, 0, Math.PI * 2);
        ctx.fillStyle = this._hexToRgba(m.color, 0.15);
        ctx.fill();

        // Dot
        ctx.beginPath();
        ctx.arc(x, y, m.r * scale, 0, Math.PI * 2);
        ctx.fillStyle = m.color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label
        if (prog > 0.5) {
          ctx.font = 'bold 10px Inter';
          ctx.fillStyle = m.color;
          ctx.textAlign = 'center';
          ctx.fillText(m.label, x, y - m.r - 8);
        }
      }
    }

    _hexToRgba(hex, a) {
      const r = parseInt(hex.slice(1,3),16);
      const g = parseInt(hex.slice(3,5),16);
      const b = parseInt(hex.slice(5,7),16);
      return `rgba(${r},${g},${b},${a})`;
    }

    destroy() { this.stop(); }
  }

  /* ---------- Exponential Growth Visualization ---------- */
  class ExponentialGrowthViz {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.animProgress = 0;
      this.running = false;
      this.maxN = opts.maxN || 12;
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
      this.animProgress = Math.min(1, this.animProgress + 0.01);
      this._draw();
      if (this.animProgress < 1) requestAnimationFrame(() => this._loop());
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const margin = { top: 55, right: 40, bottom: 55, left: 70 };
      const cw = this.W - margin.left - margin.right;
      const ch = this.H - margin.top - margin.bottom;
      const ease = t => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      const prog = ease(this.animProgress);

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('Exponential Explosion: 2ⁿ Feature Subsets', this.W / 2, 22);

      const barCount = Math.floor(this.maxN * prog);
      const barW = Math.min(50, cw / this.maxN - 4);
      const maxVal = Math.pow(2, this.maxN);

      for (let n = 1; n <= barCount; n++) {
        const val = Math.pow(2, n);
        const barH = (val / maxVal) * ch;
        const x = margin.left + ((n - 1) / (this.maxN - 1)) * (cw - barW);
        const y = this.H - margin.bottom - barH;

        // Bar color gradient based on danger
        const danger = n / this.maxN;
        const r = Math.floor(59 + (239 - 59) * danger);
        const g = Math.floor(130 + (68 - 130) * danger);
        const b = Math.floor(246 + (68 - 246) * danger);

        const grad = ctx.createLinearGradient(0, y, 0, this.H - margin.bottom);
        grad.addColorStop(0, `rgba(${r},${g},${b},0.9)`);
        grad.addColorStop(1, `rgba(${r},${g},${b},0.3)`);
        ctx.fillStyle = grad;
        this._roundRect(ctx, x, y, barW, barH, 3);
        ctx.fill();

        // Value on top
        ctx.font = 'bold 9px JetBrains Mono';
        ctx.fillStyle = COLORS.text;
        ctx.textAlign = 'center';
        ctx.fillText(val >= 1000 ? `${(val/1000).toFixed(0)}K` : val, x + barW / 2, y - 6);

        // Feature count
        ctx.font = '9px Inter';
        ctx.fillStyle = COLORS.textDim;
        ctx.fillText(`n=${n}`, x + barW / 2, this.H - margin.bottom + 15);
      }

      // Warning annotation
      if (prog > 0.6) {
        ctx.font = 'bold 12px Inter';
        ctx.fillStyle = COLORS.shap;
        ctx.textAlign = 'right';
        ctx.fillText('⚠ Infeasible for n > 15', this.W - margin.right, margin.top + 20);
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

  /* ---------- Expose ---------- */
  window.ComplexityChart = ComplexityChart;
  window.SpeedChart = SpeedChart;
  window.TradeoffChart = TradeoffChart;
  window.ExponentialGrowthViz = ExponentialGrowthViz;

})();
