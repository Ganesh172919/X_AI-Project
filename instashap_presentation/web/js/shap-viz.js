/* ============================================================
   InstaSHAP Presentation – SHAP Visualizations
   ============================================================
   Animated SHAP value displays:
   - Force plot
   - Feature importance bar chart
   - Waterfall chart
   - Heatmap
   ============================================================ */

(function () {
  'use strict';

  const COLORS = {
    positive: '#10B981',
    negative: '#EF4444',
    neutral:  '#94A3B8',
    text:     '#F8FAFC',
    textDim:  '#94A3B8',
    grid:     'rgba(71,85,105,0.3)',
    bg:       'transparent',
    accent:   '#06B6D4',
  };

  /* ---------- Animated Bar Chart ---------- */
  class SHAPBarChart {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.features = opts.features || [
        { name: 'Age',    value:  0.35 },
        { name: 'Income', value:  0.52 },
        { name: 'Score',  value: -0.28 },
        { name: 'Tenure', value:  0.15 },
        { name: 'Region', value: -0.10 },
      ];
      this.animProgress = 0;
      this.running = false;
      this.animFrame = null;
      this.title = opts.title || 'SHAP Feature Importance';

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

    stop() {
      this.running = false;
      if (this.animFrame) cancelAnimationFrame(this.animFrame);
    }

    _loop() {
      if (!this.running) return;
      this.animProgress = Math.min(1, this.animProgress + 0.02);
      this._draw();
      if (this.animProgress < 1) {
        this.animFrame = requestAnimationFrame(() => this._loop());
      }
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const margin = { top: 50, right: 40, bottom: 30, left: 90 };
      const chartW = this.W - margin.left - margin.right;
      const chartH = this.H - margin.top - margin.bottom;

      // Sort by absolute value
      const sorted = [...this.features].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
      const maxVal = Math.max(...sorted.map(f => Math.abs(f.value)));
      const barH = Math.min(35, chartH / sorted.length - 8);
      const centerX = margin.left + chartW / 2;

      // Title
      ctx.font = 'bold 14px Inter, sans-serif';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText(this.title, this.W / 2, 25);

      // Center line
      ctx.beginPath();
      ctx.moveTo(centerX, margin.top);
      ctx.lineTo(centerX, this.H - margin.bottom);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Bars
      const ease = t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      const prog = ease(this.animProgress);

      for (let i = 0; i < sorted.length; i++) {
        const f = sorted[i];
        const y = margin.top + (chartH / sorted.length) * i + (chartH / sorted.length - barH) / 2;
        const barWidth = (Math.abs(f.value) / maxVal) * (chartW / 2) * prog;
        const isPos = f.value >= 0;

        // Bar
        const x = isPos ? centerX : centerX - barWidth;
        const color = isPos ? COLORS.positive : COLORS.negative;

        // Gradient
        const grad = ctx.createLinearGradient(x, 0, x + barWidth, 0);
        if (isPos) {
          grad.addColorStop(0, this._hexToRgba(color, 0.6));
          grad.addColorStop(1, color);
        } else {
          grad.addColorStop(0, color);
          grad.addColorStop(1, this._hexToRgba(color, 0.6));
        }

        ctx.fillStyle = grad;
        this._roundRect(ctx, x, y, barWidth, barH, 4);
        ctx.fill();

        // Value label
        if (prog > 0.5) {
          ctx.font = 'bold 11px JetBrains Mono, monospace';
          ctx.fillStyle = COLORS.text;
          ctx.textAlign = isPos ? 'left' : 'right';
          const valX = isPos ? centerX + barWidth + 6 : centerX - barWidth - 6;
          ctx.fillText(f.value.toFixed(3), valX, y + barH / 2 + 4);
        }

        // Feature name
        ctx.font = '12px Inter, sans-serif';
        ctx.fillStyle = COLORS.textDim;
        ctx.textAlign = 'right';
        ctx.fillText(f.name, margin.left - 8, y + barH / 2 + 4);
      }

      // Axis labels
      ctx.font = '10px Inter, sans-serif';
      ctx.fillStyle = COLORS.textDim;
      ctx.textAlign = 'center';
      ctx.fillText('← Negative', centerX - chartW / 4, this.H - 10);
      ctx.fillText('Positive →', centerX + chartW / 4, this.H - 10);
    }

    _roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    _hexToRgba(hex, a) {
      const r = parseInt(hex.slice(1,3),16);
      const g = parseInt(hex.slice(3,5),16);
      const b = parseInt(hex.slice(5,7),16);
      return `rgba(${r},${g},${b},${a})`;
    }

    destroy() { this.stop(); }
  }

  /* ---------- Force Plot ---------- */
  class SHAPForcePlot {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.baseValue = opts.baseValue || 0.5;
      this.outputValue = opts.outputValue || 0.72;
      this.features = opts.features || [
        { name: 'Income', value: 0.12, color: COLORS.positive },
        { name: 'Age',    value: 0.08, color: COLORS.positive },
        { name: 'Score',  value: 0.05, color: COLORS.positive },
        { name: 'Region', value: -0.03, color: COLORS.negative },
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

    stop() {
      this.running = false;
    }

    _loop() {
      if (!this.running) return;
      this.animProgress = Math.min(1, this.animProgress + 0.015);
      this._draw();
      if (this.animProgress < 1) {
        requestAnimationFrame(() => this._loop());
      }
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const margin = 60;
      const plotW = this.W - margin * 2;
      const centerY = this.H / 2;
      const barH = 40;

      const ease = t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
      const prog = ease(this.animProgress);

      // Scale
      const range = Math.max(Math.abs(this.outputValue - this.baseValue) * 3, 0.3);
      const toX = val => margin + ((val - this.baseValue + range) / (range * 2)) * plotW;

      // Background
      ctx.fillStyle = 'rgba(30,41,59,0.5)';
      this._roundRect(ctx, margin - 10, centerY - barH - 20, plotW + 20, barH * 2 + 40, 12);
      ctx.fill();

      // Base value marker
      const baseX = toX(this.baseValue);
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.moveTo(baseX, centerY - barH - 10);
      ctx.lineTo(baseX, centerY + barH + 10);
      ctx.strokeStyle = COLORS.textDim;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = 'bold 11px Inter';
      ctx.fillStyle = COLORS.textDim;
      ctx.textAlign = 'center';
      ctx.fillText(`Base: ${this.baseValue.toFixed(2)}`, baseX, centerY + barH + 28);

      // Draw feature segments
      let currentX = baseX;
      const sorted = [...this.features].sort((a, b) => b.value - a.value);

      for (let i = 0; i < sorted.length; i++) {
        const f = sorted[i];
        const segW = (Math.abs(f.value) / range) * (plotW / 2) * prog;
        const isPos = f.value >= 0;
        const x = isPos ? currentX : currentX - segW;

        ctx.fillStyle = isPos ? 'rgba(16,185,129,0.6)' : 'rgba(239,68,68,0.6)';
        this._roundRect(ctx, x, centerY - barH / 2, segW, barH, 3);
        ctx.fill();

        // Border
        ctx.strokeStyle = isPos ? COLORS.positive : COLORS.negative;
        ctx.lineWidth = 1;
        this._roundRect(ctx, x, centerY - barH / 2, segW, barH, 3);
        ctx.stroke();

        // Label
        if (segW > 25 && prog > 0.6) {
          ctx.font = '10px Inter';
          ctx.fillStyle = COLORS.text;
          ctx.textAlign = 'center';
          ctx.fillText(f.name, x + segW / 2, centerY + 4);
        }

        currentX = isPos ? currentX + segW : currentX - segW;
      }

      // Output value
      const outX = toX(this.baseValue + (this.outputValue - this.baseValue) * prog);
      ctx.beginPath();
      ctx.arc(outX, centerY, 6, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.accent;
      ctx.fill();
      ctx.shadowColor = COLORS.accent;
      ctx.shadowBlur = 12;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.font = 'bold 13px JetBrains Mono';
      ctx.fillStyle = COLORS.accent;
      ctx.textAlign = 'center';
      ctx.fillText(`f(x) = ${(this.baseValue + (this.outputValue - this.baseValue) * prog).toFixed(3)}`,
                    outX, centerY - barH / 2 - 15);

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('SHAP Force Plot – Single Prediction', this.W / 2, 25);
    }

    _roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    destroy() { this.stop(); }
  }

  /* ---------- Heatmap ---------- */
  class SHAPHeatmap {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.data = opts.data || this._generateSample();
      this.features = opts.features || ['Age', 'Income', 'Score', 'Tenure', 'Region'];
      this.samples = opts.samples || 10;
      this.animProgress = 0;
      this.running = false;

      this._resize();
    }

    _generateSample() {
      const data = [];
      for (let s = 0; s < 10; s++) {
        const row = [];
        for (let f = 0; f < 5; f++) {
          row.push((Math.random() - 0.5) * 0.8);
        }
        data.push(row);
      }
      return data;
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
      this.animProgress = Math.min(1, this.animProgress + 0.02);
      this._draw();
      if (this.animProgress < 1) {
        requestAnimationFrame(() => this._loop());
      }
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const margin = { top: 50, right: 30, bottom: 30, left: 80 };
      const cellW = (this.W - margin.left - margin.right) / this.data[0].length;
      const cellH = (this.H - margin.top - margin.bottom) / this.data.length;

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('SHAP Value Heatmap', this.W / 2, 25);

      // Feature labels
      ctx.font = '11px Inter';
      ctx.fillStyle = COLORS.textDim;
      ctx.textAlign = 'center';
      for (let f = 0; f < this.features.length; f++) {
        ctx.fillText(this.features[f], margin.left + f * cellW + cellW / 2, margin.top - 8);
      }

      // Sample labels
      ctx.textAlign = 'right';
      for (let s = 0; s < this.data.length; s++) {
        ctx.fillText(`Sample ${s + 1}`, margin.left - 8, margin.top + s * cellH + cellH / 2 + 4);
      }

      // Cells
      const revealedCells = Math.floor(this.data.length * this.data[0].length * this.animProgress);
      let cellIdx = 0;

      for (let s = 0; s < this.data.length; s++) {
        for (let f = 0; f < this.data[s].length; f++) {
          if (cellIdx > revealedCells) break;
          cellIdx++;

          const val = this.data[s][f];
          const x = margin.left + f * cellW;
          const y = margin.top + s * cellH;

          // Color: positive = green, negative = red
          const intensity = Math.min(1, Math.abs(val) * 2);
          if (val >= 0) {
            ctx.fillStyle = `rgba(16,185,129,${intensity * 0.8})`;
          } else {
            ctx.fillStyle = `rgba(239,68,68,${intensity * 0.8})`;
          }
          ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);

          // Value text
          if (cellW > 40) {
            ctx.font = '10px JetBrains Mono';
            ctx.fillStyle = intensity > 0.5 ? COLORS.text : COLORS.textDim;
            ctx.textAlign = 'center';
            ctx.fillText(val.toFixed(2), x + cellW / 2, y + cellH / 2 + 4);
          }
        }
      }
    }

    destroy() { this.stop(); }
  }

  /* ---------- Expose ---------- */
  window.SHAPBarChart = SHAPBarChart;
  window.SHAPForcePlot = SHAPForcePlot;
  window.SHAPHeatmap = SHAPHeatmap;

})();
