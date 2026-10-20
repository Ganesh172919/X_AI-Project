/* ============================================================
   InstaSHAP Presentation – Failure Analysis Visualizations
   ============================================================
   Animated visualizations showing where InstaSHAP fails:
   - Feature interaction errors
   - Non-linear model misattribution
   - Distribution shift visualization
   - Expected vs actual comparison
   ============================================================ */

(function () {
  'use strict';

  const COLORS = {
    correct:   '#10B981',
    incorrect: '#EF4444',
    expected:  '#3B82F6',
    actual:    '#F59E0B',
    text:      '#F8FAFC',
    textDim:   '#94A3B8',
    grid:      'rgba(71,85,105,0.3)',
    accent:    '#06B6D4',
  };

  /* ---------- Expected vs Actual Comparison ---------- */
  class ComparisonBarChart {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.features = opts.features || [
        { name: 'Age',    expected: 0.35, actual: 0.18 },
        { name: 'Income', expected: 0.52, actual: 0.45 },
        { name: 'Score',  expected: -0.28, actual: -0.05 },
        { name: 'X₁·X₂', expected: 0.30, actual: 0.02 },
      ];
      this.title = opts.title || 'Expected vs InstaSHAP Output';
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
      if (this.animProgress < 1) {
        requestAnimationFrame(() => this._loop());
      }
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const margin = { top: 55, right: 40, bottom: 60, left: 90 };
      const chartW = this.W - margin.left - margin.right;
      const chartH = this.H - margin.top - margin.bottom;

      const maxVal = Math.max(...this.features.map(f => Math.max(Math.abs(f.expected), Math.abs(f.actual))));
      const barGroupH = chartH / this.features.length;
      const barH = barGroupH * 0.3;
      const centerX = margin.left + chartW / 2;

      const ease = t => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
      const prog = ease(this.animProgress);

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText(this.title, this.W / 2, 25);

      // Center line
      ctx.beginPath();
      ctx.moveTo(centerX, margin.top);
      ctx.lineTo(centerX, this.H - margin.bottom);
      ctx.strokeStyle = COLORS.grid;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      for (let i = 0; i < this.features.length; i++) {
        const f = this.features[i];
        const groupY = margin.top + barGroupH * i;
        const y1 = groupY + barGroupH * 0.2;
        const y2 = y1 + barH + 4;

        // Expected bar
        const expW = (Math.abs(f.expected) / maxVal) * (chartW / 2) * prog;
        const expX = f.expected >= 0 ? centerX : centerX - expW;
        ctx.fillStyle = COLORS.expected;
        this._roundRect(ctx, expX, y1, expW, barH, 3);
        ctx.fill();

        // Actual bar
        const actW = (Math.abs(f.actual) / maxVal) * (chartW / 2) * prog;
        const actX = f.actual >= 0 ? centerX : centerX - actW;
        ctx.fillStyle = COLORS.actual;
        this._roundRect(ctx, actX, y2, actW, barH, 3);
        ctx.fill();

        // Error indicator
        if (prog > 0.7) {
          const error = Math.abs(f.expected - f.actual);
          if (error > 0.15) {
            ctx.font = '10px Inter';
            ctx.fillStyle = COLORS.incorrect;
            const errX = Math.max(expX + expW, actX + actW) + 10;
            ctx.textAlign = 'left';
            ctx.fillText(`⚠ Error: ${error.toFixed(2)}`, errX, y1 + barH);
          }
        }

        // Feature name
        ctx.font = '12px Inter';
        ctx.fillStyle = COLORS.textDim;
        ctx.textAlign = 'right';
        ctx.fillText(f.name, margin.left - 8, y1 + barH + 4);
      }

      // Legend
      const legY = this.H - 25;
      ctx.font = '11px Inter';
      ctx.textAlign = 'left';
      ctx.fillStyle = COLORS.expected;
      ctx.fillRect(margin.left, legY - 8, 12, 12);
      ctx.fillText('Expected (Exact SHAP)', margin.left + 18, legY + 2);
      ctx.fillStyle = COLORS.actual;
      ctx.fillRect(margin.left + 200, legY - 8, 12, 12);
      ctx.fillText('Actual (InstaSHAP)', margin.left + 218, legY + 2);
    }

    _roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x+r,y);
      ctx.arcTo(x+w,y,x+w,y+h,r);
      ctx.arcTo(x+w,y+h,x,y+h,r);
      ctx.arcTo(x,y+h,x,y,r);
      ctx.arcTo(x,y,x+w,y,r);
      ctx.closePath();
    }

    destroy() { this.stop(); }
  }

  /* ---------- Feature Interaction Visualization ---------- */
  class InteractionViz {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.features = opts.features || ['X₁', 'X₂', 'X₃'];
      this.interactions = opts.interactions || [
        { from: 0, to: 1, strength: 0.8, label: 'Strong' },
        { from: 0, to: 2, strength: 0.3, label: 'Weak' },
        { from: 1, to: 2, strength: 0.6, label: 'Medium' },
      ];
      this.animProgress = 0;
      this.running = false;
      this.time = 0;

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
      this.time += 0.016;
      this.animProgress = Math.min(1, this.animProgress + 0.01);
      this._draw();
      requestAnimationFrame(() => this._loop());
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const centerX = this.W / 2;
      const centerY = this.H / 2;
      const radius = Math.min(this.W, this.H) * 0.3;

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('Feature Interactions (InstaSHAP Misses These)', centerX, 25);

      // Position features in a circle
      const positions = [];
      for (let i = 0; i < this.features.length; i++) {
        const angle = (i / this.features.length) * Math.PI * 2 - Math.PI / 2;
        positions.push({
          x: centerX + Math.cos(angle) * radius,
          y: centerY + Math.sin(angle) * radius,
        });
      }

      // Draw interactions
      for (const inter of this.interactions) {
        const from = positions[inter.from];
        const to = positions[inter.to];

        // Pulsing connection
        const pulse = 0.5 + 0.5 * Math.sin(this.time * 2 + inter.strength * 3);
        const alpha = inter.strength * pulse * this.animProgress;

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);

        // Curved line
        const midX = (from.x + to.x) / 2 + (from.y - to.y) * 0.2;
        const midY = (from.y + to.y) / 2 + (to.x - from.x) * 0.2;
        ctx.quadraticCurveTo(midX, midY, to.x, to.y);

        ctx.strokeStyle = `rgba(239,68,68,${alpha})`;
        ctx.lineWidth = 1 + inter.strength * 4;
        ctx.stroke();

        // Label
        if (this.animProgress > 0.5) {
          ctx.font = '10px Inter';
          ctx.fillStyle = `rgba(239,68,68,${alpha})`;
          ctx.textAlign = 'center';
          ctx.fillText(inter.label, midX, midY - 8);
        }

        // "MISSED" label for strong interactions
        if (inter.strength > 0.5 && this.animProgress > 0.8) {
          ctx.font = 'bold 10px Inter';
          ctx.fillStyle = COLORS.incorrect;
          ctx.fillText('MISSED ✗', midX, midY + 12);
        }
      }

      // Draw feature nodes
      for (let i = 0; i < this.features.length; i++) {
        const pos = positions[i];
        const nodeR = 28;

        // Outer glow
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR + 4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(59,130,246,${0.1 + 0.1 * Math.sin(this.time * 1.5 + i)})`;
        ctx.fill();

        // Node
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeR, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(59,130,246,0.2)';
        ctx.fill();
        ctx.strokeStyle = COLORS.expected;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Label
        ctx.font = 'bold 14px Inter';
        ctx.fillStyle = COLORS.text;
        ctx.textAlign = 'center';
        ctx.fillText(this.features[i], pos.x, pos.y + 5);
      }

      // Warning message
      if (this.animProgress > 0.9) {
        ctx.font = 'bold 12px Inter';
        ctx.fillStyle = COLORS.incorrect;
        ctx.textAlign = 'center';
        ctx.fillText('⚠ InstaSHAP assumes feature independence — misses interaction effects',
                      centerX, this.H - 20);
      }
    }

    destroy() { this.stop(); }
  }

  /* ---------- Distribution Shift Visualization ---------- */
  class DistributionShiftViz {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.animProgress = 0;
      this.running = false;
      this.time = 0;

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
      this.time += 0.016;
      this.animProgress = Math.min(1, this.animProgress + 0.008);
      this._draw();
      requestAnimationFrame(() => this._loop());
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const margin = { top: 50, right: 40, bottom: 50, left: 60 };
      const chartW = this.W - margin.left - margin.right;
      const chartH = this.H - margin.top - margin.bottom;

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('Distribution Shift: Training vs Inference', this.W / 2, 25);

      // Axes
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, this.H - margin.bottom);
      ctx.lineTo(this.W - margin.right, this.H - margin.bottom);
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Training distribution (Gaussian)
      ctx.beginPath();
      const trainMean = chartW * 0.35;
      const trainStd = chartW * 0.12;
      for (let px = 0; px < chartW; px++) {
        const x = margin.left + px;
        const val = Math.exp(-0.5 * Math.pow((px - trainMean) / trainStd, 2));
        const y = this.H - margin.bottom - val * chartH * 0.8 * this.animProgress;
        if (px === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = COLORS.expected;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Fill under training curve
      ctx.lineTo(this.W - margin.right, this.H - margin.bottom);
      ctx.lineTo(margin.left, this.H - margin.bottom);
      ctx.closePath();
      ctx.fillStyle = 'rgba(59,130,246,0.1)';
      ctx.fill();

      // Inference distribution (shifted Gaussian)
      const shift = this.animProgress * chartW * 0.25;
      ctx.beginPath();
      const infMean = trainMean + shift;
      const infStd = trainStd * 1.3;
      for (let px = 0; px < chartW; px++) {
        const x = margin.left + px;
        const val = Math.exp(-0.5 * Math.pow((px - infMean) / infStd, 2));
        const y = this.H - margin.bottom - val * chartH * 0.6 * this.animProgress;
        if (px === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = COLORS.actual;
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.lineTo(this.W - margin.right, this.H - margin.bottom);
      ctx.lineTo(margin.left, this.H - margin.bottom);
      ctx.closePath();
      ctx.fillStyle = 'rgba(245,158,11,0.1)';
      ctx.fill();

      // Shift arrow
      if (this.animProgress > 0.5) {
        const arrowY = margin.top + chartH * 0.15;
        const arrowX1 = margin.left + trainMean;
        const arrowX2 = margin.left + infMean;

        ctx.beginPath();
        ctx.moveTo(arrowX1, arrowY);
        ctx.lineTo(arrowX2, arrowY);
        ctx.strokeStyle = COLORS.incorrect;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Arrowhead
        ctx.beginPath();
        ctx.moveTo(arrowX2, arrowY);
        ctx.lineTo(arrowX2 - 8, arrowY - 5);
        ctx.lineTo(arrowX2 - 8, arrowY + 5);
        ctx.closePath();
        ctx.fillStyle = COLORS.incorrect;
        ctx.fill();

        ctx.font = 'bold 11px Inter';
        ctx.fillStyle = COLORS.incorrect;
        ctx.textAlign = 'center';
        ctx.fillText('Distribution Shift', (arrowX1 + arrowX2) / 2, arrowY - 10);
      }

      // Labels
      ctx.font = '11px Inter';
      ctx.textAlign = 'left';
      ctx.fillStyle = COLORS.expected;
      ctx.fillRect(margin.left + 10, this.H - 35, 12, 12);
      ctx.fillText('Training Distribution', margin.left + 28, this.H - 25);

      ctx.fillStyle = COLORS.actual;
      ctx.fillRect(margin.left + 200, this.H - 35, 12, 12);
      ctx.fillText('Inference Distribution', margin.left + 218, this.H - 25);

      // Warning
      if (this.animProgress > 0.8) {
        ctx.font = 'bold 11px Inter';
        ctx.fillStyle = COLORS.incorrect;
        ctx.textAlign = 'center';
        ctx.fillText('⚠ InstaSHAP uses training distribution for perturbation — invalid under shift',
                      this.W / 2, this.H - 8);
      }
    }

    destroy() { this.stop(); }
  }

  /* ---------- Expose ---------- */
  window.ComparisonBarChart = ComparisonBarChart;
  window.InteractionViz = InteractionViz;
  window.DistributionShiftViz = DistributionShiftViz;

})();
