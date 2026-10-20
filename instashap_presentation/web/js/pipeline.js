/* ============================================================
   InstaSHAP Presentation – Pipeline Animations
   ============================================================
   Step-by-step InstaSHAP pipeline visualization showing
   data flow through each stage of the explanation process.
   ============================================================ */

(function () {
  'use strict';

  const COLORS = {
    input:    '#3B82F6',
    model:    '#8B5CF6',
    sampling: '#F59E0B',
    shap:     '#10B981',
    error:    '#EF4444',
    accent:   '#06B6D4',
    text:     '#F8FAFC',
    textDim:  '#94A3B8',
    bg:       '#1E293B',
    line:     'rgba(71,85,105,0.4)',
  };

  /* ---------- Pipeline Flow Visualization ---------- */
  class PipelineViz {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.currentStep = opts.currentStep || 0;
      this.totalSteps = 6;
      this.animProgress = 0;
      this.running = false;
      this.animFrame = null;
      this.particles = [];
      this.time = 0;

      this.steps = [
        { label: 'Input\nSample',     icon: '📊', color: COLORS.input },
        { label: 'Feature\nPerturb',  icon: '🔀', color: COLORS.input },
        { label: 'Sampling\nSubsets',  icon: '🎲', color: COLORS.sampling },
        { label: 'Model\nPredict',    icon: '🧠', color: COLORS.model },
        { label: 'Compute\nContrib',   icon: '📐', color: COLORS.shap },
        { label: 'SHAP\nOutput',      icon: '📊', color: COLORS.shap },
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

    setStep(step) {
      this.currentStep = step;
      this.animProgress = 0;
      if (!this.running) this.start();
    }

    start() {
      this.running = true;
      this._loop();
    }

    stop() {
      this.running = false;
      if (this.animFrame) cancelAnimationFrame(this.animFrame);
    }

    _loop() {
      if (!this.running) return;
      this.time += 0.016;
      this.animProgress = Math.min(1, this.animProgress + 0.015);
      this._updateParticles();
      this._draw();
      this.animFrame = requestAnimationFrame(() => this._loop());
    }

    _updateParticles() {
      // Spawn particles flowing between active steps
      if (Math.random() < 0.15 && this.currentStep > 0) {
        const fromIdx = this.currentStep - 1;
        const toIdx = this.currentStep;
        const fromPos = this._getStepPos(fromIdx);
        const toPos = this._getStepPos(toIdx);
        this.particles.push({
          x: fromPos.x + 30,
          y: fromPos.y,
          targetX: toPos.x - 30,
          targetY: toPos.y,
          t: 0,
          speed: 0.02 + Math.random() * 0.015,
          color: this.steps[fromIdx].color,
          r: 2 + Math.random() * 2,
        });
      }

      for (let i = this.particles.length - 1; i >= 0; i--) {
        const p = this.particles[i];
        p.t += p.speed;
        if (p.t > 1) {
          this.particles.splice(i, 1);
        }
      }
    }

    _getStepPos(idx) {
      const margin = 60;
      const stepW = (this.W - margin * 2) / this.totalSteps;
      return {
        x: margin + stepW * idx + stepW / 2,
        y: this.H / 2,
      };
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const margin = 60;
      const stepW = (this.W - margin * 2) / this.totalSteps;
      const boxW = Math.min(90, stepW - 20);
      const boxH = 70;
      const centerY = this.H / 2;

      // Draw connections
      for (let i = 0; i < this.totalSteps - 1; i++) {
        const x1 = margin + stepW * i + stepW / 2 + boxW / 2;
        const x2 = margin + stepW * (i + 1) + stepW / 2 - boxW / 2;

        ctx.beginPath();
        ctx.moveTo(x1, centerY);
        ctx.lineTo(x2, centerY);

        if (i < this.currentStep) {
          ctx.strokeStyle = `rgba(6,182,212,${0.3 + 0.4 * Math.sin(this.time * 3 + i)})`;
          ctx.lineWidth = 2;
        } else {
          ctx.strokeStyle = COLORS.line;
          ctx.lineWidth = 1;
        }
        ctx.stroke();

        // Arrow
        if (i < this.currentStep) {
          const midX = (x1 + x2) / 2;
          ctx.beginPath();
          ctx.moveTo(midX + 6, centerY);
          ctx.lineTo(midX - 2, centerY - 5);
          ctx.lineTo(midX - 2, centerY + 5);
          ctx.closePath();
          ctx.fillStyle = COLORS.accent;
          ctx.fill();
        }
      }

      // Draw particles
      for (const p of this.particles) {
        const ease = t => t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
        const t = ease(p.t);
        const x = p.x + (p.targetX - p.x) * t;
        const y = p.y + (p.targetY - p.y) * t + Math.sin(t * Math.PI * 2) * 8;
        const alpha = Math.sin(p.t * Math.PI);

        ctx.beginPath();
        ctx.arc(x, y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = this._hexToRgba(p.color, alpha * 0.8);
        ctx.fill();
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 6;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Draw step boxes
      for (let i = 0; i < this.totalSteps; i++) {
        const x = margin + stepW * i + stepW / 2 - boxW / 2;
        const y = centerY - boxH / 2;
        const step = this.steps[i];
        const isActive = i === this.currentStep;
        const isCompleted = i < this.currentStep;

        // Box
        ctx.beginPath();
        this._roundRect(ctx, x, y, boxW, boxH, 10);

        if (isActive) {
          ctx.fillStyle = this._hexToRgba(step.color, 0.2);
          ctx.fill();
          ctx.strokeStyle = step.color;
          ctx.lineWidth = 2;
          ctx.stroke();

          // Glow
          ctx.shadowColor = step.color;
          ctx.shadowBlur = 15;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else if (isCompleted) {
          ctx.fillStyle = this._hexToRgba(step.color, 0.1);
          ctx.fill();
          ctx.strokeStyle = this._hexToRgba(step.color, 0.5);
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          ctx.fillStyle = 'rgba(30,41,59,0.6)';
          ctx.fill();
          ctx.strokeStyle = COLORS.line;
          ctx.lineWidth = 1;
          ctx.stroke();
        }

        // Icon
        ctx.font = `${isActive ? 22 : 18}px serif`;
        ctx.textAlign = 'center';
        ctx.fillText(step.icon, x + boxW / 2, y + 28);

        // Label
        const lines = step.label.split('\n');
        ctx.font = `${isActive ? 'bold ' : ''}10px Inter, sans-serif`;
        ctx.fillStyle = isActive ? COLORS.text : (isCompleted ? COLORS.textDim : 'rgba(148,163,184,0.5)');
        for (let li = 0; li < lines.length; li++) {
          ctx.fillText(lines[li], x + boxW / 2, y + boxH - 15 + li * 13);
        }

        // Step number
        ctx.font = 'bold 9px Inter';
        ctx.fillStyle = step.color;
        ctx.fillText(`Step ${i + 1}`, x + boxW / 2, y - 8);

        // Checkmark for completed
        if (isCompleted) {
          ctx.font = 'bold 14px serif';
          ctx.fillStyle = COLORS.shap;
          ctx.fillText('✓', x + boxW - 8, y + 14);
        }
      }
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

  /* ---------- Sampling Visualization ---------- */
  class SamplingViz {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.numFeatures = opts.numFeatures || 5;
      this.numSubsets = opts.numSubsets || 32; // 2^5
      this.numSampled = opts.numSampled || 8;  // InstaSHAP samples
      this.animProgress = 0;
      this.running = false;
      this.subsets = [];
      this.sampledIndices = new Set();

      this._generateSubsets();
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

    _generateSubsets() {
      this.subsets = [];
      for (let i = 0; i < this.numSubsets; i++) {
        const mask = [];
        for (let f = 0; f < this.numFeatures; f++) {
          mask.push((i >> f) & 1);
        }
        this.subsets.push(mask);
      }

      // Randomly select sampled subsets
      const indices = Array.from({ length: this.numSubsets }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      this.sampledIndices = new Set(indices.slice(0, this.numSampled));
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
      if (this.animProgress < 1) {
        requestAnimationFrame(() => this._loop());
      }
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      const margin = { top: 60, right: 30, bottom: 50, left: 30 };
      const cols = 8;
      const rows = Math.ceil(this.numSubsets / cols);
      const cellW = (this.W - margin.left - margin.right) / cols;
      const cellH = Math.min(50, (this.H - margin.top - margin.bottom) / rows);

      // Title
      ctx.font = 'bold 14px Inter';
      ctx.fillStyle = COLORS.text;
      ctx.textAlign = 'center';
      ctx.fillText('Exact SHAP: All 2ⁿ Subsets vs InstaSHAP: Sampled Subsets', this.W / 2, 25);

      // Subtitle
      ctx.font = '11px Inter';
      ctx.fillStyle = COLORS.textDim;
      ctx.fillText(`${this.numSubsets} total subsets → ${this.numSampled} sampled (${((this.numSampled/this.numSubsets)*100).toFixed(0)}% reduction)`, this.W / 2, 42);

      const revealCount = Math.floor(this.numSubsets * this.animProgress);

      for (let i = 0; i < Math.min(this.numSubsets, revealCount); i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = margin.left + col * cellW;
        const y = margin.top + row * cellH;

        const isSampled = this.sampledIndices.has(i);
        const featureW = (cellW - 10) / this.numFeatures;

        // Background
        if (isSampled) {
          ctx.fillStyle = 'rgba(6,182,212,0.08)';
          ctx.strokeStyle = 'rgba(6,182,212,0.4)';
        } else {
          ctx.fillStyle = 'rgba(30,41,59,0.4)';
          ctx.strokeStyle = 'rgba(71,85,105,0.3)';
        }
        ctx.lineWidth = 1;
        ctx.beginPath();
        this._roundRect(ctx, x + 2, y + 2, cellW - 4, cellH - 4, 4);
        ctx.fill();
        ctx.stroke();

        // Feature mask
        for (let f = 0; f < this.numFeatures; f++) {
          const fx = x + 5 + f * featureW;
          const fy = y + 8;
          const fw = featureW - 2;
          const fh = cellH - 22;

          if (this.subsets[i][f]) {
            ctx.fillStyle = isSampled ? COLORS.accent : 'rgba(59,130,246,0.4)';
          } else {
            ctx.fillStyle = 'rgba(71,85,105,0.2)';
          }
          ctx.fillRect(fx, fy, fw, fh);
        }

        // Crossed out if not sampled
        if (!isSampled && this.animProgress > 0.7) {
          ctx.beginPath();
          ctx.moveTo(x + 2, y + 2);
          ctx.lineTo(x + cellW - 4, y + cellH - 4);
          ctx.strokeStyle = 'rgba(239,68,68,0.4)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }

      // Legend
      const legY = this.H - 25;
      ctx.font = '11px Inter';
      ctx.textAlign = 'left';

      ctx.fillStyle = 'rgba(6,182,212,0.3)';
      ctx.fillRect(margin.left, legY - 8, 12, 12);
      ctx.fillStyle = COLORS.accent;
      ctx.fillText('Sampled (InstaSHAP)', margin.left + 18, legY + 2);

      ctx.fillStyle = 'rgba(71,85,105,0.3)';
      ctx.fillRect(margin.left + 200, legY - 8, 12, 12);
      ctx.fillStyle = COLORS.textDim;
      ctx.fillText('Skipped', margin.left + 218, legY + 2);
    }

    _roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
    }

    destroy() { this.stop(); }
  }

  /* ---------- Expose ---------- */
  window.PipelineViz = PipelineViz;
  window.SamplingViz = SamplingViz;

})();
