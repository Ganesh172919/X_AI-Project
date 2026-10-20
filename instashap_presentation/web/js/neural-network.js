/* ============================================================
   InstaSHAP Presentation – Neural Network Canvas Renderer
   ============================================================
   Real-time animated multi-layer perceptron (MLP) with:
   - Forward propagation pulse animation
   - Feature masking / dimming
   - Node activation glow
   - Connection weight visualization
   - Data particle flow along connections
   ============================================================ */

(function () {
  'use strict';

  const COLORS = {
    input:      '#3B82F6',
    hidden:     '#8B5CF6',
    output:     '#10B981',
    connection: 'rgba(148,163,184,0.15)',
    connActive: 'rgba(6,182,212,0.6)',
    particle:   '#06B6D4',
    masked:     'rgba(239,68,68,0.3)',
    maskedNode: '#EF4444',
    bg:         'transparent',
    text:       '#F8FAFC',
    textDim:    '#94A3B8',
  };

  class NeuralNetworkViz {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {Object} opts
     * @param {number[]} opts.layers - Nodes per layer, e.g. [3,5,4,1]
     * @param {string[]} [opts.inputLabels] - Labels for input nodes
     * @param {string[]} [opts.outputLabels] - Labels for output nodes
     * @param {number[]} [opts.maskedFeatures] - Indices of masked input features
     * @param {boolean} [opts.animate] - Whether to run forward-prop animation
     * @param {boolean} [opts.showWeights] - Show weight thickness
     * @param {boolean} [opts.showActivations] - Glow nodes on activation
     */
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.layers = opts.layers || [3, 5, 4, 1];
      this.inputLabels = opts.inputLabels || [];
      this.outputLabels = opts.outputLabels || [];
      this.maskedFeatures = new Set(opts.maskedFeatures || []);
      this.animate = opts.animate !== false;
      this.showWeights = opts.showWeights || false;
      this.showActivations = opts.showActivations !== false;

      // Computed positions
      this.nodes = [];   // [layer][node] => {x, y, r, activation}
      this.connections = [];
      this.particles = [];

      // Animation state
      this.propagationProgress = 0;  // 0 → 1
      this.propagationSpeed = 0.008;
      this.running = false;
      this.animFrame = null;
      this.time = 0;

      this._resize();
      this._buildNetwork();
      if (this.animate) this.start();
      else this.drawStatic();
    }

    /* ---------- Layout ---------- */
    _resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      this.W = rect.width;
      this.H = rect.height;
      this.canvas.width = this.W * dpr;
      this.canvas.height = this.H * dpr;
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    _buildNetwork() {
      const paddingX = 80;
      const paddingY = 50;
      const usableW = this.W - paddingX * 2;
      const usableH = this.H - paddingY * 2;
      const numLayers = this.layers.length;

      this.nodes = [];
      this.connections = [];

      // Position nodes
      for (let l = 0; l < numLayers; l++) {
        const layerNodes = [];
        const n = this.layers[l];
        const x = paddingX + (usableW / (numLayers - 1)) * l;

        for (let i = 0; i < n; i++) {
          const y = paddingY + (usableH / (n + 1)) * (i + 1);
          const isMasked = l === 0 && this.maskedFeatures.has(i);
          layerNodes.push({
            x, y,
            r: Math.min(18, Math.max(8, 120 / Math.max(...this.layers))),
            activation: isMasked ? 0 : 0.3 + Math.random() * 0.7,
            isMasked,
            baseActivation: isMasked ? 0 : 0.3 + Math.random() * 0.7,
            label: l === 0 ? (this.inputLabels[i] || '') :
                   l === numLayers - 1 ? (this.outputLabels[i] || '') : '',
          });
        }
        this.nodes.push(layerNodes);
      }

      // Build connections
      for (let l = 0; l < numLayers - 1; l++) {
        for (let i = 0; i < this.nodes[l].length; i++) {
          for (let j = 0; j < this.nodes[l + 1].length; j++) {
            const weight = 0.2 + Math.random() * 0.8;
            this.connections.push({
              from: this.nodes[l][i],
              to:   this.nodes[l + 1][j],
              weight,
              layer: l,
              active: false,
            });
          }
        }
      }
    }

    /* ---------- Drawing ---------- */
    drawStatic() {
      this._clear();
      this._drawConnections(1);
      this._drawNodes(1);
      this._drawLabels();
    }

    _clear() {
      this.ctx.clearRect(0, 0, this.W, this.H);
    }

    _drawConnections(progress) {
      const ctx = this.ctx;
      const numLayers = this.layers.length;

      for (const conn of this.connections) {
        const layerProgress = (progress * (numLayers - 1) - conn.layer);
        const alpha = Math.max(0, Math.min(1, layerProgress));

        ctx.beginPath();
        ctx.moveTo(conn.from.x, conn.from.y);
        ctx.lineTo(conn.to.x, conn.to.y);

        if (conn.from.isMasked) {
          ctx.strokeStyle = COLORS.masked;
          ctx.lineWidth = 0.5;
        } else if (alpha > 0 && this.animate) {
          const glow = `rgba(6,182,212,${0.1 + alpha * 0.4})`;
          ctx.strokeStyle = glow;
          ctx.lineWidth = 1 + conn.weight * 2 * alpha;
        } else {
          ctx.strokeStyle = COLORS.connection;
          ctx.lineWidth = this.showWeights ? 0.5 + conn.weight * 1.5 : 1;
        }
        ctx.stroke();
      }
    }

    _drawNodes(progress) {
      const ctx = this.ctx;
      const numLayers = this.layers.length;

      for (let l = 0; l < numLayers; l++) {
        for (const node of this.nodes[l]) {
          const layerFrac = l / (numLayers - 1);
          const isActive = progress >= layerFrac;

          ctx.beginPath();
          ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);

          if (node.isMasked) {
            // Masked node — dashed circle, red
            ctx.fillStyle = 'rgba(239,68,68,0.1)';
            ctx.fill();
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = COLORS.maskedNode;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.setLineDash([]);

            // X mark
            const s = node.r * 0.5;
            ctx.beginPath();
            ctx.moveTo(node.x - s, node.y - s);
            ctx.lineTo(node.x + s, node.y + s);
            ctx.moveTo(node.x + s, node.y - s);
            ctx.lineTo(node.x - s, node.y + s);
            ctx.strokeStyle = COLORS.maskedNode;
            ctx.lineWidth = 2;
            ctx.stroke();
          } else {
            // Determine color
            let color;
            if (l === 0) color = COLORS.input;
            else if (l === numLayers - 1) color = COLORS.output;
            else color = COLORS.hidden;

            // Base fill
            const act = isActive && this.showActivations ? node.activation : 0.3;
            ctx.fillStyle = this._hexToRgba(color, 0.15 + act * 0.6);
            ctx.fill();

            // Glow
            if (isActive && this.animate && this.showActivations) {
              ctx.shadowColor = color;
              ctx.shadowBlur = 10 + act * 15;
            }

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
          }
        }
      }
    }

    _drawLabels() {
      const ctx = this.ctx;
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';

      // Input labels
      for (const node of this.nodes[0]) {
        if (node.label) {
          ctx.fillStyle = node.isMasked ? COLORS.maskedNode : COLORS.textDim;
          ctx.fillText(node.label, node.x - node.r - 25, node.y + 4);
        }
      }

      // Output labels
      const lastLayer = this.nodes[this.nodes.length - 1];
      for (const node of lastLayer) {
        if (node.label) {
          ctx.fillStyle = COLORS.textDim;
          ctx.fillText(node.label, node.x + node.r + 30, node.y + 4);
        }
      }

      // Layer labels
      const layerNames = ['Input', ...Array(this.layers.length - 2).fill('Hidden'), 'Output'];
      ctx.font = 'bold 12px Inter, sans-serif';
      for (let l = 0; l < this.nodes.length; l++) {
        const x = this.nodes[l][0].x;
        ctx.fillStyle = COLORS.textDim;
        ctx.fillText(layerNames[l], x, 25);
      }
    }

    _drawParticles() {
      const ctx = this.ctx;
      for (let i = this.particles.length - 1; i >= 0; i--) {
        const p = this.particles[i];
        p.t += p.speed;
        if (p.t > 1) {
          this.particles.splice(i, 1);
          continue;
        }
        const x = p.from.x + (p.to.x - p.from.x) * p.t;
        const y = p.from.y + (p.to.y - p.from.y) * p.t;
        const alpha = Math.sin(p.t * Math.PI);

        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(6,182,212,${alpha})`;
        ctx.fill();
        ctx.shadowColor = COLORS.particle;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    /* ---------- Animation Loop ---------- */
    start() {
      if (this.running) return;
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
      this.propagationProgress += this.propagationSpeed;
      if (this.propagationProgress > 1.3) {
        this.propagationProgress = 0;
        this._spawnParticles();
      }

      // Update activations with some pulse
      for (const layer of this.nodes) {
        for (const node of layer) {
          if (!node.isMasked) {
            node.activation = node.baseActivation * (0.8 + 0.2 * Math.sin(this.time * 2 + node.x * 0.01));
          }
        }
      }

      this._clear();
      this._drawConnections(this.propagationProgress);
      this._drawParticles();
      this._drawNodes(this.propagationProgress);
      this._drawLabels();

      this.animFrame = requestAnimationFrame(() => this._loop());
    }

    _spawnParticles() {
      // Spawn random particles along connections
      const numParticles = Math.min(15, this.connections.length);
      for (let i = 0; i < numParticles; i++) {
        const conn = this.connections[Math.floor(Math.random() * this.connections.length)];
        if (conn.from.isMasked) continue;
        this.particles.push({
          from: conn.from,
          to: conn.to,
          t: 0,
          speed: 0.015 + Math.random() * 0.02,
        });
      }
    }

    /* ---------- Feature Masking ---------- */
    setMaskedFeatures(indices) {
      this.maskedFeatures = new Set(indices);
      for (let i = 0; i < this.nodes[0].length; i++) {
        this.nodes[0][i].isMasked = this.maskedFeatures.has(i);
        if (this.nodes[0][i].isMasked) {
          this.nodes[0][i].activation = 0;
          this.nodes[0][i].baseActivation = 0;
        } else {
          this.nodes[0][i].baseActivation = 0.3 + Math.random() * 0.7;
        }
      }
    }

    /* ---------- Utility ---------- */
    _hexToRgba(hex, alpha) {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    destroy() {
      this.stop();
    }
  }

  /* ---------- Factory function for slide canvases ---------- */
  function initNeuralNetwork(canvasId, opts) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    return new NeuralNetworkViz(canvas, opts);
  }

  /* ---------- Expose ---------- */
  window.NeuralNetworkViz = NeuralNetworkViz;
  window.initNeuralNetwork = initNeuralNetwork;

})();
