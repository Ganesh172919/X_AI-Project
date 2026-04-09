/* ============================================================
   InstaSHAP Presentation – Particle System
   ============================================================
   Background data-flow particle animation for visual ambiance.
   Particles flow across the slide to represent data movement.
   ============================================================ */

(function () {
  'use strict';

  class ParticleSystem {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.count = opts.count || 60;
      this.color = opts.color || '#3B82F6';
      this.speed = opts.speed || 0.5;
      this.connectDist = opts.connectDist || 120;
      this.showConnections = opts.showConnections !== false;
      this.particles = [];
      this.running = false;
      this.animFrame = null;

      this._resize();
      this._spawn();
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

    _spawn() {
      this.particles = [];
      for (let i = 0; i < this.count; i++) {
        this.particles.push({
          x: Math.random() * this.W,
          y: Math.random() * this.H,
          vx: (Math.random() - 0.5) * this.speed,
          vy: (Math.random() - 0.5) * this.speed,
          r: 1.5 + Math.random() * 2,
          alpha: 0.2 + Math.random() * 0.5,
        });
      }
    }

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
      this._update();
      this._draw();
      this.animFrame = requestAnimationFrame(() => this._loop());
    }

    _update() {
      for (const p of this.particles) {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = this.W;
        if (p.x > this.W) p.x = 0;
        if (p.y < 0) p.y = this.H;
        if (p.y > this.H) p.y = 0;
      }
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      // Connections
      if (this.showConnections) {
        for (let i = 0; i < this.particles.length; i++) {
          for (let j = i + 1; j < this.particles.length; j++) {
            const a = this.particles[i];
            const b = this.particles[j];
            const dx = a.x - b.x;
            const dy = a.y - b.y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < this.connectDist) {
              const alpha = (1 - dist / this.connectDist) * 0.15;
              ctx.beginPath();
              ctx.moveTo(a.x, a.y);
              ctx.lineTo(b.x, b.y);
              ctx.strokeStyle = this._colorWithAlpha(this.color, alpha);
              ctx.lineWidth = 0.5;
              ctx.stroke();
            }
          }
        }
      }

      // Particles
      for (const p of this.particles) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = this._colorWithAlpha(this.color, p.alpha);
        ctx.fill();
      }
    }

    _colorWithAlpha(hex, alpha) {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    destroy() {
      this.stop();
    }
  }

  /* ---------- Data Flow Particles (directional) ---------- */
  class DirectionalParticles {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.count = opts.count || 30;
      this.colors = opts.colors || ['#3B82F6', '#8B5CF6', '#06B6D4'];
      this.speed = opts.speed || 1.5;
      this.direction = opts.direction || 'right'; // right, left, down
      this.particles = [];
      this.running = false;
      this.animFrame = null;

      this._resize();
      this._spawn();
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

    _spawn() {
      this.particles = [];
      for (let i = 0; i < this.count; i++) {
        this.particles.push(this._createParticle());
      }
    }

    _createParticle(reset = false) {
      const color = this.colors[Math.floor(Math.random() * this.colors.length)];
      const p = {
        x: this.direction === 'right' ? (reset ? -10 : Math.random() * this.W) : Math.random() * this.W,
        y: this.direction === 'down' ? (reset ? -10 : Math.random() * this.H) : Math.random() * this.H,
        r: 2 + Math.random() * 3,
        alpha: 0.3 + Math.random() * 0.5,
        speed: this.speed * (0.5 + Math.random()),
        color,
        trail: [],
      };
      return p;
    }

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
      this._update();
      this._draw();
      this.animFrame = requestAnimationFrame(() => this._loop());
    }

    _update() {
      for (let i = 0; i < this.particles.length; i++) {
        const p = this.particles[i];

        // Store trail
        p.trail.push({ x: p.x, y: p.y });
        if (p.trail.length > 8) p.trail.shift();

        switch (this.direction) {
          case 'right': p.x += p.speed; break;
          case 'left':  p.x -= p.speed; break;
          case 'down':  p.y += p.speed; break;
        }

        // Add slight wave
        p.y += Math.sin(p.x * 0.02) * 0.3;

        // Reset if off screen
        if (p.x > this.W + 20 || p.x < -20 || p.y > this.H + 20) {
          this.particles[i] = this._createParticle(true);
        }
      }
    }

    _draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.W, this.H);

      for (const p of this.particles) {
        // Trail
        if (p.trail.length > 1) {
          ctx.beginPath();
          ctx.moveTo(p.trail[0].x, p.trail[0].y);
          for (let i = 1; i < p.trail.length; i++) {
            ctx.lineTo(p.trail[i].x, p.trail[i].y);
          }
          ctx.lineTo(p.x, p.y);
          ctx.strokeStyle = this._colorWithAlpha(p.color, p.alpha * 0.3);
          ctx.lineWidth = p.r * 0.8;
          ctx.lineCap = 'round';
          ctx.stroke();
        }

        // Main dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = this._colorWithAlpha(p.color, p.alpha);
        ctx.fill();

        // Glow
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 6;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    _colorWithAlpha(hex, alpha) {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return `rgba(${r},${g},${b},${alpha})`;
    }

    destroy() { this.stop(); }
  }

  /* ---------- Expose ---------- */
  window.ParticleSystem = ParticleSystem;
  window.DirectionalParticles = DirectionalParticles;

})();
