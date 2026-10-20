/* ============================================================
   InstaSHAP Presentation – Core Slide Engine
   ============================================================
   Handles:
   - Slide navigation (arrows, keyboard, touch)
   - Progressive reveal (click-to-reveal)
   - Progress bar
   - Section tracking
   - Fullscreen toggle
   ============================================================ */

(function () {
  'use strict';

  /* ---------- State ---------- */
  let currentSlide = 0;
  let totalSlides = 0;
  let isTransitioning = false;

  const TRANSITION_MS = 600;

  const SECTIONS = [
    { start: 0,   label: '🎯 Motivation' },
    { start: 7,   label: '💡 What is InstaSHAP' },
    { start: 12,  label: '⚙️ Full Workflow' },
    { start: 35,  label: '🔬 Internal Mechanism' },
    { start: 45,  label: '⚡ Why Fast' },
    { start: 53,  label: '❌ Failure Analysis' },
    { start: 73,  label: '⚠️ Limitations' },
    { start: 83,  label: '🔬 Applicability' },
    { start: 91,  label: '🎮 Interactive Sim' },
    { start: 101, label: '🚀 Improvements v2' },
  ];

  /* ---------- DOM References ---------- */
  let slides, counter, sectionLabel, progressBar;

  /* ---------- Init ---------- */
  function init() {
    slides      = document.querySelectorAll('.slide');
    totalSlides = slides.length;
    counter      = document.getElementById('slide-counter');
    sectionLabel = document.getElementById('section-label');
    progressBar  = document.getElementById('top-progress');

    if (totalSlides === 0) return;

    // Show first slide
    goToSlide(0, false);

    // Keyboard navigation
    document.addEventListener('keydown', handleKeyDown);

    // Touch navigation
    let touchStartX = 0;
    document.addEventListener('touchstart', e => { touchStartX = e.touches[0].clientX; });
    document.addEventListener('touchend', e => {
      const dx = e.changedTouches[0].clientX - touchStartX;
      if (Math.abs(dx) > 60) dx > 0 ? prevSlide() : nextSlide();
    });

    // HUD buttons
    document.getElementById('btn-prev')?.addEventListener('click', prevSlide);
    document.getElementById('btn-next')?.addEventListener('click', nextSlide);
    document.getElementById('btn-fs')?.addEventListener('click', toggleFullscreen);

    // Click-to-reveal within slides
    document.querySelectorAll('.reveal-group').forEach(group => {
      group.addEventListener('click', () => revealNext(group));
    });

    // Initialize all canvases when slides become active
    observeSlides();

    console.log(`[Core] Initialized with ${totalSlides} slides.`);
  }

  /* ---------- Navigation ---------- */
  function goToSlide(idx, animate = true) {
    if (idx < 0 || idx >= totalSlides) return;
    if (isTransitioning && animate) return;

    isTransitioning = true;

    // Deactivate current
    slides.forEach((s, i) => {
      s.classList.remove('active', 'prev');
      if (i < idx) s.classList.add('prev');
    });

    // Activate target
    slides[idx].classList.add('active');
    currentSlide = idx;

    updateHUD();
    fireSlideCallbacks(idx);

    if (animate) {
      setTimeout(() => { isTransitioning = false; }, TRANSITION_MS);
    } else {
      isTransitioning = false;
    }
  }

  function nextSlide() { goToSlide(currentSlide + 1); }
  function prevSlide() { goToSlide(currentSlide - 1); }

  function handleKeyDown(e) {
    switch (e.key) {
      case 'ArrowRight':
      case ' ':
      case 'Enter':
        e.preventDefault();
        nextSlide();
        break;
      case 'ArrowLeft':
      case 'Backspace':
        e.preventDefault();
        prevSlide();
        break;
      case 'Home':
        e.preventDefault();
        goToSlide(0);
        break;
      case 'End':
        e.preventDefault();
        goToSlide(totalSlides - 1);
        break;
      case 'f':
      case 'F':
        toggleFullscreen();
        break;
      case 'Escape':
        if (document.fullscreenElement) document.exitFullscreen();
        break;
    }
  }

  /* ---------- HUD ---------- */
  function updateHUD() {
    if (counter) counter.textContent = `${currentSlide + 1} / ${totalSlides}`;
    if (progressBar) progressBar.style.width = `${((currentSlide + 1) / totalSlides) * 100}%`;
    if (sectionLabel) sectionLabel.textContent = getCurrentSection();
  }

  function getCurrentSection() {
    let label = SECTIONS[0].label;
    for (const s of SECTIONS) {
      if (currentSlide >= s.start) label = s.label;
    }
    return label;
  }

  /* ---------- Fullscreen ---------- */
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen();
    }
  }

  /* ---------- Click-to-Reveal ---------- */
  function revealNext(group) {
    const items = group.querySelectorAll('.reveal-item:not(.revealed)');
    if (items.length > 0) {
      items[0].classList.add('revealed');
    }
  }

  /* ---------- Slide Observation (Canvas Init) ---------- */
  const slideCallbacks = {};

  function onSlideEnter(slideIdx, fn) {
    if (!slideCallbacks[slideIdx]) slideCallbacks[slideIdx] = [];
    slideCallbacks[slideIdx].push(fn);
  }

  function fireSlideCallbacks(idx) {
    if (slideCallbacks[idx]) {
      slideCallbacks[idx].forEach(fn => fn());
    }
    // Fire generic event for any module listening
    window.dispatchEvent(new CustomEvent('slideEnter', { detail: { index: idx } }));
  }

  function observeSlides() {
    // MutationObserver approach – modules register via onSlideEnter
  }

  /* ---------- Expose API ---------- */
  window.Presentation = {
    init,
    next: nextSlide,
    prev: prevSlide,
    goTo: goToSlide,
    onSlideEnter,
    getCurrentSlide: () => currentSlide,
    getTotalSlides: () => totalSlides,
  };

  /* ---------- Auto-init ---------- */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
