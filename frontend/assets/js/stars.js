/* ============================================================
   ProteinFold-RL — Shared Starfield
   Usage: call initStarfield() after DOM loads.
   ============================================================ */

function initStarfield() {
  const canvas = document.getElementById('stars-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let W, H, stars = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function initStars() {
    stars = Array.from({ length: 180 }, () => ({
      x:     Math.random() * W,
      y:     Math.random() * H,
      r:     Math.random() * 1.1 + 0.2,
      phase: Math.random() * Math.PI * 2,
      speed: Math.random() * 0.4 + 0.08,
      drift: (Math.random() - 0.5) * 0.12,
    }));
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const t = Date.now() / 1000;

    stars.forEach(s => {
      s.x += s.drift;
      if (s.x < 0) s.x = W;
      if (s.x > W) s.x = 0;

      const a = (Math.sin(s.phase + t * s.speed) * 0.5 + 0.5) * 0.5 + 0.08;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(80,170,255,${a.toFixed(2)})`;
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }

  resize();
  initStars();
  draw();

  window.addEventListener('resize', () => {
    resize();
    initStars();
  });
}