import { draw } from 'https://initialis.azlen.me/lib/initialis.js';

const palettes = [
  { inkA: '#6e302d', inkB: '#b68b48', inkC: '#344b70' },
  { inkA: '#30495b', inkB: '#bb914f', inkC: '#9a4738' },
  { inkA: '#43553c', inkB: '#c49b59', inkC: '#973e35' },
  { inkA: '#65472f', inkB: '#c79a56', inkC: '#394e70' },
  { inkA: '#653f4d', inkB: '#c89c5b', inkC: '#354f63' },
];

function drawInitial(element) {
  if (element.dataset.started) return;
  element.dataset.started = 'true';
  const title = element.dataset.title || '';
  const initial = element.dataset.initial || '';
  const seed = Array.from(title).reduce((value, character) => (value * 31 + character.charCodeAt(0)) % 2147483647, 17);
  const palette = palettes[Number(element.dataset.palette || 0) % palettes.length];
  const canvas = element.querySelector('canvas');
  const settings = {
    text: initial, font: 'cormorant-garamond-700', frame: 'rect', seed,
    inkA: palette.inkA, inkB: palette.inkB, inkC: palette.inkC, paper: '#f0e1c1',
    fRuleColor: 'B', oFill: 'B', lOutline: 1.5, lOutlineColor: 'C',
  };
  draw(canvas, settings, { size: 320, paper: true })
    .then(() => element.classList.add('is-ready'))
    .catch(() => { /* The plain initial remains if the artwork cannot be drawn. */ });
}

const initials = document.querySelectorAll('.project-initial');
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        observer.unobserve(entry.target);
        drawInitial(entry.target);
      }
    });
  }, { rootMargin: '300px' });
  initials.forEach((element) => observer.observe(element));
} else {
  initials.forEach(drawInitial);
}
