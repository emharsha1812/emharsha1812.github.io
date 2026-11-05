/* Render two local artwork layers as colored text. The control affects only the sky. */
(function () {
  const portrait = document.querySelector('.ascii-portrait');
  if (!portrait) return;

  const frame = portrait.querySelector('.ascii-portrait-frame');
  const sky = portrait.querySelector('.ascii-portrait-sky');
  const figure = portrait.querySelector('.ascii-portrait-figure');
  const canvas = portrait.querySelector('.ascii-portrait-canvas');
  const control = portrait.querySelector('input[type="range"]');
  const cols = 104;
  const rows = 80;
  const chars = ' .,:;irsXA253hMHGS#9B&@';
  let skyPixels;
  let figurePixels;
  let pending = false;

  function brightness() {
    return 0.48 + Number(control.value) / 100 * 1.3;
  }

  function setFallbackBrightness() {
    frame.style.setProperty('--ascii-sky-brightness', brightness().toFixed(2));
  }

  function coverImage(ctx, image, targetWidth, targetHeight) {
    const frameRatio = canvas.width / canvas.height;
    const imageRatio = image.naturalWidth / image.naturalHeight;
    let sx = 0;
    let sy = 0;
    let sw = image.naturalWidth;
    let sh = image.naturalHeight;
    if (imageRatio > frameRatio) {
      sw = sh * frameRatio;
      sx = (image.naturalWidth - sw) * 0.78;
    } else {
      sh = sw / frameRatio;
      sy = (image.naturalHeight - sh) / 2;
    }
    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, targetWidth, targetHeight);
  }

  function sampleLayers() {
    const skyCanvas = document.createElement('canvas');
    const figureCanvas = document.createElement('canvas');
    skyCanvas.width = figureCanvas.width = cols;
    skyCanvas.height = figureCanvas.height = rows;
    const skyCtx = skyCanvas.getContext('2d', { willReadFrequently: true });
    const figureCtx = figureCanvas.getContext('2d', { willReadFrequently: true });
    if (!skyCtx || !figureCtx) throw new Error('Canvas is unavailable');
    coverImage(skyCtx, sky, cols, rows);

    // The subject is sized in display-space so the painted figure keeps her proportions.
    const figureWidth = cols * 0.9;
    const figureHeight = rows * 0.9 * (canvas.width / canvas.height) * (figure.naturalHeight / figure.naturalWidth);
    figureCtx.drawImage(figure, (cols - figureWidth) / 2, rows - figureHeight, figureWidth, figureHeight);
    skyPixels = skyCtx.getImageData(0, 0, cols, rows).data;
    figurePixels = figureCtx.getImageData(0, 0, cols, rows).data;
  }

  function render() {
    pending = false;
    if (!skyPixels || !figurePixels) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;
    const skyLight = brightness();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#11182d';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = 'bold ' + Math.ceil(cellH * 0.94) + 'px Consolas, "Courier New", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const i = (y * cols + x) * 4;
        const alpha = figurePixels[i + 3] / 255;
        const r = Math.min(255, figurePixels[i] * alpha + skyPixels[i] * skyLight * (1 - alpha));
        const g = Math.min(255, figurePixels[i + 1] * alpha + skyPixels[i + 1] * skyLight * (1 - alpha));
        const b = Math.min(255, figurePixels[i + 2] * alpha + skyPixels[i + 2] * skyLight * (1 - alpha));
        const luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
        const glyph = chars[Math.min(chars.length - 1, Math.floor(luma * (chars.length - 1)))];
        const px = x * cellW;
        const py = y * cellH;
        ctx.fillStyle = 'rgb(' + Math.round(r * 0.32 + 5) + ',' + Math.round(g * 0.32 + 5) + ',' + Math.round(b * 0.32 + 7) + ')';
        ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
        ctx.fillStyle = 'rgb(' + Math.min(255, Math.round(r * 1.3 + 14)) + ',' + Math.min(255, Math.round(g * 1.3 + 14)) + ',' + Math.min(255, Math.round(b * 1.3 + 14)) + ')';
        ctx.fillText(glyph, px + cellW / 2, py + cellH / 2);
      }
    }
    canvas.hidden = false;
    frame.classList.add('is-ascii');
  }

  function requestRender() {
    setFallbackBrightness();
    if (!pending) {
      pending = true;
      requestAnimationFrame(render);
    }
  }

  function whenLoaded(image) {
    return new Promise(function (resolve, reject) {
      if (image.complete) {
        if (image.naturalWidth) resolve();
        else reject(new Error('Artwork could not be loaded'));
      } else {
        image.addEventListener('load', resolve, { once: true });
        image.addEventListener('error', reject, { once: true });
      }
    });
  }

  control.addEventListener('input', requestRender);
  control.parentElement.hidden = false;
  setFallbackBrightness();
  Promise.all([whenLoaded(sky), whenLoaded(figure)]).then(function () {
    try {
      sampleLayers();
      requestRender();
    } catch (error) {
      // The two image layers remain visible if canvas rendering is unavailable.
      console.warn('ASCII portrait could not be drawn:', error);
    }
  }).catch(function (error) {
    console.warn('Portrait artwork could not be loaded:', error);
  });
})();
