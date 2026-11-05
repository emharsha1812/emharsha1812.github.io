/*! ASCII rendering adapted from JDhruv14/ascii-diorama, MIT license.
 * Original revision: 3d3278a0f925aebddc574dfb7042eb1eb815f427.
 * See /assets/diorama/LICENSE.txt. Scene geometry is original to this site.
 */
import * as THREE from 'three';

// The footer is deliberately small: a modest render size and shadow map keep it
// usable on laptops and phones. The scene is loaded only near the footer.
const W = 720, H = 440, COLS = 100, ROWS = 40;
const CLS = { wood: 1, leather: 2, paper: 3, metal: 4, ink: 5, flame: 6 };
const time = { value: 0 };

// As in the source scene, materials write their class into the alpha channel.
function tag(material, cls) {
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <dithering_fragment>',
      `#include <dithering_fragment>\n gl_FragColor.a = ${(cls / 16).toFixed(4)};`
    );
  };
  material.customProgramCacheKey = () => `notebook-class-${cls}`;
  return material;
}

const makeMat = (color, cls, opts = {}) => tag(new THREE.MeshLambertMaterial({ color, ...opts }), cls);

function addBox(parent, width, height, depth, material, x, y, z) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function makeClosedBook(color, angle, x, y, z) {
  const book = new THREE.Group();
  const cover = makeMat(color, CLS.leather);
  const paper = makeMat('#d8c29a', CLS.paper);
  const gold = makeMat('#b5965c', CLS.metal);
  addBox(book, 1.9, .22, 1.4, paper, 0, 0, 0);
  addBox(book, 2.02, .065, 1.5, cover, 0, .145, 0);
  addBox(book, 2.02, .065, 1.5, cover, 0, -.145, 0);
  addBox(book, .12, .3, 1.5, cover, -1, 0, 0);
  for (const dz of [-.43, 0, .43]) addBox(book, .14, .32, .035, gold, -.99, 0, dz);
  for (const bx of [-.82, .82]) addBox(book, .035, .012, 1.16, gold, bx, .183, 0);
  for (const bz of [-.56, .56]) addBox(book, 1.68, .012, .035, gold, 0, .183, bz);
  book.position.set(x, y, z);
  book.rotation.y = angle;
  return book;
}

function buildStudy(scene) {
  const desk = new THREE.Group();
  const wood = makeMat('#624333', CLS.wood);
  const woodEdge = makeMat('#402c25', CLS.wood);
  const parchment = makeMat('#ddc9a7', CLS.paper);
  const writing = makeMat('#684c37', CLS.ink);
  const brass = makeMat('#bc9559', CLS.metal);
  const wax = makeMat('#eadbb8', CLS.paper);

  // A small floating desk, with three bound books to the left.
  addBox(desk, 8.9, .24, 5.4, wood, 0, 0, 0);
  addBox(desk, 9.0, .075, 5.5, woodEdge, 0, -.16, 0);
  for (const x of [-3.75, 3.75]) for (const z of [-2.05, 2.05]) {
    addBox(desk, .28, .9, .28, woodEdge, x, -.62, z);
  }
  desk.add(makeClosedBook('#8d3f36', -.14, -2.55, .33, .18));
  desk.add(makeClosedBook('#496457', .08, -2.58, .66, .2));
  desk.add(makeClosedBook('#a46c42', -.06, -2.52, .99, .16));

  // Open pages and ruled writing provide a diagonal line through the scene.
  const open = new THREE.Group();
  const binding = makeMat('#56342b', CLS.leather);
  addBox(open, 3.05, .10, 2.08, binding, 0, 0, 0);
  for (const side of [-1, 1]) {
    addBox(open, 1.43, .13, 1.95, parchment, side * .73, .10, 0);
    for (let i = 0; i < 6; i++) {
      const lineWidth = i === 5 ? .63 : 1.05;
      addBox(open, lineWidth, .012, .025, writing, side * .73, .172, -.62 + i * .22);
    }
  }
  addBox(open, .045, .155, 1.95, binding, 0, .10, 0);
  open.position.set(.05, .22, .30);
  open.rotation.y = -.22;
  desk.add(open);

  // Candle, holder, ink pot and quill. The flame moves gently when visible.
  const plate = new THREE.Mesh(new THREE.CylinderGeometry(.48, .52, .1, 24), brass);
  plate.position.set(2.85, .24, -.52); plate.castShadow = true; desk.add(plate);
  const candle = new THREE.Mesh(new THREE.CylinderGeometry(.22, .25, 1.0, 16), wax);
  candle.position.set(2.85, .79, -.52); candle.castShadow = true; desk.add(candle);
  const wick = new THREE.Mesh(new THREE.CylinderGeometry(.018, .018, .13, 6), writing);
  wick.position.set(2.85, 1.35, -.52); desk.add(wick);
  const flameMaterial = makeMat('#f5af43', CLS.flame, { emissive: '#c57516', emissiveIntensity: .8 });
  const flame = new THREE.Mesh(new THREE.SphereGeometry(.14, 12, 9), flameMaterial);
  flame.position.set(2.85, 1.55, -.52); flame.scale.set(.8, 1.65, .8); desk.add(flame);

  const pot = new THREE.Mesh(new THREE.CylinderGeometry(.23, .31, .32, 12), makeMat('#342b27', CLS.ink));
  pot.position.set(2.0, .29, 1.42); pot.castShadow = true; desk.add(pot);
  const quill = new THREE.Group();
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(.018, .025, 1.7, 6), makeMat('#e1ca9f', CLS.paper));
  shaft.position.y = .8; quill.add(shaft);
  const featherShape = new THREE.Shape();
  featherShape.moveTo(0, .48); featherShape.quadraticCurveTo(.6, .88, .08, 1.72);
  featherShape.quadraticCurveTo(-.32, 1.1, 0, .48);
  const feather = new THREE.Mesh(new THREE.ShapeGeometry(featherShape), makeMat('#d7c5a7', CLS.paper, { side: THREE.DoubleSide }));
  quill.add(feather); quill.position.set(2.0, .46, 1.42); quill.rotation.set(-.15, -.25, .47); desk.add(quill);

  scene.add(desk);
  return { flame, flameMaterial };
}

// Character atlas and the two rendering passes are adapted from the skill's
// starter: material-aware characters followed by a gently softened edge band.
const GLYPHS = " .'`,:;-~=+*oOsS$%&8#@/\\|_^v—";
const GI = Object.fromEntries([...GLYPHS].map((c, i) => [c, i]));
const gg = c => GI[c].toFixed(1);

function buildAtlas() {
  const tw = 64, th = 120, canvas = document.createElement('canvas');
  canvas.width = tw * GLYPHS.length; canvas.height = th;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.font = 'bold 100px Menlo, Monaco, monospace';
  [...GLYPHS].forEach((ch, i) => {
    if (ch === '—') ctx.fillRect(i * tw, th * .44, tw, th * .12);
    else ctx.fillText(ch, (i + .5) * tw, th * .52);
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.flipY = false;
  return texture;
}

const vertexShader = `in vec3 position; in vec2 uv; out vec2 vUv;
void main(){ vUv=uv; gl_Position=vec4(position.xy,0.,1.); }`;

const asciiFrag = /* glsl */`
precision highp float;
uniform sampler2D uScene, uAtlas; uniform vec2 uGrid; uniform float uN, uLight;
in vec2 vUv; out vec4 outColor;
float lum(vec3 c){ return dot(c,vec3(.25,.6,.15)); }
vec3 grade(vec3 c){ c=max(c,0.); c=c/(1.+c*.12); c=pow(c,vec3(1./2.2));
  c=mix(c,c*c*(3.-2.*c),.5); float l=lum(c); return clamp(mix(vec3(l),c,1.2),0.,1.); }
void main(){
  vec2 g=vUv*uGrid, cell=floor(g), cuv=(cell+.5)/uGrid;
  ivec2 sz=textureSize(uScene,0); float lod=log2(float(sz.x)/uGrid.x)+.05;
  vec4 sampleColor=textureLod(uScene,cuv,lod);
  vec3 col=grade(sampleColor.rgb);
  float cls=floor(texelFetch(uScene,ivec2(cuv*vec2(sz)),0).a*16.+.5);
  float L=lum(col); vec2 e=1./uGrid;
  float lx0=lum(grade(textureLod(uScene,cuv-vec2(e.x,0),lod).rgb));
  float lx1=lum(grade(textureLod(uScene,cuv+vec2(e.x,0),lod).rgb));
  float ly0=lum(grade(textureLod(uScene,cuv-vec2(0,e.y),lod).rgb));
  float ly1=lum(grade(textureLod(uScene,cuv+vec2(0,e.y),lod).rgb));
  vec2 grad=vec2(lx1-lx0,(ly1-ly0)*.55); float mag=length(grad);
  float a4=mod(atan(grad.y,grad.x)+.3927,3.14159)/.7854;
  float edgeG=a4<1.?${gg('|')}:a4<2.?${gg('\\')}:a4<3.?${gg('—')}:${gg('/')};
  float gl=${gg('.')};
  if(cls<1.5) gl=L<.15?${gg(':')}:L<.38?${gg('=')}:${gg('#')};
  else if(cls<2.5) gl=L<.2?${gg(':')}:L<.48?${gg('=')}:${gg('#')};
  else if(cls<3.5) gl=L<.25?${gg(':')}:L<.55?${gg('+')}:${gg('%')};
  else if(cls<4.5) gl=L<.3?${gg(':')}:L<.6?${gg('*')}:${gg('#')};
  else if(cls<5.5) gl=L<.25?${gg(':')}:${gg('—')};
  else gl=L<.4?${gg('*')}:${gg('#')};
  if(mag>.18) gl=edgeG;
  vec2 gv=g-cell, dx=dFdx(g), dy=dFdy(g);
  float ink=smoothstep(.12,.78,textureGrad(uAtlas,vec2((gl+gv.x)/uN,1.-gv.y),dx/uN,dy).r);
  float shape=smoothstep(.005,.06,sampleColor.a);
  vec3 darkInk=clamp(col*1.3+.22,0.,1.);
  vec3 lightInk=clamp(col*.68+vec3(.06,.025,.01),0.,.82);
  // Let the desktop recede so the book and candle remain distinct at footer size.
  float opacity=shape*(cls<1.5 ? (.04+.42*ink) : (.28+.72*ink));
  // The browser composites WebGL canvases with premultiplied alpha. Keeping
  // empty cells truly transparent avoids a rectangular halo around the scene.
  outColor=vec4(mix(darkInk,lightInk,uLight)*opacity,opacity);
}`;

const tiltFrag = /* glsl */`
precision highp float;
uniform sampler2D uA; uniform vec2 uRes;
in vec2 vUv; out vec4 outColor;
void main(){
  float r=pow(smoothstep(.27,.62,abs(vUv.y-.47)),1.5)*12.;
  if(r<.2){ outColor=texture(uA,vUv); return; }
  vec4 c=vec4(0.); float lod=max(log2(r/4.),0.);
  for(int i=0;i<12;i++){
    float t=float(i)+.5, a=t*2.39996, rr=sqrt(t/12.)*r;
    c+=textureLod(uA,vUv+vec2(cos(a),sin(a))*rr/uRes,lod);
  }
  outColor=c/12.;
}`;

export function startNotebookDiorama(host, button) {
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false, powerPreference: 'low-power' });
  renderer.setPixelRatio(1);
  renderer.setSize(W, H, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.setAttribute('aria-hidden', 'true');
  renderer.domElement.className = 'notebook-scene-canvas';

  const scene = new THREE.Scene();
  const sun = new THREE.DirectionalLight('#ffe8c7', 2.4);
  sun.position.set(-7, 12, 8); sun.castShadow = true;
  sun.shadow.mapSize.set(512, 512);
  Object.assign(sun.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: .1, far: 40 });
  sun.shadow.bias = -.0005; sun.shadow.normalBias = .03;
  scene.add(sun, new THREE.HemisphereLight('#f6e9d5', '#5a4a3d', 1.25));
  const { flame, flameMaterial } = buildStudy(scene);
  const camera = new THREE.OrthographicCamera(-6, 6, 6 * H / W, -6 * H / W, .1, 100);
  const target = new THREE.Vector3(0, .2, 0);

  const raw = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter });
  const art = new THREE.WebGLRenderTarget(W, H, { generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, depthBuffer: false });
  const lightTheme = { value: document.documentElement.dataset.theme === 'dark' ? 0 : 1 };
  const ascii = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader, fragmentShader: asciiFrag, depthTest: false, depthWrite: false,
    uniforms: { uScene: { value: raw.texture }, uAtlas: { value: buildAtlas() }, uGrid: { value: new THREE.Vector2(COLS, ROWS) }, uN: { value: GLYPHS.length }, uLight: lightTheme } });
  const tilt = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader, fragmentShader: tiltFrag, depthTest: false, depthWrite: false,
    uniforms: { uA: { value: art.texture }, uRes: { value: new THREE.Vector2(W, H) } } });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), ascii);
  quad.frustumCulled = false;
  const post = new THREE.Scene(); post.add(quad);
  const postCamera = new THREE.Camera();

  function drawFrame(t) {
    time.value = t;
    const a = Math.sin(t * .12) * .035;
    camera.position.set(9 * Math.sin(.64 + a), 9.8, 9 * Math.cos(.64 + a));
    camera.lookAt(target); camera.updateMatrixWorld();
    flame.scale.set(.8 + .05 * Math.sin(t * 5), 1.55 + .12 * Math.sin(t * 4.2), .8);
    flameMaterial.emissiveIntensity = .7 + .15 * Math.sin(t * 4.5);
    renderer.setRenderTarget(raw); renderer.setClearColor(0x000000, 0); renderer.clear(); renderer.render(scene, camera);
    quad.material = ascii; renderer.setRenderTarget(art); renderer.clear(); renderer.render(post, postCamera);
    quad.material = tilt; renderer.setRenderTarget(null); renderer.clear(); renderer.render(post, postCamera);
  }

  // Draw once before hiding the static fallback. Reduced-motion visitors get a
  // still image and may choose to start it. Hidden scenes do not keep drawing.
  drawFrame(0);
  host.appendChild(renderer.domElement);
  host.querySelector('.notebook-footer-art').hidden = true;
  button.hidden = false;
  let paused = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let visible = true, frame = 0, last = 0, elapsed = 0;
  const labelButton = () => { button.textContent = paused ? 'Play animation' : 'Pause animation'; };
  function schedule() {
    if (!frame && !paused && visible && !document.hidden) frame = requestAnimationFrame(tick);
  }
  function tick(now) {
    frame = 0;
    if (last && now - last < 1000 / 24) { schedule(); return; }
    elapsed += last ? Math.min(.1, (now - last) / 1000) : 0;
    last = now; drawFrame(elapsed); schedule();
  }
  button.addEventListener('click', () => { paused = !paused; last = 0; labelButton(); schedule(); });
  document.addEventListener('visibilitychange', () => { last = 0; schedule(); });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      visible = entries.some(entry => entry.isIntersecting); last = 0; schedule();
    }, { rootMargin: '80px' }).observe(host);
  }
  new MutationObserver(() => {
    lightTheme.value = document.documentElement.dataset.theme === 'dark' ? 0 : 1;
    drawFrame(elapsed);
  }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  labelButton(); schedule();
}
