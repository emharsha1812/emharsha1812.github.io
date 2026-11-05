# Notebook footer diorama

The animated study is an original scene made from simple geometry. Its ASCII
character atlas, material-class rendering, and soft-focus pass are adapted from
[JDhruv14/ascii-diorama](https://github.com/JDhruv14/ascii-diorama), revision
`3d3278a0f925aebddc574dfb7042eb1eb815f427`, under the MIT license in
`assets/diorama/LICENSE.txt`. No external image, texture, or model assets are used.
Three.js is also distributed under the MIT license; its license is included at
`assets/diorama/THREE-LICENSE.txt`.

To rebuild after changing `main.js`, run `npm ci` and `npm run build` from this
directory. The bundled output is `assets/diorama/notebook.min.js`, which is checked
in so the Jekyll build does not require Node.js.

The scene loads when the footer is close to the viewport. Visitors can pause it;
reduced-motion preferences start it paused. If loading or WebGL fails, the
static text illustration remains visible.
