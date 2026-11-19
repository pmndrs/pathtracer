interface Comparison {
  id: string;
  title: string;
  protocol: string;
  look: string;
  arms: [string, string] | [string, string, string];
}

// The numbered captures are display exports of the paired HDR images in the November 19 Mac review.
const comparisons: Comparison[] = [
  {
    id: 'red', title: 'Red card · What makes the bright wall seam?',
    protocol: 'Both images use full path tracing at 2,048 spp with the same seed, light, camera and six-bounce depth limit. A uses the scene’s rippled metal floor. B removes only the metal’s normal map and keeps its roughness. This camera recreates the wall seam from a nearby angle.',
    arms: ['Floor normal map on', 'Floor normal map off'],
    look: 'the white scallops on the gray wall just above the seam, then the red stripes on the floor below it. Flip A/B. The wall scallops disappear with flat floor normals, while the floor still reflects the red panel. The authored floor changes the illumination reaching the diffuse wall. The brightness is specific to this material, light and renderer.',
  },
  {
    id: '3', title: 'Blue card · Reflection strength',
    protocol: 'Both caches trace one bounce through the coated floor. Compare 256 accumulated samples after warm-up to a 4,096-spp full-path reference.',
    arms: ['Compact MLP', 'RGB hash', 'Full path tracing'],
    look: 'the floor directly beneath the blue card. Toggle R → A → R → B. Compare the width and blue intensity of the reflection and the narrow dark line below the card. A smooth image can still lose reflected color.',
  },
  {
    id: '1', title: 'Blank wall · Fixed grid',
    protocol: 'This test deliberately queries at the first hit to make the grid easier to see. Only the position encoding changes. Each arm has 256 accumulated samples from a warm cache.',
    arms: ['12 position bands', '4 position bands', '4,096-spp path tracing'],
    look: 'the blank gray wall beside the blue card and along its vertical edge. Check R for the underlying gradient. Does either arm add straight horizontal or vertical bands? Does reducing the bands soften the card shadow too much?',
  },
  {
    id: '2', title: 'Still camera · Residual noise',
    protocol: 'Both start cold and accumulate 512 samples. A keeps the cache active. B switches to full traced paths for new samples after 64. This is the final still image. Use the live shadow test below for flicker.',
    arms: ['Cache throughout', 'Trace full paths after 64', '4,096-spp path tracing'],
    look: 'the shadow just under the card, then an evenly lit patch on the wall. Against R, compare scattered grain, any stable dark patch, and the shadow width. Report if a smoother arm moves the edge or changes its brightness.',
  },
  {
    id: '4', title: 'Thin wall · Light leaking across a partition',
    protocol: 'A and B shade the red side of the same 8 mm partition. Both use frozen caches and 256 spp. The opposite face is blue.',
    arms: ['Compact MLP', 'RGB hash', '4,096-spp path tracing'],
    look: 'the horizontal wall/floor seam near the bottom of the orange image. Compare its color to R. Flag blue or gray bleeding onto the warm face, a broad dark band, or a seam that changes width.',
  },
  {
    id: '5', title: 'Grazing metal · Directional hash',
    protocol: 'The two hashes use the same memory budget. This is one fixed grazing pose with 64 frozen samples per pixel. Use the live motion test below to see a highlight move.',
    arms: ['4 × 4 direction bins', 'First-order spherical harmonics', '1,024-spp path tracing'],
    look: 'the bright streaks between the dark grooves. Compare their position, width and contrast to R. If grain overwhelms the difference between A and B, say so.',
  },
  {
    id: '6', title: 'Moved card · Shading the reflection hit',
    protocol: 'Both arms use the same traced hit in the reflections provider, 64 updates after the card moved. A shades it with the environment. B evaluates direct light and adds cached indirect light. R traces full paths.',
    arms: ['Environment hit shading', 'Direct + cached indirect', 'Full path tracing'],
    look: 'the blue patch on the right of this narrow mirror crop and the neutral area to its left. Compare each with R for brightness, tint and leftover color at the old location.',
  },
  {
    id: '7', title: 'Rough metal · Highlight energy',
    protocol: 'A predicts outgoing RGB. B predicts prefiltered incoming light and reconstructs the specular result. Both use the same rough-conductor fixture. The reference traces GGX at 4,096 spp.',
    arms: ['Raw outgoing RGB', 'Prefiltered incident light', 'Traced GGX'],
    look: 'the bright blue-white highlight in each strip. Flip back to R between arms. Note where a highlight gets dimmer, spreads out, or loses its bright center. Identify the strip by counting down from the top.',
  },
  {
    id: '8', title: 'Red reflection · Learned position features',
    protocol: 'The compact decoder and camera are fixed. Both arms had the same rendering time. A encodes position with fixed bands. B learns a spatial feature grid.',
    arms: ['Fixed position bands', 'Learned spatial features', 'Full path tracing'],
    look: 'the footprint of the reflected red card in the grooves and the bright wall/floor seam above it. Which arm preserves the boundary and brightness seen in R? Flag extra grid patterns separately from grooves present in R.',
  },
];

const select = document.querySelector<HTMLSelectElement>('#test')!;
const controls = document.querySelector<HTMLDivElement>('#arms')!;
const image = document.querySelector<HTMLImageElement>('#capture')!;
const caption = document.querySelector<HTMLElement>('#caption')!;
const keys = ['a', 'b', 'r'];

function displayArm(index: number): void {
  const test = comparisons[select.selectedIndex]!;
  const arm = test.arms[index];
  if (!arm) return;
  const file = test.id === 'red' ? ['red-mapped', 'red-flat'][index] : test.id + '-' + keys[index];
  image.src = '../results/nrc-review/' + file + '.png';
  image.alt = test.title + ' — ' + arm;
  caption.textContent = keys[index]!.toUpperCase() + ' · ' + arm;
  Array.from(controls.children).forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
}

function displayTest(): void {
  const test = comparisons[select.selectedIndex]!;
  document.querySelector('#protocol')!.textContent = test.protocol;
  document.querySelector('#look')!.textContent = test.look;
  controls.replaceChildren(...test.arms.map((arm, index) => {
    const button = document.createElement('button');
    button.textContent = keys[index]!.toUpperCase() + ' · ' + arm;
    button.addEventListener('click', () => displayArm(index));
    return button;
  }));
  displayArm(test.arms.length === 3 ? 2 : 0);
}

select.replaceChildren(...comparisons.map(test => new Option(test.title, test.id)));
const url = new URL(location.href);
if (comparisons.some(test => test.id === url.searchParams.get('test'))) select.value = url.searchParams.get('test')!;
select.addEventListener('change', () => {
  url.searchParams.set('test', select.value);
  url.hash = 'captures';
  history.replaceState(null, '', url);
  displayTest();
});
document.addEventListener('keydown', event => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const index = keys.indexOf(event.key.toLowerCase());
  if (index >= 0) {
    event.preventDefault();
    displayArm(index);
  }
});
displayTest();
