// One input, two ReLU neurons, a linear readout. All seven parameters are trainable.
const initial = [1, 0, 1, -0.5, 0.4, -0.6, 0.2] as const;
let weights: number[] = [...initial];
let updates = 0;
const training = Array.from({ length: 9 }, (_, i) => i / 8);
const heldOut = Array.from({ length: 8 }, (_, i) => (i + 0.5) / 8);
const measured = (x: number) => 0.2 + 0.95 * x - 1.6 * Math.max(0, x - 0.55);

function forward(x: number) {
  const [wa, ba, wb, bb, va, vb, bias] = weights as [number, number, number, number, number, number, number];
  const a = Math.max(0, wa * x + ba);
  const b = Math.max(0, wb * x + bb);
  return { a, b, y: va * a + vb * b + bias };
}

function trainStep() {
  const gradient = Array<number>(weights.length).fill(0);
  for (const x of training) {
    const { a, b, y } = forward(x);
    const dy = 2 * (y - measured(x)) / training.length;
    const da = a > 0 ? dy * weights[4]! : 0;
    const db = b > 0 ? dy * weights[5]! : 0;
    const partials = [da * x, da, db * x, db, dy * a, dy * b, dy];
    partials.forEach((value, i) => { gradient[i]! += value; });
  }
  weights = weights.map((w, i) => w - 0.12 * gradient[i]!);
  updates++;
  return gradient[4]!;
}

const el = (id: string) => document.getElementById(id)!;
const neuronInputs = ['nx', 'nw', 'nb'].map(id => el(id) as HTMLInputElement);
function showArithmetic() {
  const [x, w, b] = neuronInputs.map(field => field.valueAsNumber) as [number, number, number];
  neuronInputs.forEach(field => { el(field.id + '-value').textContent = field.valueAsNumber.toFixed(2); });
  const product = w * x, z = product + b;
  el('arithmetic').textContent = [
    `Multiply   w × x        = ${w.toFixed(2)} × ${x.toFixed(2)} = ${product.toFixed(2)}`,
    `Add bias   z            = ${product.toFixed(2)} ${b < 0 ? '−' : '+'} ${Math.abs(b).toFixed(2)} = ${z.toFixed(2)}`,
    `ReLU       y = max(0,z) = ${Math.max(0, z).toFixed(2)}`,
  ].join('\n');
}
neuronInputs.forEach(field => field.addEventListener('input', showArithmetic));
el('neuron-reset').addEventListener('click', () => {
  neuronInputs.forEach(field => { field.value = field.defaultValue; });
  showArithmetic();
});
showArithmetic();

const dial = el('dial') as HTMLInputElement;
const svg = 'http://www.w3.org/2000/svg';
const px = (x: number) => 36 + x * 468;
const py = (y: number) => 179 - y * 166;
for (const [inputs, isTraining] of [[training, true], [heldOut, false]] as const) {
  for (const x of inputs) {
    const cx = px(x), cy = py(measured(x));
    const mark = document.createElementNS(svg, isTraining ? 'circle' : 'path');
    if (isTraining) {
      mark.setAttribute('cx', String(cx)); mark.setAttribute('cy', String(cy));
      mark.setAttribute('r', '4.5'); mark.setAttribute('fill', '#ffb894');
    } else {
      mark.setAttribute('d', `M${cx-3} ${cy-3}l6 6m0 -6l-6 6`);
      mark.setAttribute('stroke', '#d3d9ff'); mark.setAttribute('stroke-width', '2');
    }
    el('samples').append(mark);
  }
}

function draw() {
  const x = dial.valueAsNumber;
  const { a, b, y } = forward(x);
  for (const [id, value] of Object.entries({ vx: x, va: a, vb: b, vy: y })) el(id).textContent = value.toFixed(3);
  el('dial-value').textContent = x.toFixed(2);
  const magnitude = (i: number) => Math.abs(weights[i]!).toFixed(3);
  const signed = (i: number) => (weights[i]! < 0 ? '− ' : '+ ') + magnitude(i);
  el('formula').textContent = `A = ReLU(${weights[0]!.toFixed(3)} × x ${signed(1)})\nB = ReLU(${weights[2]!.toFixed(3)} × x ${signed(3)})\ny = ${weights[4]!.toFixed(3)} × A ${signed(5)} × B ${signed(6)}`;
  const path = Array.from({ length: 121 }, (_, i) => `${i ? 'L' : 'M'}${px(i/120)} ${py(forward(i/120).y)}`);
  el('curve').setAttribute('d', path.join(' '));
  el('probe').setAttribute('cx', String(px(x))); el('probe').setAttribute('cy', String(py(y)));
  el('steps').textContent = updates + ' updates';
  for (const [id, inputs] of [['loss', training], ['test-loss', heldOut]] as const) {
    const error = inputs.reduce((sum, input) => sum + (forward(input).y - measured(input)) ** 2, 0) / inputs.length;
    el(id).textContent = error < 0.00001 ? error.toExponential(2) : error.toFixed(5);
  }
}
dial.addEventListener('input', draw);
function runUpdates(count: number) {
  for (let i = 1; i < count; i++) trainStep();
  const before = weights[4]!;
  const gradient = trainStep();
  el('weight-update').textContent = `Update ${updates}, output weight on A: ${before.toFixed(5)} − 0.12 × (${gradient.toFixed(5)}) = ${weights[4]!.toFixed(5)}. Here 0.12 is the learning rate. The gradient combines all nine training readings.`;
  draw();
}
el('train-one').addEventListener('click', () => runUpdates(1));
el('train').addEventListener('click', () => runUpdates(100));
el('reset').addEventListener('click', () => { weights = [...initial]; updates = 0; el('weight-update').textContent = ''; draw(); });
draw();

const sections = [...document.querySelectorAll<HTMLElement>('.chapters section')];
const links = [...document.querySelectorAll<HTMLAnchorElement>('.contents a')];
function selectChapter(hash: string) {
  for (const link of links) {
    if (link.hash === hash) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
}
function markChapter() {
  const active = sections.filter(section => section.getBoundingClientRect().top < window.innerHeight * 0.36).pop() ?? sections[0];
  if (active) selectChapter('#' + active.id);
}
for (const link of links) link.addEventListener('click', () => selectChapter(link.hash));
window.addEventListener('scroll', markChapter, { passive: true });
window.addEventListener('hashchange', markChapter);
markChapter();

// The gallery dev server does not build the heavyweight Classroom showcase.
if (import.meta.env.DEV) {
  for (const link of document.querySelectorAll<HTMLAnchorElement>('[data-classroom]')) link.href = 'https://pmndrs.github.io/pathtracer/examples/showcase/classroom/';
}
