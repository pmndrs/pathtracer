import { bendTable, fullOffset, makeModel, reach, recordSweep, type Inputs, type Pose, type Reading } from './model.ts';

const el = (id: string) => document.getElementById(id)!;
const field = el('frame') as HTMLInputElement;
const train = el('fit-model') as HTMLButtonElement;
const captureButtons = [...document.querySelectorAll<HTMLButtonElement>('[data-twist]')];
let inputs: Inputs = 'bend';
let model = makeModel(inputs);
let steps = 0;
const savedModels = new Map<Inputs, { model: ReturnType<typeof makeModel>; steps: number }>();
let data: Reading[] = [];
type Predict = (p: Pose) => number;

function errors(rows: Reading[], predict: Predict) {
  const deltas = rows.map(row => fullOffset * Math.abs(predict(row) - row.target));
  const worst = Math.max(...deltas);
  return { mean: deltas.reduce((sum, d) => sum + d, 0) / rows.length, worst, frame: deltas.indexOf(worst) };
}
function evaluate(predict: Predict) {
  return { train: errors(data, predict).mean, test: errors(reach, predict) };
}
const results = new Map<Inputs, ReturnType<typeof evaluate> & { steps: number }>();
const mm = (value: number) => value.toFixed(2) + ' mm';
const names = { bend: 'Bend only', both: 'Bend + twist' };

function displayFrame() {
  const frame = field.valueAsNumber;
  const p = reach[frame]!;
  const prediction = model.predict(p);
  const rad = p.bend * Math.PI / 180;
  const forearm = `M153 92l${86 * Math.cos(rad)} ${-86 * Math.sin(rad)}`;
  el('forearm').setAttribute('d', forearm);
  el('bone').setAttribute('d', forearm);
  // The direction follows the outside of the joint. Start at a common skin marker.
  const dx = Math.sin(rad / 2), dy = Math.cos(rad / 2);
  const point = (drive: number) => [153 + dx * (15 + 38 * drive), 92 + dy * (15 + 38 * drive)] as const;
  for (const [id, drive] of [['target-dot', p.target], ['pred-dot', prediction]] as const) {
    const [x, y] = point(drive);
    el(id).setAttribute('cx', String(x)); el(id).setAttribute('cy', String(y));
  }
  el('offset-line').setAttribute('d', `M153 92L${point(Math.max(p.target, prediction)).join(' ')}`);
  el('angles').textContent = `${p.bend.toFixed(0)}° bend · ${p.twist.toFixed(0)}° twist`;
  el('frame-value').textContent = String(frame);
  el('pose-readout').textContent = `Rig drive   ${p.target.toFixed(3)} → ${mm(p.target * fullOffset)}\nMLP drive   ${prediction.toFixed(3)} → ${mm(prediction * fullOffset)}\nFrame error ${mm(fullOffset * Math.abs(p.target - prediction))}`;
}

function scoreTable() {
  const tbody = el('scores') as HTMLTableSectionElement;
  tbody.replaceChildren();
  function row(name: string, count: string, score?: ReturnType<typeof evaluate>, inspect = false) {
    const tr = tbody.insertRow();
    for (const text of [name, count, score ? mm(score.train) : '—', score ? mm(score.test.mean) : '—']) tr.insertCell().textContent = text;
    const last = tr.insertCell();
    if (inspect && score) {
      const button = document.createElement('button');
      button.textContent = `${mm(score.test.worst)} · frame ${score.test.frame}`;
      button.addEventListener('click', () => {
        field.value = String(score.test.frame);
        displayFrame();
        field.focus();
      });
      last.append(button);
    } else last.textContent = score ? mm(score.test.worst) : '—';
  }
  const baseline = data.length ? evaluate(bendTable(data)) : undefined;
  row('Bend table', '—', baseline);
  for (const key of ['bend', 'both'] as const) {
    const score = results.get(key);
    row(names[key], score ? String(score.steps) : '—', score, key === inputs);
  }
  const current = results.get(inputs);
  el('verdict').textContent = baseline && current
    ? `${names[inputs]} has ${current.test.mean < baseline.test.mean ? 'lower' : 'equal or higher'} mean reach error than the bend table. Its worst frame ${current.test.worst <= 2 ? 'meets' : 'exceeds'} the 2 mm budget.`
    : 'Train the selected model to score it. The bend table averages the captured drives at each bend and interpolates between them.';
}
function refresh() {
  train.disabled = !data.length;
  el('count').textContent = data.length + ' training poses';
  el('updates').textContent = steps + ' updates';
  el('training-status').textContent = !data.length ? 'Capture a sweep to enable training.'
    : steps ? `Trained on ${data.length} poses. Inspect the reach below.` : 'Starting weights are ready. Train a batch to compare it with the bend table.';
  el('wiring').textContent = inputs === 'bend' ? 'x = [bend / 120]\n1 input → 8 ReLU neurons → 1 drive\n25 trainable parameters'
    : 'x = [bend / 120, twist / 60]\n2 inputs → 8 ReLU neurons → 1 drive\n33 trainable parameters';
  const tbody = el('rows') as HTMLTableSectionElement;
  tbody.replaceChildren();
  if (!data.length) {
    const cell = tbody.insertRow().insertCell(); cell.colSpan = 4; cell.textContent = 'Capture a sweep to inspect its label.';
  }
  for (const p of data.filter(p => p.bend === 90)) {
    const tr = tbody.insertRow();
    for (const s of ['90°', p.twist + '°', p.target.toFixed(2), mm(p.target * fullOffset)]) tr.insertCell().textContent = s;
  }
  for (const button of captureButtons) button.disabled = data.some(row => row.twist === Number(button.dataset.twist));
  displayFrame(); scoreTable();
}
function resetModel() {
  model = makeModel(inputs); steps = 0;
  results.delete(inputs); savedModels.delete(inputs);
  refresh();
}
for (const button of captureButtons) button.addEventListener('click', () => {
  data = data.concat(recordSweep(Number(button.dataset.twist)));
  results.clear(); savedModels.clear(); resetModel();
});
for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="features"]')) radio.addEventListener('change', () => {
  inputs = radio.value === 'both' ? 'both' : 'bend';
  const saved = savedModels.get(inputs);
  model = saved?.model ?? makeModel(inputs);
  steps = saved?.steps ?? 0;
  refresh();
});
train.addEventListener('click', () => {
  model.train(data, 500);
  steps += 500;
  savedModels.set(inputs, { model, steps });
  results.set(inputs, { ...evaluate(model.predict), steps });
  refresh();
});
el('reset-model').addEventListener('click', resetModel);
el('clear-data').addEventListener('click', () => {
  data = []; results.clear(); savedModels.clear(); resetModel(); captureButtons[0]!.focus();
});
field.addEventListener('input', displayFrame);
refresh();
