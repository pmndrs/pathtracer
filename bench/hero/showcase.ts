import { LIGHT_RECORD_VEC4S, PACKED_TABLE_BASE_VEC4S, makeSun, packLight } from '@pmndrs/raytracer';
import type { NrcPipelineSession } from '../nrc/pipeline-session.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import './showcase.css';

const scenes = {
  'hero-classroom': { name: 'Classroom', exposure: 2.5, credit: 'Christophe Seux · Blender Classroom' },
  'hero-sponza': { name: 'Sponza', exposure: 2.5, credit: 'Crytek · Khronos glTF Sample Assets' },
};
type SceneId = keyof typeof scenes;

export async function loadHeroScene(id: string): Promise<SceneDesc> {
  if (!Object.hasOwn(scenes, id)) throw new Error('Unknown showcase scene.');
  const res = await fetch('/bench/assets/generated/web/' + id + '.scene.json');
  if (!res.ok) throw new Error('Scene assets are missing. See bench/hero/README.md for the preparation command.');
  const scene: SceneDesc = await res.json();
  if (scene.name !== id) throw new Error('The prepared scene does not match the selected interior.');
  return scene;
}

export function setupHero() {
  document.title = 'Radiance cache · Interior studies';
  document.body.classList.add('hero');
  const app = document.querySelector<HTMLElement>('.app')!;
  const header = document.createElement('header');
  header.className = 'hero-header';
  header.innerHTML = '<div><p class="hero-project">WebGPU path tracer</p><h1>Radiance cache</h1></div><div class="hero-selector"><label for="scene">Interior</label></div>';
  const footer = document.createElement('footer');
  footer.className = 'hero-footer';
  footer.innerHTML = '<div class="hero-controls"><div><label for="stageSel">Pipeline stage</label></div><div><label for="view">Comparison</label></div><div><label for="exposure">Exposure</label></div><div class="hero-actions"></div></div><p class="hero-help">Drag to look · WASD to move · Q / E for height · Shift-drag the split</p><p class="hero-credit"></p> <a class="quality-link" href="./nrc-classroom-quality.html">Classroom quality bench ↗</a>';
  app.prepend(header);
  app.append(footer);
  function select(id: string, options: [string, string][], target: Element) {
    const el = document.querySelector<HTMLSelectElement>('#' + id)!;
    el.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
    target.append(el);
    return el;
  }
  const scene = select('scene', Object.entries(scenes).map(([id, info]) => [id, info.name]), header.querySelector('.hero-selector')!);
  const requested = new URLSearchParams(location.search).get('scene') ?? '';
  scene.value = Object.hasOwn(scenes, requested) ? requested : 'hero-classroom';
  select('stageSel', [['final', 'Final output'], ['denoised', 'Denoiser output'], ['raw', 'Raw radiance']], footer.querySelector('label[for=stageSel]')!.parentElement!);
  select('view', [['nrc', 'NRC'], ['plain', 'Plain path tracing'], ['split', 'Plain PT / NRC']], footer.querySelector('label[for=view]')!.parentElement!);
  const exposure = document.querySelector<HTMLInputElement>('#exposure')!;
  footer.querySelector('label[for=exposure]')!.parentElement!.append(exposure);
  footer.querySelector('.hero-actions')!.append(document.getElementById('resetView')!);
  const updateCredit = () => {
    const interior = scenes[scene.value as SceneId];
    footer.querySelector('.hero-credit')!.textContent = interior.credit;
    exposure.value = String(interior.exposure);
  };
  scene.addEventListener('change', updateCredit);
  updateCredit();
  const pauseButton = document.getElementById('pause')!;
  const panel = document.querySelector('aside')!;
  const controls = document.createElement('details');
  controls.className = 'hero-settings';
  controls.open = true;
  controls.innerHTML = '<summary>Render controls</summary>';
  const daylight = document.createElement('fieldset');
  daylight.innerHTML = `<h2>Daylight</h2>
    <label>Sun strength <input id="sunStrength" type="range" min="0" max="6" step="0.1" value="1"><output></output></label>
    <label>Sun azimuth <input id="sunAzimuth" type="range" min="-180" max="180" step="1"><output></output></label>
    <label>Sun elevation <input id="sunElevation" type="range" min="1" max="89" step="1"><output></output></label>
    <label>Sun radius <input id="sunRadius" type="range" min="0" max="2" step="0.001"><output></output></label>`;
  panel.prepend(daylight);
  for (const label of panel.querySelectorAll('label')) if (!label.children.length) label.remove();
  const denoiserGroup = document.getElementById('denoiserSel')!.closest('fieldset')!;
  denoiserGroup.querySelector('h2')!.textContent = 'Denoiser';
  document.getElementById('pause')!.parentElement!.remove();
  footer.querySelector('.hero-actions')!.prepend(pauseButton);
  document.getElementById('denoiserSel')!.querySelector('option[value=oidn]')!.textContent = 'OIDN adaptive (continues while still)';
  controls.append(...panel.querySelectorAll('fieldset'));
  footer.append(controls);
  document.querySelector<HTMLInputElement>('#accumulateStill')!.checked = true;
  document.querySelector<HTMLInputElement>('#rcasDenoise')!.checked = true;
  const native = new URLSearchParams(location.search).get('preset') === 'classroom-native';
  document.querySelector<HTMLSelectElement>('#res')!.value = native ? '1280x720' : '640x360';
  if (native) {
    document.querySelector<HTMLSelectElement>('#displayScale')!.value = '1';
    document.querySelector<HTMLInputElement>('#maxDepth')!.value = '10';
    document.querySelector<HTMLInputElement>('#fireflyClamp')!.value = '102.8';
  }
}

const sunControl = (id: string) => document.getElementById(id) as HTMLInputElement;

export function setupDaylight(session: NrcPipelineSession): void {
  const sun = session.options.scene.lights?.find(light => light.type === 'directional');
  if (!sun) return;
  const [x, y, z] = sun.direction;
  sunControl('sunAzimuth').value = String(Math.round(Math.atan2(-z, -x) * 180 / Math.PI));
  sunControl('sunElevation').value = String(Math.round(Math.atan2(-y, Math.hypot(x, z)) * 180 / Math.PI));
  sunControl('sunRadius').value = String((sun.angularRadius ?? 0) * 180 / Math.PI);
  sunControl('sunStrength').value = '1';
  if (session.options.scene.name === 'hero-classroom' && new URLSearchParams(location.search).get('preset') === 'classroom-native') {
    sunControl('sunAzimuth').value = '65';
    sunControl('sunElevation').value = '27';
    sunControl('sunRadius').value = '0.048';
  }
  updateDaylight(session);
}

export function updateDaylight(session: NrcPipelineSession): void {
  const { copy, input, sceneBuffers } = session.adapter.debug;
  if (!input.sun || !sceneBuffers.packed) return;
  const strength = sunControl('sunStrength').valueAsNumber;
  const az = sunControl('sunAzimuth').valueAsNumber * Math.PI / 180;
  const el = sunControl('sunElevation').valueAsNumber * Math.PI / 180;
  const sun = makeSun({ ...input.sun, angularRadius: sunControl('sunRadius').valueAsNumber * Math.PI / 180, direction: [-Math.cos(el) * Math.cos(az), -Math.sin(el), -Math.cos(el) * Math.sin(az)],
    irradiance: [input.sun.irradiance[0] * strength, input.sun.irradiance[1] * strength, input.sun.irradiance[2] * strength] }, copy.bounds.radius);
  const i = copy.sunLight;
  // Keep the original nonzero selection probability for this A/B. NEE and MIS
  // share that proposal; just the sampled radiance and its direction change.
  const word = 4 * (copy.sections.lights + i * LIGHT_RECORD_VEC4S);
  const record = new Uint32Array(4 * LIGHT_RECORD_VEC4S);
  packLight(sun, copy.alias.pmf[i]!, copy.alias.prob[i]!, copy.alias.alias[i]!, record, 0);
  session.device.queue.writeBuffer(sceneBuffers.packed, (4 * PACKED_TABLE_BASE_VEC4S + word) * 4, record);
  for (const id of ['sunStrength', 'sunAzimuth', 'sunElevation', 'sunRadius']) {
    const field = sunControl(id);
    field.nextElementSibling!.textContent = field.value + (id === 'sunStrength' ? '×' : '°');
  }
  session.cutNext();
}
