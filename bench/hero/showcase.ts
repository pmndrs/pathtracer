import './showcase.css';

const scenes = {
  'hero-classroom': { name: 'Classroom', exposure: 2.5, credit: 'Christophe Seux · Blender Classroom' },
  'hero-sponza': { name: 'Sponza', exposure: 2.5, credit: 'Crytek · Khronos glTF Sample Assets' },
};
type SceneId = keyof typeof scenes;

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
