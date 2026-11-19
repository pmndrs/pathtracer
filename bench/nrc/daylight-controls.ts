import { LIGHT_RECORD_VEC4S, PACKED_TABLE_BASE_VEC4S, makeSun, packLight } from '@pmndrs/raytracer';
import type { NrcPipelineSession } from './pipeline-session.ts';
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
