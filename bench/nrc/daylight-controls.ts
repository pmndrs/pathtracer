/**
 * Live lighting controls of the interior scenes (nrc-pipeline, the Classroom showcase): the sun's strength, direction and size, and
 * on / off switches for the sky and for the interior emitters (ceiling lamps, the blackboard light). Nothing is rebuilt: the light records
 * and the emissive material records are rewritten in the packed scene copy, so a change shows on the next frame and the radiance cache
 * relearns the new lighting while you watch.
 */
import { DEFAULT_LIGHT_SELECTION, LIGHT_RECORD_VEC4S, MATERIAL_RECORD_VEC4S, PACKED_TABLE_BASE_VEC4S, buildAliasTable, lightSelectionWeights, makeSun, packLight, packMaterial, type RtLight } from '@pmndrs/raytracer';
import type { NrcPipelineSession } from './pipeline-session.ts';

const field = (id: string) => document.getElementById(id) as HTMLInputElement;
const SUN_FIELDS = ['sunStrength', 'sunAzimuth', 'sunElevation', 'sunRadius'];
export const LIGHTING_FIELDS = [...SUN_FIELDS, 'interiorLights', 'lampStrength', 'skyLight'];

/** The lighting the controls describe, as a string: a saved network is only valid for the lighting it learned (N11.9). */
export const lightingState = (): string => LIGHTING_FIELDS.map((id) => (field(id).type === 'checkbox' ? Number(field(id).checked) : field(id).value)).join(',');

/** Emission scale of the interior emitters: the switch times the brightness slider. */
const lampScale = (): number => (field('interiorLights').checked ? Math.max(0, field('lampStrength').valueAsNumber) : 0);

export function setupDaylight(session: NrcPipelineSession): void {
  const { copy } = session.adapter.debug;
  const sun = session.options.scene.lights?.find(light => light.type === 'directional');
  // A switch only shows when the scene has that kind of light.
  const lamps = copy.lights.some(light => light.kind === 'triangle');
  field('interiorLights').closest('label')!.hidden = !lamps;
  field('lampStrength').closest('label')!.hidden = !lamps;
  field('lampStrength').value = '1';
  field('skyLight').closest('label')!.hidden = !copy.lights.some(light => light.kind === 'environment');
  for (const id of SUN_FIELDS) field(id).closest('label')!.hidden = !sun;
  field('interiorLights').checked = true;
  field('skyLight').checked = true;
  if (sun) {
    const [x, y, z] = sun.direction;
    field('sunAzimuth').value = String(Math.round(Math.atan2(-z, -x) * 180 / Math.PI));
    field('sunElevation').value = String(Math.round(Math.atan2(-y, Math.hypot(x, z)) * 180 / Math.PI));
    field('sunRadius').value = String((sun.angularRadius ?? 0) * 180 / Math.PI);
    field('sunStrength').value = '1';
    if (session.options.scene.name === 'hero-classroom' && new URLSearchParams(location.search).get('preset') === 'classroom-native') {
      field('sunAzimuth').value = '65';
      field('sunElevation').value = '27';
      field('sunRadius').value = '0.048';
    }
  }
  updateDaylight(session);
}

const scaled = (v: readonly number[], k: number): [number, number, number] => [v[0]! * k, v[1]! * k, v[2]! * k];

/** The scene's lights with the controls applied (`copy.lights` itself stays as built: it is what "on" means). */
export function controlledLights(session: NrcPipelineSession): RtLight[] {
  const { copy, input } = session.adapter.debug;
  const interior = lampScale();
  const sky = field('skyLight').checked ? 1 : 0;
  return copy.lights.map((light): RtLight => {
    if (light.kind === 'triangle') return { ...light, radiance: scaled(light.radiance, interior), power: light.power * interior };
    if (light.kind === 'environment') return { ...light, radiance: scaled(light.radiance, sky), power: light.power * sky };
    if (!input.sun) return light;
    const strength = field('sunStrength').valueAsNumber;
    const az = field('sunAzimuth').valueAsNumber * Math.PI / 180;
    const el = field('sunElevation').valueAsNumber * Math.PI / 180;
    return makeSun({ ...input.sun, angularRadius: field('sunRadius').valueAsNumber * Math.PI / 180, direction: [-Math.cos(el) * Math.cos(az), -Math.sin(el), -Math.cos(el) * Math.sin(az)], irradiance: scaled(input.sun.irradiance, strength) }, copy.bounds.radius);
  });
}

export function updateDaylight(session: NrcPipelineSession): void {
  const { copy, sceneBuffers } = session.adapter.debug;
  const packed = sceneBuffers.packed;
  if (!packed || copy.lights.length === 0) return;
  const lights = controlledLights(session);
  // The light picks follow the new powers (a light that is off is never picked, so next-event estimation wastes no sample on it). Next-event
  // estimation and the MIS weight of a BSDF-sampled hit both read the pmf from these records, so the estimator stays consistent.
  // If everything is off, the table as built stays: every light then returns zero radiance whichever is picked.
  const weights = lightSelectionWeights(lights, DEFAULT_LIGHT_SELECTION);
  const alias = weights.some(w => w > 0) ? buildAliasTable(weights) : copy.alias;
  const records = new Uint32Array(lights.length * LIGHT_RECORD_VEC4S * 4);
  lights.forEach((light, i) => packLight(light, alias.pmf[i]!, alias.prob[i]!, alias.alias[i]!, records, i * LIGHT_RECORD_VEC4S * 4));
  const tableByte = (vec4: number) => (PACKED_TABLE_BASE_VEC4S + vec4) * 16;
  session.device.queue.writeBuffer(packed, tableByte(copy.sections.lights), records);
  // A camera or bounce ray that hits an emitter reads the emission from its material, not from the light record.
  const interior = lampScale();
  const record = new Uint32Array(MATERIAL_RECORD_VEC4S * 4);
  copy.materials.forEach((material, i) => {
    if (!(material.emissionLuminance > 0)) return;
    packMaterial(interior === 1 ? material : { ...material, emissionColor: scaled(material.emissionColor, interior), emissionLuminance: material.emissionLuminance * interior }, record, 0);
    session.device.queue.writeBuffer(packed, tableByte(copy.sections.materials + i * MATERIAL_RECORD_VEC4S), record);
  });
  for (const id of [...SUN_FIELDS, 'lampStrength']) field(id).nextElementSibling!.textContent = field(id).value + (id.endsWith('Strength') ? '×' : '°');
  field('lampStrength').disabled = !field('interiorLights').checked;
  session.cutNext();
}
