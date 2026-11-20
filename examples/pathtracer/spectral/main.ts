/**
 * Spectral rendering and dispersion (experimental).
 *
 * With `integrator.spectral` every path carries three wavelengths instead of RGB: colours are lifted to spectra (an RGB-to-spectrum
 * table, fitted on first use), lights emit spectra, and the path's result is converted back to RGB. That lets a dielectric's index of
 * refraction depend on the wavelength (`dispersion` on the material, the glTF KHR_materials_dispersion value): white light splits.
 * Here a glass prism and a slab stand in front of bright bars, so the split shows directly in the camera's view as colour fringes.
 * In RGB mode the same scene renders without them (dispersion is ignored).
 */
import { buildGpuScene, type GpuScene, type RtMaterialInput, type SceneCopyInput, type SceneCopyMesh } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

type V3 = [number, number, number];

/** A convex closed mesh from polygons (any winding): each face is turned to face away from the centre, as a thick dielectric needs. */
function convex(faces: V3[][]): SceneCopyMesh {
  const all = faces.flat();
  const centre = [0, 1, 2].map((i) => all.reduce((s, p) => s + p[i]!, 0) / all.length);
  const positions: number[] = [], normals: number[] = [], indices: number[] = [];
  for (let face of faces) {
    const fc = [0, 1, 2].map((i) => face.reduce((s, p) => s + p[i]!, 0) / face.length);
    const [a, b, c] = face as [V3, V3, V3];
    let n = [(b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]), (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])];
    if (n[0]! * (fc[0]! - centre[0]!) + n[1]! * (fc[1]! - centre[1]!) + n[2]! * (fc[2]! - centre[2]!) < 0) {
      face = [...face].reverse();
      n = n.map((v) => -v);
    }
    const l = Math.hypot(n[0]!, n[1]!, n[2]!);
    const base = positions.length / 3;
    for (const p of face) {
      positions.push(...p);
      normals.push(n[0]! / l, n[1]! / l, n[2]! / l);
    }
    for (let k = 1; k + 1 < face.length; k++) indices.push(base, base + k, base + k + 1);
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices) };
}

function prism(cx: number, cz: number, r: number, h: number, turn: number): SceneCopyMesh {
  const corner = (k: number): [number, number] => [cx + r * Math.cos(turn + (k * 2 * Math.PI) / 3), cz + r * Math.sin(turn + (k * 2 * Math.PI) / 3)];
  const c = [0, 1, 2].map(corner);
  const sides = [0, 1, 2].map((k) => {
    const [a, b] = [c[k]!, c[(k + 1) % 3]!];
    return [[a[0], 0, a[1]], [b[0], 0, b[1]], [b[0], h, b[1]], [a[0], h, a[1]]] as V3[];
  });
  return convex([c.map(([x, z]) => [x, 0, z] as V3), c.map(([x, z]) => [x, h, z] as V3), ...sides]);
}

function slab(cx: number, cz: number, w: number, h: number, d: number, turn: number): SceneCopyMesh {
  const co = Math.cos(turn), si = Math.sin(turn);
  const p = (x: number, y: number, z: number): V3 => [cx + co * x + si * z, y, cz - si * x + co * z];
  const v = [p(-w, 0, -d), p(w, 0, -d), p(w, 0, d), p(-w, 0, d), p(-w, h, -d), p(w, h, -d), p(w, h, d), p(-w, h, d)];
  const f = (...i: number[]) => i.map((k) => v[k]!);
  return convex([f(0, 1, 2, 3), f(4, 5, 6, 7), f(0, 1, 5, 4), f(1, 2, 6, 5), f(2, 3, 7, 6), f(3, 0, 4, 7)]);
}

function scene(dispersion: number): SceneCopyInput {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const quad = (q: V3[]) => convex([q]);
  const meshes: SceneCopyMesh[] = [
    quad([[-4, 0, 3], [4, 0, 3], [4, 0, -3], [-4, 0, -3]]), // floor
    quad([[-4, 0, -2.6], [4, 0, -2.6], [4, 3, -2.6], [-4, 3, -2.6]]), // dark back wall
    prism(-0.6, 0, 0.55, 1.3, 0.4),
    slab(0.9, 0.2, 0.45, 1.1, 0.12, -0.5),
  ];
  // Thin bright bars on the back wall, facing the camera (+z): the light the glass splits.
  for (let i = 0; i < 9; i++) {
    const x = -2.4 + i * 0.6;
    meshes.push({ positions: new Float32Array([x, 0.2, -2.55, x + 0.07, 0.2, -2.55, x + 0.07, 2.4, -2.55, x, 2.4, -2.55]), indices: [0, 1, 2, 0, 2, 3] });
  }
  const glass: RtMaterialInput = { baseWeight: 0, baseColor: [0, 0, 0], transmissionWeight: 1, transmissionColor: [1, 1, 1], specularWeight: 1, ior: 1.5, specularRoughness: 0, dispersion };
  const materials: RtMaterialInput[] = [
    { baseColor: [0.35, 0.35, 0.37], specularWeight: 0 },
    { baseColor: [0.03, 0.03, 0.03], specularWeight: 0 },
    glass,
    { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0, emissionColor: [12, 12, 12] },
  ];
  return {
    meshes,
    instances: meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: identity, material: i < 2 ? i : i < 4 ? 2 : 3 })),
    materials,
    environment: { radiance: [0.05, 0.05, 0.06] },
  };
}

defineExample(async (ctx) => {
  const { device, pane, canvas } = ctx;
  const search = new URLSearchParams(location.search);
  const params = { spectral: search.get('spectral') !== '0', dispersion: 1.5 };
  const orbit = new OrbitCamera(canvas, { position: [0.2, 1.5, 4.6], target: [0.1, 0.7, 0], up: [0, 1, 0], yFovDeg: 40 });
  let gpuScene: GpuScene = await buildGpuScene(device, scene(params.dispersion), { label: 'prism' });
  let renderer: ProgressiveRenderer | null = null;

  const rebuild = async () => {
    ctx.progress(params.spectral ? 'Compiling the spectral kernel (fitting the RGB-to-spectrum table)…' : 'Compiling…');
    try {
      // res 32: a coarser uplift table than the default 64, fitted in well under a second in the page.
      const next = await ProgressiveRenderer.create(ctx, gpuScene, orbit.camera, { integrator: { spectral: params.spectral ? { res: 32 } : false } });
      next.settings.maxDepth = 16;
      if (renderer) next.settings.samplesPerFrame = renderer.settings.samplesPerFrame;
      const old = renderer;
      renderer = next;
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('Could not build the kernel', error);
    }
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);

  const folder = pane.addFolder({ title: 'Spectral' });
  folder.addBinding(params, 'spectral', { label: 'spectral (vs RGB)' }).on('change', () => void rebuild());
  folder.addBinding(params, 'dispersion', { min: 0, max: 4, step: 0.05 }).on('change', async (e) => {
    if (!e.last || !renderer) return;
    // Dispersion is a material value: a new scene copy, the same kernel.
    const old = gpuScene;
    gpuScene = await buildGpuScene(device, scene(params.dispersion), { label: 'prism' });
    renderer.setScene(gpuScene);
    await device.queue.onSubmittedWorkDone();
    old.destroy();
  });

  await rebuild();
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
