/**
 * MaterialX (experimental): a `.mtlx` document as a path tracer material, two ways.
 *
 *   codegen  the graph becomes WGSL that runs at every hit (`integrator.materialx`): exact for anything the graph reads, uv, position or images
 *   bake     inputs that vary over uv are baked into textures over the uv square: no kernel change, but position-driven graphs only average
 * `buildMaterialX(text, options)` returns a scene-copy material and its textures; `mtlxAddToScene` appends them to a scene input, and
 * `mtlxKernelOptions` gives the kernel options (spread into `ptSceneOptions`). The three documents are the bench's MaterialX test scenes.
 */
import { buildMaterialX, mtlxAddToScene, mtlxKernelOptions, mtlxSyntheticImage, type MtlxBuildMode } from '@pmndrs/pathtracer';
import { buildGpuScene, type GpuScene, type RtMaterialInput, type SceneCopyInput, type SceneCopyMesh } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { sunFrom } from '../../shared/models.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

const head = (v = '1.39') => `<?xml version="1.0"?><materialx version="${v}" colorspace="lin_rec709">`;
const surfaceMaterial = '<surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S"/></surfacematerial>';

const DOCUMENTS: Record<string, string> = {
  'Constant coat (open_pbr_surface)': `${head()}
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" value="0.7, 0.35, 0.15"/><input name="specular_roughness" type="float" value="0.35"/>
    <input name="coat_weight" type="float" value="0.6"/><input name="coat_roughness" type="float" value="0.1"/>
  </open_pbr_surface>${surfaceMaterial}</materialx>`,
  'Texture graph (standard_surface)': `${head('1.38')}
  <nodegraph name="NG">
    <texcoord name="uv" type="vector2"/>
    <place2d name="pl" type="vector2"><input name="texcoord" type="vector2" nodename="uv"/><input name="rotate" type="float" value="15"/><input name="offset" type="vector2" value="0.1, 0"/></place2d>
    <tiledimage name="col" type="color3"><input name="file" type="filename" value="gen:checker8" colorspace="srgb_texture"/><input name="texcoord" type="vector2" nodename="pl"/><input name="uvtiling" type="vector2" value="1, 1"/></tiledimage>
    <multiply name="tint" type="color3"><input name="in1" type="color3" nodename="col"/><input name="in2" type="color3" value="1, 0.9, 0.8"/></multiply>
    <image name="rings" type="float"><input name="file" type="filename" value="gen:rings"/><input name="texcoord" type="vector2" nodename="pl"/></image>
    <multiply name="r1" type="float"><input name="in1" type="float" nodename="rings"/><input name="in2" type="float" value="0.7"/></multiply>
    <add name="rough" type="float"><input name="in1" type="float" nodename="r1"/><input name="in2" type="float" value="0.12"/></add>
    <ramplr name="metal" type="float"><input name="valuel" type="float" value="0"/><input name="valuer" type="float" value="1"/><input name="texcoord" type="vector2" nodename="uv"/></ramplr>
    <smoothstep name="metal2" type="float"><input name="in" type="float" nodename="metal"/><input name="low" type="float" value="0.55"/><input name="high" type="float" value="0.75"/></smoothstep>
    <image name="bump" type="vector3"><input name="file" type="filename" value="gen:bumps4"/><input name="texcoord" type="vector2" nodename="uv"/></image>
    <normalmap name="nm" type="vector3"><input name="in" type="vector3" nodename="bump"/><input name="scale" type="float" value="1"/></normalmap>
    <output name="out_color" type="color3" nodename="tint"/><output name="out_rough" type="float" nodename="rough"/>
    <output name="out_metal" type="float" nodename="metal2"/><output name="out_normal" type="vector3" nodename="nm"/>
  </nodegraph>
  <standard_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" nodegraph="NG" output="out_color"/><input name="metalness" type="float" nodegraph="NG" output="out_metal"/>
    <input name="specular_roughness" type="float" nodegraph="NG" output="out_rough"/><input name="normal" type="vector3" nodegraph="NG" output="out_normal"/>
  </standard_surface>${surfaceMaterial}</materialx>`,
  'Procedural stripes (position-driven)': `${head()}
  <nodegraph name="NG">
    <position name="p" type="vector3"/>
    <separate3 name="ps" type="multioutput"><input name="in" type="vector3" nodename="p"/></separate3>
    <multiply name="py" type="float"><input name="in1" type="float" nodename="ps" output="outy"/><input name="in2" type="float" value="22"/></multiply>
    <multiply name="px" type="float"><input name="in1" type="float" nodename="ps" output="outx"/><input name="in2" type="float" value="7"/></multiply>
    <add name="arg" type="float"><input name="in1" type="float" nodename="py"/><input name="in2" type="float" nodename="px"/></add>
    <sin name="s" type="float"><input name="in" type="float" nodename="arg"/></sin>
    <multiply name="s2" type="float"><input name="in1" type="float" nodename="s"/><input name="in2" type="float" value="0.5"/></multiply>
    <add name="s3" type="float"><input name="in1" type="float" nodename="s2"/><input name="in2" type="float" value="0.5"/></add>
    <smoothstep name="stripe" type="float"><input name="in" type="float" nodename="s3"/><input name="low" type="float" value="0.45"/><input name="high" type="float" value="0.55"/></smoothstep>
    <mix name="color" type="color3"><input name="fg" type="color3" value="0.85, 0.75, 0.2"/><input name="bg" type="color3" value="0.1, 0.25, 0.6"/><input name="mix" type="float" nodename="stripe"/></mix>
    <remap name="rough" type="float"><input name="in" type="float" nodename="ps" output="outx"/><input name="inlow" type="float" value="-0.7"/><input name="inhigh" type="float" value="0.7"/><input name="outlow" type="float" value="0.1"/><input name="outhigh" type="float" value="0.7"/></remap>
    <output name="out_color" type="color3" nodename="color"/><output name="out_rough" type="float" nodename="rough"/><output name="out_metal" type="float" nodename="stripe"/>
  </nodegraph>
  <open_pbr_surface name="S" type="surfaceshader">
    <input name="base_color" type="color3" nodegraph="NG" output="out_color"/><input name="base_metalness" type="float" nodegraph="NG" output="out_metal"/>
    <input name="specular_roughness" type="float" nodegraph="NG" output="out_rough"/>
  </open_pbr_surface>${surfaceMaterial}</materialx>`,
};

/** A uv sphere (u around, v from the top), smooth normals. */
function sphere(cx: number, cy: number, cz: number, r: number, seg = 64, rings = 32): SceneCopyMesh {
  const positions: number[] = [], normals: number[] = [], uvs: number[] = [], indices: number[] = [];
  for (let j = 0; j <= rings; j++) {
    const v = j / rings, th = v * Math.PI;
    for (let i = 0; i <= seg; i++) {
      const u = i / seg, ph = u * 2 * Math.PI;
      const n = [Math.sin(th) * Math.sin(ph), Math.cos(th), Math.sin(th) * Math.cos(ph)];
      positions.push(cx + r * n[0]!, cy + r * n[1]!, cz + r * n[2]!);
      normals.push(...n);
      uvs.push(u, v);
    }
  }
  for (let j = 0; j < rings; j++) {
    for (let i = 0; i < seg; i++) {
      const a = j * (seg + 1) + i, b = a + seg + 1;
      if (j > 0) indices.push(a, b, a + 1);
      if (j < rings - 1) indices.push(a + 1, b, b + 1);
    }
  }
  return { positions: new Float32Array(positions), normals: new Float32Array(normals), uvs: new Float32Array(uvs), indices: new Uint32Array(indices) };
}

function scene(document: string, mode: MtlxBuildMode) {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const meshes: SceneCopyMesh[] = [
    { positions: new Float32Array([-4, 0, 4, 4, 0, 4, 4, 0, -4, -4, 0, -4]), indices: [0, 1, 2, 0, 2, 3] },
    sphere(-0.75, 0.6, 0, 0.6),
    // An upright panel facing the camera, uv over [0, 1]^2 (the square a bake covers).
    { positions: new Float32Array([0.2, 0.1, 0, 1.4, 0.1, 0, 1.4, 1.3, 0, 0.2, 1.3, 0]), indices: [0, 1, 2, 0, 2, 3], uvs: [0, 1, 1, 1, 1, 0, 0, 0], normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]) },
  ];
  const input: SceneCopyInput & { materials: RtMaterialInput[] } = {
    meshes,
    instances: [],
    materials: [{ baseColor: [0.55, 0.55, 0.55], specularWeight: 0 }],
    sun: sunFrom(48, 35, 3.2),
    environment: { radiance: [0.45, 0.5, 0.6] },
  };
  const built = buildMaterialX(document, { mode, resolution: 512, images: mtlxSyntheticImage });
  const use = mtlxAddToScene(input, built);
  input.instances = meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: identity, material: i === 0 ? 0 : use.material }));
  return { input, use };
}

defineExample(async (ctx) => {
  const { device, pane, canvas, stats } = ctx;
  const search = new URLSearchParams(location.search);
  const names = Object.keys(DOCUMENTS);
  const params = { document: names[Number(search.get('doc') ?? 2)] ?? names[2]!, mode: (search.get('mode') ?? 'codegen') as 'codegen' | 'bake' };
  const orbit = new OrbitCamera(canvas, { position: [0, 1.0, 3.8], target: [0, 0.65, 0], up: [0, 1, 0], yFovDeg: 38 });
  let renderer: ProgressiveRenderer | null = null;
  let gpuScene: GpuScene | null = null;

  const rebuild = async () => {
    ctx.progress('Building the material…');
    try {
      const doc = DOCUMENTS[params.document]!;
      // A constant document has nothing to generate or bake: build it as constants whatever the mode says.
      const constant = params.document.startsWith('Constant');
      const { input, use } = scene(doc, constant ? 'constants' : params.mode);
      const next = await buildGpuScene(device, input, { label: 'materialx' });
      ctx.progress('Compiling…');
      const kernel = mtlxKernelOptions([use]);
      const r = await ProgressiveRenderer.create(ctx, next, orbit.camera, kernel);
      r.settings.maxDepth = 6;
      stats.set('mode', use.built.mode);
      if (kernel.notes.length) console.info(`MaterialX notes:\n${kernel.notes.join('\n')}`);
      const old = renderer, oldScene = gpuScene;
      renderer = r;
      gpuScene = next;
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
      oldScene?.destroy();
    } catch (error) {
      ctx.error('Could not build the material', error);
    }
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);

  const folder = pane.addFolder({ title: 'MaterialX' });
  folder.addBinding(params, 'document', { options: Object.fromEntries(names.map((n) => [n, n])) }).on('change', () => void rebuild());
  folder.addBinding(params, 'mode', { label: 'approach', options: { 'Codegen (WGSL per hit)': 'codegen', 'Bake (textures over uv)': 'bake' } }).on('change', () => void rebuild());

  await rebuild();
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
