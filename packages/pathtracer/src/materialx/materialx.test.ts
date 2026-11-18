import { describe, expect, it } from 'vitest';
import { addressIndex, bakeMtlxMaterial, constantContext, evalMtlxParams, imageSampler, loadMtlxMaterial, MtlxUnsupportedError, mtlxConstantParams, parseMaterialX, parseXml, sampleImage, type MtlxImage, type MtlxTsContext } from './index.ts';

const wrap = (body: string, version = '1.39'): string => `<?xml version="1.0"?><materialx version="${version}" colorspace="lin_rec709">${body}</materialx>`;
const OPEN_PBR = (inputs: string): string => `<surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="S"/></surfacematerial><open_pbr_surface name="S" type="surfaceshader">${inputs}</open_pbr_surface>`;
const ctxAt = (u: number, v: number, sample?: MtlxTsContext['sample']): MtlxTsContext => ({ ...constantContext(sample), uv: [u, v] });

describe('xml', () => {
  it('reads elements, attributes, comments and self-closing tags', () => {
    const r = parseXml('<?xml version="1.0"?><!-- c --><a x="1" y=\'two &amp; three\'><b/><c k="v"> text </c></a>');
    expect(r.name).toBe('a');
    expect(r.attributes).toEqual({ x: '1', y: 'two & three' });
    expect(r.children.map((c) => c.name)).toEqual(['b', 'c']);
    expect(r.children[1]!.text).toBe('text');
  });
  it('rejects mismatched tags with a line number', () => {
    expect(() => parseXml('<a>\n<b></a>')).toThrow(/line 2/);
    expect(() => parseXml('<a x=1/>')).toThrow(/quoted/);
  });
});

describe('document and surfaces', () => {
  it('reads an open_pbr_surface with constants only', () => {
    const m = loadMtlxMaterial(wrap(OPEN_PBR('<input name="base_color" type="color3" value="0.2, 0.4, 0.6"/><input name="specular_roughness" type="float" value="0.15"/><input name="transmission_depth" type="float" value="2"/><input name="geometry_thin_walled" type="boolean" value="true"/>')));
    const p = mtlxConstantParams(m);
    expect(p.baseColor).toEqual([0.2, 0.4, 0.6]);
    expect(p.specularRoughness).toBe(0.15);
    expect(p.transmissionDepth).toBeCloseTo(0.02, 12); // cm -> m (V262)
    expect(p.geometryThinWalled).toBe(true);
    expect(Object.keys(m.varying)).toEqual([]);
  });
  it('maps standard_surface to OpenPBR with its own defaults', () => {
    const m = loadMtlxMaterial(wrap('<standard_surface name="S" type="surfaceshader"><input name="base" type="float" value="0.5"/><input name="metalness" type="float" value="1"/><input name="opacity" type="color3" value="0.2, 0.4, 0.6"/><input name="thin_film_thickness" type="float" value="500"/><input name="emission" type="float" value="3"/></standard_surface>', '1.38'), { material: 'S' });
    const p = mtlxConstantParams(m);
    expect(p.baseWeight).toBe(0.5);
    expect(p.baseMetalness).toBe(1);
    expect(p.geometryOpacity).toBeCloseTo(0.4, 12);
    expect(p.thinFilmThickness).toBeCloseTo(0.5, 12); // nm -> um
    expect(p.thinFilmWeight).toBe(1);
    expect(p.emissionLuminance).toBe(3);
    expect(p.baseColor).toEqual([1, 1, 1]); // standard_surface default, not OpenPBR's 0.8
    expect(p.specularRoughness).toBe(0.2);
  });
  it('folds constant graphs and keeps geometry-dependent ones', () => {
    const m = loadMtlxMaterial(wrap(`<multiply name="k" type="color3"><input name="in1" type="color3" value="0.5, 0.25, 1"/><input name="in2" type="float" value="2"/></multiply>
      <texcoord name="uv" type="vector2"/>
      <ramplr name="ramp" type="float"><input name="valuel" type="float" value="0.1"/><input name="valuer" type="float" value="0.9"/><input name="texcoord" type="vector2" nodename="uv"/></ramplr>
      ${OPEN_PBR('<input name="base_color" type="color3" nodename="k"/><input name="specular_roughness" type="float" nodename="ramp"/>')}`));
    expect(Object.keys(m.constants)).toEqual(['baseColor']);
    expect(Object.keys(m.varying)).toEqual(['specularRoughness']);
    expect(mtlxConstantParams(m).baseColor).toEqual([1, 0.5, 2]);
    expect(m.categories).toEqual(['ramplr', 'texcoord']);
  });
  it('names the unsupported node', () => {
    expect(() => loadMtlxMaterial(wrap(`<fractal3d name="n" type="float"/>${OPEN_PBR('<input name="specular_roughness" type="float" nodename="n"/>')}`))).toThrow(MtlxUnsupportedError);
    expect(() => loadMtlxMaterial(wrap(`<fractal3d name="n" type="float"/>${OPEN_PBR('<input name="specular_roughness" type="float" nodename="n"/>')}`))).toThrow(/fractal3d/);
  });
  it('resolves nodegraph outputs, interface defaults and multi-output nodes', () => {
    const m = loadMtlxMaterial(wrap(`<nodegraph name="NG"><input name="k" type="float" value="3"/>
        <texcoord name="uv" type="vector2"/>
        <separate2 name="sp" type="multioutput"><input name="in" type="vector2" nodename="uv"/></separate2>
        <multiply name="m" type="float"><input name="in1" type="float" nodename="sp" output="outx"/><input name="in2" type="float" interfacename="k"/></multiply>
        <output name="out" type="float" nodename="m"/></nodegraph>
      ${OPEN_PBR('<input name="specular_roughness" type="float" nodegraph="NG" output="out"/>')}`));
    const p = evalMtlxParams(m, ctxAt(0.25, 0.5));
    expect(p.specularRoughness).toBeCloseTo(0.75, 12);
  });
});

describe('node semantics (closed forms)', () => {
  const evalNode = (nodeXml: string, u = 0.3, v = 0.6, sample?: MtlxTsContext['sample']) => {
    const m = loadMtlxMaterial(wrap(`<texcoord name="uv" type="vector2"/>${nodeXml}${OPEN_PBR('<input name="base_color" type="color3" nodename="N"/>')}`));
    return evalMtlxParams(m, ctxAt(u, v, sample)).baseColor as number[];
  };
  it('mix, multiply with a float operand, clamp, remap', () => {
    expect(evalNode('<mix name="N" type="color3"><input name="fg" type="color3" value="1, 0, 0"/><input name="bg" type="color3" value="0, 0, 1"/><input name="mix" type="float" value="0.25"/></mix>')).toEqual([0.25, 0, 0.75]);
    expect(evalNode('<multiply name="N" type="color3"><input name="in1" type="color3" value="0.5, 1, 2"/><input name="in2" type="float" value="0.5"/></multiply><texcoord name="x" type="vector2"/>')).toEqual([0.25, 0.5, 1]);
    expect(evalNode('<clamp name="N" type="color3"><input name="in" type="color3" value="-1, 0.5, 3"/></clamp>')).toEqual([0, 0.5, 1]);
    const r = evalNode('<remap name="N" type="color3"><input name="in" type="color3" value="0.5, 0.5, 0.5"/><input name="inlow" type="float" value="0"/><input name="inhigh" type="float" value="1"/><input name="outlow" type="float" value="2"/><input name="outhigh" type="float" value="4"/></remap>');
    expect(r[0]).toBeCloseTo(3, 12);
  });
  it('place2d: scale, rotate, offset, pivot (SRT)', () => {
    const m = loadMtlxMaterial(wrap(`<texcoord name="uv" type="vector2"/>
      <place2d name="p" type="vector2"><input name="texcoord" type="vector2" nodename="uv"/><input name="pivot" type="vector2" value="0.5, 0.5"/><input name="scale" type="vector2" value="2, 4"/><input name="rotate" type="float" value="90"/><input name="offset" type="vector2" value="0.1, 0.2"/></place2d>
      <separate2 name="s" type="multioutput"><input name="in" type="vector2" nodename="p"/></separate2>
      <combine3 name="N" type="color3"><input name="in1" type="float" nodename="s" output="outx"/><input name="in2" type="float" nodename="s" output="outy"/><input name="in3" type="float" value="0"/></combine3>
      ${OPEN_PBR('<input name="base_color" type="color3" nodename="N"/>')}`));
    const [u, v] = [0.7, 0.9];
    const c = evalMtlxParams(m, ctxAt(u, v)).baseColor as number[];
    // (uv - pivot) / scale, rotated by 90 degrees (x' = ca x + sa y, y' = -sa x + ca y), minus offset, plus pivot.
    const sx = (u - 0.5) / 2, sy = (v - 0.5) / 4;
    expect(c[0]).toBeCloseTo(sy - 0.1 + 0.5, 12);
    expect(c[1]).toBeCloseTo(-sx - 0.2 + 0.5, 12);
  });
  it('normalmap: tangent space decodes, scales and rotates into the frame', () => {
    const m = loadMtlxMaterial(wrap(`<normalmap name="NM" type="vector3"><input name="in" type="vector3" value="0.75, 0.5, 1"/><input name="scale" type="float" value="2"/></normalmap>
      ${OPEN_PBR('<input name="base_color" type="color3" nodename="NM"/>')}`));
    const c = evalMtlxParams(m, { ...ctxAt(0, 0), normal: [0, 0, 1], tangent: [1, 0, 0] }).baseColor as number[];
    // (0.75, 0.5, 1) decodes to (0.5, 0, 1); scale 2 makes xy (1, 0): normalised (1, 0, 1) / sqrt 2.
    expect(c[0]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(c[1]).toBeCloseTo(0, 12);
    expect(c[2]).toBeCloseTo(Math.SQRT1_2, 12);
  });
  it('tiledimage reads texel centres exactly and applies uvtiling', () => {
    const img: MtlxImage = { width: 2, height: 2, data: new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255]) };
    const sample = imageSampler((f) => (f === 'img.png' ? img : undefined));
    const node = '<tiledimage name="N" type="color3"><input name="file" type="filename" value="img.png"/><input name="uvtiling" type="vector2" value="2, 2"/></tiledimage>';
    // uv (0.125, 0.125) * 2 = (0.25, 0.25) -> texel (0, 0) centre: red.
    expect(evalNode(node, 0.125, 0.125, sample)).toEqual([1, 0, 0]);
    // uv (0.375, 0.125) * 2 = (0.75, 0.25) -> texel (1, 0): green.
    expect(evalNode(node, 0.375, 0.125, sample)).toEqual([0, 1, 0]);
    // v grows downwards: uv (0.125, 0.375) * 2 = (0.25, 0.75) -> texel (0, 1): blue.
    expect(evalNode(node, 0.125, 0.375, sample)).toEqual([0, 0, 1]);
    // A missing file reads as the node's default.
    expect(evalNode('<image name="N" type="color3"><input name="file" type="filename" value="nope.png"/><input name="default" type="color3" value="0.1, 0.2, 0.3"/></image>', 0.5, 0.5, sample)).toEqual([0.1, 0.2, 0.3]);
  });
  it('sRGB images are decoded before filtering, data images are not', () => {
    const img: MtlxImage = { width: 1, height: 1, data: new Uint8Array([188, 188, 188, 255]) };
    const sample = imageSampler(() => img);
    const srgb = evalNode('<image name="N" type="color3"><input name="file" type="filename" value="a.png" colorspace="srgb_texture"/></image>', 0.5, 0.5, sample);
    expect(srgb[0]).toBeCloseTo(0.5, 2);
    const linear = evalNode('<image name="N" type="color3"><input name="file" type="filename" value="a.png"/></image>', 0.5, 0.5, sample);
    expect(linear[0]).toBeCloseTo(188 / 255, 12);
  });
});

describe('image sampling', () => {
  it('address modes', () => {
    expect([-1, 0, 3, 4].map((i) => addressIndex(i, 4, 'periodic'))).toEqual([3, 0, 3, 0]);
    expect([-1, 0, 3, 4].map((i) => addressIndex(i, 4, 'clamp'))).toEqual([0, 0, 3, 3]);
    expect([-1, 0, 3, 4, 5].map((i) => addressIndex(i, 4, 'mirror'))).toEqual([0, 0, 3, 3, 2]);
  });
  it('bilinear halfway between two texel centres is their mean', () => {
    const img: MtlxImage = { width: 2, height: 1, data: new Uint8Array([0, 0, 0, 255, 255, 255, 255, 255]) };
    expect(sampleImage(img, false, 'clamp', 'clamp', 0.5, 0.5)[0]).toBeCloseTo(0.5, 12);
  });
});

describe('bake (approach a)', () => {
  const rampMaterial = loadMtlxMaterial(wrap(`<texcoord name="uv" type="vector2"/>
    <ramplr name="ramp" type="color3"><input name="valuel" type="color3" value="0, 0, 0"/><input name="valuer" type="color3" value="1, 0.5, 0.25"/><input name="texcoord" type="vector2" nodename="uv"/></ramplr>
    <ramptb name="rough" type="float"><input name="valuet" type="float" value="0.2"/><input name="valueb" type="float" value="0.8"/><input name="texcoord" type="vector2" nodename="uv"/></ramptb>
    <ramplr name="metal" type="float"><input name="valuel" type="float" value="0"/><input name="valuer" type="float" value="1"/><input name="texcoord" type="vector2" nodename="uv"/></ramplr>
    <ramplr name="coat" type="float"><input name="valuel" type="float" value="0"/><input name="valuer" type="float" value="1"/><input name="texcoord" type="vector2" nodename="uv"/></ramplr>
    ${OPEN_PBR('<input name="base_color" type="color3" nodename="ramp"/><input name="specular_roughness" type="float" nodename="rough"/><input name="base_metalness" type="float" nodename="metal"/><input name="coat_weight" type="float" nodename="coat"/><input name="base_weight" type="float" value="0.5"/>')}`));
  it('writes colour, roughness and metalness textures that match the graph at texel centres', () => {
    const n = 8;
    const baked = bakeMtlxMaterial(rampMaterial, { resolution: n });
    const bc = baked.baseColorTexture!;
    const mr = baked.metallicRoughnessTexture!;
    expect(bc.colorSpace).toBe('srgb');
    expect(mr.colorSpace).toBe('linear');
    for (const [x, y] of [[0, 0], [3, 5], [7, 7]] as const) {
      const u = (x + 0.5) / n, v = (y + 0.5) / n;
      const o = (y * n + x) * 4;
      // base weight 0.5 is a constant here, so it stays a factor; the texture holds the colour.
      const lin = (code: number) => (code / 255 <= 0.04045 ? code / 255 / 12.92 : Math.pow((code / 255 + 0.055) / 1.055, 2.4));
      expect(lin(bc.data[o]!)).toBeCloseTo(u * 1, 1);
      expect(lin(bc.data[o + 1]!)).toBeCloseTo(u * 0.5, 1);
      expect(mr.data[o + 1]! / 255).toBeCloseTo(0.2 + 0.6 * v, 2);
      expect(mr.data[o + 2]! / 255).toBeCloseTo(u, 2);
    }
    expect(baked.params.baseColor).toEqual([1, 1, 1]);
    expect(baked.params.specularRoughness).toBe(1);
    expect(baked.params.baseWeight).toBe(0.5);
  });
  it('averages a varying input that has no texture slot and says so', () => {
    const baked = bakeMtlxMaterial(rampMaterial, { resolution: 4 });
    expect(baked.params.coatWeight).toBeCloseTo(0.5, 1);
    expect(baked.approximations.some((a) => a.startsWith('coatWeight'))).toBe(true);
  });
  it('folds a varying base weight into the colour texture and puts opacity in alpha', () => {
    const m = loadMtlxMaterial(wrap(`<texcoord name="uv" type="vector2"/>
      <ramplr name="w" type="float"><input name="valuel" type="float" value="0"/><input name="valuer" type="float" value="1"/><input name="texcoord" type="vector2" nodename="uv"/></ramplr>
      ${OPEN_PBR('<input name="base_weight" type="float" nodename="w"/><input name="base_color" type="color3" value="1, 1, 1"/><input name="geometry_opacity" type="float" nodename="w"/>')}`));
    const baked = bakeMtlxMaterial(m, { resolution: 4 });
    expect(baked.params.baseWeight).toBe(1);
    expect(baked.alphaMode).toBe('blend');
    const t = baked.baseColorTexture!;
    expect(t.data[(0 * 4 + 3) * 4 + 3]).toBe(Math.round(0.875 * 255));
    expect(t.data[3]).toBe(Math.round(0.125 * 255));
  });
  it('bakes a normalmap node to a tangent-space texture', () => {
    const m = loadMtlxMaterial(wrap(`<normalmap name="NM" type="vector3"><input name="in" type="vector3" value="0.75, 0.5, 1"/><input name="scale" type="float" value="1"/></normalmap>
      <standard_surface name="S" type="surfaceshader"><input name="normal" type="vector3" nodename="NM"/></standard_surface>`, '1.38'), { material: 'S' });
    const baked = bakeMtlxMaterial(m, { resolution: 2 });
    const l = Math.hypot(0.5, 0, 1);
    expect(baked.normalTexture!.data[0]).toBe(Math.round((0.5 / l) * 0.5 * 255 + 127.5));
    expect(baked.normalTexture!.data[2]).toBe(Math.round((1 / l) * 0.5 * 255 + 127.5));
  });
  it('parses documents with a fileprefix and colorspace on the file input', () => {
    const doc = parseMaterialX(wrap('<image name="i" type="color3"><input name="file" type="filename" value="a.png" colorspace="srgb_texture" fileprefix="img/"/></image>'));
    const input = doc.nodes.get('i')!.inputs.file!;
    expect(input.colorspace).toBe('srgb_texture');
    expect(input.fileprefix).toBe('img/');
  });
});
