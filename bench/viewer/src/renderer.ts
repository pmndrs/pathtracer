/**
 * WebGL2 renderer for the A/B viewer. Images live on the GPU as RGB32F textures (row 0 = top, exactly as in the
 * Image type); one full-screen triangle does tonemap / diff / false colour / swipe / side-by-side per pixel, so
 * panning, zooming and the EV slider cost one draw regardless of image size.
 */
import { colormapLUT } from '../../metrics/colormap.ts';
import { AGX_INSET, AGX_MAX_EV, AGX_MIN_EV, AGX_OUTSET, REC2020_TO_SRGB_MATRIX, SRGB_TO_REC2020_MATRIX } from '../../metrics/tonemap.ts';
import type { Image } from '../../src/adapter.ts';

export const LAYOUT = { swipe: 0, flip: 1, side: 2 } as const;
export const CONTENT = { image: 0, absdiff: 1, relerr: 2, flip: 3, bias: 4 } as const;
export const TONEMAP_ID = { clamp: 0, reinhard: 1, aces: 2, agx: 3 } as const;

export interface ViewState {
  layout: number; content: number; toggle: number; split: number; ev: number; tonemap: number;
  /** Image pixel at the centre of the view, and device pixels per image pixel. */
  cx: number; cy: number; zoom: number;
  /** Texture index (0 A, 1 B, 2 R) shown in slot 0 / 1 for content=image; -1 = no slot. */
  img: [number, number];
  /** Pair index (0 = A vs B, 1 = A vs R, 2 = B vs R) for error contents; -1 = no slot. */
  pair: [number, number];
}

// GLSL mat3 is column-major; the TS arrays are row-major.
const glslMat = (r: number[]) => `mat3(${[0, 1, 2].map((j) => `vec3(${[0, 1, 2].map((i) => r[i * 3 + j]!.toPrecision(9)).join(',')})`).join(',')})`;

const VERT = `#version 300 es
void main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uT0, uT1, uT2, uF0, uF1, uF2, uLutMagma, uLutTurbo;
uniform vec2 uRes, uSize, uCenter;
uniform float uZoom, uSplit, uGain;
uniform int uLayout, uContent, uToggle, uTonemap;
uniform ivec2 uImg, uPair;
out vec4 outColor;

const mat3 SRGB_TO_2020 = ${glslMat(SRGB_TO_REC2020_MATRIX)};
const mat3 REC2020_TO_SRGB = ${glslMat(REC2020_TO_SRGB_MATRIX)};
const mat3 AGX_IN = ${glslMat(AGX_INSET)};
const mat3 AGX_OUT = ${glslMat(AGX_OUTSET)};

float agxContrast(float x){
  float x2 = x*x, x4 = x2*x2;
  return 15.5*x4*x2 - 40.14*x4*x + 31.96*x4 - 6.868*x2*x + 0.4298*x2 + 0.1191*x - 0.00232;
}
vec3 tonemap(vec3 c){
  if (uTonemap == 0) return clamp(c, 0.0, 1.0);
  if (uTonemap == 1) return c / (1.0 + c);
  if (uTonemap == 2) return clamp((c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14), 0.0, 1.0);
  vec3 v = AGX_IN * (SRGB_TO_2020 * c);
  v = (log2(max(v, vec3(1e-10))) - ${AGX_MIN_EV.toPrecision(9)}) / ${(AGX_MAX_EV - AGX_MIN_EV).toPrecision(9)};
  v = clamp(v, 0.0, 1.0);
  v = vec3(agxContrast(v.x), agxContrast(v.y), agxContrast(v.z));
  v = pow(max(AGX_OUT * v, vec3(0.0)), vec3(2.2));
  return clamp(REC2020_TO_SRGB * v, 0.0, 1.0);
}
vec3 oetf(vec3 c){ c = clamp(c, 0.0, 1.0); return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4)) - 0.055, step(vec3(0.0031308), c)); }

vec3 img(int i, vec2 uv){ if (i == 0) return texture(uT0, uv).rgb; if (i == 1) return texture(uT1, uv).rgb; return texture(uT2, uv).rgb; }
float flipv(int i, vec2 uv){ if (i == 0) return texture(uF0, uv).r; if (i == 1) return texture(uF1, uv).r; return texture(uF2, uv).r; }
bool bad(vec3 c){ return any(isnan(c)) || any(isinf(c)); }
vec3 lut(sampler2D s, float t){ return texture(s, vec2(clamp(t, 0.0, 1.0) * (255.0/256.0) + 0.5/256.0, 0.5)).rgb; }

vec3 shade(int slot, vec2 uv){
  if (uContent == 0) {
    int i = slot == 0 ? uImg.x : uImg.y;
    vec3 c = img(i, uv);
    if (bad(c)) return vec3(1.0, 0.0, 1.0);
    return oetf(tonemap(c * uGain));
  }
  int p = slot == 0 ? uPair.x : uPair.y;
  int ti = p == 2 ? 1 : 0;
  int ri = p == 0 ? 1 : 2;
  if (uContent == 3) return lut(uLutMagma, flipv(p, uv));
  vec3 t = img(ti, uv), r = img(ri, uv);
  if (bad(t) || bad(r)) return vec3(1.0, 0.0, 1.0);
  vec3 d = t - r;
  if (uContent == 1) return oetf(tonemap(abs(d) * uGain));
  if (uContent == 2) {
    vec3 q = d*d / (r*r + 0.01);
    float e = sqrt((q.x + q.y + q.z) / 3.0);
    if (e <= 0.0) return vec3(0.0);
    return lut(uLutTurbo, (log(e)/2.302585093 + 3.0) / 3.0);
  }
  float b = clamp(((d.x + d.y + d.z) / 3.0) * uGain, -1.0, 1.0);
  return oetf(b > 0.0 ? mix(vec3(1.0), vec3(1.0, 0.1, 0.05), b) : mix(vec3(1.0), vec3(0.05, 0.25, 1.0), -b));
}

void main(){
  vec2 fp = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  bool two = (uContent == 0 ? uImg.y : uPair.y) >= 0;
  int slot = 0;
  vec2 origin = vec2(0.0), panel = uRes;
  if (two) {
    if (uLayout == 2) {
      float h = floor(uRes.x * 0.5);
      if (fp.x >= h) { slot = 1; origin = vec2(h, 0.0); }
      panel = vec2(slot == 0 ? h : uRes.x - h, uRes.y);
    } else if (uLayout == 1) slot = uToggle;
    else slot = fp.x < uSplit * uRes.x ? 0 : 1;
  }
  vec2 ip = uCenter + (fp - origin - panel * 0.5) / uZoom;
  vec2 uv = ip / uSize;
  vec3 col;
  if (any(lessThan(ip, vec2(0.0))) || any(greaterThanEqual(ip, uSize))) {
    float ck = mod(floor(fp.x / 12.0) + floor(fp.y / 12.0), 2.0);
    col = vec3(0.075 + 0.02 * ck);
  } else col = shade(slot, uv);
  if (two && uLayout == 0 && abs(fp.x - uSplit * uRes.x) < 1.0) col = vec3(1.0);
  if (two && uLayout == 2 && abs(fp.x - floor(uRes.x * 0.5)) < 1.0) col = vec3(0.35);
  outColor = vec4(col, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`shader: ${gl.getShaderInfoLog(s)}`);
  return s;
}

export class GLView {
  readonly gl: WebGL2RenderingContext;
  readonly linearFloat: boolean;
  private prog: WebGLProgram;
  private imgTex: WebGLTexture[] = [];
  private flipTex: WebGLTexture[] = [];
  private lutTex: WebGLTexture[] = [];
  private u: Record<string, WebGLUniformLocation | null> = {};
  private linear = false;
  private dirty = true;
  size: [number, number] = [1, 1];
  maxTextureSize: number;

  constructor(readonly canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available');
    this.gl = gl;
    this.linearFloat = !!gl.getExtension('OES_texture_float_linear');
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`link: ${gl.getProgramInfoLog(prog)}`);
    this.prog = prog;
    for (const n of ['uT0', 'uT1', 'uT2', 'uF0', 'uF1', 'uF2', 'uLutMagma', 'uLutTurbo', 'uRes', 'uSize', 'uCenter', 'uZoom', 'uSplit', 'uGain', 'uLayout', 'uContent', 'uToggle', 'uTonemap', 'uImg', 'uPair']) {
      this.u[n] = gl.getUniformLocation(prog, n);
    }
    for (let i = 0; i < 3; i++) {
      this.imgTex.push(this.makeFloatTex(1, 1, null, false));
      this.flipTex.push(this.makeFloatTex(1, 1, null, true));
    }
    for (const name of ['magma', 'turbo'] as const) {
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB8, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, colormapLUT(name));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.lutTex.push(t);
    }
  }

  private makeFloatTex(w: number, h: number, data: Float32Array | null, single: boolean, existing?: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const t = existing ?? gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, single ? gl.R32F : gl.RGB32F, w, h, 0, single ? gl.RED : gl.RGB, gl.FLOAT, data ?? new Float32Array(w * h * (single ? 1 : 3)));
    for (const [p, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]] as const) {
      gl.texParameteri(gl.TEXTURE_2D, p, v);
    }
    return t;
  }

  /** slot 0 = A, 1 = B, 2 = R. RGB data (channels 3) or RGBA (repacked). */
  setImage(slot: number, img: Image | null): void {
    if (!img) { this.makeFloatTex(1, 1, null, false, this.imgTex[slot]); this.dirty = true; return; }
    if (img.width > this.maxTextureSize || img.height > this.maxTextureSize) throw new Error(`image ${img.width}x${img.height} exceeds MAX_TEXTURE_SIZE ${this.maxTextureSize}`);
    let data = img.data;
    if (img.channels === 4) {
      data = new Float32Array(img.width * img.height * 3);
      for (let i = 0; i < img.width * img.height; i++) { data[3 * i] = img.data[4 * i]!; data[3 * i + 1] = img.data[4 * i + 1]!; data[3 * i + 2] = img.data[4 * i + 2]!; }
    }
    this.makeFloatTex(img.width, img.height, data, false, this.imgTex[slot]);
    this.dirty = true; // re-apply the filter state on the next draw
  }

  /** pair 0 = A vs B, 1 = A vs R, 2 = B vs R. */
  setFlipMap(pair: number, map: Float32Array | null, w = 1, h = 1): void {
    this.makeFloatTex(w, h, map, true, this.flipTex[pair]);
    this.dirty = true;
  }

  private applyFilter(linear: boolean) {
    if (!this.dirty && linear === this.linear) return;
    this.dirty = false;
    this.linear = linear;
    const gl = this.gl;
    const f = linear && this.linearFloat ? gl.LINEAR : gl.NEAREST;
    for (const t of [...this.imgTex, ...this.flipTex]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    }
  }

  draw(s: ViewState): void {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.prog);
    this.applyFilter(s.zoom < 1);
    const units = [...this.imgTex, ...this.flipTex, ...this.lutTex];
    const names = ['uT0', 'uT1', 'uT2', 'uF0', 'uF1', 'uF2', 'uLutMagma', 'uLutTurbo'];
    units.forEach((t, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.uniform1i(this.u[names[i]!]!, i);
    });
    gl.uniform2f(this.u.uRes!, this.canvas.width, this.canvas.height);
    gl.uniform2f(this.u.uSize!, this.size[0], this.size[1]);
    gl.uniform2f(this.u.uCenter!, s.cx, s.cy);
    gl.uniform1f(this.u.uZoom!, s.zoom);
    gl.uniform1f(this.u.uSplit!, s.split);
    gl.uniform1f(this.u.uGain!, Math.pow(2, s.ev));
    gl.uniform1i(this.u.uLayout!, s.layout);
    gl.uniform1i(this.u.uContent!, s.content);
    gl.uniform1i(this.u.uToggle!, s.toggle);
    gl.uniform1i(this.u.uTonemap!, s.tonemap);
    gl.uniform2i(this.u.uImg!, s.img[0], s.img[1]);
    gl.uniform2i(this.u.uPair!, s.pair[0], s.pair[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
