/** Provider trace, one mirror BSDF, then a Lambert receiver with full direct and a strict indirect cache. */
export function reflectionConsumerWgsl(environment: boolean): string {
  return /* wgsl */ `
struct RcFrame { origin: vec4f, right: vec4f, up: vec4f, forward: vec4f, dims: vec4u, options: vec4u }
@group(0) @binding(0) var<uniform> frame: RcFrame;
@group(0) @binding(1) var<storage, read_write> result: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> records: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> count: array<atomic<u32>>;

var<private> hsoPixel: vec2u;
fn traceLightVisible(P: vec3f, N: vec3f, L: vec3f, tMax: f32) -> bool {
  return !traceAny(rtOffsetOrigin(P, rtSideNormal(N, L)), L, tMax);
}
fn rtHitSpecularOn() -> bool { return true; }

fn rcAngles(dir: vec3f) -> vec2f {
  let azimuth = select(atan2(dir.z, dir.x), 0.0, all(dir.xz == vec2f(0.0)));
  return vec2f(acos(clamp(dir.y, -1.0, 1.0)) / 3.14159265359, azimuth / 6.28318530718 + 0.5);
}
fn rcStoreRecord(p: RtShadePoint, radiance: vec3f, pixel: u32) {
  let i = atomicAdd(&count[1], 1u) * 4u;
  records[i] = vec4f(p.position, f32(pixel));
  records[i + 1u] = vec4f(rcAngles(p.wo), rcAngles(p.geometricNormal));
  records[i + 2u] = vec4f(p.albedo, 1.0);
  records[i + 3u] = vec4f(radiance, 0.0);
}

@compute @workgroup_size(8, 8)
fn renderConsumer(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= frame.dims.xy)) { return; }
  let pixel = id.y * frame.dims.x + id.x;
  let i = pixel * 5u;
  let seed = rcHash(pixel * 2654435761u + frame.dims.z * 0x68bc21ebu);
  let train = frame.options.x == 0u;
  let useCache = frame.options.x == 1u;
  let jitter = select(vec2f(rcRandom(seed, 0u), rcRandom(seed, 1u)), vec2f(0.5), frame.options.x == 3u);
  let ndc = 2.0 * (vec2f(id.xy) + jitter) / vec2f(frame.dims.xy) - 1.0;
  let dir = normalize(frame.forward.xyz + frame.right.xyz * ndc.x - frame.up.xyz * ndc.y);
  let ray = RtRay(frame.origin.xyz, 0.0, dir, 1e30);
  let primary = traceFirstHit(ray.origin, ray.direction, ray.tMax);
  if (!train) { result[i + 2u].x += 1.0; }
  if (primary.t < 0.0) { return; }
  var p = rtShadePoint(rtpRtHit(primary), ray);
  var weight = vec3f(1.0);
  var maxBounces = frame.dims.w;
  var receiver = primary;
  var incoming = dir;
  if (rcMirror(p)) {
    incoming = reflect(dir, p.normal);
    let origin = rtOffsetOrigin(p.position, rtSideNormal(p.geometricNormal, incoming));
    weight = rcMirrorWeight(p);
    receiver = traceFirstHit(origin, incoming, 100.0);
    if (!train) {
      result[i].w += 1.0;
      result[i + 2u].y += 1.0;
    }
    maxBounces -= 1u;
    if (receiver.t < 0.0) {
      if (!train) {
        result[i] += vec4f(weight * rtShadeMiss(incoming), 0.0);
        result[i + 4u].x += 1.0;
      }
      return;
    }
    p = rtShadePoint(rtpRtHit(receiver), RtRay(origin, 0.0, incoming, receiver.t));
  } else if (!train) {
    return; // the benchmark reports the reflection plane only
  }

  if (train) {
    if (!rcMirror(p) && ptMax3(p.albedo) > 0.0) {
      rcStoreRecord(p, rcSuffix(p, seed, maxBounces, false).indirect, pixel);
    }
    return;
  }

  // Match Reflections' geometric confidence: a traced hit is 1, a traced miss is 0.
  // Cache confidence is separate, including whether a lookup needed a coarser level.
  result[i + 1u] += vec4f(receiver.t, 1.0, f32(receiver.instance == frame.options.y), 0.0);
  result[i + 4u].y += f32(ptMax3(rtShadeEmission(p)) > 0.0);
  ${environment ? /* wgsl */ `
  hsoPixel = id.xy;
  let light = shadeTracedHit(receiver, p.position, p.normal, incoming);
  result[i] += vec4f(weight * light, 0.0);
  ` : /* wgsl */ `
  var light = rtShadeEmission(p);
  if (!rcMirror(p)) {
    let ld = rcDirect(p, seed, 2u);
    light += ld.direct;
    result[i + 2u].w += f32(ld.light != RT_NONE);
  }
  if (frame.options.x != 4u && (ptMax3(p.albedo) > 0.0 || rcMirror(p))) {
    var prediction = vec4f(0.0, 0.0, 0.0, -1.0);
    if (useCache && !rcMirror(p)) {
      prediction = hgSample(p.position, vec4f(rcAngles(p.wo), rcAngles(p.geometricNormal)), 1.0, pixel, frame.dims.z);
      result[i + 3u] += vec4f(1.0, f32(prediction.w >= 0.0), f32(prediction.w > 0.0), 0.0);
    }
    if (prediction.w >= 0.0) {
      light += prediction.xyz;
    } else {
      let suffix = rcSuffix(p, seed, maxBounces, false);
      light += suffix.indirect;
      result[i + 2u] += vec4f(0.0, 0.0, f32(suffix.continuationRays), f32(suffix.shadowRays));
    }
  }
  result[i] += vec4f(weight * light, 0.0);
  `}
}
`;
}
