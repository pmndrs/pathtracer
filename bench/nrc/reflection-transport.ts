import { PT_MIS_WGSL, ptBsdfWgsl } from '@pmndrs/pathtracer';

/** Lambert strict indirect, with a separately measured BSDF-sampled MIS direct tail. */
export const REFLECTION_TRANSPORT_WGSL = /* wgsl */ `
${PT_MIS_WGSL}
${ptBsdfWgsl()}
fn rcHash(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rcRandom(seed: u32, dimension: u32) -> f32 {
  return f32(rcHash(seed ^ rcHash(dimension + 0x9e3779b9u)) >> 8u) * 5.9604644775390625e-8;
}

fn rcDirect(p: RtShadePoint, seed: u32, dimension: u32) -> RtShaded {
  return rtShadeDirect(p, RT_SHADE_LAMBERT,
    vec3f(rcRandom(seed, dimension), rcRandom(seed, dimension + 1u), rcRandom(seed, dimension + 2u)), RT_MASK_ALL);
}

fn rcMirror(p: RtShadePoint) -> bool {
  return p.material.conductorEtaK && p.material.specularRoughness == 0.0;
}

fn rcMirrorWeight(p: RtShadePoint) -> vec3f {
  return ptFrComplex(dot(p.wo, p.normal), p.material.conductorEta, p.material.conductorK);
}

struct RcSuffix {
  indirect: vec3f,
  directComplement: vec3f,
  continuationRays: u32,
  shadowRays: u32,
}

fn rcSuffix(receiver: RtShadePoint, seed: u32, maxBounces: u32, measureMisComplement: bool) -> RcSuffix {
  var p = receiver;
  var beta = vec3f(1.0);
  var result: RcSuffix;
  for (var i = 1u; i <= maxBounces; i++) {
    let dim = 7u + 5u * i;
    let disk = rtShadeConcentricDisk(vec2f(rcRandom(seed, dim), rcRandom(seed, dim + 1u)));
    let cosine = sqrt(max(0.0, 1.0 - dot(disk, disk)));
    let delta = rcMirror(p);
    var wi: vec3f;
    if (delta) {
      beta *= rcMirrorWeight(p);
      wi = reflect(-p.wo, p.normal);
    } else {
      beta *= p.albedo;
      wi = rtOrthonormalA(p.normal) * disk.x + rtOrthonormalB(p.normal) * disk.y + p.normal * cosine;
    }
    if (ptMax3(beta) <= 0.0) { break; }
    let origin = rtOffsetOrigin(p.position, rtSideNormal(p.geometricNormal, wi));
    let next = traceFirstHit(origin, wi, 1e30);
    result.continuationRays++;
    if (next.t < 0.0) {
      if (delta) {
        result.indirect += beta * rtShadeMiss(wi);
      } else if (i == 1u && measureMisComplement && rtHasEnvironment()) {
        let e = rtEnvironment();
        result.directComplement = beta * rtEnvironmentRadiance(e) * ptMis(cosine * RT_SHADE_INV_PI, e.pmf * RT_ENV_UNIFORM_PDF);
      }
      break;
    }
    let previous = p.position;
    p = rtShadePoint(rtpRtHit(next), RtRay(origin, 0.0, wi, next.t));
    if (delta) {
      result.indirect += beta * rtShadeEmission(p);
    } else if (i == 1u && measureMisComplement) {
      let le = rtShadeEmission(p);
      let lightIndex = rtLightIndexOfHit(next.instance, next.triangle);
      if (ptMax3(le) > 0.0 && lightIndex != RT_NONE) {
        let light = rtLight(lightIndex);
        let pLight = light.pmf * rtTriangleSolidAnglePdf(light.area, length(p.position - previous), abs(dot(next.normal, wi)));
        result.directComplement = beta * le * ptMis(cosine * RT_SHADE_INV_PI, pLight);
      }
    }
    // A diffuse predecessor already samples visible emitters with full NEE. Delta
    // vertices cannot do that, so their sampled emission and sky stay in the tail.
    if (i < maxBounces && !rcMirror(p)) {
      let direct = rcDirect(p, seed, dim + 2u);
      result.indirect += beta * direct.direct;
      result.shadowRays += u32(direct.light != RT_NONE);
    }
  }
  return result;
}
`;
