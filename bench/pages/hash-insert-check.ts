/**
 * Hash-grid insert under contention (pmndrs/pathtracer#1): many threads insert the same handful of cells into one bucket at once.
 * Each cell must end up in exactly one slot, every thread must get the slot that holds its own key, and no insert may be refused
 * while the bucket has room. A second case has more cells than slots: a refused cell must then be absent from a full bucket.
 *
 * The `current` arm compiles `hashGridInsertWgsl()`, the text the cache's update kernel runs. The `dd771c8` arm is the insert gi
 * ported (it let a cell take two slots): a control that shows whether this device exposes the race at all. Only `current` gates
 * the run. `node bench/runner/run.ts --page hash-insert-check [--args rounds=32 buckets=64 threadsPerBucket=1024]`.
 */
import { hashGridInsertWgsl } from '../nrc/hash-grid.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';

const BUCKET = 8;

const DD771C8_INSERT = /* wgsl */ `
fn hgInsert(k: vec2u) -> i32 {
  for (var i = 0u; i < ${BUCKET}u; i++) {
    if (atomicLoad(&keys[k.x + i]) == k.y) { return i32(k.x + i); }
  }
  for (var i = 0u; i < ${BUCKET}u; i++) {
    let slot = k.x + i;
    if (atomicLoad(&keys[slot]) == 0u) {
      var res = atomicCompareExchangeWeak(&keys[slot], 0u, k.y);
      if (!res.exchanged && res.old_value == 0u) { res = atomicCompareExchangeWeak(&keys[slot], 0u, k.y); }
      if (res.exchanged || res.old_value == k.y) { return i32(slot); }
    }
  }
  return -1;
}
`;

const kernel = (insert: string) => /* wgsl */ `
struct Params { buckets: u32, threadsPerBucket: u32, distinct: u32, round: u32 }
@group(0) @binding(0) var<storage, read_write> keys: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> results: array<i32>;
@group(0) @binding(2) var<uniform> params: Params;
${insert}
fn mix32(x0: u32) -> u32 {
  var x = x0 ^ (x0 >> 16u);
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  return x ^ (x >> 16u);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= params.buckets * params.threadsPerBucket) { return; }
  // Interleave buckets across neighbouring workgroups so every bucket sees threads from many groups at once.
  let bucket = (t / 64u) % params.buckets;
  let cell = mix32(t ^ (params.round * 0x9e3779b9u)) % params.distinct;
  results[t] = hgInsert(vec2u(bucket * ${BUCKET}u, bucket * 256u + cell + 1u));
}
`;

/** TS twin of the kernel's `mix32` (u32 arithmetic). */
function mix32(x0: number): number {
  let x = (x0 ^ (x0 >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const rounds = numberArg(args, 'rounds', 32);
  const buckets = numberArg(args, 'buckets', 64);
  const threadsPerBucket = numberArg(args, 'threadsPerBucket', 1024);
  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);

  const threads = buckets * threadsPerBucket;
  const keys = storageBuffer(device, buckets * BUCKET * 4, 'insert keys');
  const results = storageBuffer(device, threads * 4, 'insert results');
  const params = device.createBuffer({ label: 'insert params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const metrics: Record<string, number> = {};
  const failures: string[] = [];

  const arms = { current: hashGridInsertWgsl(BUCKET), dd771c8: DD771C8_INSERT };
  for (const [arm, insert] of Object.entries(arms)) {
    const module = device.createShaderModule({ label: `insert ${arm}`, code: kernel(insert) });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(`${arm}: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);
    const pipeline = await device.createComputePipelineAsync({ label: `insert ${arm}`, layout: 'auto', compute: { module, entryPoint: 'main' } });
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: keys } }, { binding: 1, resource: { buffer: results } }, { binding: 2, resource: { buffer: params } }],
    });

    for (const distinct of [3, 12]) {
      const name = `${arm}_cells${distinct}`;
      let duplicateSlots = 0;
      let wrongSlot = 0;
      let refusedWithRoom = 0;
      let refused = 0;
      for (let round = 0; round < rounds; round++) {
        device.queue.writeBuffer(params, 0, new Uint32Array([buckets, threadsPerBucket, distinct, round]));
        const encoder = device.createCommandEncoder({ label: `${name} round ${round}` });
        encoder.clearBuffer(keys);
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(threads / 64));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const table = new Uint32Array(await readBuffer(device, keys, buckets * BUCKET * 4));
        const slots = new Int32Array(await readBuffer(device, results, threads * 4));

        for (let b = 0; b < buckets; b++) {
          const bucket = table.subarray(b * BUCKET, (b + 1) * BUCKET);
          const used = [...bucket].filter((k) => k !== 0);
          duplicateSlots += used.length - new Set(used).size;
        }
        for (let t = 0; t < threads; t++) {
          const b = Math.floor(t / 64) % buckets;
          const slot = slots[t]!;
          const key = b * 256 + (mix32((t ^ Math.imul(round, 0x9e3779b9)) >>> 0) % distinct) + 1;
          if (slot >= 0) {
            if (slot < b * BUCKET || slot >= (b + 1) * BUCKET || table[slot] !== key) wrongSlot++;
            continue;
          }
          refused++;
          const bucket = table.subarray(b * BUCKET, (b + 1) * BUCKET);
          if (bucket.includes(0) || bucket.includes(key)) refusedWithRoom++;
        }
      }
      metrics[`${name}_duplicate_slots`] = duplicateSlots;
      metrics[`${name}_wrong_slot`] = wrongSlot;
      metrics[`${name}_refused`] = refused;
      metrics[`${name}_refused_with_room`] = refusedWithRoom;
      if (arm === 'current' && duplicateSlots + wrongSlot + refusedWithRoom > 0) {
        failures.push(`${name}: ${duplicateSlots} duplicate slots, ${wrongSlot} wrong slots, ${refusedWithRoom} refusals with room`);
      }
      if (arm === 'current' && distinct <= BUCKET && refused > 0) failures.push(`${name}: ${refused} refusals with fewer cells than slots`);
    }
  }
  for (const buffer of [keys, results, params]) buffer.destroy();

  await drainDevice(device);
  if (deviceErrors.length) throw new Error(deviceErrors.join('\n'));
  if (failures.length) throw new Error(`hash insert check failed:\n${failures.join('\n')}\n${JSON.stringify(metrics)}`);
  reportResult({
    scene: 'hash-insert-check',
    renderer: 'ours',
    config: { rounds, buckets, threadsPerBucket, bucket: BUCKET },
    valid: true,
    metrics,
    notes: { control: 'dd771c8_* rows are the pre-fix insert; a non-zero duplicate count there shows this device exposes the race' },
  }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
