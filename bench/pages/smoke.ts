/**
 * Smoke page: proves the whole runner path on a real adapter.
 *  - makes a WebGPU device and reports adapter info;
 *  - renders a gradient with a compute shader into a storage buffer (RGBA f32, row 0 = top);
 *  - reads it back ONCE, at the end, checks it against the analytic gradient;
 *  - uploads it as an image (POST to the runner, written as PFM) and reports a JSON result.
 *
 * Args: width=256 height=256
 */

import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';

const SHADER = /* wgsl */ `
struct Params { width: u32, height: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> image: array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.width || id.y >= params.height) { return; }
  // Row 0 = top: r grows to the right, g grows downward.
  let u = (f32(id.x) + 0.5) / f32(params.width);
  let v = (f32(id.y) + 0.5) / f32(params.height);
  image[id.y * params.width + id.x] = vec4f(u, v, 0.5, 1.0);
}
`;

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const width = numberArg(args, 'width', 256);
  const height = numberArg(args, 'height', 256);

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const device = await adapter.requestDevice();
  deviceErrors = watchDevice(device);

  const bytes = width * height * 16;
  const storage = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniform, 0, new Uint32Array([width, height, 0, 0]));

  const module = device.createShaderModule({ code: SHADER });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: storage } },
    ],
  });

  // Everything on the GPU first; the single readback comes after the work, never inside it.
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();
  encoder.copyBufferToBuffer(storage, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  // Check against the analytic gradient.
  let maxAbsError = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const errors = [
        Math.abs((data[i] as number) - (x + 0.5) / width),
        Math.abs((data[i + 1] as number) - (y + 0.5) / height),
        Math.abs((data[i + 2] as number) - 0.5),
        Math.abs((data[i + 3] as number) - 1),
      ];
      maxAbsError = Math.max(maxAbsError, ...errors);
    }
  }
  if (!(maxAbsError < 1e-6)) throw new Error(`Gradient differs from the analytic one by ${maxAbsError}.`);

  const upload = await uploadImage('gradient', { width, height, channels: 4, data });
  await drainDevice(device);

  reportResult(
    {
      scene: 'gradient',
      renderer: 'smoke',
      config: { width, height },
      metrics: { pixels: width * height, max_abs_error: maxAbsError, upload_bytes: upload.bytes },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
