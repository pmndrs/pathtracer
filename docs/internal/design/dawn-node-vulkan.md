# Subgroup matrices on Windows: Dawn on Vulkan through dawn.node

How we ran WebGPU subgroup-matrix kernels on the RTX 3060 Laptop (track N, M6), and why it takes a detour. Written
2026-11-17 for anyone who asks "how did you get the Vulkan backend on Windows? I couldn't find the switch in Dawn".

## Short answer

You can't get there through Chrome. We run the pages outside the browser in **dawn.node** (the npm `webgpu` package:
Dawn and Tint built as a Node addon). Its `create()` call takes the backend as a flag, `backend=vulkan`, and Dawn's
Vulkan backend exposes subgroup matrices on NVIDIA.

## Why Chrome can't do it

- Chrome on Windows runs WebGPU on Dawn's **D3D12** backend only (D3D11 as a fallback).
- Dawn's D3D12 backend has **no subgroup matrices**: the `chromium-experimental-subgroup-matrix` feature is never
  offered there.
- Dawn's **Vulkan** backend has them on NVIDIA (through `VK_KHR_cooperative_matrix`), but Chrome on Windows never
  gives WebGPU a hardware Vulkan adapter. With Chrome 157 stable and Canary 160, every switch we tried left
  chrome://gpu on D3D12/D3D11:
  - `--use-webgpu-adapter=vulkan`
  - `--force-webgpu-adapter=...`
  - `--enable-features=Vulkan`
  - `--use-vulkan=native`

So the switch you're looking for isn't a Chrome flag. It's an argument to dawn.node's `create()`.

## Setup

1. **Install dawn.node** into its own folder (git-ignored):

   ```bash
   mkdir -p .cache/dawn-node && cd .cache/dawn-node && npm install webgpu
   ```

   We used `webgpu` 0.6.2. Our host also accepts `DAWN_NODE=<path to webgpu/index.js>`.

2. **Put the Vulkan loader next to the addon.** Without this, `create()` fails with
   `vulkan-1.dll Windows Error: 87`. Copy `C:\Windows\System32\vulkan-1.dll` into
   `.cache/dawn-node/node_modules/webgpu/dist/win32-x64/`, next to `dawn.node`.

3. **Create the GPU with the backend and toggles:**

   ```js
   const { create, globals } = await import('webgpu');
   Object.assign(globalThis, globals); // GPUBufferUsage etc.
   const gpu = create([
     'backend=vulkan',
     'enable-dawn-features=allow_unsafe_apis,vulkan_enable_f16_on_nvidia',
     'disable-dawn-features=timestamp_quantization',
   ]);
   const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
   const device = await adapter.requestDevice({
     requiredFeatures: ['chromium-experimental-subgroup-matrix', 'shader-f16', 'subgroups', 'timestamp-query'],
   });
   ```

   WGSL then starts with `enable chromium_experimental_subgroup_matrix;` (plus `enable f16;` / `enable subgroups;`).

## What each switch does

| Flag | Why |
|---|---|
| `backend=vulkan` | Picks Dawn's Vulkan backend. `backend=d3d12` gives what Chrome uses, for comparison. |
| `allow_unsafe_apis` | The subgroup-matrix feature is experimental; Dawn offers it only with unsafe APIs allowed. |
| `vulkan_enable_f16_on_nvidia` | Without it Dawn hides `shader-f16` on NVIDIA Vulkan (the driver reports `storageInputOutput16 = false`), and with it every f16 matrix config, which leaves only int8. |
| `disable-dawn-features=timestamp_quantization` | Dawn rounds timestamps to 65,536 ns by default (a 0.6 ms step read 0.655 or 0.721 ms). With it off NVIDIA steps at 1,024 ns. This is the twin of Chrome's `--enable-webgpu-developer-features`. |
| `powerPreference: 'high-performance'` | dawn.node isn't tied to one GPU; a default request on this laptop returned the Intel iGPU. |

Our host uses `disable-dawn-features` for any toggle written with a leading `-` (for example `-use_dxc` on D3D12).

## Pitfalls we hit

- **The process exits early.** dawn.node doesn't keep Node's event loop alive while GPU promises are pending; a
  script that only awaits `mapAsync` or `onSubmittedWorkDone` can just end. Hold a timer (`setInterval`) until the
  work is done.
- **Broken log lines.** Tint prints compile logs to stdout without a final newline, so anything you print next
  lands on the same line. Our host starts each of its own records on a new line.
- **What the RTX 3060 offers.** NVIDIA lists 16×16×16 configs (f16 → f16, f16 → f32, int8), and no f32 → f32 or 8×8
  config. f16 sums reach 42 TFLOP/s, f32 sums 21 (`--page sgm-roofline mode=peak`).
- **Canary hands out the iGPU.** Canary 160 ignores `--use-adapter-luid`. Use Chrome stable for the browser arms.

## In this repo

The runner wraps all of the above, so every bench page runs unchanged in dawn.node:

```bash
node bench/runner/run.ts --page mlp-timing --gpu nvidia --dawn vulkan --dawn-toggles allow_unsafe_apis,vulkan_enable_f16_on_nvidia
```

- [bench/runner/dawn-host.ts](../../../bench/runner/dawn-host.ts) builds the `create()` flags (`dawnCreateFlags`). It
  gives the page a fake `window`, `location` and `document` and dawn.node's `navigator.gpu`, and makes relative
  `fetch` go to the runner's Vite server. One process runs per page, so each run gets a fresh device.
- `bench/runner/dawn-session.ts` spawns it. The runner keeps its load checks, ABBA order and CSV output; rows say
  `dawn.node/<version> vulkan +toggles` in the `browser` column.
- To bisect a D3D12-only problem, the same path takes `--dawn d3d12` with toggles such as `disable_robustness`,
  `d3d12_dont_use_shader_model_66_or_higher`, `-use_dxc`, or `dump_shaders` (with `--verbose`).

## What it is not

This is an **out-of-browser** result. A Windows Chrome user today gets D3D12 and our portable kernels, not these
matrix numbers. The numbers show what WebGPU's subgroup matrices can do on NVIDIA hardware with the same Dawn and
Tint, and what Chrome on Windows would gain if it offered them. `docs/internal/design/nrc-diverges.md` records the split.
