/**
 * The example template: one call, `defineExample(async (ctx) => { ... })`, gives a page a WebGPU device with the library's capability
 * tier, a full-window canvas, a Tweakpane panel, a stats box, progress and error overlays, and a frame loop that never queues more than
 * one frame on the GPU. The nav (logo, breadcrumb, docs and source links) is added by the build, not here.
 */
import { detectCaps, requestDeviceForCaps, type Caps } from '@pmndrs/raytracer';
import { Pane } from 'tweakpane';
import './theme.css';
import './example.css';

export interface ExampleStats {
  /** Sets one line of the stats box (insertion order is display order). */
  set(key: string, value: string | number): void;
  delete(key: string): void;
}

export interface ExampleContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  /** The library's view of the device (tier, subgroups, f16, timestamps). */
  caps: Caps;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
  pane: Pane;
  stats: ExampleStats;
  /** Shows a progress message over the canvas; `null` hides it. */
  progress(message: string | null): void;
  /** Shows an error over the canvas (the loop keeps running). */
  error(title: string, detail?: unknown): void;
  /** Canvas size in pixels: device pixels times the render scale. A renderer's film is this size. */
  filmSize(): [number, number];
  /** Renders below device resolution (0.25-1; the canvas is stretched). Fires the resize callbacks. */
  setRenderScale(scale: number): void;
  /**
   * The texture to draw this frame into: the canvas's current texture, or with `?offscreen=1` in the URL an offscreen texture of the
   * same size and format (headless checks: canvas presentation is not available everywhere, e.g. SwiftShader in a container).
   */
  target(): GPUTextureView;
  /** Called after the canvas changes size. */
  onResize(callback: () => void): void;
  /**
   * Starts the frame loop. `frame(encoder, dt)` records the frame; the template submits it and waits for the GPU before the next one
   * (a headless or hidden page must not queue work without bound). Return false when there was nothing to do.
   */
  loop(frame: (encoder: GPUCommandEncoder, dtMs: number) => boolean | void): void;
  /** Calls `onFile` with a file dropped onto the page (`accept` lists lower-case extensions such as `.glb`). */
  onDrop(accept: readonly string[], onFile: (file: File) => void): void;
}

function overlay(kind: 'progress' | 'error'): { root: HTMLElement; box: HTMLElement } {
  const root = document.createElement('div');
  root.className = `ex-overlay ${kind}`;
  root.hidden = true;
  const box = document.createElement('div');
  box.className = 'box';
  box.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  root.append(box);
  document.body.append(root);
  return { root, box };
}

function describe(detail: unknown): string {
  if (detail instanceof Error) return detail.stack ?? detail.message;
  return detail === undefined ? '' : String(detail);
}

/** Runs `setup` once the page has a device; shows a clear message where WebGPU is missing. */
export function defineExample(setup: (ctx: ExampleContext) => Promise<void> | void): void {
  const progressBox = overlay('progress');
  const errorBox = overlay('error');
  const showError = (title: string, detail?: unknown) => {
    errorBox.box.replaceChildren();
    const h = document.createElement('h2');
    h.textContent = title;
    errorBox.box.append(h);
    const text = describe(detail);
    if (text) {
      const pre = document.createElement('pre');
      pre.textContent = text;
      errorBox.box.append(pre);
    }
    errorBox.root.hidden = false;
    progressBox.root.hidden = true;
    console.error(title, detail);
  };
  const progress = (message: string | null) => {
    progressBox.root.hidden = message === null;
    progressBox.box.textContent = message ?? '';
  };

  void (async () => {
    progress('Starting WebGPU…');
    if (!('gpu' in navigator)) {
      showError('This browser has no WebGPU', 'The examples need WebGPU: Chrome or Edge 113+ is what we test; Safari 26 and Firefox 141+ on Windows also ship it.');
      return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      showError('No GPU adapter', 'WebGPU is present but returned no adapter: hardware acceleration may be off, or the driver is blocklisted.');
      return;
    }
    const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'example' });
    device.lost.then((info) => {
      if (info.reason !== 'destroyed') showError('The GPU device was lost', info.message);
    });
    device.addEventListener('uncapturederror', (event) => showError('WebGPU error', (event as GPUUncapturedErrorEvent).error.message));

    const canvas = document.createElement('canvas');
    canvas.id = 'view';
    document.body.prepend(canvas);
    const context = canvas.getContext('webgpu');
    if (!context) {
      showError('No WebGPU canvas context');
      return;
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    const offscreen = new URLSearchParams(location.search).has('offscreen');
    if (!offscreen) context.configure({ device, format, alphaMode: 'opaque' });
    let offscreenTexture: GPUTexture | null = null;
    const target = (): GPUTextureView => {
      if (!offscreen) return context.getCurrentTexture().createView();
      if (!offscreenTexture || offscreenTexture.width !== canvas.width || offscreenTexture.height !== canvas.height) {
        offscreenTexture?.destroy();
        offscreenTexture = device.createTexture({ label: 'offscreen target', size: [canvas.width, canvas.height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
      }
      return offscreenTexture.createView();
    };

    const paneHost = document.createElement('div');
    paneHost.className = 'ex-pane';
    document.body.append(paneHost);
    const pane = new Pane({ container: paneHost, title: 'Controls', expanded: innerWidth > 720 });

    const statsBox = document.createElement('dl');
    statsBox.className = 'ex-stats';
    document.body.append(statsBox);
    const statLines = new Map<string, HTMLElement>();
    const stats: ExampleStats = {
      set(key, value) {
        let dd = statLines.get(key);
        if (!dd) {
          const dt = document.createElement('dt');
          dt.textContent = key;
          dd = document.createElement('dd');
          statsBox.append(dt, dd);
          statLines.set(key, dd);
        }
        dd.textContent = typeof value === 'number' ? (Number.isInteger(value) ? String(value) : value.toFixed(2)) : value;
      },
      delete(key) {
        const dd = statLines.get(key);
        if (!dd) return;
        dd.previousElementSibling?.remove();
        dd.remove();
        statLines.delete(key);
      },
    };

    const resizeCallbacks: (() => void)[] = [];
    let renderScale = 1;
    const resize = () => {
      const dpr = Math.min(devicePixelRatio || 1, 2) * renderScale;
      const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width === w && canvas.height === h) return;
      canvas.width = w;
      canvas.height = h;
      for (const cb of resizeCallbacks) cb();
    };
    new ResizeObserver(resize).observe(canvas);
    resize();

    let busy = false;
    let last = performance.now();
    const loop: ExampleContext['loop'] = (frame) => {
      const tick = (now: number) => {
        requestAnimationFrame(tick);
        if (busy) return;
        const dt = now - last;
        last = now;
        const encoder = device.createCommandEncoder({ label: 'example frame' });
        let worked: boolean | void;
        try {
          worked = frame(encoder, dt);
        } catch (error) {
          showError('Frame failed', error);
          return;
        }
        if (worked === false) return;
        device.queue.submit([encoder.finish()]);
        busy = true;
        device.queue.onSubmittedWorkDone().then(() => (busy = false), () => (busy = false));
      };
      requestAnimationFrame(tick);
    };

    const drop = document.createElement('div');
    drop.className = 'ex-drop';
    drop.hidden = true;
    document.body.append(drop);
    const onDrop: ExampleContext['onDrop'] = (accept, onFile) => {
      drop.textContent = `Drop a ${accept.join(' or ')} file`;
      addEventListener('dragover', (e) => {
        e.preventDefault();
        drop.hidden = false;
      });
      addEventListener('dragleave', (e) => {
        if (e.relatedTarget === null) drop.hidden = true;
      });
      addEventListener('drop', (e) => {
        e.preventDefault();
        drop.hidden = true;
        const file = [...(e.dataTransfer?.files ?? [])].find((f) => accept.some((ext) => f.name.toLowerCase().endsWith(ext)));
        if (file) onFile(file);
        else showError('Unsupported file', `Drop a ${accept.join(' or ')} file.`);
      });
    };

    const ctx: ExampleContext = {
      adapter, device, caps, canvas, context, format, pane, stats, progress, error: showError, loop, onDrop, target,
      filmSize: () => [canvas.width, canvas.height],
      setRenderScale: (scale) => {
        renderScale = Math.min(1, Math.max(0.25, scale));
        resize();
      },
      onResize: (cb) => resizeCallbacks.push(cb),
    };
    await setup(ctx);
  })().catch((error) => showError('The example failed to start', error));
}
