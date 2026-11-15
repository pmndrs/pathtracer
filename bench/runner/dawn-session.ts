/**
 * The runner's session for `--dawn <backend>`: same surface as BrowserSession (preflight, browserLabel, runPage, close),
 * but each page runs in its own dawn.node process (dawn-host.ts). Load checks, ABBA plans, the clock sampler, CSV rows
 * and uploads are the runner's, unchanged. See dawn-host.ts for why this exists (M6: Vulkan on Windows).
 */
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { registerCleanup, throwIfShuttingDown } from './cleanup.ts';
import { QUERY_GPU, QUERY_TOKEN, type AdapterSummary, type BenchReport, type GpuChoice } from './contract.ts';
import { DAWN_HOST_PREFIX, defaultDawnNodePath } from './dawn-host.ts';
import { adapterMismatch, evaluatePage, type BrowserMessage } from './evaluate.ts';
import { WebGpuUnavailableError, WrongAdapterError, type PageResult, type PageRunOptions } from './session.ts';

const HOST = resolve(import.meta.dirname, 'dawn-host.ts');
const REPO_ROOT = resolve(import.meta.dirname, '..', '..');

export interface DawnSessionOptions {
  serverOrigin: string;
  backend: string;
  toggles: string[];
  gpu: GpuChoice | null;
}

interface HostOutput {
  events: Array<Record<string, unknown>>;
  stderr: string;
  code: number | null;
  timedOut: boolean;
}

export class DawnSession {
  private readonly options: DawnSessionOptions;
  private readonly running = new Set<ReturnType<typeof spawn>>();

  private constructor(options: DawnSessionOptions) {
    this.options = options;
    registerCleanup(async () => {
      for (const child of this.running) child.kill();
    });
  }

  static launch(options: DawnSessionOptions): DawnSession {
    throwIfShuttingDown();
    return new DawnSession(options);
  }

  /** `dawn.node <webgpu version> vulkan [toggles]`, for the CSV `browser` column. */
  async browserLabel(): Promise<string> {
    let version = '?';
    try {
      const pkg = JSON.parse(readFileSync(join(dirname(defaultDawnNodePath(REPO_ROOT)), 'package.json'), 'utf8')) as { version?: string };
      version = pkg.version ?? '?';
    } catch {
      // keep '?'
    }
    return `dawn.node/${version} ${this.options.backend}${this.options.toggles.length ? ` +${this.options.toggles.join('+')}` : ''}`;
  }

  async preflight(): Promise<AdapterSummary> {
    const out = await this.host(['--preflight'], 60_000);
    const event = out.events.find((e) => e.kind === 'preflight');
    if (!event) throw new WebGpuUnavailableError(`dawn.node preflight failed (exit ${out.code}): ${this.describeFailure(out)}`);
    if (!event.adapter) throw new WebGpuUnavailableError(`WebGPU is not available: dawn.node found no ${this.options.backend} adapter.`);
    const adapter = event.adapter as AdapterSummary;
    const mismatch = adapterMismatch(this.options.gpu, adapter, false);
    if (mismatch) throw new WrongAdapterError(mismatch.replace('Chrome gave', 'dawn.node gave'));
    return adapter;
  }

  async runPage(pageName: string, run: PageRunOptions): Promise<PageResult> {
    const startTime = Date.now();
    const token = `${process.pid}-${startTime}-${pageName}`;
    const query = new URLSearchParams({ [QUERY_TOKEN]: token });
    if (this.options.gpu) query.set(QUERY_GPU, this.options.gpu);
    for (const [key, value] of Object.entries(run.args)) query.set(key, value);
    const out = await this.host(['--page', pageName, '--query', query.toString()], run.timeoutMs);
    const messages: BrowserMessage[] = out.events
      .filter((e) => e.kind === 'message')
      .map((e) => ({ origin: 'console', level: e.level as BrowserMessage['level'], text: String(e.text) }));
    const reportEvent = out.events.find((e) => e.kind === 'report');
    const report = (reportEvent?.report as BenchReport | undefined) ?? null;
    let runnerProblem: string | null = null;
    if (out.timedOut) runnerProblem = `Timed out after ${run.timeoutMs} ms without a report (dawn.node page hung).`;
    else if (!report) runnerProblem = `dawn.node host exited (code ${out.code}) without a report: ${this.describeFailure(out)}`;
    else if (report.token !== token) runnerProblem = 'dawn.node host reported a different token.';
    const verdict = evaluatePage({
      report,
      runnerProblem,
      messages,
      expectFailure: false,
      expectMessage: null,
      wantedGpu: this.options.gpu,
      allowSoftware: false,
    });
    return { pageName, verdict, report, messages, durationMs: Date.now() - startTime, screenshotPath: null };
  }

  async close(): Promise<void> {
    for (const child of this.running) child.kill();
  }

  private describeFailure(out: HostOutput): string {
    const errors = out.events.filter((e) => e.kind === 'message' && e.level === 'error').map((e) => String(e.text));
    return [...errors, out.stderr.trim()].filter((t) => t !== '').join(' | ').slice(0, 2000) || '(no output)';
  }

  private host(extra: string[], timeoutMs: number): Promise<HostOutput> {
    const args = [HOST, '--backend', this.options.backend, '--toggles', this.options.toggles.join(','), '--origin', this.options.serverOrigin, ...extra];
    return new Promise((resolvePromise) => {
      const child = spawn(process.execPath, args, { cwd: REPO_ROOT, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
      this.running.add(child);
      const events: Array<Record<string, unknown>> = [];
      let stdoutRest = '';
      let stderr = '';
      let timedOut = false;
      const timer = setTimeout(() => {
        timedOut = true;
        child.kill();
      }, timeoutMs);
      child.stdout.setEncoding('utf8');
      child.stdout.on('data', (chunk: string) => {
        stdoutRest += chunk;
        const lines = stdoutRest.split('\n');
        stdoutRest = lines.pop() ?? '';
        for (const line of lines) {
          // Dawn's own stdout (compilation logs) has no final newline, so a record can start mid-line.
          const at = line.indexOf(DAWN_HOST_PREFIX);
          if (at < 0) {
            if (line.trim()) events.push({ kind: 'message', level: 'log', text: line.trim() });
            continue;
          }
          try {
            events.push(JSON.parse(line.slice(at + DAWN_HOST_PREFIX.length)) as Record<string, unknown>);
          } catch {
            events.push({ kind: 'message', level: 'warning', text: `unparsable host line: ${line.slice(0, 200)}` });
          }
        }
      });
      child.stderr.setEncoding('utf8');
      child.stderr.on('data', (chunk: string) => {
        stderr += chunk;
      });
      child.on('close', (code) => {
        clearTimeout(timer);
        this.running.delete(child);
        resolvePromise({ events, stderr, code, timedOut });
      });
    });
  }
}
