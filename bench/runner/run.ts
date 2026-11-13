/**
 * `node bench/runner/run.ts --page <name> [--gpu apple|nvidia|intel|amd] [--headed] [--repeat 3] [--args k=v ...]`
 *
 * Starts the vite bench site (repo root) with the image upload endpoint, launches installed Chrome (WebGPU, developer
 * features on so timestamp queries are not quantized), runs bench/pages/<name>.html once or N times, collects the
 * JSON results and the uploaded PFM images, and appends CSV rows to bench/results/<suite>.csv.
 * Ported from ~/Developer/bvh/tests/gpu/run.ts (MIT, same author); the page contract is different: one result per run.
 *
 * Exit code 0 only if every run passed. A run that passed but whose timing is not VALID still exits 0; its CSV rows
 * carry valid=0 and the reason.
 */

import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseRunnerArgs, UsageError, USAGE, type RunnerOptions } from './args.ts';
import { EXIT_CODE_SIGINT as EXIT_CODE_INTERRUPTED, installShutdownHandlers, registerCleanup, shutdownSignal } from './cleanup.ts';
import { startClockSampler, type ClockSampler } from './clock-note.ts';
import { appendCsv } from './csv.ts';
import { buildRows, deviceLabel, makeRunId, planRuns, utcDate } from './plan.ts';
import { type AdapterSummary, type BenchReport, type RunRecord, type RunSummary } from './contract.ts';
import { acquireGpuLock } from './gpu-lock.ts';
import { gitCommit } from './measured-commit.ts';
import { BrowserSession, WebGpuUnavailableError, WrongAdapterError } from './session.ts';
import { INCOMING_DIR, REPOSITORY_ROOT, RESULTS_DIR, startBenchServer, type BenchServer } from './server.ts';
import { DEFAULT_PAGE_TIMEOUT_MS } from './contract.ts';

function describeAdapter(adapter: AdapterSummary): string {
  const parts = [`vendor=${adapter.vendor}`, `architecture=${adapter.architecture}`];
  if (adapter.device) parts.push(`device=${adapter.device}`);
  if (adapter.description) parts.push(`description=${adapter.description}`);
  return parts.join(' ');
}

const indent = (text: string): string =>
  text
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => `      ${line}`)
    .join('\n');

async function main(): Promise<number> {
  let options: RunnerOptions;
  try {
    options = parseRunnerArgs(process.argv.slice(2));
  } catch (error) {
    if (error instanceof UsageError) {
      console.error(`${error.message}\n\n${USAGE}`);
      return 2;
    }
    throw error;
  }
  if (options.help) {
    console.log(USAGE);
    return 0;
  }
  if (!options.page) {
    console.error(`--page is required.\n\n${USAGE}`);
    return 2;
  }
  const pageFile = join(REPOSITORY_ROOT, 'bench', 'pages', `${options.page}.html`);
  if (!existsSync(pageFile)) {
    console.error(`No such page: ${pageFile}`);
    return 2;
  }

  const runStart = new Date();
  const runId = makeRunId(runStart, options.page);
  const suite = options.suite ?? options.page;
  const csvPath = join(RESULTS_DIR, `${suite}.csv`);
  const plan = planRuns(options);
  const summary: RunSummary = {
    schema: 1,
    runId,
    startedAt: runStart.toISOString(),
    commit: recordedCommit(),
    page: options.page,
    gpuRequested: options.gpu,
    adapter: null,
    browser: null,
    ok: false,
    totalMs: 0,
    error: null,
    timingVerdict: null,
    runs: [],
    uploads: [],
    csvFile: null,
  };
  let failures = 0;
  let session: BrowserSession | null = null;
  let server: BenchServer | null = null;
  let sampler: ClockSampler | null = null;
  let releaseLock: () => void = () => {};
  let closeServer: () => Promise<void> = async () => {};

  try {
    if (!options.noLock) {
      releaseLock = await acquireGpuLock({
        who: `bench ${options.page}`,
        signal: shutdownSignal,
        onWait: (owner) => console.log(`waiting for the GPU lock (held by: ${owner}); --no-lock skips it`),
      });
      registerCleanup(async () => releaseLock());
    }

    server = await startBenchServer(runId);
    closeServer = registerCleanup(server.close);
    console.log(`bench site: ${server.origin}   gpu requested: ${options.gpu ?? 'any'}   run id: ${runId}`);

    session = await BrowserSession.launch({
      serverOrigin: server.origin,
      chromePath: options.chromePath,
      gpu: options.gpu,
      withoutGpu: options.withoutWebGpu,
      allowSoftware: options.allowSoftware,
      headed: options.headed,
    });
    summary.adapter = await session.preflight();
    summary.browser = await session.browserLabel();
    console.log(`adapter: ${describeAdapter(summary.adapter)}`);
    console.log(`browser: ${summary.browser}`);
    if (summary.adapter.isFallbackAdapter) console.log('SOFTWARE ADAPTER: this run is on a CPU fallback (allowed by --allow-software), not a real GPU.');

    sampler = await startClockSampler();
    console.log(`page: ${options.page}   runs: ${plan.map((run) => run.variant).join(' ')}   clock sampler: ${sampler.kind}`);

    for (const [index, planned] of plan.entries()) {
      const slot = `${index + 1}-${planned.variant === '-' ? 'run' : planned.variant}`;
      server.setSlot(slot);
      const result = await session.runPage(options.page, {
        timeoutMs: options.timeoutMs ?? DEFAULT_PAGE_TIMEOUT_MS,
        args: planned.args,
        screenshotPath: index === plan.length - 1 ? join(INCOMING_DIR, runId, 'screenshot.png') : null,
      });
      const ok = result.verdict.outcome === 'pass';
      const record: RunRecord = {
        index: index + 1,
        variant: planned.variant,
        args: planned.args,
        outcome: ok ? 'pass' : 'fail',
        reasons: result.verdict.reasons,
        durationMs: result.durationMs,
        result: result.report?.result ?? null,
      };
      summary.runs.push(record);
      if (!ok) failures++;
      printRun(record, result.report, options.verbose ? result.messages : []);
    }

    const samplerKind = sampler.kind;
    const verdict = await sampler.stop();
    sampler = null;
    summary.timingVerdict = { kind: verdict.kind, reason: verdict.reason };
    const label = verdict.kind === 'valid' ? 'VALID' : verdict.kind === 'invalid' ? 'INVALID' : 'NO VERDICT';
    console.log(`\nTiming ${label}: ${verdict.reason}${verdict.detail ? `\n  ${verdict.detail}` : ''}`);

    if (!options.noCsv && summary.adapter && failures === 0) {
      const rows = buildRows({
        date: utcDate(runStart),
        commit: summary.commit ?? 'unknown',
        device: deviceLabel(summary.adapter),
        browser: summary.browser ?? 'Chrome',
        page: options.page,
        runs: summary.runs,
        timingValid: verdict.kind === 'valid',
        verdictNote: verdict.kind === 'valid' ? `timing valid (${samplerKind})` : `timing ${verdict.kind}: ${verdict.reason}`.slice(0, 160),
      });
      const written = appendCsv(csvPath, rows);
      if (written > 0) summary.csvFile = csvPath;
      console.log(`csv: ${written} rows appended to ${csvPath}`);
    }
    summary.uploads = server.uploads;
    for (const upload of server.uploads) {
      console.log(`image: ${upload.pfm}  ${upload.width}x${upload.height}x${upload.channels}${upload.nonFinite > 0 ? `  WARNING ${upload.nonFinite} non-finite floats` : ''}`);
    }
  } catch (error) {
    // Ctrl-C closed Chrome under our feet; the shutdown handler is already exiting the process.
    if (shutdownSignal.aborted) return EXIT_CODE_INTERRUPTED;
    summary.error = error instanceof Error ? error.message : String(error);
    if (error instanceof WebGpuUnavailableError || error instanceof WrongAdapterError) console.error(`\nFAILED: ${error.message}`);
    else console.error(`\nFAILED: ${error instanceof Error ? (error.stack ?? error.message) : error}`);
    failures++;
  } finally {
    if (sampler) await sampler.stop();
    await session?.close();
    await closeServer();
    releaseLock();
  }

  summary.ok = failures === 0;
  summary.totalMs = Date.now() - runStart.getTime();
  // The run JSON always lands next to the images, and wherever --json says.
  const json = `${JSON.stringify(summary, null, 2)}\n`;
  mkdirSync(join(INCOMING_DIR, runId), { recursive: true });
  writeFileSync(join(INCOMING_DIR, runId, 'run.json'), json);
  if (options.jsonPath) {
    writeFileSync(resolve(options.jsonPath), json);
    console.log(`json: ${resolve(options.jsonPath)}`);
  }
  console.log(`\n${summary.ok ? 'OK' : 'FAILED'}  page=${options.page}  gpu=${options.gpu ?? 'default'}  total ${(summary.totalMs / 1000).toFixed(1)} s  results: ${join(INCOMING_DIR, runId)}`);
  return summary.ok ? 0 : 1;
}

function printRun(record: RunRecord, report: BenchReport | null, messages: Array<{ origin: string; level: string; text: string }>): void {
  const seconds = (record.durationMs / 1000).toFixed(2);
  const args = Object.keys(record.args).length > 0 ? `  ${JSON.stringify(record.args)}` : '';
  console.log(`  ${record.outcome === 'pass' ? 'PASS' : 'FAIL'}  run ${record.index}${record.variant === '-' ? '' : ` (${record.variant})`}  ${seconds} s${args}`);
  if (record.result) console.log(indent(`metrics: ${JSON.stringify(record.result.metrics)}`));
  if (record.result?.valid === false) console.log(indent('page says its numbers are NOT valid'));
  if (record.result?.notes && Object.keys(record.result.notes).length > 0) console.log(indent(`notes: ${JSON.stringify(record.result.notes)}`));
  if (record.outcome === 'fail') for (const reason of record.reasons) console.log(indent(reason));
  if (report?.adapter === null && record.outcome === 'pass') console.log(indent('note: the page reported no adapter, so it could not be checked against --gpu'));
  for (const message of messages) console.log(indent(`[${message.origin}/${message.level}] ${message.text}`));
}

installShutdownHandlers();
process.exitCode = await main();

/** The commit being measured; undefined when git can't tell. */
function recordedCommit(): string | undefined {
  try {
    return gitCommit();
  } catch {
    return undefined;
  }
}
