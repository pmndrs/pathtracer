/**
 * Ported from ~/Developer/bvh/tests/gpu/chrome.ts (MIT, same author). Adapted: macOS first, Apple/AMD choices,
 * developer features flag, headed mode.
 */
/**
 * Finding, configuring, launching and stopping Chrome.
 *
 * Choosing a GPU (measured on this laptop, Chrome 156, Windows 11; see NOTES):
 *  - Headless Chrome gives the NVIDIA GPU whatever `powerPreference` says.
 *  - `--use-webgpu-power-preference=force-low-power` makes requestAdapter()
 *    return null, so it cannot be used to get the Intel GPU.
 *  - `--use-adapter-luid=<high>,<low>` works: Dawn then sees only that adapter.
 *    The LUID changes on every boot, so we look it up from the Windows device
 *    properties (DEVPKEY_Gpu_Luid) each run. No registry or settings changes.
 */

import { spawn, execFile } from 'node:child_process';
import { existsSync } from 'node:fs';
import { readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { type GpuChoice } from './contract.ts';

export function browserCandidates(
    platform: NodeJS.Platform = process.platform,
    env: NodeJS.ProcessEnv = process.env,
): string[] {
    if (platform === 'win32') {
        const programFiles = env.ProgramFiles ?? 'C:/Program Files';
        const programFilesX86 = env['ProgramFiles(x86)'] ?? 'C:/Program Files (x86)';
        const local = env.LOCALAPPDATA;
        return [
            join(programFiles, 'Google/Chrome/Application/chrome.exe'),
            join(programFilesX86, 'Google/Chrome/Application/chrome.exe'),
            ...(local ? [join(local, 'Google/Chrome/Application/chrome.exe')] : []),
            join(programFilesX86, 'Microsoft/Edge/Application/msedge.exe'),
            join(programFiles, 'Microsoft/Edge/Application/msedge.exe'),
        ];
    }
    if (platform === 'darwin') {
        return [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
            '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
        ];
    }
    return ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser'];
}

/** An explicit path (or CHROME_PATH) is authoritative: a typo must not quietly pick another browser. */
export function findBrowser(explicit: string | null, env: NodeJS.ProcessEnv = process.env): string {
    const requested = explicit ?? env.CHROME_PATH;
    if (requested) {
        if (!existsSync(requested)) throw new Error(`Browser executable does not exist: ${requested}`);
        return requested;
    }
    const found = browserCandidates().find((candidate) => existsSync(candidate));
    if (!found) throw new Error('Chrome or Edge not found. Pass --chrome <path> or set CHROME_PATH.');
    return found;
}

export interface AdapterLuid {
    high: number;
    low: number;
}

export interface ChromeLaunchOptions {
    profileDir: string;
    /** Pins Dawn to one physical adapter (see the file comment). */
    luid: AdapterLuid | null;
    /** Disables the GPU entirely, so WebGPU has no adapter. Used to test the failure path. */
    withoutGpu: boolean;
    /** Show the window instead of running headless. */
    headed?: boolean;
}

export function buildChromeArgs(options: ChromeLaunchOptions): string[] {
    const args = [
        ...(options.headed ? [] : ['--headless=new']),
        // WebGPU is on by default in current Chrome; this keeps older builds working.
        '--enable-unsafe-webgpu',
        // Without this Chrome rounds timestamp-query results to 100 us (a side-channel mitigation). The flag gives
        // raw GPU ticks. Source: Denoiser/tools/rt-capture/lib/headless.mjs ("unquantized timestamp queries").
        // PT_NO_DEV_FEATURES=1 leaves it off, to prove the calibration page detects quantization.
        // Always verify per session with bench/pages/timer-calibration.html: the page, not the flag, is the evidence.
        ...(process.env.PT_NO_DEV_FEATURES === '1' ? [] : ['--enable-webgpu-developer-features']),
        // Port 0: Chrome picks a free port and writes it to DevToolsActivePort in the profile.
        '--remote-debugging-port=0',
        `--user-data-dir=${options.profileDir}`,
        // A hidden or backgrounded page gets no GPU time, which would skew any timing test.
        '--disable-background-timer-throttling',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows',
        '--no-first-run',
        '--no-default-browser-check',
        '--window-size=1280,720',
        '--force-device-scale-factor=1',
    ];
    // Chrome on macOS defaults to Metal; stating it keeps the backend from changing under us.
    if (process.platform === 'darwin') args.push('--use-angle=metal');
    if (options.luid) args.push(`--use-adapter-luid=${options.luid.high},${options.luid.low}`);
    if (options.withoutGpu) args.push('--disable-gpu');
    // CI on a machine without a GPU sets e.g. "--no-sandbox --use-webgpu-adapter=swiftshader --use-angle=swiftshader --enable-unsafe-swiftshader".
    const extra = process.env.PT_CHROME_FLAGS?.trim();
    if (extra) args.push(...extra.split(/\s+/));
    args.push('about:blank');
    return args;
}

const PCI_VENDOR_ID: Record<GpuChoice, string> = { nvidia: '10DE', intel: '8086', amd: '1002', apple: '106B' };

interface GpuLuidEntry {
    instanceId: string;
    luid: AdapterLuid;
}

/** Reads lines of the form `PCI\VEN_10DE&DEV_...|80901` (device id, LUID as one 64-bit number). */
export function parseGpuLuidLines(output: string): GpuLuidEntry[] {
    const entries: GpuLuidEntry[] = [];
    for (const line of output.split(/\r?\n/)) {
        const [instanceId, luidText] = line.trim().split('|');
        if (!instanceId || !luidText || !/^\d+$/.test(luidText)) continue;
        const luid = BigInt(luidText);
        entries.push({ instanceId, luid: { high: Number(luid >> 32n), low: Number(luid & 0xffffffffn) } });
    }
    return entries;
}

/** The LUID of the first listed device from that vendor's PCI vendor id, or null. */
export function findGpuLuid(entries: GpuLuidEntry[], gpu: GpuChoice): AdapterLuid | null {
    const vendorTag = `VEN_${PCI_VENDOR_ID[gpu]}`;
    return entries.find((entry) => entry.instanceId.toUpperCase().includes(vendorTag))?.luid ?? null;
}

const LIST_GPU_LUIDS_SCRIPT = `
Get-PnpDevice -Class Display -PresentOnly | ForEach-Object {
    $luid = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName '{60B193CB-5276-4D0F-96FC-F173ABAD3EC6} 2' -ErrorAction SilentlyContinue
    if ($luid) { '{0}|{1}' -f $_.InstanceId, $luid.Data }
}`;

/** The PowerShell query normally takes about 2 s; this only guards against a hung PowerShell. */
const LUID_LOOKUP_TIMEOUT_MS = 20_000;

/**
 * Finds the LUID Chrome needs to be pinned to `gpu` (Windows only).
 * Throws an error in plain language if the platform is not Windows, if the
 * PowerShell query fails or times out, or if no such GPU is listed.
 */
export async function discoverGpuLuid(gpu: GpuChoice, signal?: AbortSignal): Promise<AdapterLuid> {
    if (process.platform !== 'win32') {
        throw new Error(
            `Pinning the ${gpu} GPU works on Windows only (it needs the adapter LUID from Windows). ` +
                'On this platform run without pinning; --gpu then only checks which adapter Chrome gave.',
        );
    }
    const queryName = 'powershell.exe Get-PnpDevice -Class Display (GPU adapter list)';
    const output = await new Promise<string>((resolve, reject) => {
        execFile(
            'powershell.exe',
            ['-NoProfile', '-NonInteractive', '-Command', LIST_GPU_LUIDS_SCRIPT],
            { timeout: LUID_LOOKUP_TIMEOUT_MS, windowsHide: true, signal },
            (error, stdout) => {
                if (!error) return resolve(stdout);
                const timedOut = error.killed === true;
                reject(
                    new Error(
                        `Could not list the GPUs to find the ${gpu} one: "${queryName}" ` +
                            (timedOut ? `did not finish within ${LUID_LOOKUP_TIMEOUT_MS} ms` : `failed: ${error.message}`),
                    ),
                );
            },
        );
    });
    const entries = parseGpuLuidLines(output);
    const luid = findGpuLuid(entries, gpu);
    if (!luid) {
        const seen = entries.length > 0 ? entries.map((entry) => entry.instanceId).join(', ') : 'none';
        throw new Error(`No ${gpu} GPU found among the Windows display devices (found: ${seen}). Check Device Manager.`);
    }
    return luid;
}

export interface ChromeProcess {
    /** DevTools HTTP endpoint, for example http://127.0.0.1:51234. Empty until {@link ready} resolves. */
    devtoolsOrigin: string;
    /** Resolves once Chrome has opened its DevTools port; rejects if it exits or is too slow. */
    ready(): Promise<void>;
    /** Kills the whole process tree and waits for it to exit. Safe to call more than once. */
    stop(): Promise<void>;
}

/** Chrome normally opens its DevTools port in well under a second; this covers a cold, busy machine. */
const DEVTOOLS_PORT_TIMEOUT_MS = 20_000;
const DEVTOOLS_PORT_POLL_MS = 50;

/** How long to wait for taskkill / the exit event before giving up on a process. */
const KILL_WAIT_MS = 5_000;

/**
 * Starts Chrome and returns at once, before it is ready, so the caller can
 * register `stop()` for cleanup immediately. A Chrome that fails to start
 * (bad path, crash) rejects `ready()` rather than throwing here.
 */
export function startChrome(executable: string, options: ChromeLaunchOptions): ChromeProcess {
    const child = spawn(executable, buildChromeArgs(options), {
        stdio: ['ignore', 'ignore', 'pipe'],
        windowsHide: true,
        // Own process group on Unix, so the whole tree can be killed at once.
        detached: process.platform !== 'win32',
    });
    let stderr = '';
    const startFailure: { error: Error | null } = { error: null };
    child.stderr.on('data', (chunk: Buffer) => {
        stderr = (stderr + chunk.toString()).slice(-4000);
    });
    // Without this handler a missing or non-executable file would crash Node with an unhandled 'error' event.
    child.on('error', (error) => {
        startFailure.error = error;
    });
    const exited = new Promise<void>((resolve) => child.once('exit', () => resolve()));
    let stopping: Promise<void> | null = null;

    const handle: ChromeProcess = {
        devtoolsOrigin: '',
        ready: async () => {
            const portFile = join(options.profileDir, 'DevToolsActivePort');
            const deadline = Date.now() + DEVTOOLS_PORT_TIMEOUT_MS;
            while (Date.now() < deadline) {
                if (startFailure.error) throw new Error(`Could not start ${executable}: ${startFailure.error.message}`);
                if (child.exitCode !== null) {
                    throw new Error(`Chrome exited during startup (code ${child.exitCode}). ${stderr.trim()}`);
                }
                const port = await readFile(portFile, 'utf8')
                    .then((text) => text.split('\n')[0]?.trim() ?? '')
                    .catch(() => '');
                if (/^\d+$/.test(port)) {
                    handle.devtoolsOrigin = `http://127.0.0.1:${port}`;
                    return;
                }
                await new Promise((resolve) => setTimeout(resolve, DEVTOOLS_PORT_POLL_MS));
            }
            throw new Error(`Chrome did not open its DevTools port within ${DEVTOOLS_PORT_TIMEOUT_MS} ms.`);
        },
        stop: () => {
            stopping ??= killProcessTree(child.pid, exited).then(() => {
                child.stderr.destroy();
            });
            return stopping;
        },
    };
    return handle;
}

async function killProcessTree(pid: number | undefined, exited: Promise<void>): Promise<void> {
    if (pid === undefined) return;
    // unref: a pending give-up timer must not keep Node alive after everything else is done.
    const timeout = (ms: number) =>
        new Promise<void>((resolve) => {
            setTimeout(resolve, ms).unref();
        });
    if (process.platform === 'win32') {
        // /T kills Chrome's helper processes too; plain kill() leaves them running.
        const taskkill = spawn('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
        taskkill.on('error', () => {}); // taskkill missing: the exit wait below simply times out
        await Promise.race([new Promise((resolve) => taskkill.once('exit', resolve)), timeout(KILL_WAIT_MS)]);
    } else {
        killUnix(pid);
    }
    await Promise.race([exited, timeout(KILL_WAIT_MS)]);
}

/** Kills the process group (negative pid), or just the process; "no such process" means it is already gone. */
function killUnix(pid: number): void {
    for (const target of [-pid, pid]) {
        try {
            process.kill(target, 'SIGKILL');
            return;
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error;
        }
    }
}

/** Removes the throwaway profile. Chrome may hold files for a moment after exit, so retry. */
export async function removeProfile(profileDir: string): Promise<void> {
    try {
        await rm(profileDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
    } catch (error) {
        console.warn(`warning: could not remove ${profileDir}: ${error instanceof Error ? error.message : error}`);
    }
}
