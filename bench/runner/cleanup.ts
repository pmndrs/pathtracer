/**
 * Ported from ~/Developer/bvh/tests/gpu/cleanup.ts (MIT, same author). Adapted for the pathtracer bench runner.
 */
/**
 * Everything the runner starts (Chrome, its profile folder, the vite server)
 * registers a cleanup here the moment it exists. Normal exit, a failed run,
 * Ctrl-C and crashes all go through {@link runCleanups}, so nothing is left behind.
 *
 * Limits: a hard kill (taskkill /F on the runner, power loss) cannot run any
 * code. On Windows only Ctrl-C and Ctrl-Break (and closing the console) are
 * delivered to Node as signals; `process.kill(pid, 'SIGTERM')` from another
 * process just terminates it.
 */

type CleanupTask = () => Promise<void>;

const tasks = new Set<CleanupTask>();

/** Aborted as soon as shutdown starts, so slow startup steps (PowerShell lookup) can give up. */
const shutdownController = new AbortController();
export const shutdownSignal: AbortSignal = shutdownController.signal;

/**
 * Registers a cleanup and returns the function to close the resource normally.
 * The task runs at most once, whoever asks first (normal close or a signal), and
 * stays registered until it has FINISHED. That matters: a Ctrl-C that arrives while a
 * normal close is half done must wait for it, not exit and leave the profile folder behind.
 */
export function registerCleanup(task: CleanupTask): () => Promise<void> {
    let running: Promise<void> | null = null;
    const runOnce = (): Promise<void> => {
        running ??= task().finally(() => {
            tasks.delete(runOnce);
        });
        return running;
    };
    tasks.add(runOnce);
    return runOnce;
}

/**
 * Starting something new (a Chrome, a server) while the process is shutting down would
 * leave it behind: the main flow keeps running until process.exit. Call this first.
 */
export function throwIfShuttingDown(): void {
    if (shutdownSignal.aborted) throw new Error('The runner is shutting down.');
}

/**
 * Runs every registered cleanup, newest first. Each task runs once; failures are warned, not thrown.
 * Loops until nothing is registered, so something registered while shutting down is still caught.
 */
export async function runCleanups(): Promise<void> {
    shutdownController.abort();
    while (tasks.size > 0) {
        for (const task of [...tasks].reverse()) {
            try {
                await task();
            } catch (error) {
                console.warn(`warning: cleanup failed: ${error instanceof Error ? error.message : error}`);
            }
        }
    }
}

/** Ctrl-C and its relatives are 128 plus the signal number by convention. */
export const EXIT_CODE_SIGINT = 130;
const EXIT_CODE_SIGTERM = 143;
const EXIT_CODE_CRASH = 1;

let shuttingDown = false;

/** Clean up and exit when the process is interrupted or crashes. Call once, at startup. */
export function installShutdownHandlers(): void {
    const shutDown = (reason: string, exitCode: number) => {
        if (shuttingDown) return;
        shuttingDown = true;
        console.error(`\n${reason}: closing Chrome and the bench server.`);
        void runCleanups().finally(() => process.exit(exitCode));
    };
    process.on('SIGINT', () => shutDown('Interrupted', EXIT_CODE_SIGINT));
    process.on('SIGTERM', () => shutDown('Terminated', EXIT_CODE_SIGTERM));
    process.on('SIGHUP', () => shutDown('Console closed', EXIT_CODE_SIGTERM));
    process.on('uncaughtException', (error) => {
        console.error(error);
        shutDown('Crashed', EXIT_CODE_CRASH);
    });
    process.on('unhandledRejection', (error) => {
        console.error(error);
        shutDown('Crashed (unhandled rejection)', EXIT_CODE_CRASH);
    });
}
