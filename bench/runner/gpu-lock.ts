/**
 * Fair (first-in, first-out) GPU lock for heavy runs on a shared Mac. Same protocol as
 * ~/Developer/Denoiser/tools/gpu-lock.sh (TASKS rule 11 there): a ticket in $RT_GPU_QUEUE (default /tmp/rt-gpu.queue),
 * a lock directory $RT_GPU_LOCK (default /tmp/rt-gpu.lock) holding an `owner` file. Old shell callers and this one
 * exclude each other. Differences: a lock whose owner process is dead is taken over (the shell script's trap
 * cannot run after SIGKILL), and the ticket name is `<ms>-<pid>` like the script's.
 */

import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export interface GpuLockOptions {
  who: string;
  lockDir?: string;
  queueDir?: string;
  pollMs?: number;
  /** Called once when we have to wait, with the current owner text. */
  onWait?: (owner: string) => void;
  signal?: AbortSignal;
}

export function defaultLockDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.RT_GPU_LOCK ?? '/tmp/rt-gpu.lock';
}
export function defaultQueueDir(env: NodeJS.ProcessEnv = process.env): string {
  return env.RT_GPU_QUEUE ?? '/tmp/rt-gpu.queue';
}

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/** The pid in a ticket name (`<digits>-<pid>`), or null. */
export function ticketPid(name: string): number | null {
  const match = /-(\d+)$/.exec(name);
  return match ? Number(match[1]) : null;
}

/** The pid in an owner file (`<who> pid <n>`), or null. */
export function ownerPid(owner: string): number | null {
  const match = /pid (\d+)\s*$/.exec(owner);
  return match ? Number(match[1]) : null;
}

/** The oldest live ticket: tickets sort by their time prefix; dead ones are removed first. */
function oldestLiveTicket(queueDir: string): string | null {
  const names = readdirSync(queueDir).sort();
  for (const name of names) {
    const pid = ticketPid(name);
    if (pid !== null && !alive(pid)) {
      rmSync(join(queueDir, name), { force: true });
      continue;
    }
    return name;
  }
  return null;
}

let ticketSequence = 0;

/** Waits for our turn and takes the lock. Returns a release function (idempotent). */
export async function acquireGpuLock(options: GpuLockOptions): Promise<() => void> {
  const lockDir = options.lockDir ?? defaultLockDir();
  const queueDir = options.queueDir ?? defaultQueueDir();
  const pollMs = options.pollMs ?? 1000;
  mkdirSync(queueDir, { recursive: true });
  // 19 digits of nanoseconds like the shell script's `date +%s%N`, then a 3-digit per-process sequence where the
  // script pads 000: two acquires from one process in the same millisecond must not share a ticket file (they did,
  // and the first release deleted the second caller's ticket, so it waited forever).
  ticketSequence = (ticketSequence + 1) % 1000;
  const ticket = `${String(BigInt(Date.now()) * 1_000_000n).padStart(19, '0')}${String(ticketSequence).padStart(3, '0')}-${process.pid}`;
  const ticketPath = join(queueDir, ticket);
  writeFileSync(ticketPath, options.who);
  let held = false;
  const release = (): void => {
    rmSync(ticketPath, { force: true });
    if (held) {
      try {
        if (readFileSync(join(lockDir, 'owner'), 'utf8').trim() === `${options.who} pid ${process.pid}`) rmSync(lockDir, { recursive: true, force: true });
      } catch {
        // already gone
      }
      held = false;
    }
  };
  let announced = false;
  try {
    for (;;) {
      if (options.signal?.aborted) throw new Error('Interrupted while waiting for the GPU lock.');
      if (oldestLiveTicket(queueDir) === ticket) {
        try {
          mkdirSync(lockDir);
          writeFileSync(join(lockDir, 'owner'), `${options.who} pid ${process.pid}\n`);
          held = true;
          return release;
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
          // Locked. A dead owner's lock is stale (killed with SIGKILL, so its trap never ran): take it over.
          let owner = '';
          try {
            owner = readFileSync(join(lockDir, 'owner'), 'utf8').trim();
          } catch {
            // owner file not written yet
          }
          const pid = ownerPid(owner);
          if (pid !== null && !alive(pid)) {
            rmSync(lockDir, { recursive: true, force: true });
            continue;
          }
          if (!announced) {
            announced = true;
            options.onWait?.(owner || 'unknown owner');
          }
        }
      } else if (!announced) {
        announced = true;
        let owner = '';
        try {
          owner = readFileSync(join(lockDir, 'owner'), 'utf8').trim();
        } catch {
          // nobody holds it yet; an older ticket is about to
        }
        options.onWait?.(owner ? `${owner}; queued behind an older ticket` : 'queued behind an older ticket');
      }
      await new Promise((resolve) => setTimeout(resolve, pollMs));
    }
  } catch (error) {
    release();
    throw error;
  }
}
