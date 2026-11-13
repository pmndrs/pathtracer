import { mkdtempSync, readdirSync, rmSync, existsSync, writeFileSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { acquireGpuLock, ownerPid, ticketPid } from './gpu-lock.ts';

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});
function sandbox(): { lockDir: string; queueDir: string } {
  const root = mkdtempSync(join(tmpdir(), 'pt-lock-'));
  dirs.push(root);
  return { lockDir: join(root, 'lock'), queueDir: join(root, 'queue') };
}

describe('gpu lock', () => {
  it('parses ticket and owner pids', () => {
    expect(ticketPid('1789000000000000000000-4242')).toBe(4242);
    expect(ticketPid('nonsense')).toBeNull();
    expect(ownerPid('agent task pid 99\n')).toBe(99);
    expect(ownerPid('')).toBeNull();
  });

  it('takes the lock, removes its ticket on release, and frees the lock', async () => {
    const dirsForLock = sandbox();
    const release = await acquireGpuLock({ who: 'test one', ...dirsForLock, pollMs: 10 });
    expect(existsSync(dirsForLock.lockDir)).toBe(true);
    release();
    release(); // idempotent
    expect(existsSync(dirsForLock.lockDir)).toBe(false);
    expect(readdirSync(dirsForLock.queueDir)).toEqual([]);
  });

  it('makes a second caller wait until the first releases', async () => {
    const dirsForLock = sandbox();
    const first = await acquireGpuLock({ who: 'first', ...dirsForLock, pollMs: 10 });
    let waited = '';
    let secondHeld = false;
    const second = acquireGpuLock({ who: 'second', ...dirsForLock, pollMs: 10, onWait: (owner) => (waited = owner) }).then((release) => {
      secondHeld = true;
      return release;
    });
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(secondHeld).toBe(false);
    expect(waited).toContain('first');
    first();
    (await second)();
    expect(secondHeld).toBe(true);
  });

  it('takes over a lock whose owner process is gone', async () => {
    const dirsForLock = sandbox();
    mkdirSync(dirsForLock.lockDir, { recursive: true });
    writeFileSync(join(dirsForLock.lockDir, 'owner'), 'ghost pid 2147483646\n'); // no such process
    const release = await acquireGpuLock({ who: 'live', ...dirsForLock, pollMs: 10 });
    release();
  });
});
