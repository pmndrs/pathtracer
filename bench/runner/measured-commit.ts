/// <reference types="node" />
/**
 * Which commit a GPU measurement belongs to. Ported from ~/Developer/bvh/tests/gpu/measured-commit.ts (MIT, same
 * author). The runner records it when a run starts, so rows written later still name the code that was measured.
 */

import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '..', '..');

/**
 * Files whose changes can move a number: the packages, the bench source, runner and pages, and the lockfile.
 * bench/results is output and not counted; other agents' scene/metric/viewer work does not mark every row dirty
 * either, but anything under bench/src, bench/runner, bench/pages, bench/adapters and packages does.
 */
export const MEASURED_PATHS = ['packages', 'bench/src', 'bench/runner', 'bench/pages', 'bench/adapters', 'pnpm-lock.yaml'] as const;

/** HEAD's short hash, with "-dirty" when a measured file differs from HEAD (modified, staged or new and not ignored). */
export function gitCommit(): string {
  const git = (...args: string[]) => execFileSync('git', args, { cwd: REPOSITORY_ROOT, encoding: 'utf8' }).trim();
  const hash = git('rev-parse', '--short', 'HEAD');
  const changes = git('status', '--porcelain', '--untracked-files=all', '--', ...MEASURED_PATHS);
  return changes === '' ? hash : `${hash}-dirty`;
}
