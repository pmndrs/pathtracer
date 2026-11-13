import { describe, expect, it } from 'vitest';
import { extractPageArgs, parseRunnerArgs, UsageError } from './args.ts';

describe('parseRunnerArgs', () => {
  it('has sane defaults', () => {
    const options = parseRunnerArgs([], {});
    expect(options.page).toBeNull();
    expect(options.gpu).toBeNull();
    expect(options.headed).toBe(false);
    expect(options.repeat).toBe(1);
    expect(options.args).toEqual({});
    expect(options.ab).toBeNull();
    expect(options.allowSoftware).toBe(false);
  });

  it('reads page, gpu, headed and repeat', () => {
    const options = parseRunnerArgs(['--page', 'smoke', '--gpu', 'apple', '--headed', '--repeat', '3'], {});
    expect(options).toMatchObject({ page: 'smoke', gpu: 'apple', headed: true, repeat: 3 });
    expect(parseRunnerArgs(['--gpu=amd'], {}).gpu).toBe('amd');
  });

  it('rejects an unknown gpu, a bad repeat and a bad page name', () => {
    expect(() => parseRunnerArgs(['--gpu', 'voodoo'], {})).toThrow(UsageError);
    expect(() => parseRunnerArgs(['--repeat', '0'], {})).toThrow(UsageError);
    expect(() => parseRunnerArgs(['--repeat', '2.5'], {})).toThrow(UsageError);
    expect(() => parseRunnerArgs(['--page', '../etc/passwd'], {})).toThrow(UsageError);
    expect(() => parseRunnerArgs(['--nope'], {})).toThrow(UsageError);
  });

  it('collects --args k=v words up to the next flag', () => {
    const options = parseRunnerArgs(['--page', 'p', '--args', 'spp=64', 'scene=cornell', 'eq=a=b', '--repeat', '2'], {});
    expect(options.args).toEqual({ spp: '64', scene: 'cornell', eq: 'a=b' });
    expect(options.repeat).toBe(2);
  });

  it('rejects an --args word without =', () => {
    expect(() => parseRunnerArgs(['--args', 'oops'], {})).toThrow(UsageError);
  });

  it('parses --ab key=A,B', () => {
    expect(parseRunnerArgs(['--ab', 'sched=wavefront,megakernel'], {}).ab).toEqual({ key: 'sched', values: ['wavefront', 'megakernel'] });
    expect(() => parseRunnerArgs(['--ab', 'sched=one'], {})).toThrow(UsageError);
    expect(() => parseRunnerArgs(['--ab', 'a=1,2,3'], {})).toThrow(UsageError);
    expect(() => parseRunnerArgs(['--ab', 'k=1,2', '--args', 'k=3'], {})).toThrow(UsageError);
  });

  it('allows a software adapter only by flag or PT_ALLOW_SOFTWARE_ADAPTER=1', () => {
    expect(parseRunnerArgs(['--allow-software'], {}).allowSoftware).toBe(true);
    expect(parseRunnerArgs([], { PT_ALLOW_SOFTWARE_ADAPTER: '1' }).allowSoftware).toBe(true);
    expect(parseRunnerArgs([], { PT_ALLOW_SOFTWARE_ADAPTER: '0' }).allowSoftware).toBe(false);
  });
});

describe('extractPageArgs', () => {
  it('leaves the other words alone and keeps the first = as the separator', () => {
    const { rest, args } = extractPageArgs(['--page', 'x', '--args', 'a=1', 'b=', '--headed']);
    expect(rest).toEqual(['--page', 'x', '--headed']);
    expect(args).toEqual({ a: '1', b: '' });
  });
});
