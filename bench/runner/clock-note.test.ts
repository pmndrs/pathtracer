import { describe, expect, it } from 'vitest';
import { judgeMacThermal, judgeTiming, parseClockLine, parsePmsetTherm, summarizeClockSamples, type ClockSample } from './clock-note.ts';

describe('mac thermal', () => {
  it('reads a quiet pmset output as no limit', () => {
    const quiet = 'Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded\n';
    expect(parsePmsetTherm(quiet)).toEqual({ cpuSpeedLimit: 100, warning: false });
  });

  it('reads a speed limit and a warning level', () => {
    expect(parsePmsetTherm('CPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 12\nCPU_Speed_Limit \t= 70\n').cpuSpeedLimit).toBe(70);
    expect(parsePmsetTherm('Thermal warning level set to 2').warning).toBe(true);
  });

  it('judges: invalid on any limited sample, valid otherwise, no verdict for none', () => {
    const ok = { cpuSpeedLimit: 100, warning: false };
    expect(judgeMacThermal([ok, ok]).kind).toBe('valid');
    expect(judgeMacThermal([ok, { cpuSpeedLimit: 80, warning: false }]).kind).toBe('invalid');
    expect(judgeMacThermal([{ cpuSpeedLimit: 100, warning: true }]).kind).toBe('invalid');
    expect(judgeMacThermal([]).kind).toBe('no-verdict');
  });
});

describe('nvidia clock rule (ported)', () => {
  const loaded = (reasons: number): ClockSample => ({ smMhz: 1500, memMhz: 6000, utilizationPercent: 99, temperatureC: 60, throttleReasons: reasons });

  it('parses the nvidia-smi line', () => {
    expect(parseClockLine('1500, 6000, 99, 61, 0x0000000000000020\n')).toEqual({ smMhz: 1500, memMhz: 6000, utilizationPercent: 99, temperatureC: 61, throttleReasons: 0x20 });
    expect(parseClockLine('garbage')).toBeNull();
  });

  it('is invalid on a hot start, no-verdict with few loaded samples, valid when cool and not throttled', () => {
    const samples = Array.from({ length: 10 }, () => loaded(0));
    const summary = summarizeClockSamples(samples);
    expect(summary).not.toBeNull();
    if (!summary) return;
    expect(judgeTiming(summary).kind).toBe('no-verdict'); // no start temperature
    expect(judgeTiming({ ...summary, start: { temperatureC: 50, coolBelowC: 70, waitedSeconds: 0, capped: false } }).kind).toBe('valid');
    expect(judgeTiming({ ...summary, start: { temperatureC: 75, coolBelowC: 70, waitedSeconds: 0, capped: false } }).kind).toBe('invalid');
    const thermal = summarizeClockSamples(Array.from({ length: 10 }, () => loaded(0x20)));
    expect(thermal && judgeTiming({ ...thermal, start: { temperatureC: 50, coolBelowC: 70, waitedSeconds: 0, capped: false } }).kind).toBe('invalid');
    const few = summarizeClockSamples([loaded(0)]);
    expect(few && judgeTiming({ ...few, start: { temperatureC: 50, coolBelowC: 70, waitedSeconds: 0, capped: false } }).kind).toBe('no-verdict');
  });
});
