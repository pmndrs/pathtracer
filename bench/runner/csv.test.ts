import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { appendCsv, canonicalJson, configHash, csvField, csvLine, csvText, CSV_HEADER, type CsvRow } from './csv.ts';

const row: CsvRow = {
  date: '2026-11-13',
  commit: 'abc1234',
  device: 'Apple M5 Pro',
  browser: 'Chrome/157 headless',
  scene: 'cornell',
  renderer: 'ours',
  config_hash: 'deadbeef',
  metric: 'gpu_ms_total',
  value: 1.25,
  valid: 1,
  run: 'median',
  note: '',
};

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe('csv', () => {
  it('has the documented header', () => {
    expect(CSV_HEADER).toBe('date,commit,device,browser,scene,renderer,config_hash,metric,value,valid,run,note');
  });

  it('quotes only when needed', () => {
    expect(csvField('plain')).toBe('plain');
    expect(csvField('a,b')).toBe('"a,b"');
    expect(csvField('say "hi"')).toBe('"say ""hi"""');
    expect(csvField('two\nlines')).toBe('"two\nlines"');
    expect(csvField(NaN)).toBe('');
    expect(csvField(0.5)).toBe('0.5');
  });

  it('writes a row in column order', () => {
    expect(csvLine(row)).toBe('2026-11-13,abc1234,Apple M5 Pro,Chrome/157 headless,cornell,ours,deadbeef,gpu_ms_total,1.25,1,median,');
    expect(csvLine({ ...row, note: 'a, b' }).endsWith(',"a, b"')).toBe(true);
  });

  it('adds the header only for a new file', () => {
    expect(csvText([row], true).split('\n')[0]).toBe(CSV_HEADER);
    expect(csvText([row], false)).toBe(`${csvLine(row)}\n`);
  });

  it('appends, creating the file and parent folders, never rewriting', () => {
    const dir = mkdtempSync(join(tmpdir(), 'pt-csv-'));
    dirs.push(dir);
    const path = join(dir, 'nested', 'suite.csv');
    expect(appendCsv(path, [row])).toBe(1);
    expect(appendCsv(path, [{ ...row, value: 2 }])).toBe(1);
    expect(appendCsv(path, [])).toBe(0);
    const lines = readFileSync(path, 'utf8').trimEnd().split('\n');
    expect(lines).toHaveLength(3);
    expect(lines[0]).toBe(CSV_HEADER);
    expect(lines[1]).toContain(',1.25,');
    expect(lines[2]).toContain(',2,');
  });

  it('hashes configs independent of key order', () => {
    expect(canonicalJson({ b: 1, a: { d: [1, 2], c: undefined } })).toBe('{"a":{"d":[1,2]},"b":1}');
    expect(configHash({ spp: 64, scene: 'x' })).toBe(configHash({ scene: 'x', spp: 64 }));
    expect(configHash({ spp: 64 })).not.toBe(configHash({ spp: 65 }));
    expect(configHash({})).toMatch(/^[0-9a-f]{8}$/);
  });
});
