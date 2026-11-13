import { describe, expect, it } from 'vitest';
import { parseUploadQuery, safeName } from './server.ts';

describe('upload query', () => {
  it('accepts a good query and defaults to 4 channels', () => {
    expect(parseUploadQuery(new URLSearchParams('name=beauty&w=4&h=2&c=3'))).toEqual({ name: 'beauty', width: 4, height: 2, channels: 3 });
    expect(parseUploadQuery(new URLSearchParams('name=a.b-c_1&w=1&h=1'))).toMatchObject({ channels: 4 });
  });

  it('rejects path tricks, bad sizes and bad channels', () => {
    expect(typeof parseUploadQuery(new URLSearchParams('name=../x&w=1&h=1'))).toBe('string');
    expect(typeof parseUploadQuery(new URLSearchParams('w=1&h=1'))).toBe('string');
    expect(typeof parseUploadQuery(new URLSearchParams('name=a&w=0&h=1'))).toBe('string');
    expect(typeof parseUploadQuery(new URLSearchParams('name=a&w=1.5&h=1'))).toBe('string');
    expect(typeof parseUploadQuery(new URLSearchParams('name=a&w=1&h=1&c=2'))).toBe('string');
  });

  it('restricts names to plain file stems', () => {
    expect(safeName('ok-1.2_x')).toBe('ok-1.2_x');
    expect(safeName('.hidden')).toBeNull();
    expect(safeName('a/b')).toBeNull();
    expect(safeName(null)).toBeNull();
  });
});
