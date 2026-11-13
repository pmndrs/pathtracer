import { TgptAdapter, type TgptOptions } from '../tgpt-common/index.ts';

/** three-gpu-pathtracer WebGPU, wavefront scheduler (the library default). */
export const createTgptWavefront = (options?: TgptOptions) => new TgptAdapter('wavefront', options);
