import { TgptAdapter, type TgptOptions } from '../tgpt-common/index.ts';

/** three-gpu-pathtracer WebGPU, megakernel scheduler (`useMegakernel(true)`). */
export const createTgptMegakernel = (options?: TgptOptions) => new TgptAdapter('megakernel', options);
