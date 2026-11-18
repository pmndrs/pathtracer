import type { SceneDesc } from '../schema.ts';
import { sponzaGlossy } from './sponza-glossy.ts';
import { sponzaWhite, sponzaWhiteVknrc } from './sponza-white.ts';

export { sponzaGlossy, sponzaWhite, sponzaWhiteVknrc };

export const tier2Scenes: readonly SceneDesc[] = [sponzaWhite, sponzaWhiteVknrc, sponzaGlossy];
