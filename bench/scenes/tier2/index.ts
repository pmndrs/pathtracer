import type { SceneDesc } from '../schema.ts';
import { sponzaGlossy } from './sponza-glossy.ts';
import { sponzaLamps, sponzaLampsVisible } from './sponza-lamps.ts';
import { sponzaWhite, sponzaWhiteVknrc } from './sponza-white.ts';

export { sponzaGlossy, sponzaLamps, sponzaLampsVisible, sponzaWhite, sponzaWhiteVknrc };

export const tier2Scenes: readonly SceneDesc[] = [sponzaWhite, sponzaWhiteVknrc, sponzaGlossy, sponzaLamps, sponzaLampsVisible];
