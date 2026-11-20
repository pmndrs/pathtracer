// The local bench runs at the origin root. The Pages build injects its configured site path.
declare const __BENCH_SITE_BASE__: string | undefined;
export const pagesBase = typeof __BENCH_SITE_BASE__ === 'string' ? __BENCH_SITE_BASE__ : undefined;
export const benchAssets = (pagesBase ?? '/') + 'bench/assets';
export const blueNoiseAssets = (pagesBase ?? '/') + 'packages/raytracer/assets/bluenoise';
