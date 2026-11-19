import type { Image } from '../src/adapter.ts';

// Pixel boxes in the 320 × 180 blue-card reference. Scale them with the film for the same camera and aspect ratio.
type Box = readonly [number, number, number, number];
const wall: Box = [228, 16, 302, 107];
const edges: Box[] = [[110, 120, 212, 125], [218, 16, 223, 111]];
const reflection: Box = [86, 131, 234, 168];

export function cardMetrics(image: Image, ref: Image): Record<string, number> {
  function region(box: Box, fn: (a: number, b: number, x: number, y: number) => void) {
    const left = Math.floor(box[0] * image.width / 320), right = Math.floor(box[2] * image.width / 320);
    const top = Math.floor(box[1] * image.height / 180), bottom = Math.floor(box[3] * image.height / 180);
    for (let y = top; y < bottom; y++) {
      for (let x = left; x < right; x++) {
        const pixel = y * image.width + x;
        fn(pixel * image.channels, pixel * ref.channels, x - left, y - top);
      }
    }
  }
  function relativeRms(boxes: Box[]) {
    let error = 0, energy = 0;
    for (const box of boxes) region(box, (a, b) => {
      for (let c = 0; c < 3; c++) {
        error += (image.data[a + c]! - ref.data[b + c]!) ** 2;
        energy += ref.data[b + c]! ** 2;
      }
    });
    return Math.sqrt(error / energy);
  }
  function blueContrast(im: Image) {
    const blueMinusRed = (box: Box) => {
      let sum = 0, count = 0;
      region(box, (a, b) => { const i = im === image ? a : b; sum += im.data[i + 2]! - im.data[i]!; count++; });
      return sum / count;
    };
    return blueMinusRed([118, 131, 202, 158]) - (blueMinusRed([58, 131, 90, 158]) + blueMinusRed([230, 131, 262, 158])) / 2;
  }
  // Adjacent-column and adjacent-row changes of the mean residual emphasize coherent stripes over pixel noise.
  const cols: number[] = [], rows: number[] = [];
  const luma = (data: Float32Array, i: number) => 0.2126 * data[i]! + 0.7152 * data[i + 1]! + 0.0722 * data[i + 2]!;
  let referenceLight = 0, grainEnergy = 0, grainPairs = 0;
  region(wall, (a, b, x, y) => {
    const light = luma(ref.data, b), error = luma(image.data, a) - light;
    referenceLight += light;
    // Adjacent residual differences suppress slowly varying bias. Division by two gives per-pixel variance for white noise.
    if (x > 0) {
      const previous = luma(image.data, a - image.channels) - luma(ref.data, b - ref.channels);
      grainEnergy += (error - previous) ** 2;
      grainPairs++;
    }
    if (y > 0) {
      const previous = luma(image.data, a - image.width * image.channels) - luma(ref.data, b - ref.width * ref.channels);
      grainEnergy += (error - previous) ** 2;
      grainPairs++;
    }
    cols[x] = (cols[x] ?? 0) + error;
    rows[y] = (rows[y] ?? 0) + error;
  });
  const differences = (values: number[], n: number) => {
    let sum = 0;
    for (let i = 1; i < values.length; i++) sum += ((values[i]! - values[i - 1]!) / n) ** 2;
    return sum / Math.max(1, values.length - 1);
  };
  return {
    wall_grain_rms: Math.sqrt(grainEnergy / (2 * Math.max(1, grainPairs))) * rows.length * cols.length / referenceLight,
    wall_stripe_rms: Math.sqrt((differences(cols, rows.length) + differences(rows, cols.length)) / 2) * rows.length * cols.length / referenceLight,
    wall_rrms: relativeRms([wall]), edge_rrms: relativeRms(edges), reflection_rrms: relativeRms([reflection]),
    reflection_contrast_ratio: blueContrast(image) / blueContrast(ref),
  };
}
