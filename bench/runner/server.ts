/**
 * Serves the repo root with vite's dev server (so a page can import TypeScript and the workspace packages with no
 * build step, as in ~/Developer/bvh/tests/gpu/server.ts, MIT, same author) and adds the image upload endpoint:
 *
 *   POST /__bench/upload?name=<n>&w=<w>&h=<h>&c=<1|3|4>[&token=<t>]   body: raw float32 (row 0 = top)
 *
 * It writes `bench/results/incoming/<run-id>/<slot>/<name>.pfm` and `<name>.json` (header). Big images never go
 * through CDP `Runtime.evaluate`: a 30 MB Float32Array as JSON is slow and blows the message limit.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { relative, resolve } from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { createServer, type ViteDevServer } from 'vite';
import { UPLOAD_PATH, type UploadRecord } from './contract.ts';
import { bytesToFloats, countNonFinite, encodePfm, expectedFloats } from './pfm.ts';

export const REPOSITORY_ROOT = resolve(import.meta.dirname, '..', '..');
export const RESULTS_DIR = resolve(REPOSITORY_ROOT, 'bench', 'results');
export const INCOMING_DIR = resolve(RESULTS_DIR, 'incoming');

/** A 8192 x 8192 RGBA float image is 1 GiB; nothing in the bench is bigger than a few thousand pixels a side. */
export const MAX_UPLOAD_BYTES = 512 * 1024 * 1024;

export interface BenchServer {
  /** For example http://127.0.0.1:5291 */
  origin: string;
  /** Uploads go to `<incoming>/<runId>/<slot>/`. Call before each page run. */
  setSlot(slot: string): void;
  uploads: UploadRecord[];
  close(): Promise<void>;
}

/** Image names become file names: keep them plain. */
export function safeName(name: string | null): string | null {
  return name !== null && /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/.test(name) ? name : null;
}

export interface UploadParams {
  name: string;
  width: number;
  height: number;
  channels: 1 | 3 | 4;
}

/** Validates the query of an upload; returns a message for the 400 response on failure. */
export function parseUploadQuery(query: URLSearchParams): UploadParams | string {
  const name = safeName(query.get('name'));
  if (!name) return 'name must match [A-Za-z0-9][A-Za-z0-9._-]* (max 100 characters).';
  const width = Number(query.get('w'));
  const height = Number(query.get('h'));
  const channels = Number(query.get('c') ?? '4');
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) return 'w and h must be positive integers.';
  if (channels !== 1 && channels !== 3 && channels !== 4) return 'c must be 1, 3 or 4.';
  return { name, width, height, channels };
}

function readBody(request: IncomingMessage, limit: number): Promise<Buffer> {
  return new Promise((resolveBody, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    request.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error(`Upload is larger than ${limit} bytes.`));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => resolveBody(Buffer.concat(chunks)));
    request.on('error', reject);
  });
}

function reply(response: ServerResponse, status: number, body: unknown): void {
  response.statusCode = status;
  response.setHeader('Content-Type', 'application/json');
  response.end(JSON.stringify(body));
}

export async function startBenchServer(runId: string): Promise<BenchServer> {
  let slot = 'run';
  const uploads: UploadRecord[] = [];

  const server: ViteDevServer = await createServer({
    root: REPOSITORY_ROOT,
    // Do not pick up a vite config from the repo: the bench site is self-contained.
    configFile: false,
    logLevel: 'warn',
    appType: 'mpa',
    // Pages have no dependencies that need pre-bundling; skipping the scan keeps startup fast and quiet.
    optimizeDeps: { noDiscovery: true, include: [] },
    server: {
      host: '127.0.0.1',
      port: 5291,
      strictPort: false,
      // The runner reloads pages itself; hot reload and file watching would only add noise (and watch research/).
      hmr: false,
      watch: null,
    },
    plugins: [
      {
        name: 'bench-upload',
        configureServer(devServer) {
          devServer.middlewares.use(UPLOAD_PATH, (request, response) => {
            void (async () => {
              if (request.method !== 'POST') return reply(response, 405, { error: 'POST only' });
              try {
                const query = new URL(request.url ?? '/', 'http://localhost').searchParams;
                const params = parseUploadQuery(query);
                if (typeof params === 'string') return reply(response, 400, { error: params });
                const body = await readBody(request, MAX_UPLOAD_BYTES);
                const want = expectedFloats(params.width, params.height, params.channels) * 4;
                if (body.length !== want) {
                  return reply(response, 400, { error: `Body is ${body.length} bytes, expected ${want} for ${params.width}x${params.height}x${params.channels} float32.` });
                }
                const data = bytesToFloats(body);
                const dir = resolve(INCOMING_DIR, runId, slot);
                mkdirSync(dir, { recursive: true });
                const pfmPath = resolve(dir, `${params.name}.pfm`);
                writeFileSync(pfmPath, encodePfm({ ...params, data }));
                const record: UploadRecord = {
                  ...params,
                  token: query.get('token'),
                  slot,
                  pfm: relative(REPOSITORY_ROOT, pfmPath),
                  bytes: body.length,
                  nonFinite: countNonFinite(data),
                  receivedAt: new Date().toISOString(),
                };
                writeFileSync(resolve(dir, `${params.name}.json`), `${JSON.stringify(record, null, 2)}\n`);
                uploads.push(record);
                reply(response, 200, record);
              } catch (error) {
                reply(response, 500, { error: error instanceof Error ? error.message : String(error) });
              }
            })();
          });
        },
      },
      {
        // Chrome asks for /favicon.ico on every page and logs a 404 as a console error, which the runner would count as a failure.
        name: 'no-favicon-404',
        configureServer(devServer) {
          devServer.middlewares.use('/favicon.ico', (_request, response) => {
            response.statusCode = 204;
            response.end();
          });
        },
      },
    ],
  });
  await server.listen();
  const origin = server.resolvedUrls?.local[0]?.replace(/\/$/, '');
  if (!origin) {
    await server.close();
    throw new Error('The vite bench server started but reported no local address.');
  }
  return {
    origin,
    setSlot: (next) => {
      slot = next;
    },
    uploads,
    close: () => server.close(),
  };
}
