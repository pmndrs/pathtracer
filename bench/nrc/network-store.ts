/**
 * Saved network per scene (N11.9): the trained weights of the radiance cache, kept in the browser (IndexedDB) so a scene the visitor has
 * seen before starts warm instead of showing plain path tracing through the warm-up gate.
 *
 * An entry is found by its key (scene description hash + everything that changes what the network means) and carries the lighting it was
 * trained under: a network trained for other lighting is stale and is not loaded. One entry per key: a save replaces it.
 */

/** Bump when the encoding, the network inputs or the training target change meaning: every older entry is then ignored. */
export const NETWORK_STORE_VERSION = 1;

export interface SavedNetwork {
  key: string;
  /** The lighting the network learned (the controls' state as a string); a different value on load means stale. */
  lighting: string;
  weights: Float32Array;
  trained: { frames: number; steps: number };
  savedAt: number;
}

export interface NetworkKeyParts {
  /** sha256 of the scene description, as `export-scenes.ts` computes it. */
  sceneHash: string;
  /** Network shape name, cache model, factorisation, material inputs, direction input: anything that changes the weights' meaning. */
  network: string;
  /** Path length the cache's training target is rendered at. */
  maxDepth: number;
}

export function networkKey(parts: NetworkKeyParts): string {
  return `v${NETWORK_STORE_VERSION}|${parts.sceneHash}|${parts.network}|d${parts.maxDepth}`;
}

/** A saved entry is usable for this session: same key, same lighting, the right number of finite weights. */
export function usableNetwork(entry: SavedNetwork | undefined, key: string, lighting: string, weightCount: number): entry is SavedNetwork {
  if (!entry || entry.key !== key || entry.lighting !== lighting) return false;
  if (!(entry.weights instanceof Float32Array) || entry.weights.length !== weightCount) return false;
  return entry.weights.every(Number.isFinite) && entry.trained.steps > 0;
}

export async function sceneHash(scene: unknown): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(scene)));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

const DB = 'pathtracer-nrc';
const STORE = 'networks';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: 'key' });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function run<T>(mode: IDBTransactionMode, work: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const request = work(db.transaction(STORE, mode).objectStore(STORE));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  } finally {
    db.close();
  }
}

// Storage can be missing or blocked (private window, cleared site data): every call fails soft, the demo then simply starts cold.
export async function loadNetwork(key: string): Promise<SavedNetwork | undefined> {
  try {
    return await run<SavedNetwork | undefined>('readonly', (store) => store.get(key));
  } catch {
    return undefined;
  }
}

export async function saveNetwork(entry: SavedNetwork): Promise<boolean> {
  try {
    await run('readwrite', (store) => store.put(entry));
    return true;
  } catch {
    return false;
  }
}

export async function forgetNetwork(key: string): Promise<void> {
  try {
    await run('readwrite', (store) => store.delete(key));
  } catch {
    // nothing to forget
  }
}
