/**
 * Ported from ~/Developer/bvh/tests/gpu/cdp.ts (MIT, same author). Adapted for the pathtracer bench runner.
 */
/**
 * A small Chrome DevTools Protocol client over Node's built-in WebSocket.
 * Same idea as the upscaler's `cdp-client.mjs`, with event subscriptions
 * instead of a growing event array.
 */

/** The DevTools socket of a local Chrome opens at once; this only guards against a stuck connection. */
const CONNECT_TIMEOUT_MS = 10_000;

/** Default for one DevTools call. Page.navigate and Runtime.evaluate on a busy GPU can take a few seconds. */
const DEFAULT_CALL_TIMEOUT_MS = 30_000;

type EventHandler = (params: Record<string, unknown>) => void;

interface PendingRequest {
    method: string;
    resolve: (result: Record<string, unknown>) => void;
    reject: (error: Error) => void;
}

export class CdpClient {
    private nextId = 0;
    private readonly pending = new Map<number, PendingRequest>();
    private readonly handlers = new Map<string, EventHandler[]>();

    private readonly socket: WebSocket;

    private constructor(socket: WebSocket) {
        // No constructor parameter properties: Node's type stripping rejects them.
        this.socket = socket;
        socket.addEventListener('message', (event) => this.handleMessage(String(event.data)));
        socket.addEventListener('close', () => this.rejectAll(new Error('DevTools connection closed')));
    }

    static async connect(webSocketUrl: string): Promise<CdpClient> {
        const socket = new WebSocket(webSocketUrl);
        await new Promise<void>((resolve, reject) => {
            const timer = setTimeout(() => {
                socket.close();
                reject(new Error(`Connecting to ${webSocketUrl} took longer than ${CONNECT_TIMEOUT_MS} ms`));
            }, CONNECT_TIMEOUT_MS);
            socket.addEventListener('open', () => {
                clearTimeout(timer);
                resolve();
            }, { once: true });
            socket.addEventListener('error', () => {
                clearTimeout(timer);
                reject(new Error(`Could not connect to ${webSocketUrl}`));
            }, { once: true });
        });
        return new CdpClient(socket);
    }

    send<Result = Record<string, unknown>>(
        method: string,
        params: Record<string, unknown> = {},
        timeoutMs = DEFAULT_CALL_TIMEOUT_MS,
    ): Promise<Result> {
        const id = ++this.nextId;
        return new Promise<Result>((resolve, reject) => {
            const timer = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error(`DevTools call ${method} timed out after ${timeoutMs} ms`));
            }, timeoutMs);
            this.pending.set(id, {
                method,
                resolve: (result) => {
                    clearTimeout(timer);
                    resolve(result as Result);
                },
                reject: (error) => {
                    clearTimeout(timer);
                    reject(error);
                },
            });
            this.socket.send(JSON.stringify({ id, method, params }));
        });
    }

    on(method: string, handler: EventHandler): void {
        const list = this.handlers.get(method) ?? [];
        list.push(handler);
        this.handlers.set(method, list);
    }

    /** Runs `expression` in the page and returns its value (awaiting promises). */
    async evaluate<Value = unknown>(expression: string): Promise<Value> {
        const response = await this.send<{
            result: { value?: Value };
            exceptionDetails?: { text: string; exception?: { description?: string } };
        }>('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
        if (response.exceptionDetails) {
            throw new Error(response.exceptionDetails.exception?.description ?? response.exceptionDetails.text);
        }
        return response.result.value as Value;
    }

    close(): void {
        this.socket.close();
    }

    private handleMessage(raw: string): void {
        const message = JSON.parse(raw) as {
            id?: number;
            method?: string;
            params?: Record<string, unknown>;
            result?: Record<string, unknown>;
            error?: unknown;
        };
        if (message.id !== undefined) {
            const request = this.pending.get(message.id);
            this.pending.delete(message.id);
            if (!request) return;
            if (message.error) {
                request.reject(new Error(`DevTools call ${request.method} failed: ${JSON.stringify(message.error)}`));
            } else {
                request.resolve(message.result ?? {});
            }
            return;
        }
        if (message.method) {
            for (const handler of this.handlers.get(message.method) ?? []) handler(message.params ?? {});
        }
    }

    private rejectAll(error: Error): void {
        for (const request of this.pending.values()) request.reject(error);
        this.pending.clear();
    }
}
