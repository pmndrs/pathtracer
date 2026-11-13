/**
 * Ported from ~/Developer/bvh/tests/gpu/evaluate.ts (MIT, same author). Adapted for the pathtracer bench runner.
 */
/**
 * Pure decisions the runner makes: is this adapter the one we asked for, which
 * browser messages count as errors, and does a page pass. Kept free of Chrome
 * and the file system so it can be unit-tested.
 */

import { type AdapterSummary, type GpuChoice, type BenchReport } from './contract.ts';

/** One thing the browser told us while a page ran. */
export interface BrowserMessage {
    origin: 'log' | 'console' | 'exception';
    level: 'error' | 'warning' | 'info' | 'debug' | 'log';
    text: string;
    /** Source location (url:line), when the browser gave one. */
    where?: string;
}

/**
 * Dawn reports WGSL and other WebGPU validation problems as console/log
 * warnings in some Chrome versions and as errors in others, so match on the
 * text too. Any of these in a page's output fails it.
 */
const WEBGPU_ERROR_PATTERN =
    /WGSL|Tint|Dawn|\[Invalid |validation error|GPUValidationError|GPUOutOfMemoryError|GPUInternalError|device (was )?lost/i;

/** Messages that are not about the test, but Chrome prints anyway. */
const IGNORED_MESSAGE_PATTERN = /^\[vite\]|Download the React DevTools|GPU stall due to ReadPixels/i;

export function isFailureMessage(message: BrowserMessage): boolean {
    if (IGNORED_MESSAGE_PATTERN.test(message.text)) return false;
    if (message.origin === 'exception') return true;
    if (message.level === 'error') return true;
    return message.level === 'warning' && WEBGPU_ERROR_PATTERN.test(message.text);
}

export function failureMessages(messages: BrowserMessage[]): BrowserMessage[] {
    return messages.filter(isFailureMessage);
}

/**
 * Returns a sentence describing the mismatch, or null when the adapter is
 * acceptable. With no GPU requested, anything is acceptable.
 */
export function adapterMismatch(wanted: GpuChoice | null, adapter: AdapterSummary, allowSoftware = false): string | null {
    if (adapter.isFallbackAdapter && !allowSoftware) {
        return `Chrome gave a software fallback adapter (${adapter.vendor}/${adapter.architecture}), not a real GPU.`;
    }
    if (wanted === null) return null;
    if (adapter.vendor.toLowerCase() !== wanted) {
        return `Asked for --gpu ${wanted}, but Chrome gave ${adapter.vendor}/${adapter.architecture}.`;
    }
    return null;
}

export type PageOutcome =
    /** The page passed and nothing went wrong in the browser. */
    | 'pass'
    /** The page failed, and it was not supposed to. */
    | 'fail'
    /** The page is a self-test, it failed, and the runner noticed. */
    | 'caught-expected-failure'
    /** The page is a self-test and it did NOT fail: the runner is blind to something. */
    | 'missed-expected-failure'
    /** The page is a self-test and it failed, but not with the failure it was supposed to show. */
    | 'wrong-failure';

export interface PageVerdict {
    outcome: PageOutcome;
    /** Why the page failed (or why the failure was caught), one sentence each. */
    reasons: string[];
    /** True when this page should make the whole run exit 0. */
    acceptable: boolean;
}

export interface PageEvidence {
    /** Null when the page never wrote a report (timeout, crash, navigation error). */
    report: BenchReport | null;
    /** Set by the runner when it gave up or the page died. */
    runnerProblem: string | null;
    messages: BrowserMessage[];
    expectFailure: boolean;
    /** With expectFailure: some failure reason must contain (or match) this. */
    expectMessage: string | RegExp | null;
    /** The GPU the run asked for; the page's own adapter must be that one too. */
    wantedGpu: GpuChoice | null;
    /** The run opted in to a software (fallback) adapter; the page's own adapter may be one too. */
    allowSoftware?: boolean;
}

/** Collects every reason a page counts as failed; none means it passed. */
export function failureReasons(evidence: PageEvidence): string[] {
    const reasons: string[] = [];
    if (evidence.runnerProblem) reasons.push(evidence.runnerProblem);
    if (evidence.report === null) {
        if (!evidence.runnerProblem) reasons.push('The page finished without reporting a result.');
    } else {
        if (evidence.report.status === 'fail') {
            reasons.push(`The page reported failure: ${evidence.report.message ?? '(no message)'}`);
        }
        for (const error of evidence.report.deviceErrors) addUnlessKnown(reasons, 'GPU device error', error);
        // The page may have made its own adapter (a three.js renderer, say); it must be the requested GPU too.
        if (evidence.report.adapter) {
            const mismatch = adapterMismatch(evidence.wantedGpu, evidence.report.adapter, evidence.allowSoftware ?? false);
            if (mismatch) reasons.push(`The page's own adapter is wrong: ${mismatch}`);
        }
    }
    for (const message of failureMessages(evidence.messages)) {
        addUnlessKnown(reasons, `Browser ${message.origin} (${message.level})`, message.text);
    }
    return reasons;
}

/**
 * One WGSL error reaches us three ways (the page's report, the device's error event and
 * the browser log). Say it once: skip a message whose text an earlier reason already holds.
 */
function addUnlessKnown(reasons: string[], label: string, text: string): void {
    if (reasons.some((reason) => reason.includes(text.trim()))) return;
    reasons.push(`${label}: ${text}`);
}

export function matchesExpectation(reason: string, expected: string | RegExp): boolean {
    return typeof expected === 'string' ? reason.includes(expected) : expected.test(reason);
}

/** "/pattern/flags" is a regular expression; anything else is a plain substring. */
export function parseExpectedMessage(text: string): string | RegExp {
    const match = /^\/(.+)\/([a-z]*)$/s.exec(text);
    return match ? new RegExp(match[1] as string, match[2]) : text;
}

export function evaluatePage(evidence: PageEvidence): PageVerdict {
    const reasons = failureReasons(evidence);
    const failed = reasons.length > 0;
    if (evidence.expectFailure) {
        if (!failed) {
            return {
                outcome: 'missed-expected-failure',
                reasons: ['This page is supposed to fail, but the runner saw a clean pass.'],
                acceptable: false,
            };
        }
        const expected = evidence.expectMessage;
        if (expected !== null && !reasons.some((reason) => matchesExpectation(reason, expected))) {
            return {
                outcome: 'wrong-failure',
                reasons: [`The page failed, but not with the expected message (${expected}). What the runner saw:`, ...reasons],
                acceptable: false,
            };
        }
        return { outcome: 'caught-expected-failure', reasons, acceptable: true };
    }
    return failed ? { outcome: 'fail', reasons, acceptable: false } : { outcome: 'pass', reasons, acceptable: true };
}
