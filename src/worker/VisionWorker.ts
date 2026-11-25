/// <reference lib="webworker" />
import { createWorkerHandler } from "./VisionWorkerHost";
import type { WorkerRequest } from "./protocol";

/** Web Worker entry point (spec §39): the handler lives in VisionWorkerHost. */
const ctx = self as unknown as DedicatedWorkerGlobalScope;
const handle = createWorkerHandler((msg, transfer = []) => ctx.postMessage(msg, transfer));
ctx.onmessage = (ev: MessageEvent<WorkerRequest>) => handle(ev.data);
