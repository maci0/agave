/**
 * Runtime smoke test for the standalone browser WASM shell.
 *
 * `web/shell.tsx` builds a `Shell` tree that only exists once it is mounted, so
 * this drives the real entry point against a stubbed `AgaveEngine` in a DOM. The
 * property it pins is engine identity: the shell owns an instantiated WASM
 * module and a loaded model, and a state change re-renders the tree. If the
 * component body constructs the engine instead of holding one per mount, every
 * render discards the module and the composer falls back to "Load a GGUF model
 * first." even though a model is loaded, and the reader never sees a reply.
 *
 * Run: bun test src/web (also wired into scripts/lint-web.sh).
 */

import { afterAll, expect, test } from 'bun:test';
import { GlobalRegistrator } from '@happy-dom/global-registrator';

GlobalRegistrator.register({ url: 'http://127.0.0.1:49453' });

/** The bytes `use-model-loader` requires before it accepts a model. */
const GGUF = [0x47, 0x47, 0x55, 0x46, 1, 2, 3, 4];

/** A model URL the loader accepts (`isHttpUrl`) and then fetches. */
const MODEL_URL = 'https://example.com/model.gguf';

/** Every engine the shell built, with the prompts that reached it. An engine
 *  that answered a prompt is the one holding the model. */
const engines: Array<{ hasModel: boolean; downloads: number; prompted: Array<string> }> = [];

/** Resolves when the transcript holds `text`. Keeps the wait out of the test
 *  body, which the lint rules read as a conditional in a test. */
const transcriptHolds = async (text: string): Promise<boolean> => {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (document.querySelector('#chat')?.textContent.includes(text) === true) { return true; }
    // oxlint-disable-next-line promise/avoid-new -- a timer is the only clock the test needs
    await new Promise((resolve) => { setTimeout(resolve, 10); });
  }
  return false;
};

/** Stands in for `web/agave.ts`'s `AgaveEngine`, which needs agave.wasm. It
 *  keeps the one property the shell depends on: `hasModel` is false until
 *  `loadModel()` and true after, so an engine rebuilt on re-render is
 *  detectable without a WebAssembly instance. `generate()` returns the shape
 *  the real one does while the wasm32 forward pass is blocked: a tokenization
 *  report, not model output. Every method is async because the shell awaits
 *  it; the bodies resolve on the microtask queue. */
class StubEngine {
  public ready = false;
  public hasModel = false;
  public initMessage = '';
  public downloads = 0;
  public readonly prompted: Array<string> = [];
  /** True once `init()` has run to completion, whether it was gated or not. */
  public initialized = false;
  /** True when `fetchModel()` ran before `init()` finished. */
  public downloadStartedBeforeInit = false;
  /** The latch a gated `init()` parks on; null until that call arrives. */
  public releaseInit: (() => void) | null = null;
  /** True holds `init()` open until the test calls `releaseInit`, so the
   *  model's download can be observed against an unfinished WASM handshake. */
  public gated = false;

  public constructor() { engines.push(this); }

  public async init(): Promise<void> {
    await (this.gated
      // oxlint-disable-next-line promise/avoid-new -- a deferred latch is the point; no library promise defers a later call
      ? new Promise<void>((resolve) => { this.releaseInit = resolve; })
      : Promise.resolve());
    this.initialized = true;
    this.ready = true;
  }

  public async fetchModel(): Promise<ArrayBuffer> {
    await Promise.resolve();
    this.downloadStartedBeforeInit = !this.initialized;
    this.downloads += 1;
    return new Uint8Array(GGUF).buffer;
  }

  public async loadModel(): Promise<void> {
    await Promise.resolve();
    this.hasModel = true;
    this.initMessage = 'Loaded: stub-model';
  }

  public async generate(prompt: string): Promise<string> {
    await Promise.resolve();
    if (!this.hasModel) { throw new Error('Engine not initialized'); }
    this.prompted.push(prompt);
    return '[stub-model] Tokenized 1 tokens from prompt.';
  }
}

/** The value, or a thrown error naming what was missing, so a test body reads
 *  as straight-line assertions. */
const present = <T,>(value: T | null | undefined, what: string): T => {
  if (value === null || value === undefined) { throw new Error(`${what} missing`); }
  return value;
};

/** Let Preact flush its render, its effects and the load promises. */
const settle = (ms = 10): Promise<void> =>
  // oxlint-disable-next-line promise/avoid-new -- a timer is the only clock the test needs
  new Promise((resolve) => { setTimeout(resolve, ms); });

/* The framework installs its own value setter on the node, so the native
   descriptor is the only way to write a value the controlled input observes. */
const typeInto = (input: HTMLInputElement, text: string): void => {
  // oxlint-disable-next-line typescript-eslint/unbound-method -- `.call` below binds the node, which is the point
  const nativeSetter = present(Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set, 'input value setter');
  // SAFETY: the descriptor is the DOM value setter of an input.
  nativeSetter.call(input, text);
  input.dispatchEvent(new Event('input', { bubbles: true }));
};

const clickButtonLabelled = (text: string): void => {
  const button = [...document.querySelectorAll('button')].find((node) => node.textContent.includes(text));
  present(button, `button reading "${text}"`).click();
};

const stubEngineHost = (): void => {
  document.body.innerHTML = '<div id="root"></div>';
  engines.length = 0;
  Object.assign(globalThis, { AgaveEngine: StubEngine });
  // SAFETY: the stub answers only the model fetch the shell makes on this path.
  globalThis.fetch = function (_input: RequestInfo | URL): Promise<Response> {
    return Promise.resolve(new Response(new Uint8Array(GGUF)));
  } as typeof fetch;
};

afterAll(async () => {
  /* Preact's scheduler drains through `window`, so let the pending work from
     the mounted tree land before the registrator takes the DOM globals away. */
  for (let index = 0; index < 40; index += 1) { await settle(5); }
  await GlobalRegistrator.unregister();
});

/** The transcript, as one line of text, for a copy assertion. */
const transcriptText = (): string => present(document.querySelector('#chat'), 'chat log').textContent;

/** Mount the shell against whatever `globalThis.AgaveEngine` currently is. The
 *  cache-busting query re-runs the module's top-level `createRoot`, so each
 *  test owns its own tree instead of sharing the first one. */
let mountSeq = 0;
const mountShell = async (): Promise<void> => {
  mountSeq += 1;
  document.body.innerHTML = '<div id="root"></div>';
  await import(`../../web/shell?mount=${String(mountSeq)}`);
  await settle(30);
};

test('the model download overlaps the WASM handshake instead of queueing behind it', async () => {
  engines.length = 0;
  const engine = new StubEngine();
  engine.gated = true;
  // oxlint-disable-next-line eslint/object-shorthand -- the shell calls `new AgaveEngine()`, which a method shorthand cannot be
  Object.assign(globalThis, { AgaveEngine: function () { return engine; } });
  await mountShell();

  typeInto(present(document.querySelector<HTMLInputElement>('#model-url'), 'model url field'), MODEL_URL);
  await settle();
  clickButtonLabelled('Load model');
  await settle();

  /* `init()` has not settled, yet the fetch is already in flight: a load that
     awaited init() up front could not have started the download yet. */
  expect(engine.initialized).toBe(false);
  expect(engine.downloadStartedBeforeInit).toBe(true);

  present(engine.releaseInit, 'init() latch')();
  expect(await transcriptHolds('Loaded: stub-model')).toBe(true);
  /* The bytes land only once the module exists: loadModel rejects otherwise. */
  expect(engine.hasModel).toBe(true);
}, 30_000);

test('a loaded model survives the re-render a prompt triggers', async () => {
  stubEngineHost();
  await mountShell();

  /* Docs/brand/README.md ("Honest status") forbids presenting the WASM build
     as a working chat, so the header and the empty state say what this build
     does before a prompt is written. */
  expect(transcriptText()).toContain('Forward pass pending a Zig wasm32 fix');
  expect(document.querySelector('header')?.textContent).toContain('blocked in this build');

  typeInto(present(document.querySelector<HTMLInputElement>('#model-url'), 'model url field'), MODEL_URL);
  await settle();
  clickButtonLabelled('Load model');
  expect(await transcriptHolds('Loaded: stub-model')).toBe(true);
  /* The model banner is a model-load report, not a reply, so it lands under
     the Engine label too. */

  /* Typing is the re-render: every keystroke runs the Shell body again. */
  typeInto(present(document.querySelector<HTMLInputElement>('#prompt'), 'prompt field'), 'hi');
  await settle();
  present(document.querySelector('form'), 'composer form')
    .dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));

  expect(await transcriptHolds('Tokenized 1 tokens')).toBe(true);
  expect(engines.filter((engine) => engine.prompted.length > 0)).toHaveLength(1);

  /* The engine's report is not model output, so it is labelled Engine and the
     log never claims a reply from "Agave" the way the serve UI's turns do. */
  expect(transcriptText()).toContain('Engine');
  expect(transcriptText()).not.toContain('Agave');

  /* One engine for the whole mount: a second one would be a second module. */
  expect(engines).toHaveLength(1);
  expect(engines[0]?.prompted).toEqual(['hi']);
  /* One download for the one load: a rebuilt engine would fetch the model again. */
  expect(engines[0]?.downloads).toBe(1);
}, 30_000);
