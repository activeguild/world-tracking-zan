// -*- mode: Bluespec; -*-
/**
 * ARSession lifecycle × vision worker protocol.
 *
 * What Lean does for single rules (lean/Guardrails), this does for the
 * *interplay*: `start()` / `stop()` / `reset()` called at any moment, camera
 * and worker start-up finishing (or failing) later, one frame in flight,
 * responses arriving in order but at arbitrary times.
 *
 * Modelled code:
 *   - src/ar/ARSession.ts          start / doStart / stop / reset /
 *                                  onVideoFrame / handleResult
 *   - src/worker/VisionWorkerClient.ts  busy flag, init, onmessage, onerror
 *   - src/worker/VisionWorker.ts        init / frame / reset handlers
 *
 * Not modelled: what the engine computes (a result is just "frame id,
 * computed before or after which reset"), the main-thread backend's frames
 * (synchronous: nothing is ever in flight), grabber switching, rendering.
 *
 * `fix` switches each repair off again, so the checker can show that the
 * invariant it protects really fails without it (scripts/quint-check.sh).
 */
module session {
  // ---------------------------------------------------------------- bounds
  /// Frames sent per run and `reset()` calls per run (keeps the state space finite).
  pure val MAX_FRAMES = 3
  pure val MAX_RESETS = 2

  // ----------------------------------------------------------------- types
  type Phase = Idle | StartingCamera | StartingBackend | Running
  type Backend = NoBackend | WorkerBackend | MainBackend
  type Call = NoCall | StartCall | StopCall

  /// Main thread → worker (protocol.ts WorkerRequest).
  type Request = Init | Frame(int) | Reset

  /// Worker → main thread. `Uncaught` is the worker's `error` event (an
  /// exception no handler caught); the payload says whether a frame died with it.
  type Response =
    | Ready
    | Result({ id: int, epoch: int })
    | FrameError(int)
    | ResetError
    | Uncaught(bool)

  /// How the worker handles the message at the head of its queue.
  type Outcome = Ok | Caught | Crash

  type Fixes = {
    // worker `error` event releases `busy`; the worker catches a failing reset
    workerError: bool,
    // `start()` during a start that `stop()` asked to abort cancels the abort
    restart: bool,
    // results of frames sent before `reset()` are not applied
    staleReset: bool,
  }

  type State = {
    fix: Fixes,
    // ARSession
    phase: Phase,
    stopRequested: bool,
    cameraOn: bool,
    backend: Backend,
    nextFrameId: int,
    // frames below this id were sent before the last `reset()`
    resetBeforeFrame: int,
    // VisionWorkerClient
    busy: bool,
    // VisionWorker
    engineEpoch: int,
    // message queues (postMessage keeps order in each direction)
    toWorker: List[Request],
    toMain: List[Response],
    // ghosts: not in the code, only for stating the properties
    lastCall: Call,
    startFailed: bool,
    resets: int,
    staleApplied: bool,
    applied: int,
  }

  pure val ALL_FIXES: Fixes = { workerError: true, restart: true, staleReset: true }

  var s: State

  // ------------------------------------------------------------- helpers
  pure def initial(fix: Fixes): State = {
    fix: fix,
    phase: Idle,
    stopRequested: false,
    cameraOn: false,
    backend: NoBackend,
    nextFrameId: 0,
    resetBeforeFrame: 0,
    busy: false,
    engineEpoch: 0,
    toWorker: [],
    toMain: [],
    lastCall: NoCall,
    startFailed: false,
    resets: 0,
    staleApplied: false,
    applied: 0,
  }

  /// ARSession.releaseResources(): backend.dispose() terminates the worker —
  /// which empties both message queues (HTML: "terminate a worker") — and the
  /// camera stops.
  pure def released(st: State): State = {
    ...st,
    phase: Idle,
    stopRequested: false,
    cameraOn: false,
    backend: NoBackend,
    busy: false,
    toWorker: [],
    toMain: [],
  }

  /// doStart() after `await createBackend()`: honour a stop() that came
  /// meanwhile, else the run begins.
  pure def backendUp(st: State, kind: Backend): State =
    if (st.stopRequested) released(st)
    else { ...st, phase: Running, backend: kind, nextFrameId: 0, resetBeforeFrame: 0, resets: 0, applied: 0 }

  pure def isFrame(r: Request): bool = match r {
    | Frame(_) => true
    | _ => false
  }

  /// A response that ends a frame's flight.
  pure def endsFlight(r: Response): bool = match r {
    | Result(_) => true
    | FrameError(_) => true
    | Uncaught(frame) => frame
    | _ => false
  }

  pure def inFlight(st: State): int =
    st.toWorker.select(isFrame).length() + st.toMain.select(endsFlight).length()

  // ----------------------------------------------------- application calls
  /// ARSession.start()
  action start = all {
    s' = match s.phase {
      | Running => { ...s, lastCall: StartCall }
      | Idle => { ...s, phase: StartingCamera, stopRequested: false, startFailed: false, lastCall: StartCall }
      // a start in flight: the same promise is returned
      | _ => { ...s, lastCall: StartCall, stopRequested: if (s.fix.restart) false else s.stopRequested }
    },
  }

  /// ARSession.stop()
  action stop = all {
    s' = match s.phase {
      | Running => { ...released(s), lastCall: StopCall }
      | Idle => { ...s, lastCall: StopCall }
      | _ => { ...s, stopRequested: true, lastCall: StopCall }
    },
  }

  /// ARSession.reset(): `this.backend?.reset()` — the backend field is only
  /// set while running.
  action reset = all {
    s.phase == Running,
    s.resets < MAX_RESETS,
    s' = {
      ...s,
      resets: s.resets + 1,
      resetBeforeFrame: s.nextFrameId,
      toWorker: if (s.backend == WorkerBackend) s.toWorker.append(Reset) else s.toWorker,
    },
  }

  // ------------------------------------------------------- start-up awaits
  /// `await this.camera.start()` settles.
  action cameraSettles = all {
    s.phase == StartingCamera,
    nondet ok = oneOf(Set(true, false))
    s' = if (not(ok)) { ...s, phase: Idle, stopRequested: false, startFailed: true }
      else if (s.stopRequested) released({ ...s, cameraOn: true })
      else { ...s, cameraOn: true, phase: StartingBackend, toWorker: [Init], toMain: [], engineEpoch: 0 },
  }

  // ------------------------------------------------------------- per frame
  /// ARSession.onVideoFrame() with the worker backend (backpressure: a frame
  /// is dropped while one is in flight).
  action videoFrame = all {
    s.phase == Running,
    s.backend == WorkerBackend,
    not(s.busy),
    s.nextFrameId < MAX_FRAMES,
    s' = { ...s, busy: true, nextFrameId: s.nextFrameId + 1, toWorker: s.toWorker.append(Frame(s.nextFrameId)) },
  }

  // --------------------------------------------------------------- worker
  /// VisionWorker onmessage for the head of its queue.
  action workerHandles = all {
    s.toWorker.length() > 0,
    nondet outcome = oneOf(Set(Ok, Caught, Crash))
    val rest = s.toWorker.tail()
    s' = match s.toWorker.head() {
      | Init =>
        if (outcome == Ok) { ...s, toWorker: rest, toMain: s.toMain.append(Ready) }
        else { ...s, toWorker: rest, toMain: s.toMain.append(Uncaught(false)) }
      | Frame(id) =>
        if (outcome == Ok) { ...s, toWorker: rest, toMain: s.toMain.append(Result({ id: id, epoch: s.engineEpoch })) }
        else if (outcome == Caught) { ...s, toWorker: rest, toMain: s.toMain.append(FrameError(id)) }
        else { ...s, toWorker: rest, toMain: s.toMain.append(Uncaught(true)) }
      | Reset =>
        if (outcome == Ok) { ...s, toWorker: rest, engineEpoch: s.engineEpoch + 1 }
        // a reset that throws still counts: the properties are about order, not about what the engine kept
        else if (s.fix.workerError) { ...s, toWorker: rest, engineEpoch: s.engineEpoch + 1, toMain: s.toMain.append(ResetError) }
        else { ...s, toWorker: rest, engineEpoch: s.engineEpoch + 1, toMain: s.toMain.append(Uncaught(false)) }
    },
  }

  // ---------------------------------------------------------- main thread
  /// VisionWorkerClient onmessage / onerror for the head of the queue, and
  /// what ARSession does with it.
  action mainHandles = all {
    s.toMain.length() > 0,
    val rest = s.toMain.tail()
    s' = match s.toMain.head() {
      | Ready => backendUp({ ...s, toMain: rest }, WorkerBackend)
      | Uncaught(_) =>
        if (s.phase == StartingBackend)
          // init() rejects → client.dispose() → main-thread backend
          backendUp({ ...s, toMain: [], toWorker: [], busy: false }, MainBackend)
        else { ...s, toMain: rest, busy: if (s.fix.workerError) false else s.busy }
      | Result(r) =>
        // handleResult(): only while running (always true here: stop() empties the queue)
        val stale = r.epoch < s.resets
        val skip = s.fix.staleReset and r.id < s.resetBeforeFrame
        { ...s, toMain: rest, busy: false,
          applied: if (skip or s.applied == MAX_FRAMES) s.applied else s.applied + 1,
          staleApplied: s.staleApplied or (stale and not(skip)) }
      | FrameError(_) => { ...s, toMain: rest, busy: false }
      | ResetError => { ...s, toMain: rest }
    },
  }

  // ------------------------------------------------------------ the system
  action init = s' = initial(ALL_FIXES)
  /// The code before the repairs, and with one repair missing at a time.
  action initShipped = s' = initial({ workerError: false, restart: false, staleReset: false })
  action initNoWorkerErrorFix = s' = initial({ ...ALL_FIXES, workerError: false })
  action initNoRestartFix = s' = initial({ ...ALL_FIXES, restart: false })
  action initNoStaleResetFix = s' = initial({ ...ALL_FIXES, staleReset: false })

  action step = any {
    start,
    stop,
    reset,
    cameraSettles,
    videoFrame,
    workerHandles,
    mainHandles,
  }

  /// A sub-behaviour of `step` for trace generation only: random walks of
  /// `step` are stopped before they get deep into a run.
  action stepWithoutStop = any {
    start,
    reset,
    cameraSettles,
    videoFrame,
    workerHandles,
    mainHandles,
  }

  // ------------------------------------------------------------ properties
  /// Backpressure is exact: `busy` iff exactly one frame is somewhere between
  /// processFrame() and its result / error. (Without the worker-error repair
  /// `busy` stays set with nothing in flight: no frame is ever sent again.)
  val busyIffOneInFlight = inFlight(s) == (if (s.busy) 1 else 0)

  /// A session that is not running and not starting holds nothing.
  val idleHoldsNothing = s.phase == Idle implies and {
    not(s.cameraOn),
    s.backend == NoBackend,
    not(s.busy),
    not(s.stopRequested),
    s.toWorker.length() == 0,
    s.toMain.length() == 0,
  }

  /// The camera is on exactly from the camera await to the release.
  val cameraMatchesPhase = s.cameraOn == (s.phase == StartingBackend or s.phase == Running)

  /// A backend exists exactly while running; messages exist only with a worker.
  val backendMatchesPhase = and {
    (s.backend != NoBackend) == (s.phase == Running),
    s.backend == MainBackend implies (s.toWorker.length() == 0 and s.toMain.length() == 0 and not(s.busy)),
  }

  /// Once nothing is pending, the session is in the state the application
  /// asked for last (unless that start failed).
  val lastCallWins = and {
    (s.phase == Idle and s.lastCall == StartCall) implies s.startFailed,
    s.phase == Running implies s.lastCall == StartCall,
  }

  /// After `reset()`, nothing computed before the reset is applied.
  val noStaleResult = not(s.staleApplied)

  /// The id-based filter drops exactly the pre-reset results: every frame
  /// sent after the last reset is computed after it.
  val filterIsExact = s.toMain.select(r => match r {
    | Result(x) => x.id >= s.resetBeforeFrame and x.epoch < s.resets
    | _ => false
  }).length() == 0

  val safety = and {
    busyIffOneInFlight,
    idleHoldsNothing,
    cameraMatchesPhase,
    backendMatchesPhase,
    lastCallWins,
    noStaleResult,
    filterIsExact,
  }

  // ---------------------------------------------- the model is not vacuous
  // States the checker must be able to reach (else the invariants hold for
  // the wrong reason).
  observation appliesThreeFrames = s.applied >= 3
  observation runsOnMainBackend = s.backend == MainBackend
  observation dropsStaleResult = s.toMain.length() > 0 and s.phase == Running and (match s.toMain.head() {
    | Result(x) => x.id < s.resetBeforeFrame
    | _ => false
  })
  observation abortsStart = s.phase == StartingBackend and s.stopRequested
}
