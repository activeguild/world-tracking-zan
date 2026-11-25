// -*- mode: Bluespec; -*-
/**
 * Tracking pipeline: map loss, relocalization, tracking state, the world,
 * object visibility, plane recovery.
 *
 * One step is one processed frame: the engine (VisionEngine.updateMap, plane
 * recovery, the tracking state machine), then the session handling that
 * frame's result (world creation / drop, object visibility). What the frame
 * looks like is nondeterministic: how many features track, whether the map
 * PnP locates the camera, whether the plane detector reports a plane, what a
 * relocalization attempt returns, whether the camera moved fast.
 *
 * Modelled code:
 *   - src/vision/RelocalizationLedger.ts  pending candidate → confirmation →
 *                                         apply → monitor window, map reset rule
 *   - src/vision/VisionEngine.ts          updateMap (drives the ledger), worldEstablished
 *   - src/vision/RelocalizationSchedule.ts  shouldAttempt / candidateStale / candidateAction
 *   - src/vision/PlaneRecovery.ts         update
 *   - src/ar/ARState.ts                   TrackingStateMachine.update
 *   - src/ar/ObjectVisibility.ts          decideObjectVisibility
 *   - src/ar/ARSession.ts                 updateWorld
 *
 * The frame counts of the configuration are scaled down (a tolerance of 3
 * frames is 2 here, a 300-frame reset is 4): the rules are the same, the
 * state space stays small. Not modelled: what a relocalization validates,
 * the keyframes, poses.
 */
module tracking {
  // ------------------------------------------------- configuration (scaled)
  /// state.mapLostFrameTolerance
  pure val MAP_LOST_TOLERANCE = 2
  /// state.lostFrameTolerance
  pure val LOST_FRAME_TOLERANCE = 2
  /// relocalization.startAfterLostFrames / attemptEveryNFrames
  pure val START_AFTER = 1
  pure val ATTEMPT_EVERY = 2
  /// relocalization.confirmationFrames / candidateMaxAgeFrames / postRelocMonitorFrames
  pure val CONFIRMATIONS = 1
  pure val CANDIDATE_MAX_AGE = 2
  pure val MONITOR_FRAMES = 2
  /// landmarks.preWorldLostResetFrames / lostResetFrames, relocalization.minAttemptsBeforeReset
  pure val PRE_WORLD_RESET = 3
  pure val LOST_RESET = 4
  pure val MIN_ATTEMPTS = 2
  /// Maps a run may build (keeps the state space finite).
  pure val MAX_MAPS = 2
  /// Frames of good tracking after which objects must be visible again.
  pure val RECOVER_WITHIN = MONITOR_FRAMES + 2

  // ----------------------------------------------------------------- types
  /// Frame-to-frame inliers: below `lostBelow`, between, at least `minTrackedForTracking`.
  type Inliers = Low | Mid | High

  type TrackingState = Initializing | Searching | Tracking | PlaneDetecting | PlaneFound | TrackingLost | Relocalizing

  /// What a relocalization attempt returns.
  type Reloc =
    | NoCandidate
    | Strong          // validated, error within the strict bound
    | StrongImmediate // … and good enough to apply at once
    | Acceptable      // validated only within the relaxed bound

  type Input = {
    inliers: Inliers,
    // tryInitialize succeeds (no map yet)
    canInit: bool,
    // the map PnP locates the camera
    located: bool,
    // PlaneDetector.found after this frame
    planeDetected: bool,
    reloc: Reloc,
    // the candidate lands where the pending one did
    near: bool,
    // fast motion / two-view trigger of the plane recovery
    motion: bool,
  }

  type Option[a] = Some(a) | None

  /// A candidate waiting for its confirmation: frames since it was found.
  type Pending = { age: int, confirmations: int }

  type Fixes = {
    // the session creates the world only when the engine has established it
    worldGate: bool,
    // a new world starts hidden: nothing is shown before a frame with a fresh pose
    freshShow: bool,
    // objects that are shown stay shown through an applied relocalization
    keepShown: bool,
  }

  type State = {
    fix: Fixes,
    // ---- engine
    mapId: Option[int],
    maps: int,
    framesSinceTracked: int,
    worldEstablished: bool,
    pending: Option[Pending],
    monitorLeft: int,
    attempts: int,
    relocalized: bool,
    planeDetected: bool,
    recovery: bool,
    state: TrackingState,
    badFrames: int,
    // ---- session
    world: Option[int],
    visible: bool,
    // ---- ghosts
    // this frame's inliers / whether the map PnP located the camera
    inliers: Inliers,
    located: bool,
    // what the state machine saw this frame
    obsMapLost: bool,
    obsPlaneFound: bool,
    obsWorldEstablished: bool,
    wasVisible: bool,
    sinceApply: int,
    goodRun: int,
    staleCandidate: bool,
    unconfirmedAcceptable: bool,
    appliedByConfirmation: bool,
    hiddenInsideTolerance: bool,
    recoveries: int,
    resets: int,
    resetWithWorld: bool,
  }

  pure val ALL_FIXES: Fixes = { worldGate: true, freshShow: true, keepShown: true }

  var s: State

  pure def initial(fix: Fixes): State = {
    fix: fix,
    mapId: None, maps: 0, framesSinceTracked: 0, worldEstablished: false, pending: None,
    monitorLeft: 0, attempts: 0, relocalized: false, planeDetected: false, recovery: false,
    state: Initializing, badFrames: 0,
    world: None, visible: false,
    inliers: High, located: false, obsMapLost: false, obsPlaneFound: false, obsWorldEstablished: false,
    wasVisible: false, sinceApply: MONITOR_FRAMES + 1, goodRun: 0,
    staleCandidate: false, unconfirmedAcceptable: false, appliedByConfirmation: false,
    hiddenInsideTolerance: false, recoveries: 0, resets: 0, resetWithWorld: false,
  }

  pure def cap(n: int, limit: int): int = if (n > limit) limit else n

  // ---------------------------------------------------- tracking state machine
  /// TrackingStateMachine.update (ARState.ts); AR_ACTIVE is the session's view of PlaneFound.
  pure def machine(
    st: TrackingState, bad: int, inl: Inliers,
    mapInit: bool, mapLost: bool, planeFound: bool, world: bool
  ): { state: TrackingState, bad: int } =
    if (st == Initializing) { state: Searching, bad: 0 }
    else if (st == Searching or st == TrackingLost or st == Relocalizing) {
      val relocatable = mapInit and world
      if (inl == High) {
        state: if (relocatable and mapLost) Relocalizing else if (relocatable and planeFound) PlaneFound else Tracking,
        bad: 0,
      }
      else if (relocatable and not(mapLost) and planeFound and inl == Mid) { state: PlaneFound, bad: 0 }
      else if (st == TrackingLost) { state: if (relocatable) Relocalizing else Searching, bad: bad }
      else if (st == Relocalizing and not(relocatable)) { state: Searching, bad: bad }
      else { state: st, bad: bad }
    }
    else if (inl == Low) {
      if (bad + 1 >= LOST_FRAME_TOLERANCE) { state: TrackingLost, bad: 0 }
      else { state: st, bad: bad + 1 }
    }
    else if (mapInit and mapLost) { state: if (world) Relocalizing else Tracking, bad: 0 }
    else if (planeFound) { state: PlaneFound, bad: 0 }
    else if (mapInit) { state: if (st == PlaneFound) PlaneFound else PlaneDetecting, bad: 0 }
    else { state: Tracking, bad: 0 }

  // ------------------------------------------------------------------ engine
  /// The map is gone (tracker.reset): everything attached to it goes with it.
  pure def mapDropped(e: State): State = {
    ...e,
    mapId: None, framesSinceTracked: 0, worldEstablished: false, pending: None,
    monitorLeft: 0, attempts: 0, relocalized: false, planeDetected: false, recovery: false,
    resets: cap(e.resets + 1, 2),
  }

  /// VisionEngine.updateMap with a map: relocalization, PnP, monitor window, reset.
  pure def mapFrame(e: State, i: Input): State = {
    // A candidate only lives while the camera is lost in an established world.
    val kept = if (e.framesSinceTracked == 0 or not(e.worldEstablished)) None else e.pending
    // candidateStale: older than its age limit → dropped, counted as a failed attempt
    val stale = match kept {
      | None => false
      | Some(p) => p.age > CANDIDATE_MAX_AGE
    }
    val pending = if (stale) None else kept
    val attempts0 = if (stale) e.attempts + 1 else e.attempts
    // shouldAttemptRelocalization
    val attempt = e.worldEstablished and e.framesSinceTracked > 0 and (
      pending != None or (e.framesSinceTracked >= START_AFTER and (e.framesSinceTracked - START_AFTER) % ATTEMPT_EVERY == 0)
    )
    val candidate = attempt and i.reloc != NoCandidate
    // requiredConfirmations / candidateAction
    val needed = if (i.reloc == Acceptable and CONFIRMATIONS < 1) 1 else CONFIRMATIONS
    val immediate = needed <= 0 or i.reloc == StrongImmediate
    val confirmed = match pending {
      | None => false
      | Some(p) => candidate and i.near and p.confirmations + 1 >= needed
    }
    val apply = candidate and (confirmed or (pending == None and immediate))
    val nextPending: Option[Pending] =
      if (not(attempt)) pending
      else if (not(candidate) or apply) None
      else match pending {
        // the newer candidate starts its own confirmation
        | None => Some({ age: 0, confirmations: 0 })
        | Some(p) => if (i.near) Some({ age: p.age, confirmations: p.confirmations + 1 }) else Some({ age: 0, confirmations: 0 })
      }
    val failed = attempt and (not(candidate) or (pending != None and not(i.near)))
    val attempts1 = if (failed) attempts0 + 1 else attempts0
    // applyRelocalization re-seeds the pose (framesSinceTracked = 0), then the map PnP runs
    val fst0 = if (apply) 0 else e.framesSinceTracked
    val fst = if (i.located) 0 else fst0 + 1
    // post-relocalization monitor: the apply frame and the frames it is still open
    val monitored = apply or e.monitorLeft > 0
    val monitorLeft = if (apply) MONITOR_FRAMES else if (e.monitorLeft > 0) e.monitorLeft - 1 else 0
    val tracked: State = {
      ...e,
      framesSinceTracked: fst,
      pending: match nextPending {
        | None => None
        | Some(p) => Some({ ...p, age: p.age })
      },
      monitorLeft: monitorLeft,
      relocalized: monitored,
      attempts: if (i.located) 0 else cap(attempts1, MIN_ATTEMPTS),
      sinceApply: if (apply) 0 else cap(e.sinceApply + 1, MONITOR_FRAMES + 1),
      staleCandidate: e.staleCandidate or stale,
      unconfirmedAcceptable: e.unconfirmedAcceptable or (apply and i.reloc == Acceptable and not(confirmed)),
      appliedByConfirmation: e.appliedByConfirmation or (apply and confirmed),
    }
    // Reset: the last resort in an established world, quick before one exists.
    val reset = not(i.located) and (
      if (e.worldEstablished) fst > LOST_RESET and tracked.attempts >= MIN_ATTEMPTS
      else fst > PRE_WORLD_RESET
    )
    if (reset) { ...mapDropped(tracked), resetWithWorld: e.resetWithWorld or e.worldEstablished } else tracked
  }

  /// One frame of the engine.
  pure def engineFrame(e0: State, i: Input): State = {
    // 1. map
    val e1 = match e0.mapId {
      | None =>
        if (i.canInit and e0.maps < MAX_MAPS)
          { ...mapDropped(e0), resets: e0.resets, mapId: Some(e0.maps), maps: e0.maps + 1 }
        else { ...e0, sinceApply: cap(e0.sinceApply + 1, MONITOR_FRAMES + 1) }
      | Some(_) => mapFrame(e0, i)
    }
    val mapInit = e1.mapId != None
    // a pending candidate ages by one frame
    val e2 = { ...e1, pending: match e1.pending {
      | None => None
      | Some(p) => Some({ ...p, age: p.age + 1 })
    } }
    val mapLost = mapInit and e2.framesSinceTracked > MAP_LOST_TOLERANCE
    // 2. plane recovery (PlaneRecovery.update): never in an established world,
    //    ends when the map is lost, starts on motion with a healthy map
    val healthy = mapInit and i.located
    val recovery =
      if (e2.worldEstablished) false
      else if (e2.recovery) not(mapLost)
      else i.motion and healthy
    val started = recovery and not(e2.recovery)
    // 3. plane detector
    val detected = mapInit and i.planeDetected
    // 4. tracking state
    val planeFound = e2.worldEstablished or detected
    val m = machine(e2.state, e2.badFrames, i.inliers, mapInit, mapLost, planeFound, e2.worldEstablished)
    {
      ...e2,
      recovery: recovery,
      recoveries: if (started) cap(e2.recoveries + 1, 2) else e2.recoveries,
      planeDetected: detected,
      state: m.state,
      badFrames: m.bad,
      // established the first time a plane is found for this map
      worldEstablished: e2.worldEstablished or (m.state == PlaneFound and mapInit),
      inliers: i.inliers,
      located: mapInit and i.located,
      obsMapLost: mapLost,
      obsPlaneFound: planeFound,
      obsWorldEstablished: e2.worldEstablished,
    }
  }

  // ----------------------------------------------------------------- session
  /// decideObjectVisibility (ObjectVisibility.ts)
  pure def decide(
    st: TrackingState, worldReady: bool, established: bool,
    fst: int, relocalized: bool, currentlyVisible: bool, keepShown: bool
  ): bool =
    if (not(worldReady) or not(established)) false
    else if (st != PlaneFound) false
    // a relocalized pose being confirmed never *starts* showing; what is
    // already shown (a dropout inside the tolerance) stays shown
    else if (relocalized and not(currentlyVisible and keepShown)) false
    else if (fst == 0) true
    else currentlyVisible and fst <= MAP_LOST_TOLERANCE

  /// ARSession.updateWorld for the frame's result.
  pure def sessionFrame(e: State, before: State): State = {
    // WorldAnchor.checkMap: the world of a map that is gone is dropped (objects with it)
    val kept = match e.world {
      | None => None
      | Some(w) => if (e.mapId == Some(w)) e.world else None
    }
    val create = kept == None and e.mapId != None and e.planeDetected and (e.worldEstablished or not(e.fix.worldGate))
    val world = if (create) e.mapId else kept
    // ARWorld starts a new world un-hidden; a kept world keeps what it showed
    val currentlyVisible = if (create) not(e.fix.freshShow) else if (kept == None) false else before.visible
    val visible =
      if (world == None or e.mapId == None) false
      else decide(e.state, true, e.worldEstablished, e.framesSinceTracked, e.relocalized, currentlyVisible, e.fix.keepShown)
    val good = e.mapId != None and before.worldEstablished and e.located and e.inliers != Low
    {
      ...e,
      world: world,
      visible: visible,
      wasVisible: before.visible,
      goodRun: if (good) cap(before.goodRun + 1, RECOVER_WITHIN) else 0,
      hiddenInsideTolerance: e.hiddenInsideTolerance or
        (before.visible and not(visible) and e.relocalized and kept != None and e.worldEstablished
          and e.state == PlaneFound and e.framesSinceTracked <= MAP_LOST_TOLERANCE),
    }
  }

  // ------------------------------------------------------------ the system
  action init = s' = initial(ALL_FIXES)
  action initNoWorldGate = s' = initial({ ...ALL_FIXES, worldGate: false })
  action initNoFreshShow = s' = initial({ ...ALL_FIXES, freshShow: false })
  action initNoKeepShown = s' = initial({ ...ALL_FIXES, keepShown: false })

  action step = {
    nondet inliers = oneOf(Set(Low, Mid, High))
    nondet canInit = oneOf(Set(true, false))
    nondet located = oneOf(Set(true, false))
    nondet planeDetected = oneOf(Set(true, false))
    nondet reloc = oneOf(Set(NoCandidate, Strong, StrongImmediate, Acceptable))
    nondet near = oneOf(Set(true, false))
    nondet motion = oneOf(Set(true, false))
    val i = { inliers: inliers, canInit: canInit, located: located, planeDetected: planeDetected, reloc: reloc, near: near, motion: motion }
    s' = sessionFrame(engineFrame(s, i), s)
  }

  // ------------------------------------------------------------ properties
  /// The session has a world exactly when the engine has established one, for the same map.
  val worldAgrees = (s.world != None) == s.worldEstablished and (s.world == None or s.world == s.mapId)

  /// Objects are only shown on a pose that can be trusted: world tracking in
  /// an established world, located within the tolerance; on a relocalized
  /// pose still being confirmed only if they were shown already.
  val visibleOnlyWhenTrusted = s.visible implies and {
    s.world != None,
    s.worldEstablished,
    s.state == PlaneFound,
    s.framesSinceTracked <= MAP_LOST_TOLERANCE,
    s.relocalized implies s.wasVisible,
  }

  /// Showing starts on a frame whose pose came from the map, never on a held pose.
  val showStartsOnFreshPose = (s.visible and not(s.wasVisible)) implies s.framesSinceTracked == 0

  /// Showing never starts in the frame a relocalization is applied nor while
  /// its monitor window is open (no one-frame flash of an unconfirmed pose).
  /// (Was `hiddenWhileConfirming`: nothing shown at all in that window. Given
  /// up for objects that were already shown, see `shownThroughRelocalization`.)
  val noShowStartWhileConfirming = (s.sinceApply <= MONITOR_FRAMES and s.visible) implies s.wasVisible

  /// Objects shown through a tolerated PnP dropout are not hidden because a
  /// relocalization is applied inside that tolerance: while the state machine
  /// stays in world tracking and the camera is located within the tolerance,
  /// a relocalization never hides what is shown.
  val shownThroughRelocalization = not(s.hiddenInsideTolerance)

  /// A pending candidate is confirmed or replaced in the very next frame:
  /// it never reaches its age limit (`candidateStale` cannot fire).
  val pendingIsYoung = not(s.staleCandidate) and (match s.pending {
    | None => true
    | Some(p) => p.age == 1 and s.worldEstablished
  })

  /// A candidate inside only the relaxed error bound is never applied without a confirmation.
  val acceptableIsConfirmed = not(s.unconfirmedAcceptable)

  /// A plane recovery runs only before the world exists and while the map is not lost.
  val recoveryOnlyBeforeWorld = s.recovery implies and {
    s.mapId != None,
    not(s.obsWorldEstablished),
    s.framesSinceTracked <= MAP_LOST_TOLERANCE,
  }

  /// Bounded recovery: once the camera is located and enough features track
  /// for RECOVER_WITHIN frames in an established world, the objects are visible.
  val recoversWhenTrackingIsGood = (s.goodRun >= RECOVER_WITHIN and s.worldEstablished) implies s.visible

  val safety = and {
    worldAgrees,
    visibleOnlyWhenTrusted,
    showStartsOnFreshPose,
    noShowStartWhileConfirming,
    shownThroughRelocalization,
    pendingIsYoung,
    acceptableIsConfirmed,
    recoveryOnlyBeforeWorld,
    recoversWhenTrackingIsGood,
  }

  // ---------------------------------------------- the model is not vacuous
  observation relocalizedByConfirmation = s.appliedByConfirmation and s.visible
  observation worldLostAndRebuilt = s.resets >= 1 and s.world == Some(1) and s.visible
  observation planeRecoveryRan = s.recoveries >= 1 and s.worldEstablished
  observation resetInEstablishedWorld = s.resetWithWorld
  /// A relocalization applied while the objects are shown, and they stay shown.
  observation relocalizedWhileShown = s.sinceApply == 0 and s.visible and s.wasVisible

  // ------------------------------------------------------------- liveness
  /// Frames keep coming.
  temporal fairness = weakFair(step, s)

  // "Good tracking from some frame on ⇒ shown from some frame on" needs no
  // temporal property: the invariant `recoversWhenTrackingIsGood` bounds the
  // number of frames, which is stronger.

  /// A loss does not last forever: the camera is located again, a
  /// relocalization is applied, or the lost map is given up (and the scan
  /// starts over).
  ///
  /// "Located again" alone does not hold: there is a run in which the map PnP
  /// never locates the camera while a relocalization is applied in every
  /// frame (each re-seeds the pose, so the loss never gets long and no
  /// attempt counts as failed). The engine then tracks by relocalization.
  temporal lossEnds = fairness implies
    always(s.framesSinceTracked > 0 implies eventually(s.framesSinceTracked == 0 or s.sinceApply == 0 or s.mapId == None))
}
