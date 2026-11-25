// -*- mode: Bluespec; -*-
/**
 * Map corrections × world anchor.
 *
 * The engine refines its map in place (init stage, keyframe BA): the whole
 * map moves by a rigid correction. The session holds the world anchor in map
 * coordinates and must move it by the same corrections — each exactly once,
 * in the order the engine applied them — or placed objects end up somewhere
 * else. Rigid transforms do not commute, so a correction is modelled as an
 * opaque tag and "the anchor is where the map is" as equality of tag lists.
 *
 * Modelled code:
 *   - src/vision/VisionEngine.ts   updateMap: map init / reset, init stage
 *                                  then keyframe BA in one frame, the anchor
 *                                  points a correction is fitted around
 *   - src/ar/ARSession.ts          handleResult (a frame's corrections, in
 *                                  order) / updateWorld, onVideoFrame
 *   - src/ar/WorldAnchor.ts        create / applyMapCorrection / checkMap
 *
 * Not modelled: the numbers (what a correction is, whether a BA is accepted),
 * lost results (session.qnt: a running session handles every result except
 * those of before a reset(), after which the map is new anyway).
 */
module mapcorrection {
  // ---------------------------------------------------------------- bounds
  pure val MAX_FRAMES = 5
  pure val MAX_CORRECTIONS = 3
  /// Map ids are frame ids of the reference view; a later map may get the
  /// same one again (same reference frame), so ids are drawn from a small set.
  pure val MAP_IDS = Set(0, 1)

  // ----------------------------------------------------------------- types
  /// A solver's outcome in one frame.
  type Solve = Skipped | Rejected | Accepted

  /// What the engine does to its map in one frame.
  type MapEvent =
    | Idle
    | InitMap(int)
    | ResetMap
    | Refine({ stage: Solve, ba: Solve, planeFound: bool })

  type LandmarkMap = { id: int, corr: List[int] }

  /// The session's world anchor: the map it is attached to and (ghost) the
  /// corrections of that map it reflects.
  type Anchor = { mapId: int, view: List[int] }

  type Option[a] = Some(a) | None

  /// Frame request: the anchor points are the anchor at send time.
  type Request = { id: int, anchor: Option[Anchor] }

  /// Result: the map after the frame and the corrections made in it.
  type Result = {
    id: int,
    map: Option[LandmarkMap],
    stage: Option[int],
    ba: Option[int],
    planeFound: bool,
  }

  type Fixes = {
    // the session applies a frame's corrections in the engine's order (stage, then BA)
    order: bool,
    // the engine moves its copy of the anchor points along with an accepted stage
    anchorPoints: bool,
    // one frame in flight (session.qnt busyIffOneInFlight); not a repair, an assumption
    oneInFlight: bool,
  }

  type State = {
    fix: Fixes,
    // engine
    map: Option[LandmarkMap],
    nextTag: int,
    // session
    anchor: Option[Anchor],
    nextFrameId: int,
    // transport (in order)
    toEngine: List[Request],
    toSession: List[Result],
    // ghosts
    lastHandled: Option[Result],
    staleAnchorPoints: bool,
    corrections: int,
    worlds: int,
  }

  pure val ALL_FIXES: Fixes = { order: true, anchorPoints: true, oneInFlight: true }

  var s: State

  pure def initial(fix: Fixes): State = {
    fix: fix,
    map: None,
    nextTag: 0,
    anchor: None,
    nextFrameId: 0,
    toEngine: [],
    toSession: [],
    lastHandled: None,
    staleAnchorPoints: false,
    corrections: 0,
    worlds: 0,
  }

  // --------------------------------------------------------------- session
  /// ARSession.onVideoFrame(): the frame carries the anchor points (world
  /// origin and placed objects in map coordinates, by the current anchor).
  action sendFrame = all {
    s.nextFrameId < MAX_FRAMES,
    s.toEngine.length() + s.toSession.length() < (if (s.fix.oneInFlight) 1 else 2),
    s' = { ...s, nextFrameId: s.nextFrameId + 1, toEngine: s.toEngine.append({ id: s.nextFrameId, anchor: s.anchor }) },
  }

  /// WorldAnchor.applyMapCorrection for one correction of a result.
  pure def moved(anchor: Option[Anchor], r: Result, tag: Option[int]): Option[Anchor] =
    match anchor {
      | None => None
      | Some(a) => match r.map {
        | None => anchor
        | Some(m) => match tag {
          | None => anchor
          // only an anchor attached to the map the correction belongs to
          | Some(t) => if (a.mapId == m.id) Some({ ...a, view: a.view.append(t) }) else anchor
        }
      }
    }

  /// ARSession.handleResult(): corrections, then updateWorld (drop the world
  /// of a map that is gone, create it on a found plane).
  action handleResult = all {
    s.toSession.length() > 0,
    val r = s.toSession.head()
    val corrected =
      if (s.fix.order) moved(moved(s.anchor, r, r.stage), r, r.ba)
      else moved(moved(s.anchor, r, r.ba), r, r.stage)
    // WorldAnchor.checkMap
    val kept = match corrected {
      | None => None
      | Some(a) => match r.map {
        | None => None
        | Some(m) => if (m.id == a.mapId) corrected else None
      }
    }
    // WorldAnchor.create: from the plane and pose of this result (after its corrections)
    val created = match r.map {
      | None => false
      | Some(_) => kept == None and r.planeFound
    }
    val anchorAfter = match r.map {
      | None => kept
      | Some(m) => if (created) Some({ mapId: m.id, view: m.corr }) else kept
    }
    s' = { ...s, toSession: s.toSession.tail(), anchor: anchorAfter, lastHandled: Some(r), worlds: if (created) s.worlds + 1 else s.worlds },
  }

  // ---------------------------------------------------------------- engine
  /// The anchor points of a request are usable for a correction of `m` with
  /// corrections `corr` so far: taken from an anchor of this map that reflects
  /// exactly those corrections. (No anchor yet: nothing to fit around.)
  pure def fresh(points: Option[Anchor], m: LandmarkMap, corr: List[int]): bool = match points {
    | None => true
    | Some(a) => a.mapId == m.id and a.view == corr
  }

  /// VisionEngine.process() for the head of the queue.
  action engineProcesses = all {
    s.toEngine.length() > 0,
    val req = s.toEngine.head()
    val rest = s.toEngine.tail()
    nondet event = oneOf(
      Set(Idle, ResetMap)
        .union(MAP_IDS.map(i => InitMap(i)))
        .union(tuples(Set(Skipped, Rejected, Accepted), Set(Skipped, Rejected, Accepted), Set(true, false))
          .map(t => Refine({ stage: t._1, ba: t._2, planeFound: t._3 })))
    )
    val budget = MAX_CORRECTIONS - s.corrections
    match s.map {
      | None => all {
        // tryInitialize (never in the frame of a reset: that frame's result has no map)
        match event {
          | Idle => true
          | InitMap(_) => true
          | _ => false
        },
        s' = match event {
          | InitMap(i) => { ...s, toEngine: rest, map: Some({ id: i, corr: [] }),
              toSession: s.toSession.append({ id: req.id, map: Some({ id: i, corr: [] }), stage: None, ba: None, planeFound: false }) }
          | _ => { ...s, toEngine: rest,
              toSession: s.toSession.append({ id: req.id, map: None, stage: None, ba: None, planeFound: false }) }
        },
      }
      | Some(m) => all {
        match event {
          | InitMap(_) => false
          | Refine(e) => (if (e.stage == Accepted) 1 else 0) + (if (e.ba == Accepted) 1 else 0) <= budget
          | _ => true
        },
        s' = match event {
          | ResetMap => { ...s, toEngine: rest, map: None,
              toSession: s.toSession.append({ id: req.id, map: None, stage: None, ba: None, planeFound: false }) }
          | Refine(e) =>
            // 1. init stage (finishInitStage), fitted around the request's anchor points
            val stageOk = e.stage == Accepted
            val staleAtStage = stageOk and not(fresh(req.anchor, m, m.corr))
            val afterStage = if (stageOk) m.corr.append(s.nextTag) else m.corr
            val tagsUsed = if (stageOk) 1 else 0
            // the engine's copy of the anchor points after the stage
            val points = match req.anchor {
              | None => None
              | Some(a) => if (stageOk and s.fix.anchorPoints) Some({ ...a, view: a.view.append(s.nextTag) }) else req.anchor
            }
            // 2. keyframe BA (runBundleAdjustment), same frame
            val baOk = e.ba == Accepted
            val staleAtBa = baOk and not(fresh(points, m, afterStage))
            val afterBa = if (baOk) afterStage.append(s.nextTag + tagsUsed) else afterStage
            val newMap = { id: m.id, corr: afterBa }
            { ...s, toEngine: rest, map: Some(newMap),
              nextTag: s.nextTag + tagsUsed + (if (baOk) 1 else 0),
              corrections: s.corrections + tagsUsed + (if (baOk) 1 else 0),
              staleAnchorPoints: s.staleAnchorPoints or staleAtStage or staleAtBa,
              toSession: s.toSession.append({
                id: req.id, map: Some(newMap),
                stage: if (stageOk) Some(s.nextTag) else None,
                ba: if (baOk) Some(s.nextTag + tagsUsed) else None,
                planeFound: e.planeFound,
              }) }
          | _ => { ...s, toEngine: rest,
              toSession: s.toSession.append({ id: req.id, map: Some(m), stage: None, ba: None, planeFound: false }) }
        },
      }
    },
  }

  // ------------------------------------------------------------ the system
  action init = s' = initial(ALL_FIXES)
  action initNoOrderFix = s' = initial({ ...ALL_FIXES, order: false })
  action initNoAnchorPointsFix = s' = initial({ ...ALL_FIXES, anchorPoints: false })
  action initTwoInFlight = s' = initial({ ...ALL_FIXES, oneInFlight: false })

  action step = any {
    sendFrame,
    engineProcesses,
    handleResult,
  }

  // ------------------------------------------------------------ properties
  /// After a result is handled, the anchor (if any) is attached to that
  /// result's map and reflects exactly the corrections that map had after the
  /// frame: none missed, none twice, in the engine's order.
  val anchorFollowsMap = match s.anchor {
    | None => true
    | Some(a) => match s.lastHandled {
      | None => false
      | Some(r) => match r.map {
        | None => false
        | Some(m) => a.mapId == m.id and a.view == m.corr
      }
    }
  }

  /// A correction is never fitted around anchor points of another map or of
  /// an earlier state of this one.
  val anchorPointsFresh = not(s.staleAnchorPoints)

  val safety = and {
    anchorFollowsMap,
    anchorPointsFresh,
  }

  // ---------------------------------------------- the model is not vacuous
  observation anchorAfterTwoCorrections = match s.anchor {
    | None => false
    | Some(a) => a.view.length() >= 2
  }
  observation stageAndBaInOneFrame = s.toSession.length() > 0 and s.anchor != None
    and s.toSession.head().stage != None and s.toSession.head().ba != None
  observation worldDroppedAndRecreated = s.worlds >= 2
  observation mapIdReused = s.worlds >= 2 and (match s.anchor {
    | None => false
    | Some(a) => match s.map {
      | None => false
      | Some(m) => a.mapId == m.id and m.corr.length() == 0 and s.corrections > 0
    }
  })
}
