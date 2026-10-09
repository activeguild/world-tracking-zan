/**
 * Sparse 3D landmark map (spec §19, §48).
 *
 * Positions are in the map frame: the camera frame of the frame in which the
 * map was initialized, with the initialization baseline as the unit of
 * length (monocular: scale-free until the plane fixes it).
 */
export interface Landmark {
  id: number;
  /** Position in the map frame. */
  position: Float64Array;
  /** Number of frames in which the landmark was observed as a PnP inlier. */
  observations: number;
  firstFrame: number;
  lastSeenFrame: number;
  /** Track currently observing this landmark (-1 when the track died). */
  trackId: number;
  /** Consecutive frames the landmark was a PnP outlier. */
  outlierCount: number;
  /** Set by the plane detector for debug rendering. */
  planeInlier: boolean;
  /**
   * Anchor observation used for (re-)triangulation: camera pose and pixel
   * of the first view. Null for landmarks without a usable anchor.
   */
  anchorPose: { rotation: Float64Array; translation: Float64Array } | null;
  anchorX: number;
  anchorY: number;
  /** Ray parallax angle (radians) of the last triangulation. */
  parallax: number;
  /**
   * Last known image position (processing pixels) and how many frames ago it
   * was observed directly (-1 = unknown). While the landmark has no track the
   * position is carried along by the frame-to-frame image motion so that it
   * can be re-linked to a re-detected corner without a camera pose.
   */
  lastX: number;
  lastY: number;
  imageAge: number;
  /**
   * Phase 7 (v18): the keyframe that was the newest when this landmark was
   * triangulated (-1 = none yet). A landmark no stored keyframe observes
   * cannot enter the bundle adjustment; when that keyframe's pose is
   * corrected the landmark is carried along with it instead, so the whole
   * map stays one frame (ORB-SLAM's reference keyframe).
   */
  refKeyframeId: number;
  /** How the last accepted bundle adjustment treated this landmark (diagnostics). */
  baMode: LandmarkBaMode;
}

/** `adjusted` = solved in the BA, `propagated` = moved with its reference keyframe, `none` = untouched. */
export type LandmarkBaMode = "none" | "adjusted" | "propagated";

export class LandmarkMap {
  private readonly landmarks = new Map<number, Landmark>();
  private nextId = 1;

  get size(): number {
    return this.landmarks.size;
  }

  add(position: ArrayLike<number>, trackId: number, frameId: number): Landmark {
    const lm: Landmark = {
      id: this.nextId++,
      position: new Float64Array([position[0], position[1], position[2]]),
      observations: 1,
      firstFrame: frameId,
      lastSeenFrame: frameId,
      trackId,
      outlierCount: 0,
      planeInlier: false,
      anchorPose: null,
      anchorX: 0,
      anchorY: 0,
      parallax: 0,
      lastX: 0,
      lastY: 0,
      imageAge: -1,
      refKeyframeId: -1,
      baMode: "none",
    };
    this.landmarks.set(lm.id, lm);
    return lm;
  }

  get(id: number): Landmark | undefined {
    return this.landmarks.get(id);
  }

  remove(id: number): void {
    this.landmarks.delete(id);
  }

  values(): IterableIterator<Landmark> {
    return this.landmarks.values();
  }

  clear(): void {
    this.landmarks.clear();
  }

  /**
   * Drop landmarks not seen for `maxAge` frames and, if still above
   * `maxCount`, the least recently seen ones (spec §48).
   *
   * `protectedIds` (v15): landmarks a stored keyframe observes are exempt
   * from both rules — a keyframe whose landmarks have been pruned can no
   * longer relocalize (on device a keyframe matched at NCC 0.91 had 26 of
   * its observations left after 5 s out of view). `maxCount` applies to the
   * *unprotected* landmarks: on device the protected set alone filled the
   * 1000 cap, the cap then removed the current view's fresh landmarks and
   * blocked new triangulation (PnP fell to 32 inliers while exploring). The
   * protected set is bounded by the keyframe store (maxKeyframes ×
   * observations per keyframe).
   */
  prune(frameId: number, maxAge: number, maxCount: number, protectedIds: ReadonlySet<number> | null = null): number {
    let removed = 0;
    const unprotected: Landmark[] = [];
    for (const lm of this.landmarks.values()) {
      if (protectedIds && protectedIds.has(lm.id)) continue;
      if (frameId - lm.lastSeenFrame > maxAge) {
        this.landmarks.delete(lm.id);
        removed++;
      } else {
        unprotected.push(lm);
      }
    }
    if (unprotected.length > maxCount) {
      unprotected.sort((a, b) => a.lastSeenFrame - b.lastSeenFrame);
      const excess = unprotected.length - maxCount;
      for (let i = 0; i < excess; i++) {
        this.landmarks.delete(unprotected[i].id);
        removed++;
      }
    }
    return removed;
  }

  /** Landmarks not in `protectedIds` (the set `maxLandmarks` bounds). */
  countUnprotected(protectedIds: ReadonlySet<number> | null): number {
    if (!protectedIds) return this.landmarks.size;
    let n = 0;
    for (const id of this.landmarks.keys()) if (!protectedIds.has(id)) n++;
    return n;
  }

  /**
   * Copy positions into a flat array (x,y,z interleaved) with the matching
   * ids. Landmarks with fewer than `minObservations` observations, not seen
   * as a PnP inlier within `maxAgeFrames` of `frameId`, or with a non-finite
   * position are skipped (plane seed conditions, v11 §10).
   */
  collect(minObservations = 1, maxAgeFrames = Number.POSITIVE_INFINITY, frameId = 0): { points: Float64Array; ids: number[] } {
    const ids: number[] = [];
    const pts: number[] = [];
    for (const lm of this.landmarks.values()) {
      if (lm.observations < minObservations) continue;
      if (frameId - lm.lastSeenFrame > maxAgeFrames) continue;
      const p = lm.position;
      if (!Number.isFinite(p[0] + p[1] + p[2])) continue;
      ids.push(lm.id);
      pts.push(p[0], p[1], p[2]);
    }
    return { points: Float64Array.from(pts), ids };
  }

  /** Landmarks seen as a PnP inlier within `maxAgeFrames` of `frameId` (any observation count). */
  countRecent(frameId: number, maxAgeFrames: number): number {
    let n = 0;
    for (const lm of this.landmarks.values()) if (frameId - lm.lastSeenFrame <= maxAgeFrames) n++;
    return n;
  }
}
