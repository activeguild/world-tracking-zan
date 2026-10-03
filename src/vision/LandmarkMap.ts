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
}

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
   */
  prune(frameId: number, maxAge: number, maxCount: number): number {
    let removed = 0;
    for (const lm of this.landmarks.values()) {
      if (frameId - lm.lastSeenFrame > maxAge) {
        this.landmarks.delete(lm.id);
        removed++;
      }
    }
    if (this.landmarks.size > maxCount) {
      const sorted = [...this.landmarks.values()].sort((a, b) => a.lastSeenFrame - b.lastSeenFrame);
      const excess = this.landmarks.size - maxCount;
      for (let i = 0; i < excess; i++) {
        this.landmarks.delete(sorted[i].id);
        removed++;
      }
    }
    return removed;
  }

  /**
   * Copy positions into a flat array (x,y,z interleaved) with the matching
   * ids. Landmarks with fewer than `minObservations` observations are skipped.
   */
  collect(minObservations = 1, maxAgeFrames = Number.POSITIVE_INFINITY, frameId = 0): { points: Float64Array; ids: number[] } {
    const ids: number[] = [];
    const pts: number[] = [];
    for (const lm of this.landmarks.values()) {
      if (lm.observations < minObservations) continue;
      if (frameId - lm.lastSeenFrame > maxAgeFrames) continue;
      ids.push(lm.id);
      pts.push(lm.position[0], lm.position[1], lm.position[2]);
    }
    return { points: Float64Array.from(pts), ids };
  }
}
