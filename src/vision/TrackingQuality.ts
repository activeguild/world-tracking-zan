/**
 * Per-frame tracking quality (spec §32).
 *
 * Phase 1 fills featureCount / trackedCount / inlierCount / trackingConfidence.
 * reprojectionError, poseDelta and planeConfidence are reserved for later
 * phases and stay 0 so the type does not change when they arrive.
 */
export interface TrackingQuality {
  /** Features present after this frame (tracked inliers + replenished). */
  featureCount: number;
  /** Features successfully tracked from the previous frame (before RANSAC). */
  trackedCount: number;
  /** Tracked features that survived RANSAC. */
  inlierCount: number;
  /** Mean RANSAC reprojection error of inliers (pixels). */
  reprojectionError: number;
  /** Reserved (Phase 2). */
  poseDelta: number;
  /** Reserved (Phase 3). */
  planeConfidence: number;
  /** 0–1 heuristic confidence of frame-to-frame tracking. */
  trackingConfidence: number;
  /** True when too few features are available (LOW_FEATURE, spec §50). */
  lowFeature: boolean;
}

export function emptyQuality(): TrackingQuality {
  return {
    featureCount: 0,
    trackedCount: 0,
    inlierCount: 0,
    reprojectionError: 0,
    poseDelta: 0,
    planeConfidence: 0,
    trackingConfidence: 0,
    lowFeature: true,
  };
}

/**
 * Combine counts into a 0–1 confidence.
 *
 *   inlier ratio (inliers / previous features) × saturation(inliers / target)
 */
export function computeTrackingConfidence(
  inlierCount: number,
  previousCount: number,
  targetCount: number,
): number {
  if (previousCount <= 0 || targetCount <= 0) return 0;
  const ratio = Math.min(1, inlierCount / previousCount);
  const saturation = Math.min(1, inlierCount / targetCount);
  return ratio * saturation;
}
