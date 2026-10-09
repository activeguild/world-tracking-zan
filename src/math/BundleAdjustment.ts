import { mat3Invert, mat3Multiply, solveLinearSystem, type Mat3 } from "./Matrix";
import { rotationAxisAngle, rotationDistance, type RigidTransform } from "./Pose";

/**
 * Local bundle adjustment (Phase 7, spec §37): jointly refine keyframe poses
 * and landmark positions so that every keyframe observation is explained by
 * one consistent map.
 *
 * Minimizes Σ ρ(‖π(R_k X_i + t_k) − x_ki‖²) over the free poses (δω, δt per
 * keyframe, R ← exp(δω) R) and the landmark positions with
 * Levenberg–Marquardt and a Huber kernel, in normalized camera coordinates.
 * The pose block is kept dense (≤ 8 keyframes → ≤ 48 parameters) and the
 * landmarks are eliminated with the Schur complement (3×3 blocks), so one
 * iteration is linear in the number of observations.
 *
 * Gauge: at least one keyframe must be fixed (position + rotation). The
 * monocular scale stays free in the normal equations (damping handles the
 * null direction) and is re-normalized afterwards so that the median depth
 * of the landmarks seen from the first fixed keyframe is unchanged.
 *
 * Convention: X_cam = R · X_map + t.
 */
export interface BAKeyframe {
  pose: RigidTransform;
  fixed: boolean;
}

export interface BAObservation {
  /** Index into `keyframes`. */
  keyframe: number;
  /** Index into the landmark array (position = landmarks[3i..3i+2]). */
  landmark: number;
  /** Normalized observation (x/z, y/z). */
  x: number;
  y: number;
}

export interface BAProblem {
  keyframes: BAKeyframe[];
  /** Landmark positions, x,y,z interleaved; refined in place. */
  landmarks: Float64Array;
  landmarkCount: number;
  observations: BAObservation[];
  /**
   * Landmark indices whose median depth in the first fixed keyframe defines
   * the scale gauge (default: all landmarks). Pass the landmarks that
   * keyframe observes so the gauge does not drift as the optimized set
   * changes from run to run.
   */
  gaugeLandmarks?: ArrayLike<number>;
  /**
   * Weak priors that anchor the poorly constrained directions (keyframes
   * related by little translation leave a landmark's depth almost free, and
   * the solve would otherwise slide it along its rays and drag the poses
   * with it). A landmark that moves `landmarkPriorSigma[i]` map units from
   * its start, or a free pose that moves `posePriorTranslationSigma` /
   * rotates `posePriorRotationSigma` radians, costs as much as one
   * observation off by `opts.huber`. 0 / undefined = no prior.
   */
  landmarkPriorSigma?: Float64Array;
  posePriorTranslationSigma?: number;
  posePriorRotationSigma?: number;
}

export interface BAOptions {
  /** Huber threshold (normalized coordinates); also the unit the priors are scaled to. */
  huber: number;
  maxIterations: number;
  /** Stop when the parameter update norm is below this. */
  epsilon: number;
}

/** Rotation vector of R · R0ᵀ (small-angle log map; exact axis, exact angle). */
function rotationVectorBetween(r: Mat3, r0: Mat3, out: Float64Array): void {
  // d = R R0ᵀ
  const d = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) d[i * 3 + j] = r[i * 3] * r0[j * 3] + r[i * 3 + 1] * r0[j * 3 + 1] + r[i * 3 + 2] * r0[j * 3 + 2];
  const tr = d[0] + d[4] + d[8];
  const angle = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
  const ax = d[7] - d[5], ay = d[2] - d[6], az = d[3] - d[1];
  const n = Math.hypot(ax, ay, az);
  if (angle < 1e-9 || n < 1e-12) {
    out[0] = out[1] = out[2] = 0;
    return;
  }
  out[0] = (ax / n) * angle;
  out[1] = (ay / n) * angle;
  out[2] = (az / n) * angle;
}

export interface BAResult {
  iterations: number;
  converged: boolean;
  /** Mean observation error before / after (normalized coordinates, all observations). */
  errorBefore: number;
  errorAfter: number;
  /** Largest landmark displacement (map units). */
  maxLandmarkShift: number;
  /** Largest free-keyframe camera-center displacement (map units) and rotation (degrees). */
  maxPoseShift: number;
  maxPoseRotationDeg: number;
  freeKeyframes: number;
}

function cameraCenter(p: RigidTransform): number[] {
  const r = p.rotation, t = p.translation;
  return [-(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]), -(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]), -(r[2] * t[0] + r[5] * t[1] + r[8] * t[2])];
}

function huberCost(e2: number, h: number): number {
  if (e2 <= h * h) return e2;
  const e = Math.sqrt(e2);
  return h * (2 * e - h);
}

/** Mean reprojection error (normalized units) of all observations with the given state. */
export function bundleError(problem: BAProblem, poses: RigidTransform[] = problem.keyframes.map((k) => k.pose), landmarks = problem.landmarks): number {
  let sum = 0;
  let n = 0;
  for (const o of problem.observations) {
    const p = poses[o.keyframe];
    const r = p.rotation, t = p.translation;
    const X = landmarks[o.landmark * 3], Y = landmarks[o.landmark * 3 + 1], Z = landmarks[o.landmark * 3 + 2];
    const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
    if (z <= 1e-6) continue;
    const u = (r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z;
    const v = (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z;
    sum += Math.hypot(u - o.x, v - o.y);
    n++;
  }
  return n ? sum / n : 0;
}

/** Prior weights in units of (observation σ = huber)², see `BAProblem`. */
function priorWeights(problem: BAProblem, huber: number): { lm: Float64Array | null; wt: number; ww: number } {
  const lm = problem.landmarkPriorSigma ? new Float64Array(problem.landmarkCount) : null;
  if (lm && problem.landmarkPriorSigma) {
    for (let i = 0; i < problem.landmarkCount; i++) {
      const s = problem.landmarkPriorSigma[i];
      lm[i] = s > 0 ? (huber / s) ** 2 : 0;
    }
  }
  const st = problem.posePriorTranslationSigma ?? 0;
  const sw = problem.posePriorRotationSigma ?? 0;
  return { lm, wt: st > 0 ? (huber / st) ** 2 : 0, ww: sw > 0 ? (huber / sw) ** 2 : 0 };
}

function priorCost(
  problem: BAProblem,
  poses: RigidTransform[],
  landmarks: Float64Array,
  startPoses: RigidTransform[],
  startLandmarks: Float64Array,
  w: { lm: Float64Array | null; wt: number; ww: number },
  rv: Float64Array,
): number {
  let cost = 0;
  if (w.lm) {
    for (let i = 0; i < problem.landmarkCount; i++) {
      if (w.lm[i] <= 0) continue;
      const dx = landmarks[i * 3] - startLandmarks[i * 3];
      const dy = landmarks[i * 3 + 1] - startLandmarks[i * 3 + 1];
      const dz = landmarks[i * 3 + 2] - startLandmarks[i * 3 + 2];
      cost += w.lm[i] * (dx * dx + dy * dy + dz * dz);
    }
  }
  if (w.wt > 0 || w.ww > 0) {
    for (let k = 0; k < poses.length; k++) {
      if (problem.keyframes[k].fixed) continue;
      if (w.wt > 0) {
        const t = poses[k].translation, t0 = startPoses[k].translation;
        cost += w.wt * ((t[0] - t0[0]) ** 2 + (t[1] - t0[1]) ** 2 + (t[2] - t0[2]) ** 2);
      }
      if (w.ww > 0) {
        rotationVectorBetween(poses[k].rotation, startPoses[k].rotation, rv);
        cost += w.ww * (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
      }
    }
  }
  return cost;
}

function totalCost(problem: BAProblem, poses: RigidTransform[], landmarks: Float64Array, huber: number): number {
  let cost = 0;
  for (const o of problem.observations) {
    const p = poses[o.keyframe];
    const r = p.rotation, t = p.translation;
    const X = landmarks[o.landmark * 3], Y = landmarks[o.landmark * 3 + 1], Z = landmarks[o.landmark * 3 + 2];
    const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
    if (z <= 1e-6) {
      cost += huber * huber * 4;
      continue;
    }
    const u = (r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z;
    const v = (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z;
    cost += huberCost((u - o.x) ** 2 + (v - o.y) ** 2, huber);
  }
  return cost;
}

export function bundleAdjust(problem: BAProblem, opts: BAOptions): BAResult {
  const kfs = problem.keyframes;
  const nL = problem.landmarkCount;
  const obs = problem.observations;
  // Free pose parameter offsets.
  const poseOffset = new Int32Array(kfs.length).fill(-1);
  let P = 0;
  for (let k = 0; k < kfs.length; k++) if (!kfs[k].fixed) poseOffset[k] = (P += 6) - 6;
  const freeKeyframes = P / 6;
  const errorBefore = bundleError(problem);
  const startPoses = kfs.map((k) => ({ rotation: Float64Array.from(k.pose.rotation), translation: Float64Array.from(k.pose.translation) }));
  const startLandmarks = Float64Array.from(problem.landmarks.subarray(0, nL * 3));
  const result: BAResult = {
    iterations: 0,
    converged: false,
    errorBefore,
    errorAfter: errorBefore,
    maxLandmarkShift: 0,
    maxPoseShift: 0,
    maxPoseRotationDeg: 0,
    freeKeyframes,
  };
  if (obs.length === 0 || nL === 0 || (P === 0 && nL === 0)) return result;

  // Working state.
  let poses: RigidTransform[] = kfs.map((k) => ({ rotation: Float64Array.from(k.pose.rotation), translation: Float64Array.from(k.pose.translation) }));
  let landmarks = Float64Array.from(problem.landmarks.subarray(0, nL * 3));

  // Normal-equation blocks.
  const Hpp = new Float64Array(P * P);
  const bp = new Float64Array(P);
  const Hll = new Float64Array(nL * 9);
  const bl = new Float64Array(nL * 3);
  // One 6×3 pose–landmark block per observation of a free keyframe.
  const Hpl = new Float64Array(obs.length * 18);
  // Landmark → observations index (for the Schur complement).
  const obsByLandmark: number[][] = Array.from({ length: nL }, () => []);
  for (let i = 0; i < obs.length; i++) obsByLandmark[obs[i].landmark].push(i);

  const S = new Float64Array(P * P);
  const g = new Float64Array(P);
  const dp = new Float64Array(P);
  const HllInv = new Float64Array(9);
  const tmp3 = new Float64Array(3);
  const ju = new Float64Array(9); // [δω(3), δt(3), δX(3)]
  const jv = new Float64Array(9);

  const prior = priorWeights(problem, opts.huber);
  const rv = new Float64Array(3);
  const costOf = (p: RigidTransform[], l: Float64Array): number =>
    totalCost(problem, p, l, opts.huber) + priorCost(problem, p, l, startPoses, startLandmarks, prior, rv);
  let lambda = 1e-3;
  let cost = costOf(poses, landmarks);
  const h2 = opts.huber * opts.huber;
  let iter = 0;
  for (; iter < opts.maxIterations; iter++) {
    Hpp.fill(0);
    bp.fill(0);
    Hll.fill(0);
    bl.fill(0);
    Hpl.fill(0);
    // ---- Priors (Gauss–Newton terms of the quadratic penalties) ----
    if (prior.lm) {
      for (let i = 0; i < nL; i++) {
        const w = prior.lm[i];
        if (w <= 0) continue;
        for (let a = 0; a < 3; a++) {
          Hll[i * 9 + a * 3 + a] += w;
          bl[i * 3 + a] += w * (landmarks[i * 3 + a] - startLandmarks[i * 3 + a]);
        }
      }
    }
    if (prior.wt > 0 || prior.ww > 0) {
      for (let k = 0; k < kfs.length; k++) {
        const po = poseOffset[k];
        if (po < 0) continue;
        if (prior.ww > 0) {
          rotationVectorBetween(poses[k].rotation, startPoses[k].rotation, rv);
          for (let a = 0; a < 3; a++) {
            Hpp[(po + a) * P + po + a] += prior.ww;
            bp[po + a] += prior.ww * rv[a];
          }
        }
        if (prior.wt > 0) {
          for (let a = 0; a < 3; a++) {
            Hpp[(po + 3 + a) * P + po + 3 + a] += prior.wt;
            bp[po + 3 + a] += prior.wt * (poses[k].translation[a] - startPoses[k].translation[a]);
          }
        }
      }
    }
    // ---- Build the normal equations ----
    for (let oi = 0; oi < obs.length; oi++) {
      const o = obs[oi];
      const p = poses[o.keyframe];
      const r = p.rotation, t = p.translation;
      const li = o.landmark;
      const X = landmarks[li * 3], Y = landmarks[li * 3 + 1], Z = landmarks[li * 3 + 2];
      const xc = r[0] * X + r[1] * Y + r[2] * Z + t[0];
      const yc = r[3] * X + r[4] * Y + r[5] * Z + t[1];
      const zc = r[6] * X + r[7] * Y + r[8] * Z + t[2];
      if (zc <= 1e-6) continue;
      const iz = 1 / zc;
      const u = xc * iz, v = yc * iz;
      const ex = u - o.x, ey = v - o.y;
      const e2 = ex * ex + ey * ey;
      const w = e2 > h2 ? opts.huber / Math.sqrt(e2) : 1;
      // ∂π/∂Xc = [[iz, 0, -u·iz], [0, iz, -v·iz]]; ∂Xc/∂δω = -[Xc]×; ∂Xc/∂δt = I; ∂Xc/∂X = R.
      // Pose part (as in refinePosePnP):
      ju[0] = -u * iz * yc; ju[1] = iz * zc + u * iz * xc; ju[2] = -iz * yc;
      ju[3] = iz; ju[4] = 0; ju[5] = -u * iz;
      jv[0] = -iz * zc - v * iz * yc; jv[1] = v * iz * xc; jv[2] = iz * xc;
      jv[3] = 0; jv[4] = iz; jv[5] = -v * iz;
      // Landmark part: [iz, 0, -u iz] · R and [0, iz, -v iz] · R.
      ju[6] = iz * r[0] - u * iz * r[6]; ju[7] = iz * r[1] - u * iz * r[7]; ju[8] = iz * r[2] - u * iz * r[8];
      jv[6] = iz * r[3] - v * iz * r[6]; jv[7] = iz * r[4] - v * iz * r[7]; jv[8] = iz * r[5] - v * iz * r[8];
      const po = poseOffset[o.keyframe];
      // Landmark block.
      for (let a = 0; a < 3; a++) {
        bl[li * 3 + a] += w * (ju[6 + a] * ex + jv[6 + a] * ey);
        for (let b = 0; b < 3; b++) Hll[li * 9 + a * 3 + b] += w * (ju[6 + a] * ju[6 + b] + jv[6 + a] * jv[6 + b]);
      }
      if (po >= 0) {
        for (let a = 0; a < 6; a++) {
          bp[po + a] += w * (ju[a] * ex + jv[a] * ey);
          for (let b = 0; b < 6; b++) Hpp[(po + a) * P + po + b] += w * (ju[a] * ju[b] + jv[a] * jv[b]);
          for (let b = 0; b < 3; b++) Hpl[oi * 18 + a * 3 + b] = w * (ju[a] * ju[6 + b] + jv[a] * jv[6 + b]);
        }
      }
    }
    // ---- Damping ----
    for (let a = 0; a < P; a++) Hpp[a * P + a] *= 1 + lambda;
    for (let i = 0; i < nL; i++) for (let a = 0; a < 3; a++) Hll[i * 9 + a * 3 + a] *= 1 + lambda;
    // ---- Schur complement: S = Hpp − Σ Hpl Hll⁻¹ Hplᵀ, g = bp − Σ Hpl Hll⁻¹ bl ----
    S.set(Hpp);
    g.set(bp);
    const dl = new Float64Array(nL * 3);
    let singular = false;
    for (let i = 0; i < nL; i++) {
      const list = obsByLandmark[i];
      if (list.length === 0) continue;
      const inv = mat3Invert(Hll.subarray(i * 9, i * 9 + 9) as Mat3, HllInv as Mat3);
      if (!inv) {
        singular = true;
        break;
      }
      // y = Hll⁻¹ bl
      for (let a = 0; a < 3; a++) tmp3[a] = HllInv[a * 3] * bl[i * 3] + HllInv[a * 3 + 1] * bl[i * 3 + 1] + HllInv[a * 3 + 2] * bl[i * 3 + 2];
      for (const oa of list) {
        const pa = poseOffset[obs[oa].keyframe];
        if (pa < 0) continue;
        const A = Hpl.subarray(oa * 18, oa * 18 + 18); // 6×3
        // g -= A y
        for (let a = 0; a < 6; a++) g[pa + a] -= A[a * 3] * tmp3[0] + A[a * 3 + 1] * tmp3[1] + A[a * 3 + 2] * tmp3[2];
        // S[pa, pb] -= A Hll⁻¹ Bᵀ for every observing free keyframe b (including a itself)
        for (const ob of list) {
          const pb = poseOffset[obs[ob].keyframe];
          if (pb < 0) continue;
          const B = Hpl.subarray(ob * 18, ob * 18 + 18);
          for (let a = 0; a < 6; a++) {
            // row a of A·Hll⁻¹
            const c0 = A[a * 3] * HllInv[0] + A[a * 3 + 1] * HllInv[3] + A[a * 3 + 2] * HllInv[6];
            const c1 = A[a * 3] * HllInv[1] + A[a * 3 + 1] * HllInv[4] + A[a * 3 + 2] * HllInv[7];
            const c2 = A[a * 3] * HllInv[2] + A[a * 3 + 1] * HllInv[5] + A[a * 3 + 2] * HllInv[8];
            for (let b = 0; b < 6; b++) S[(pa + a) * P + pb + b] -= c0 * B[b * 3] + c1 * B[b * 3 + 1] + c2 * B[b * 3 + 2];
          }
        }
      }
    }
    if (singular) {
      lambda *= 10;
      if (lambda > 1e8) break;
      continue;
    }
    // ---- Solve the reduced pose system ----
    if (P > 0) {
      for (let a = 0; a < P; a++) dp[a] = -g[a];
      if (!solveLinearSystem(S, dp, P)) {
        lambda *= 10;
        if (lambda > 1e8) break;
        continue;
      }
    }
    // ---- Back-substitute the landmarks: δl = Hll⁻¹ (−bl − Hplᵀ δp) ----
    for (let i = 0; i < nL; i++) {
      const list = obsByLandmark[i];
      if (list.length === 0) continue;
      tmp3[0] = -bl[i * 3];
      tmp3[1] = -bl[i * 3 + 1];
      tmp3[2] = -bl[i * 3 + 2];
      for (const oa of list) {
        const pa = poseOffset[obs[oa].keyframe];
        if (pa < 0) continue;
        const A = Hpl.subarray(oa * 18, oa * 18 + 18);
        for (let b = 0; b < 3; b++) {
          let s = 0;
          for (let a = 0; a < 6; a++) s += A[a * 3 + b] * dp[pa + a];
          tmp3[b] -= s;
        }
      }
      mat3Invert(Hll.subarray(i * 9, i * 9 + 9) as Mat3, HllInv as Mat3);
      for (let a = 0; a < 3; a++) dl[i * 3 + a] = HllInv[a * 3] * tmp3[0] + HllInv[a * 3 + 1] * tmp3[1] + HllInv[a * 3 + 2] * tmp3[2];
    }
    // ---- Candidate state ----
    const newPoses: RigidTransform[] = poses.map((p, k) => {
      const po = poseOffset[k];
      if (po < 0) return p;
      const dw = Math.hypot(dp[po], dp[po + 1], dp[po + 2]);
      const dR = dw > 1e-12 ? rotationAxisAngle([dp[po] / dw, dp[po + 1] / dw, dp[po + 2] / dw], dw) : null;
      return {
        rotation: dR ? mat3Multiply(dR, p.rotation) : Float64Array.from(p.rotation),
        translation: new Float64Array([p.translation[0] + dp[po + 3], p.translation[1] + dp[po + 4], p.translation[2] + dp[po + 5]]),
      };
    });
    const newLandmarks = Float64Array.from(landmarks);
    for (let i = 0; i < nL * 3; i++) newLandmarks[i] += dl[i];
    const newCost = costOf(newPoses, newLandmarks);
    if (newCost < cost) {
      let upd = 0;
      for (let a = 0; a < P; a++) upd += dp[a] * dp[a];
      for (let i = 0; i < nL * 3; i++) upd += dl[i] * dl[i];
      const rel = (cost - newCost) / Math.max(cost, 1e-18);
      poses = newPoses;
      landmarks = newLandmarks;
      cost = newCost;
      lambda = Math.max(1e-9, lambda / 3);
      if (Math.sqrt(upd) < opts.epsilon || rel < 1e-6) {
        result.converged = true;
        iter++;
        break;
      }
    } else {
      lambda *= 10;
      if (lambda > 1e8) break;
    }
  }
  result.iterations = iter;

  // ---- Scale gauge: keep the median depth from the first fixed keyframe ----
  const anchorIdx = kfs.findIndex((k) => k.fixed);
  if (anchorIdx >= 0 && nL > 0) {
    const gauge = problem.gaugeLandmarks && problem.gaugeLandmarks.length >= 10 ? Array.from(problem.gaugeLandmarks) : null;
    const medianDepth = (lm: Float64Array, p: RigidTransform): number => {
      const r = p.rotation, t = p.translation;
      const zs: number[] = [];
      const push = (i: number) => zs.push(r[6] * lm[i * 3] + r[7] * lm[i * 3 + 1] + r[8] * lm[i * 3 + 2] + t[2]);
      if (gauge) for (const i of gauge) push(i);
      else for (let i = 0; i < nL; i++) push(i);
      zs.sort((a, b) => a - b);
      return zs[zs.length >> 1];
    };
    const d0 = medianDepth(startLandmarks, startPoses[anchorIdx]);
    const d1 = medianDepth(landmarks, poses[anchorIdx]);
    if (d0 > 1e-9 && d1 > 1e-9 && Number.isFinite(d0 / d1)) {
      const s = d0 / d1;
      const C0 = cameraCenter(poses[anchorIdx]);
      for (let i = 0; i < nL; i++) for (let a = 0; a < 3; a++) landmarks[i * 3 + a] = C0[a] + s * (landmarks[i * 3 + a] - C0[a]);
      for (let k = 0; k < kfs.length; k++) {
        if (kfs[k].fixed) continue;
        const C = cameraCenter(poses[k]);
        const Cn = [C0[0] + s * (C[0] - C0[0]), C0[1] + s * (C[1] - C0[1]), C0[2] + s * (C[2] - C0[2])];
        const r = poses[k].rotation;
        poses[k].translation = new Float64Array([
          -(r[0] * Cn[0] + r[1] * Cn[1] + r[2] * Cn[2]),
          -(r[3] * Cn[0] + r[4] * Cn[1] + r[5] * Cn[2]),
          -(r[6] * Cn[0] + r[7] * Cn[1] + r[8] * Cn[2]),
        ]);
      }
    }
  }

  // ---- Write back and measure the shifts ----
  for (let i = 0; i < nL; i++) {
    const dx = landmarks[i * 3] - startLandmarks[i * 3];
    const dy = landmarks[i * 3 + 1] - startLandmarks[i * 3 + 1];
    const dz = landmarks[i * 3 + 2] - startLandmarks[i * 3 + 2];
    result.maxLandmarkShift = Math.max(result.maxLandmarkShift, Math.hypot(dx, dy, dz));
  }
  problem.landmarks.set(landmarks.subarray(0, nL * 3));
  for (let k = 0; k < kfs.length; k++) {
    if (kfs[k].fixed) continue;
    const C0 = cameraCenter(startPoses[k]);
    const C1 = cameraCenter(poses[k]);
    result.maxPoseShift = Math.max(result.maxPoseShift, Math.hypot(C1[0] - C0[0], C1[1] - C0[1], C1[2] - C0[2]));
    result.maxPoseRotationDeg = Math.max(result.maxPoseRotationDeg, (rotationDistance(startPoses[k].rotation, poses[k].rotation) * 180) / Math.PI);
    kfs[k].pose.rotation.set(poses[k].rotation);
    kfs[k].pose.translation.set(poses[k].translation);
  }
  result.errorAfter = bundleError(problem);
  return result;
}
