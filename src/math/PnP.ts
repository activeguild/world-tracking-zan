import { solveLinearSystem, type Mat3, mat3Multiply } from "./Matrix";
import { rotationAxisAngle, type RigidTransform } from "./Pose";

/**
 * Motion-only pose refinement (PnP from a prior) in normalized camera
 * coordinates.
 *
 * Minimizes Σ ρ(‖π(R X_i + t) − x_i‖²) over (R, t) with Levenberg–Marquardt
 * and a Huber robust kernel, starting from `prior`. This is the standard
 * "track against the local map" step: with a frame-to-frame prior the
 * problem is well conditioned for both general and planar point sets, which
 * a closed-form DLT is not.
 *
 * Convention: X_cam = R · X_map + t.
 */
export interface PnPOptions {
  /** Huber threshold on the reprojection error, normalized coordinates. */
  huber: number;
  /** Final inlier gate on the reprojection error, normalized coordinates. */
  inlierThreshold: number;
  maxIterations: number;
  /** Stop when the parameter update is below this. */
  epsilon: number;
}

export interface PnPResult {
  pose: RigidTransform;
  inliers: Uint8Array;
  inlierCount: number;
  /** Mean reprojection error of inliers (normalized coordinates). */
  meanError: number;
  iterations: number;
  converged: boolean;
}

const JTJ = new Float64Array(36);
const JTr = new Float64Array(6);
const step = new Float64Array(6);

/**
 * @param points3 map-frame 3D points (x,y,z interleaved)
 * @param obsX/obsY normalized observations (x/z, y/z)
 * @param mask optional: points with mask[i] === 0 are ignored
 */
export function refinePosePnP(
  prior: RigidTransform,
  points3: Float64Array | Float32Array,
  obsX: ArrayLike<number>,
  obsY: ArrayLike<number>,
  n: number,
  opts: PnPOptions,
  mask: Uint8Array | null = null,
): PnPResult {
  let r: Mat3 = Float64Array.from(prior.rotation);
  let t: Float64Array = Float64Array.from(prior.translation);
  const huberSq = opts.huber * opts.huber;
  let lambda = 1e-3;
  let prevCost = Number.POSITIVE_INFINITY;
  let converged = false;
  let iter = 0;

  const xc = new Float64Array(3);
  for (; iter < opts.maxIterations; iter++) {
    JTJ.fill(0);
    JTr.fill(0);
    let cost = 0;
    let used = 0;
    for (let i = 0; i < n; i++) {
      if (mask && mask[i] === 0) continue;
      const X = points3[i * 3];
      const Y = points3[i * 3 + 1];
      const Z = points3[i * 3 + 2];
      xc[0] = r[0] * X + r[1] * Y + r[2] * Z + t[0];
      xc[1] = r[3] * X + r[4] * Y + r[5] * Z + t[1];
      xc[2] = r[6] * X + r[7] * Y + r[8] * Z + t[2];
      const z = xc[2];
      if (z <= 1e-6) continue;
      const iz = 1 / z;
      const u = xc[0] * iz;
      const v = xc[1] * iz;
      const ex = u - obsX[i];
      const ey = v - obsY[i];
      const e2 = ex * ex + ey * ey;
      // Huber weight
      let w = 1;
      if (e2 > huberSq) {
        const e = Math.sqrt(e2);
        w = opts.huber / e;
        cost += opts.huber * (2 * e - opts.huber);
      } else {
        cost += e2;
      }
      used++;

      // Jacobian of (u, v) w.r.t. [δω (3), δt (3)] with R ← exp(δω) R:
      //   ∂π/∂Xc = [[iz, 0, -u·iz], [0, iz, -v·iz]]
      //   ∂Xc/∂δω = -[Xc]×,  ∂Xc/∂δt = I
      const x = xc[0], y = xc[1];
      // ∂Xc/∂δω = -[Xc]× = [[0, z, -y], [-z, 0, x], [y, -x, 0]]
      // row u:
      const ju0 = iz * 0 + 0 * -z + -u * iz * y;      // d u / d ω_x
      const ju1 = iz * z + 0 * 0 + -u * iz * -x;      // d u / d ω_y
      const ju2 = iz * -y + 0 * x + -u * iz * 0;      // d u / d ω_z
      const ju3 = iz, ju4 = 0, ju5 = -u * iz;
      // row v:
      const jv0 = 0 * 0 + iz * -z + -v * iz * y;
      const jv1 = 0 * z + iz * 0 + -v * iz * -x;
      const jv2 = 0 * -y + iz * x + -v * iz * 0;
      const jv3 = 0, jv4 = iz, jv5 = -v * iz;
      const ju = [ju0, ju1, ju2, ju3, ju4, ju5];
      const jv = [jv0, jv1, jv2, jv3, jv4, jv5];
      for (let a = 0; a < 6; a++) {
        JTr[a] += w * (ju[a] * ex + jv[a] * ey);
        for (let b = 0; b < 6; b++) JTJ[a * 6 + b] += w * (ju[a] * ju[b] + jv[a] * jv[b]);
      }
    }
    if (used < 3) break;

    // Levenberg–Marquardt damping and solve.
    const A = Float64Array.from(JTJ);
    for (let a = 0; a < 6; a++) A[a * 6 + a] *= 1 + lambda;
    for (let a = 0; a < 6; a++) step[a] = -JTr[a];
    if (!solveLinearSystem(A, step, 6)) {
      lambda *= 10;
      if (lambda > 1e6) break;
      continue;
    }
    // Candidate update
    const dw = Math.hypot(step[0], step[1], step[2]);
    const dR = dw > 1e-12 ? rotationAxisAngle([step[0] / dw, step[1] / dw, step[2] / dw], dw) : null;
    const rNew = dR ? mat3Multiply(dR, r) : Float64Array.from(r);
    const tNew = new Float64Array([t[0] + step[3], t[1] + step[4], t[2] + step[5]]);
    const newCost = evaluateCost(rNew, tNew, points3, obsX, obsY, n, mask, opts.huber);
    if (newCost < cost) {
      r = rNew;
      t = tNew;
      lambda = Math.max(1e-9, lambda / 3);
      const upd = Math.hypot(step[0], step[1], step[2], step[3], step[4], step[5]);
      if (upd < opts.epsilon || Math.abs(prevCost - newCost) < 1e-12) {
        converged = true;
        iter++;
        break;
      }
      prevCost = newCost;
    } else {
      lambda *= 10;
      if (lambda > 1e6) break;
    }
  }

  // Classify inliers with the final pose.
  const inliers = new Uint8Array(n);
  let count = 0;
  let errSum = 0;
  const thrSq = opts.inlierThreshold * opts.inlierThreshold;
  for (let i = 0; i < n; i++) {
    if (mask && mask[i] === 0) continue;
    const X = points3[i * 3], Y = points3[i * 3 + 1], Z = points3[i * 3 + 2];
    const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
    if (z <= 1e-6) continue;
    const u = (r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z;
    const v = (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z;
    const e2 = (u - obsX[i]) ** 2 + (v - obsY[i]) ** 2;
    if (e2 < thrSq) {
      inliers[i] = 1;
      count++;
      errSum += Math.sqrt(e2);
    }
  }
  return {
    pose: { rotation: r as Mat3, translation: t },
    inliers,
    inlierCount: count,
    meanError: count ? errSum / count : 0,
    iterations: iter,
    converged,
  };
}

function evaluateCost(
  r: Float64Array,
  t: Float64Array,
  points3: Float64Array | Float32Array,
  obsX: ArrayLike<number>,
  obsY: ArrayLike<number>,
  n: number,
  mask: Uint8Array | null,
  huber: number,
): number {
  const huberSq = huber * huber;
  let cost = 0;
  for (let i = 0; i < n; i++) {
    if (mask && mask[i] === 0) continue;
    const X = points3[i * 3], Y = points3[i * 3 + 1], Z = points3[i * 3 + 2];
    const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
    if (z <= 1e-6) {
      cost += huber * huber * 4; // behind the camera: heavy penalty
      continue;
    }
    const u = (r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z;
    const v = (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z;
    const e2 = (u - obsX[i]) ** 2 + (v - obsY[i]) ** 2;
    if (e2 > huberSq) {
      const e = Math.sqrt(e2);
      cost += huber * (2 * e - huber);
    } else {
      cost += e2;
    }
  }
  return cost;
}

/** Project a map point with a pose; returns false if behind the camera. */
export function projectPoint(pose: RigidTransform, X: number, Y: number, Z: number, out: Float64Array): boolean {
  const r = pose.rotation;
  const t = pose.translation;
  const z = r[6] * X + r[7] * Y + r[8] * Z + t[2];
  if (z <= 1e-6) return false;
  out[0] = (r[0] * X + r[1] * Y + r[2] * Z + t[0]) / z;
  out[1] = (r[3] * X + r[4] * Y + r[5] * Z + t[1]) / z;
  out[2] = z;
  return true;
}
