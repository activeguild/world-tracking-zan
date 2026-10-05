# WebAR 自前平面検出・Visual Tracking Engine — Claude Code 実装指示書

> このファイルはプロジェクトの正本となる仕様書です。実装・レビュー時は必ずこの指示書に従うこと。
> 現在の進捗は末尾の「実装状況」を参照。

## 0. 目的

ブラウザ上で動作する WebAR 向けの、独自の平面検出・カメラトラッキングエンジンを実装する。
8th Wall 等の汎用 AR エンジンの完全再現は目標にしない。

対象ユースケース:

- iOS Safari / Android Chrome
- `getUserMedia()` によるカメラ入力、単眼カメラ
- 主に床・机などの水平面
- 数秒〜30 秒程度の AR セッション、最大 2 つ程度の対象
- Three.js による 3D コンテンツ表示、GLB モデル配置
- カメラを動かしてもワールド空間上のオブジェクトを固定
- WebAssembly を利用した画像処理、Web Worker による Vision 処理分離

最初から完全な SLAM を作らない。まず **Visual Tracking + Plane Detection + World Coordinate + Three.js Integration** を成立させる。

## 1. 最終的なユーザー体験

カメラ起動 → 水平面探索 → 平面検出 → タップ → タップ位置を平面上の 3D 座標へ変換 → GLB 配置 → スマートフォンを動かす → カメラ Pose を継続推定 → GLB は同じ World 座標に固定。

## 2. 非目標（初期実装では行わない）

完全な汎用 SLAM / 大規模 3D Map / Loop Closure / 複雑な Relocalization / Face・Body Tracking / Sky Segmentation / Dense・Mesh Reconstruction / LiDAR・ARKit・ARCore・WebXR Plane Detection・8th Wall SDK 依存。
WebXR/ARKit/ARCore の平面検出を利用せず、自前の画像処理で実装する。

## 3. 基本アーキテクチャ

```
Camera → getUserMedia() → VideoFrame → Image Preprocessing
 → Feature Detection (FAST / ORB) → Feature Tracking (Pyramidal LK) → RANSAC
 → Homography / Essential Matrix → Camera Pose → 3D Landmark Map
 → Plane Fitting (RANSAC) → World Coordinate → Three.js Camera → GLB Model
```

## 4. 技術スタック

- Frontend: TypeScript / Three.js / Vite / WebGL
- Camera: `navigator.mediaDevices.getUserMedia()` / HTMLVideoElement / Canvas
- Computer Vision: FAST, ORB, Pyramidal Lucas-Kanade, RANSAC, Homography, Essential Matrix, recoverPose, Triangulation, Plane RANSAC（初期は OpenCV.js/WASM 利用可。独自 WASM へ移行可能な構造にする）
- Performance: Web Worker / WebAssembly / WASM SIMD / OffscreenCanvas

## 5. ディレクトリ構成

```
src/
├── ar/         ARSession.ts, ARState.ts, ARConfig.ts
├── camera/     CameraManager.ts, CameraFrame.ts, CameraIntrinsics.ts
├── vision/     FeatureDetector.ts, FeatureTracker.ts, PoseEstimator.ts,
│               LandmarkMap.ts, PlaneDetector.ts, TrackingQuality.ts
├── math/       Pose.ts, Plane.ts, Ray.ts, Matrix.ts, CoordinateSystem.ts
├── worker/     VisionWorker.ts, VisionWorkerClient.ts
├── rendering/  ARCamera.ts, ARWorld.ts, ARObject.ts
├── debug/      DebugOverlay.ts, FeatureRenderer.ts, PlaneRenderer.ts
└── main.ts
wasm/vision/
tests/  vision/ math/ plane/ integration/
```

## 6–7. 座標系

- World 座標は Three.js と整合させる（Y-up）。基本平面は `Y = 0`。
- Camera 座標は一般的な CV 座標系（X right, Y down, Z forward）。
- Three.js への座標変換はコード全体に分散させず **`CoordinateSystem.ts` に集約**する。

## 8. Camera Intrinsics

```ts
interface CameraIntrinsics { fx; fy; cx; cy; width; height }
```
初期段階は `fx ≈ fy ≈ width, cx = width/2, cy = height/2` の近似でよいが、端末ごとのカメラ特性を考慮できる設計にする。

## 9–10. Camera 入力 / Processing Frame Rate

- `facingMode: "environment"`。処理解像度は 640×480 または 640×360 程度。高解像度映像をそのまま Vision 処理しない。
- Camera は 30fps 以上でよい。Vision 処理は 15〜30fps、1 frame ≈ 33ms 以下（可能なら 15〜20ms）。

## 11–14. Feature Detection / Tracking / 失敗判定 / RANSAC

- 初期フレームでは FAST。100〜500 points。設定値は `FeatureConfig { maxFeatures; minFeatures; qualityLevel; minDistance }` として設定可能にする（固定値を埋め込まない）。
- 毎フレーム全特徴点を再検出しない。Pyramidal LK（640×480 / 320×240 / 160×120）で追跡。
- 各特徴点について LK status / tracking error / image boundary / **forward-backward error** を確認する。
- 特徴点対応には必ず外れ値除去（Essential Matrix + RANSAC または Homography + RANSAC）。平面が支配的な場合は Homography が有効。

## 15–18. Homography / Essential Matrix / Pose Recovery / Scale

- Homography `p' = H p`: 平面追跡、平面安定性評価、小さな移動推定。
- Essential Matrix `E = Kᵀ F K` から `R, t` を復元（recoverPose 相当）。
  `RelativePose { rotation: Matrix3; translationDirection: Vector3; inlierCount }`
- 単眼では translation の絶対スケールが求まらない。初期実装では scale-free とし、AR 開始時の Plane を World 座標の基準にする。

## 19–24. Landmark Map / Triangulation / Plane Detection

- `Landmark { id; position; descriptor?; observations; lastSeenFrame }`。Triangulation で 3D 化。深度が不安定な点は採用しない。
- 3D Landmark 群に RANSAC Plane Fitting (`ax + by + cz + d = 0`)。
  `Plane { normal; distance; center; inlierCount; areaEstimate; confidence }`
- 水平面判定: `abs(normal.y) > 0.90`（実機で調整）。
- Plane Confidence は inlier count + plane residual + horizontalness + temporal stability で評価し、複数フレーム安定した場合のみ `PLANE_FOUND` へ遷移。

## 25–30. World Origin / 座標生成 / Tap Placement / Three.js 統合

- 最初に確定した Plane center を World Origin、Plane normal を World +Y にする。
  `PlaneCoordinateSystem { origin; right; up; forward; matrix }`
- タップ: Screen → NDC → Camera Ray → World Ray → Ray ∩ Plane。`t = -(n·o + d) / (n·r)`、`n·r ≈ 0` は交差なし。
- Three.js 側は `ARCamera / ARWorld / ARObject` を分離。Camera Pose を `camera.position / quaternion` に反映し、GLB は World 座標に置く。
- **絶対にやらないこと**: `GLB position += camera delta`。必ず Camera Pose → World Coordinate → GLB 固定。

## 31–34. Tracking State / Quality / Lost / Smoothing

```
INITIALIZING → SEARCHING_FEATURES → TRACKING → PLANE_DETECTING → PLANE_FOUND
→ AR_ACTIVE → TRACKING_LOST → RELOCALIZING → TRACKING
```
- `TrackingQuality { featureCount; trackedCount; inlierCount; reprojectionError; poseDelta; planeConfidence; trackingConfidence }` を毎フレーム計算。
- `trackedCount < minimum` または `inlierCount < minimum` で Tracking Lost。いきなり AR オブジェクトを消さず最後の Pose を短時間保持。
- Pose smoothing（One Euro Filter 等）。過剰な平滑化で Latency を増やさない。

## 35–38. Keyframe / Relocalization / Bundle Adjustment / IMU（後期 Phase）

- Keyframe は Phase 2 以降。移動・回転・品質変化が閾値を超えたら作成。
- Relocalization: Feature Detection → Keyframe Matching → PnP / Homography / Pose Recovery。
- BA は Local Bundle Adjustment から（最近の Keyframe + 周辺 Landmark のみ）。
- IMU（Gyro / Accel）は将来的に。初期実装は Vision-only で動作すること。

## 39–41. Web Worker

Vision 処理を Main Thread で実行し続けない。Main: Three.js / UI / Camera、Worker: Vision Engine。
`VisionInput { frameId; timestamp; image; intrinsics }` / `VisionOutput { frameId; pose; plane?; trackingQuality }`。
毎フレーム大量の JS Object をコピーせず Transferable / SharedArrayBuffer / TypedArray / ImageBitmap を使う。

## 42–44. Debug

- Debug Mode 必須: FPS / Vision FPS / Feature Count / Tracked Count / Inlier Count / Plane Confidence / Tracking State / Pose を表示。
- 特徴点と移動方向（● ───→ ●）を描画。検出 Plane を Three.js 上に透明 Grid 等で可視化。

## 45–47. テスト / 実機 / パフォーマンス目標

Test 1 静止（jitter 小） / 2 左右移動（GLB 固定） / 3 前後移動（見かけサイズ変化） / 4 回転（World 不動） / 5 床 / 6 机（水平 Plane 検出） / 7 壁（採用しない） / 8 低テクスチャ（confidence 低下） / 9 モーションブラー（Lost 検出） / 10 30 秒 AR（大きな Drift なし）。
実機: iPhone Safari / Android Chrome（可能なら低・中性能 Android + 高性能 iPhone）。
目標: Camera 30fps+ / Vision 15〜30fps / Pose latency 100ms 以下 / Vision < 20ms/frame。

## 48–53. メモリ / Feature 管理 / Low Texture / 優先順位 / エラー

- Landmark 上限 500〜2000。古い Landmark は削除可能に。
- `tracked < target` または coverage が悪い場合に Feature を追加。Grid-based feature distribution で分散。
- 特徴点が少ない場合 `LOW_FEATURE` を返し、UI で「周囲をゆっくり動かしてください」等を表示できる設計に。
- 複数平面は Horizontal + Feature count + Area + Stability で選択。初期版は Active Plane 最大 1 つ。最大 2 対象だが Vision Map は共有。
- エラーコード: `CAMERA_PERMISSION_DENIED / CAMERA_UNAVAILABLE / INSUFFICIENT_FEATURES / TRACKING_LOST / PLANE_NOT_FOUND`。

## 54. API 設計

```ts
const ar = new ARSession({ canvas, video, threeScene, threeCamera });
await ar.start();
ar.on("planeFound", plane => {});
ar.on("trackingStateChanged", state => {});
const hit = ar.hitTest(x, y);
```
内部アルゴリズムを外部 API に露出させない。

## 55–66. Phase 計画

| Phase | 内容 | 完了条件 |
|---|---|---|
| 1 | Camera → Grayscale → FAST → LK → Forward-Backward → RANSAC → Debug Overlay | 300 程度の Feature から 100 以上を安定追跡、● → ● が正しく表示 |
| 2 | Essential Matrix → recoverPose → Relative Pose | 左右前後回転で Pose が連続的に変化 |
| 3 | Triangulation → 3D Landmarks → Plane RANSAC | 床・机で normal ≈ World Y、壁は不採用 |
| 4 | World Coordinate → Raycast → Plane Intersection → Three.js Cube/GLB 配置 | タップで配置、カメラを動かしても同じ場所 |
| 5 | Keyframe + Relocalization | 短時間 Lost から復帰 |
| 6 | IMU + Pose Fusion (Madgwick 等) | — |
| 7 | Local Bundle Adjustment (Ceres → WASM) | — |
| 8 | 性能最適化 (WASM SIMD / Worker / OffscreenCanvas / Transferable) | — |

## 67. 実装上の重要原則

1. 一度に全部実装しない。Phase → Test → Phase → Test。
2. アルゴリズムをブラックボックス化しない（Input / Processing / Output / Quality / Failure condition を明確に）。
3. Vision と Rendering を分離する。Vision は Pose / Plane / TrackingQuality だけを返す。
4. GLB を Vision 処理に依存させない。
5. 実機で成立しない最適化を先に行わない。まず iPhone Safari で成立させる。

## 68. 最初に作る Demo

Camera Start → Plane Detect → Tap → Cube/GLB Placement → Move Camera → Object remains fixed。最初は Cube でよい。

## 69–72. Claude Code への実装ルール

- 実装開始時に既存 Repository（package.json / tsconfig / vite.config / Three.js version / 既存 camera・WebGL・WASM コード）を調査し、不用意に破壊しない。
- 実装順序: Repository analysis → Camera → Grayscale → Feature detection → LK → RANSAC → Essential Matrix → Pose recovery → Triangulation → Plane detection → World coordinate → Ray/Plane hit test → Three.js integration → Cube placement → Tracking quality → Keyframe → Relocalization → IMU → BA → Performance optimization。
- 各 Step 終了時に TypeScript compile / Unit test / Browser test / Debug visualization / Performance measurement を行う。テスト失敗状態で次の Step へ進まない。
- Debug mode では `[AR] state=TRACKING features=283 tracked=219 inliers=184 planeConfidence=0.92 visionFPS=24` 形式のログ。Production では無効化可能に。

## 73–75. 完成条件 / 最重要事項

床・机を検出 / タップ位置に GLB 配置 / 動かしても World 固定 / 短時間 AR で大きな Jitter なし / iPhone Safari で実用的 FPS / 短時間 Lost から復帰。

> **最初の成功条件**: iPhone Safari 上で、机・床を検出し、タップした場所に Cube を配置し、スマートフォンを動かしても Cube が現実空間上の同じ場所に留まり続けること。この最小機能が成立してから GLB → Keyframe → Relocalization → IMU → BA へ進む。

**ユーザーの承認なしに次の Phase へ進まないこと。**

---

## 実装状況

### 修正指示書 v13 対応 — Lost / Relocalization 中の AR オブジェクト非表示（2026-10-05、実機確認待ち）

実機ログ（`State RELOCALIZING`、`Source LOST`、`Lost 2068ms`、`PnP 0i t HELD`、`Rot yaw +86.8 pitch -35.5 roll +73.6`）のように Camera Pose が信頼できない間も、`world.holdPoseOnLostMs`（10 s、v3 のデバッグ値）で古い Pose のまま Cube が表示され続けていた。**表示状態と World / Map / Anchor / Keyframe の状態を分離**し、オブジェクトは「最後に得た Pose」ではなく「現在の Pose が信頼できるか」に従わせる（§1、§39）。Map reset / World reset / Relocalization / Confirmation / v12 の Validation / PnP / LK / Jump Gate は変更していない（§13、§34）。

- **判定（§3–§11、§16）**: `src/ar/ObjectVisibility.ts` の純関数 `decideObjectVisibility(input) → { visible, reason }`。`reason: TRACKING_ACTIVE | TRACKING_LOST | RELOCALIZING | CONFIRMING | WORLD_NOT_READY | POSE_INVALID`。World 未確立 / 未生成 → `WORLD_NOT_READY`、`RELOCALIZING` → `RELOCALIZING`、`TRACKING_LOST` / `SEARCHING_FEATURES` / World 未確立相当の状態 → `TRACKING_LOST`、非有限 Pose → `POSE_INVALID`、`mapPose.relocalized`（再局所化の Apply フレーム + 事後監視 `postRelocMonitorFrames` の間）→ `CONFIRMING`、`PLANE_FOUND` / `AR_ACTIVE` で `framesSinceTracked === 0` → `TRACKING_ACTIVE`（表示）
- **ヒステリシス（§5、§9、§11）**: 新しい閾値は追加せず状態機械の `state.mapLostFrameTolerance`（3）を使う。表示中は PnP 失敗が許容フレーム内なら表示を維持（ちらつき防止）、**再表示は Map 由来の新しい Pose（`framesSinceTracked === 0`）のフレームだけ**（保持 Pose での再表示なし）。1 フレームだけの表示（§22）は `relocalized` ウィンドウで防ぐ: Apply フレームは state が WORLD_TRACKING に戻り Pose も新しいが `CONFIRMING` で非表示、監視 3 フレームが閉じた次のフレームで表示
- **ARWorld / ARObject（§12、§18–§20、§24）**: `ARWorld.updateTracking(tracking, nowMs)`（10 s 保持）を廃止し、`setObjectsVisible(visible)`（冪等、変化時のみ Three.js を触る）と `markTracking(tracking, nowMs)`（`lostDurationMs` 用）に分離。Transform は保持、`root`（World フレーム）は非表示にしない、平面グリッドは非表示に連動（HUD トグルの要求は `planeGridWanted` で記憶）。`ARObject` に `shown` を持たせ、非表示中に `place()` されたオブジェクトも非表示のまま。Render loop / Camera / Relocalization は継続
- **ARSession**: `updateWorld` で毎フレーム判定し `world.setObjectsVisible`。`ARStats.objectVisibility { visible, reason }`。ログ `OBJECTS hidden (RELOCALIZING) at frame N` は変化時のみ（debug 時）。`world.holdPoseOnLostMs` は未使用（互換のため設定キーは残す）
- **HUD（§15）**: `=== OBJECT ===` に `Visible YES|NO` / `Reason RELOCALIZING` を追加、非表示中の `ObjN` は `hidden`（Transform は保持しているが古い位置を見せない）。Debug 既定 OFF は維持
- テスト 210 件（+9）: `decideObjectVisibility` Test 1 / 6（WORLD_TRACKING + 新 Pose → 表示）、Test 2（LOST）、Test 3 / 7（RELOCALIZING、候補合格でも Apply 前は RELOCALIZING）、Test 4 / 5 / 13 / 14 / 15（Apply フレームと監視 3 フレームは `CONFIRMING`、strong / acceptable 共通、1 フレームの表示なし）、Test 9（許容 3 フレーム内の PnP 失敗は表示維持・保持 Pose での再表示なし）、Test 8 / §30（visible → lost → hidden → reloc → confirming → visible を 3 周）、ARWorld（Transform / root / 配置数の保持、冪等、非表示中の配置、非表示中もアニメーション継続 = Test 10–12）。`CameraObjectSeparation` は `new ARWorld()` に更新。ブラウザテスト 2 件合格
- **実機で読むべきもの（§27–§29、§40）**: 失探中に `OBJECT` 節が `Visible NO / Reason TRACKING_LOST → RELOCALIZING` となり Cube が消えること、再局所化 Apply 後に `CONFIRMING` を経て `Visible YES / TRACKING_ACTIVE` で元の位置に再表示されること、再度失探で再び消えること

### 修正指示書 v12 対応 — Relocalization Validation の Strong / Acceptable / Reject 化（2026-10-05、実機確認待ち）

実機ログ（`Lost 8443ms`、`KF15 50i 2.44px ncc 0.67`、`Inlier 50/25 OK`、`Error 2.44/1.50px NG`、`Ratio 0.85/0.50 OK`、`Cells 7/9 OK`、`VAL 0`）: 強い候補が再投影誤差の単一閾値だけで落ちていた。**strict 1.5 px は動かさず**、限定的な relaxed 範囲を追加し、複数の品質条件 + Relocalization 専用の Jump 検証 + 既存の確認 / 事後監視で安全性を担保する（§27）。**PnP / LK / NCC 粗ゲート / Plane / Map reset / Keyframe 生成 / 通常 Tracking の Jump Gate は変更していない**（§1、§24）。

- **3 段階（§3–§7, §17）**: `validateRelocalizationCandidate` が `level: strong | acceptable | reject` を返す。判定順序は §17 どおり pose → inliers → inlier ratio → spatial / coverage → NCC → translation jump → rotation jump → reprojection。`strong` = 誤差 ≤ `maxMeanErrorPx`（1.5、従来の合格）、`acceptable` = 1.5 超〜`relocalization.relaxedMeanErrorPx`（3.0）で**他条件すべて合格**、それ以外 `reject`。「○個 OK なら合格」方式ではない（§6）
- **acceptable の追加条件（§5）**: `relaxedMinMatchScore`（0.5）: relaxed 範囲の候補は粗 NCC がこれ以上であること（strong は粗ゲート `coarseMinScore` 0.25 のまま、§24-4 で NCC 全体は下げない）。Relocalization 専用 Jump（§9）: `maxTranslationJumpDepthRatio`（1.0 × Landmark 中央奥行き）/ `maxRotationJumpDeg`（90°）を保持姿勢との差に適用（strong / acceptable 共通）。長時間失探後の正しい復帰は保持姿勢から遠くてよい（v5 §9）ので意図的に緩く、「別の場所に置く」候補だけを落とす。v9 §16–§17 の「診断のみ」はこの専用上限で上書き（上限なし = 診断のみは `0` で可）。通常 Tracking の Jump Gate は再局所化に適用しない（従来どおり）
- **拒否理由（§8）**: `RelocValidationRejectReason` に `ncc` / `translation_jump` / `rotation_jump`。段階 `RelocalizationStage` に `ncc` / `jump`、`RelocalizationRejectCode` に `pose_jump`。relaxed 範囲内で他条件が落ちた場合は**その条件**が理由、relaxed 超過だけ `reprojection_error`。`stageOfValidation` / 順位（ratio 4 < spatial 5 < ncc 6 < jump 7 < reprojection 8 < ok 9）
- **診断（§13, §16）**: `RelocValidationDiagnostics` に `relaxedReprojectionErrorPx` / `level` / `reprojectionStrictOk` / `reprojectionRelaxedOk` / `nccScore` / `requiredNccScore` / `nccPassed` / `translationJumpPassed` / `rotationJumpPassed` / `maxTranslationJump` / `maxRotationJumpDeg`。集計に `validatedAcceptable` / `nccRejected` / `jumpRejected`。`RelocalizationResult.level`、`RelocalizationOutput.level`。`Relocalizer.relocalize(current, map, k, heldPose, sceneDepth)` で保持姿勢と奥行きを受け取り Jump を検証
- **Apply / Confirmation（§10–§11）**: `requiredConfirmations(level, confirmationFrames)` = acceptable は `max(1, 設定値)`（即 Apply なし、`confirmationFrames` 0 でも 1 回）。即 Apply（60i / 1.0 px）は strong のみ。事後監視（`postRelocMonitorFrames`）は両方。Best 選択（§12）: 検証通過同士では strong > acceptable、同じ level なら inlier 数（既存）。`goodInliers` 早期打切りは strong のみ
- **HUD（§14）**: `Error 2.44/1.50px strict NG` / `  2.44/3.00px relaxed OK` / `NCC 0.67/0.50 OK`（relaxed 要件があるとき）/ `Jump translation NG (reloc limit)`（NG 時のみ）/ `Level ACCEPTABLE` / `Reject -`、`Cand KF15 50i 2.44px ACCEPTABLE`。ログ `RELOCALIZATION CANDIDATE … error = 2.44px (ACCEPTABLE: relaxed ≤ 3px, confirmation required)`。Debug 既定 OFF は維持（§15）
- **閾値（§19–§20）**: strict 1.5 / relaxed 3.0 / relaxed NCC 0.5 / jump 1.0×奥行き・90° は初期候補値。relaxed は 5 px 以上にしない（§24-12）。実機で 2.0 / 2.5 / 3.0 / 3.5 px 付近の候補と False Positive を収集して調整する（§22）
- テスト 201 件（+5、内容更新）: v12 Test 1 / 11（1.2 px → strong、NCC 0.3 でも strong）、Test 2（実機 50i / 2.44 / 0.85 / 7 / 0.67 → strict NG・relaxed OK・`acceptable`・合格）、Test 3（3.5 px → `reprojection_error`、3.0 ちょうどは acceptable、relaxed ≤ strict / undefined で無効）、Test 4（ratio 0.35 → `inlier_ratio`）、Test 5（cells 2 → `spatial_distribution`）、Test 6（NCC 0.3 → `ncc`、inlier 不足はそれより先）、Test 7（非有限 Pose → `pose_invalid`）、Test 8 / 9（専用上限超過の translation / rotation jump → `translation_jump` / `rotation_jump`、strong にも適用、上限内 / 不明 / 上限なしは通過）、`requiredConfirmations`。v9 Test 6 は §17 の順序で理由が `inlier_ratio` に、Test 8 は上限なしで診断のみを維持。Test 10（acceptable 後の事後監視の不整合）は既存 v5 の `RELOC INCONSISTENT` 経路がそのまま働く（level に依存しない）。ブラウザテスト 2 件合格（ループカットの再局所化は専用 Jump 上限内）
- **実機で読むべきもの（§22–§23）**: RELOC 節の `Level` と `Error … relaxed`。`ACCEPTABLE` の次の試行で `Apply … ACCEPTABLE` になれば AC-1 / AC-2。`Jump … NG (reloc limit)` が正しい復帰で出るなら上限 1.0×奥行き / 90° が狭すぎる。`post map Δ … INCONSISTENT` が出れば acceptable 候補が誤りだった可能性（AC-3 / AC-6）

### 修正指示書 v11.1 対応 — Recovery の 1 回リセット化・Two-view トリガー・候補診断の現フレーム化（2026-10-05、実機確認待ち）

v11 のコードレビュー指摘への対応。**PnP / LK / Jump Gate / Relocalization / Plane RANSAC / Extent の閾値は変更していない**（AC-17、AC-18）。新しい固定値も追加していない（§7: 既存設定のみ再利用）。

- **Start と Active の分離（§3–§5, §31, AC-4〜AC-6）**: `PlaneRecovery.update()` は **inactive → active の瞬間だけ** true を返し、`resetForRecovery()` はそのフレームの 1 回のみ。Active 中に fast / two-view が再発しても restart せず `lastMotionTimestamp` を記録するだけ（reason は開始時のもの）。合成: 3 フレーム連続の高速移動で `recoveries` 1（v11 では 3）、高速フレーム中も stable が 0→1→2→3→4 と単調増加し PLANE_FOUND が 31 → 29 フレーム
- **Significant Motion（§6–§10, AC-1〜AC-3）**: `PlaneRecovery.trigger()` の優先順位 `manual > fast_motion > two_view_motion`。`significantTwoViewMotion(pose, prev, thresholds)` = 二視点 `parallaxPx ≥ pose.fullConfidenceParallaxPx`（25）**かつ** `confidence` / `translationConfidence ≥ landmarks.initMinTranslationConfidence`（0.5）**かつ** `inlierCount ≥ pose.minCorrespondences`（20）**かつ** モデルが essential / homography、**かつ同一参照フレーム内で閾値を跨いだフレーム**（前フレームの視差が閾値未満、または参照更新直後）。視差は参照フレームからの累積量なので「水準」で判定すると低速スキャンでも視差が育った時点で毎回発火する。「跨いだ瞬間」= 視点が大きく変わったイベントとして扱う。実機ログ（`Motion MED 6.3px / 2view par 45px conf 1.00/1.00 n116`）は条件を満たす。視差だけでは発火しない（§8）
- **終了条件（§32）**: `found` / World 確立 / Map 失探（`framesSinceTracked > state.mapLostFrameTolerance`、通常の失探処理へ引き渡し）。Timeout は既存設定がないため追加していない（§33）
- **候補診断の現フレーム化（§21–§24, AC-11〜AC-13）**: `candidateFound = lastSearch.stage === "candidate"`（今フレームの RANSAC 結果）、`candidateCommitted = candidateFound && 水平`、新設 `previousCandidateHeld`（今フレームは失敗だが猶予期間で保持中の候補あり）。`stableFrames` は detector のカウンタ。v11 の `candidate !== null` 判定は保持候補を「発見」と誤表示していた
- **state（§27）**: `phase` を `state: inactive | starting | warmup | candidate | stable` に変更（`starting` = 開始フレームと motion fast 継続中、`warmup` = 現フレーム候補なし、`candidate` = 現フレーム候補あり、`stable` = 安定カウント開始後）。Guidance / `worldPhase` は `starting → SLOW_DOWN / PLANE_RECOVERY`、`warmup → PLANE_WARMUP`、`candidate | stable → PLANE_DETECTING / PLANE_CANDIDATE`
- **HUD（§25–§26）**: `Recov FAST|TWO_VIEW|PLANE_POINTS|MANUAL <state> 1.2s`、`Cand YES commit YES` は現フレーム基準、保持のみは `Cand NO commit NO (held)`。ログは state/段階が変わったときのみ
- テスト 192 件（+3、内容更新）: Test 1（fast → start、state 遷移）、Test 2/3/4（MED + 視差 45 / conf 1.0 / n116 → `two_view_motion` 開始、視差小・信頼度低・inlier 不足・rotation モデルでは不発、水準ではなくイベント、参照更新後は再発火可）、Test 5〜9（3 フレーム fast + medium + normal + two-view 再発で start は `[true, false, false, false, false, false]`、`count` 1、Map 失探で終了）、Test 13〜16（PlaneDetector の保持候補は `stage: points` で現フレーム候補ではない）、合成（`recoveries` 全フレーム 1、stable 単調増加、`candidateFound === (planeSearch.stage === "candidate")`、`candidateCommitted ⇒ candidateFound`、`previousCandidateHeld ⇒ !candidateFound`、低速のみは視差 ≥ 25・conf ≥ 0.5 に達しても Recovery 0 回）。Guidance Test 9/10 は新 state 名で全組合せ。ブラウザテスト 2 件合格
- **実機で読むべきもの（§41）**: `Recov TWO_VIEW` が出れば MED でも二視点イベントで開始できている（ケース B）。`Stable` が 1/5 のまま進まないならリセットの繰り返しを疑うが、v11.1 では `recoveries`（`×N` 表示）が増えない限り reset は起きていない

### 修正指示書 v11 対応 — Fast Motion 後の Plane Detection Recovery（2026-10-05、実機確認待ち）

実機ログ（`LK 284/285`、`PnP 42i 0.38px`、`Source MAP`、`Lost 0ms`、`LM 47 plane 0`、`Plane search 42pt best 32/20 thr 1.645`、`World Established NO`）は Camera / Map Tracking が良好なまま Plane Detection だけが成立していない状態。**PnP / LK / Jump Gate / Relocalization / Plane RANSAC の閾値は変更していない**（AC-5〜AC-9）。

- **Phase 1–2 の特定結果（§37–§38, §58）**: HUD の `Plane search …` 行は `PlaneDetector.update()` が **候補を返さなかった（null）** ときだけ表示される。候補が存在すれば非水平・不安定でも `Plane n(…)` 2 行になるため、実機は「候補化の前」で止まっている。`best 32` は `fitHorizontalPlane` の最密高さウィンドウの支持数で、そこから候補になるまでに null を返す段階は (A) 窓の平均高さでの再分類後 inlier が `minInliers` 20 未満（`reclassify`）、(B) **inlier の面内分布** `s2 < max(2·thr, 0.15·s1)`（`extent`、壁の水平スライス除外。thr 1.645 → 第 2 主軸 σ ≥ 3.29 map 単位が必要）の 2 つだけ。Commit / Stability / 水平判定には到達していない（指示書の A または G）。合成では重力フィットした壁も `extent`（best 25 / inliers 24）で止まる。`kf 1`・`LM 47 = 成熟 42 = 観測 42` から Map は移動後に再初期化された小さな Map（v10 の 30 フレーム reset → 新しい場所で二視点初期化）で、全 Landmark が今見えている。点数を増やす経路は三角測量だけで、そのゲート（視差 20 px・残差 2.5 px・奥行き比 3 倍）は変更しない
- **段階診断（§22, AC-14）**: `PlaneSearchInfo` に `stage: points | support | reclassify | extent | candidate`、`inliers`（再分類後）、`extentMajor / extentMinor / extentRequired` を追加。`MapTrackingResult.triangulation`（`candidates / parallaxRejected / cheiralityRejected / angleRejected / errorRejected / depthRejected / added`）を `MapPoseOutput.triangulation` で出力（Map が増えない理由）
- **Plane Recovery（§5–§8, §13–§14, §28, AC-1〜AC-3）**: `src/vision/PlaneRecovery.ts`（純粋な判定器）。開始条件 = v7 の motion level `fast` **かつ** Map が健全（今フレーム PnP で位置決め・`mapInlierCount ≥ landmarks.minPnPInliers` 12・姿勢が有限）**かつ** World 未確立。開始時は `PlaneDetector.resetForRecovery()` だけ（前候補・安定カウンタ・miss カウンタ。`found` と Map / 正準 Pose / Keyframe / World は不変。`reset()` は使わない）。Recovery 中の Plane RANSAC 入力は `map.collect(minLandmarkObservations, plane.recoverySeedMaxAgeFrames (15), frameId)` = 直近 15 フレーム以内に PnP inlier として観測された成熟 Landmark（今見ている場所。移動前の Landmark は Map に残るが平面には投票しない）。通常時の seed 集合は従来どおり。seed は Map と Map 姿勢からのみで Plane Pose からは生成しない（§11, AC-4）。終了は既存の安定条件（`stableFramesRequired` 5）で `found` になったとき（§16, §44）。World 確立後は何もしない（§30: 確定平面が基準、Map 追跡だけで乗り切る）。高速移動が続く間は毎フレーム再開（`recoveries` カウント）。`VisionEngine.requestPlaneRecovery()`（reason `manual`、テスト用）
- **フェーズ / 誘導（§15, §45–§46, AC-10〜AC-11）**: `WorldPhase` に `PLANE_RECOVERY`（まだ fast）/ `PLANE_WARMUP`（候補なし）を追加、候補ありは `PLANE_CANDIDATE`。Guidance に `SLOW_DOWN`「スマホをゆっくり動かしてください」/ `PLANE_WARMUP`「平らな場所をゆっくり映してください」。World 未確立では全状態・全フェーズで `RELOCALIZE` も「先ほど見ていた場所」も出ない（v10 維持）。状態 enum は追加していない
- **診断 / HUD（§23–§27, AC-14〜AC-17）**: `VisionOutput.planeRecovery: PlaneRecoveryDiagnostics`（`active / reason / phase / mapHealthy / mapInliers / trackedFeatures / seedCandidates / seededPoints / searchPoints / bestInliers / requiredInliers / searchStage / candidateFound / candidateCommitted（= 候補が水平で安定カウント対象）/ stableFrames / requiredStableFrames / recoveryElapsedMs / recoveries`）。数値のみ、整形は HUD 表示時だけ。HUD に `=== PLANE ===`（World 未確立の間のみ）: `Recov FAST_MOTION warmup 1.2s` / `Map 42i healthy` / `Seed 42 of 47 recent (window)` / `Search 42pt best 32/20 thr 1.645` / `Stage extent s2 1.10 < 3.29 (s1 8.2)` / `Cand NO commit NO` / `Stable 0/5` / `Tri cand 237 +0 par 180 ang 40 err 5 dep 12`。ログ（debug のみ）: `PLANE RECOVERY start reason = fast_motion …`、フェーズ / 段階が変わったとき 1 行、`PLANE RECOVERY done`。Debug 既定 OFF は v9 のまま
- テスト 189 件（+11）: PlaneRecovery Test 1（fast + 健全 Map → 開始、フェーズ遷移、manual）/ Test 2（Map 未位置決め・inlier 不足・非有限 Pose・normal/medium・World 確立では開始しない）、seed Test 5/6（直近観測の成熟 Landmark のみ、古い・若い・NaN は除外）、PlaneDetector Test 7/8（`resetForRecovery` 後の 1 フレーム候補は `found` にならず 5 フレームで `found`）、段階診断（points / extent / wall / candidate）、VisionEngine 合成（机で Map 初期化後に 3 フレーム高速移動 → Recovery 24〜30 フレーム・map id 不変・RELOCALIZING 0・PnP source 常に map・`planePose` null → 新しい視点で stable 1→4 → 31 フレームで PLANE_FOUND・World 確立。低速のみでは Recovery 0 回で回帰なし）、Guidance Test 9/10 + `worldPhase` の回復フェーズ。ブラウザテスト 2 件合格
- **実機で読むべきもの**: `PLANE` 節の `Stage`。`extent` なら点は足りているが一塊（§18 の空間分布）で、スマホを左右に振って視野内の平面の広い範囲に Landmark を作る必要がある。`reclassify` / `support` なら高さ方向にばらついている（床と机が混在、または三角測量の奥行き精度）。`Tri` 行の `par / ang / err / dep` のどれが大きいかで Map が増えない理由（視差不足 / 回転のみ / 残差 / 奥行き比）が分かる

### 修正指示書 v10 対応 — 初期スキャンと Relocalization の分離（2026-10-05、実機確認待ち）

起動直後に別の場所へ移動すると「さっき見ていた場所に戻してください」が出ていた。原因は状態機械が `mapInitialized && mapLost` だけで `RELOCALIZING` に入っていたこと（Map は PLANE_DETECTING の二視点初期化で早期にできる）。World 未確立の失探は「スキャン継続」、確立後の失探だけが「World への復帰」。**既存の状態 enum は維持し、新しい状態は追加していない**（§41）。閾値（PnP / LK / NCC / Plane / Jump Gate / v9 Validation）は未変更（AC-8、AC-9）。

- **`worldEstablished`（§5–§7）**: `VisionEngine` が「現在の Map で初めて `PLANE_FOUND` になった」時点で true（ARSession はその同じフレームで WorldAnchor を生成するので `World ready` と同義）。Map 再初期化・リセットで false。`FrameObservation.worldEstablished` として状態機械へ渡し、`VisionOutput / ARStats.worldEstablished` で出力
- **状態機械（§8–§10、§43）**: `RELOCALIZING` に入る条件を `mapInitialized && mapLost && worldEstablished` に限定。World 未確立で Map を見失ったら TRACKING（SURFACE_SCAN）を継続、特徴も失えば TRACKING_LOST → SEARCHING_FEATURES。v10 の呼称との対応: INITIALIZING / SEARCHING_FEATURES = INITIAL_SCAN、TRACKING / PLANE_DETECTING = SURFACE_SCAN（平面候補ありで PLANE_CANDIDATE）、PLANE_FOUND / AR_ACTIVE = WORLD_TRACKING、確立後の TRACKING_LOST / RELOCALIZING = WORLD_LOST（`relocGuidanceDelayMs` 未満）/ RELOCALIZING
- **World 未確立の Map 失探（§10–§12、§18）**: 再局所化は試行しない（World がないので戻る先がない）。`landmarks.preWorldLostResetFrames`（30 = 1 s）で Map / Keyframe を捨てて現在見ている場所で再初期化（短い猶予は同じ面の速い振りを再関連付けで拾うため）。確立後は従来どおり 300 フレーム + 失敗 10 回でのみリセット
- **Guidance（§13–§17、§25）**: `src/ar/Guidance.ts` の純関数 `getGuidance(ctx)`（UX Controller、エンジン閾値には触れない）。World 未確立: `SHOW_FLAT_SURFACE`（特徴不足）/ `SCAN_SURFACE` / `MOVE_SLOWLY`（Map ありで視差待ち）/ `PLANE_DETECTING`（候補あり）。確立後: `TAP_TO_PLACE` / `NONE` / 失探は `RECOVER`（「カメラをゆっくり動かしてください」）→ `world.relocGuidanceDelayMs`（2000 ms）経過で `RELOCALIZE`（「先ほど見ていた場所にカメラを戻してください」）。この文言はそれ以外では生成されない。`main.ts` は毎フレーム評価し文言が変わったときだけ DOM 更新
- **HUD（§26–§28）**: `Phase INITIAL_SCAN | SURFACE_SCAN | PLANE_CANDIDATE | WORLD_TRACKING | WORLD_LOST | RELOCALIZING`（`worldPhase()`）、`World Established YES/NO`、RELOC 節に `Reason WORLD_LOST` / `scan continues (no world)`。Debug 既定 OFF は v9 のまま
- テスト 178 件（+9）: 状態機械 Test 1（World なしの Map 失探で RELOCALIZING に入らず TRACKING / SEARCHING_FEATURES）、Test 3/4（World ありで TRACKING_LOST → RELOCALIZING → 復帰で PLANE_FOUND）、Guidance Test 2/5（World なしでは全状態・全条件で RELOCALIZE も戻る文言も出ない）、Test 6/7（遅延前は RECOVER、経過後に RELOCALIZE）、`worldPhase` 対応表、VisionEngine Test 8（机 A で Map 初期化後に無関係な机 B へ移動 → RELOCALIZING 0 フレーム・再局所化試行 0 回・新しい Map で 37 フレーム後に PLANE_FOUND）。ブラウザテスト 2 件合格

### 修正指示書 v9 対応 — Relocalization Validation の分解・可視化（2026-10-05、実機確認待ち）

実機ログ `Best KF3 35i 2.93px / Ratio 0.61 / cells 4/9 / Stage error / Fail error` の `error` は再投影誤差の段階（2.93 px > `maxMeanErrorPx` 1.5 px）だったが名前が曖昧だった。各検証条件を独立に評価して値・閾値・PASS/FAIL を出す。**閾値（`minInliers` 25 / `maxMeanErrorPx` 1.5 / `minInlierRatio` 0.5 / `minSpatialCells` 4 / NCC / LK / PnP / Jump Gate）は変更していない**（§30）。

- **検証の分解（§3–§6）**: `validateRelocalizationCandidate(measures, thresholds)` が `RelocValidationDiagnostics`（inliers / reprojection / ratio / spatial cells / coverage / pose の値・閾値・`*Passed`、`passed`、`rejectReason: inliers | reprojection_error | inlier_ratio | spatial_distribution | pose_invalid | confirmation | unknown`）を返す。短絡せず全条件を評価し、`rejectReason` は pose → inliers → reprojection → ratio → spatial の順で最初に落ちた条件。段階名は `pnp_inliers / reprojection / ratio / spatial / invalid / ok`（`error` を廃止、§29）
- **Spatial Coverage（§12–§14）**: inlier の外接矩形 / 画像面積（`spatialCoverage`、`spatialCoverageOf`）を追加。`relocalization.minSpatialCoverage` は **0（診断のみ）**。4/9 セルでも「広く散った 0.6」と「一塊の 0.08」を区別できる。将来の再局所化専用閾値はこの設定で独立に調整可能（§15、§31）
- **Pose Jump は診断のみ（§16–§18）**: 通常 Tracking の Jump Gate は再局所化候補に適用しない（v5 と同じ）。非有限の Pose だけ `pose_invalid` で拒否。保持姿勢との差は `translationJump / rotationJumpDeg` として記録
- **最良候補（§7–§8）**: 検証通過候補 > 未通過候補、未通過同士は段階 → inlier 数 → 再投影誤差 → coverage → NCC の順で比較。`RelocalizationKeyframeTrial.validation` / `RelocalizationResult.validation` に全内訳を保持
- **HUD（§21–§22）**: RELOC 節に `Best KF3 35i 2.93px ncc 0.85` / `Inlier 35/25 OK` / `Error 2.93/1.50px NG` / `Ratio 0.61/0.50 OK` / `Cells 4/9 (min 4) OK` / `Cover 0.58` / `Reject reprojection_error`。閾値は設定値から表示
- **Debug 既定 OFF（§23–§28、AC-9–AC-12）**: `?debug=1` のときだけ HUD・オーバーレイ・`[AR]` ログ ON（`debug=true` は OFF）。HUD 非表示中は `refreshHud` が整形も DOM 更新も行わない。起動時の `[AR] build` ログも debug 時のみ。`?hud=1/0` で HUD の初期状態だけを個別に上書き可能
- テスト 169 件（+9）: `validateRelocalizationCandidate` Test 1–8（合格 / inlier 不足 / 再投影誤差過大 = 実機の 35i・2.93px ケース / ratio 不足 / cells 不足と coverage 不足の分離 / 複数 FAIL の全保持 / NaN Pose / 大きな Pose Jump でも拒否しない）、`spatialCoverageOf`、Test 9（拒否後も best の内訳を保持）。ブラウザテスト Test 10–11: 通常 URL で HUD・オーバーレイ非表示・`[AR]` ログなし・診断は生成、☰ で ON/OFF、`?debug=true` は OFF、`?debug=1` で HUD と `Motion` 行とログ
- **今回の実機ログの結論**: `Inlier 35/25 OK`、`Error 2.93/1.50 NG`、`Ratio 0.61/0.50 OK`、`Cells 4/9 OK` → 拒否理由は再投影誤差のみ。v10 で判断する候補: 再局所化専用の `maxMeanErrorPx`（Keyframe→現フレームの LK は追跡中の 1 フレーム LK より誤差が大きいのが自然）

### 修正指示書 v7 対応 — 高速移動の追跡強化 + Debug HUD トグル（2026-10-05、実機確認待ち）

通常速度の安定性を維持したまま、高速カメラ移動時の特徴点対応を「LK 以前〜LK」で強化する。**PnP / RANSAC / Jump Gate / Lost Gate 上限 / 再局所化の検証・確認 / WorldAnchor / ARObject は変更していない**（§2、§10–§11）。

- **Fast Motion 検出（§3）**: 前フレームで LK を生き残った追跡点の中央変位（`medianDisplacementPx`）で判定。`tracker.mediumMotionPx`（8）未満 = normal、`fastMotionPx`（20）未満 = medium、以上 = fast。参照フレームとの 2view parallax は累積量なのでフレーム間の速さには使わず、フレーム間の実測変位を使う
- **Adaptive LK（§4）**: LK の変位ゲート（`maxDisplacement` 60 px、予測位置からの距離）に段階別の倍率 `mediumMotionSearchScale`（1.5）/ `fastMotionSearchScale`（2.0）を掛ける。通常時は従来どおり 1 倍。ピラミッド（4 段、coarse-to-fine）はそのまま（§5）
- **Homography 予測（§6–§7）**: 前フレームの外れ値除去で得たフレーム間 Homography（`lastImageMotion`）を各追跡点に適用した位置を LK の開始点にする（パン・回転・ズームを 1 つの大域モデルで予測）。前フレームの RANSAC が `predictionMinInliers`（30）以上かつ inlier 比 `predictionMinInlierRatio`（0.6）以上のときだけ使い、それ以外は従来の追跡点ごとの等速予測（`predictMotion`）。合成 32 px/frame パンの維持率: Homography 予測 平均 0.91 / 等速 0.88 / 予測なし 3 段 0.77
- **診断（§12–§13）**: `VisionOutput.motion`（`MotionDiagnostics { level, medianDisplacementPx, maxDisplacementPx, trackedBefore, trackedAfter, forwardBackwardRejects, tooFarRejects, meanResidual, predictionMode: homography|velocity|none, searchScale }`）。HUD `Motion FAST 30.3px (max 41) H ×2` / `LK 231/265 fb 12 far 3`。Worker プロトコル経由で転送
- **Debug HUD トグル（§15–§26）**: 右上に 44×44 の `☰` ボタン（`#debug-toggle`、safe-area 考慮、HUD は `max-width: min(420px, 100vw − 76px)` でボタンに重ならない）。OFF で HUD・特徴点オーバーレイ（`#overlay`）・平面グリッドを非表示、エンジン・診断・stats は継続（`ARSession.setDebugVisualization(visible)`、`ARWorld.setPlaneGridVisible`。グリッドは設定どおり生成しておき表示だけ切替）。**既定 OFF**、`?debug=1` で ON（ログも ON）、`?hud=1 / ?hud=0` で初期状態を個別指定
- **変更しなかったもの**: PnP 品質基準、RANSAC、Jump Gate、Lost Gate 上限、再局所化の検証・確認、WorldAnchor、ARObject、v6 の RELOC 診断（OFF 時は隠すだけ）
- テスト 160 件（+4）: 動き段階（3 / 12 / 30 px で normal / medium / fast、倍率 1 / 1.5 / 2.0）、Homography 予測が成立する条件と LK 集計の整合（`trackedAfter == trackedCount`、`trackedBefore == 前フレームの featureCount`）、支持不足時は等速へフォールバック・`predictMotion: false` で `none`、低速では level normal・倍率 1・維持率 95% 超（回帰なし）。ブラウザテスト全サンプル `PLANE_FOUND`

### 修正指示書 v6 対応 — Relocalization 診断強化 + HUD 整理（2026-10-05、実機確認待ち）

実機（Keyframe 8、Lost 3.6 s、`RELOC none`、ok×0、Map 候補 14i / 3.18 px）で「8 枚の Keyframe がどの段階で落ちているか」を読めるようにする。**閾値（`minInliers` / `maxMeanErrorPx` / `minInlierRatio` / `minSpatialCells` / `coarseMinScore` / `lkMaxDisplacementPx`）、Jump Gate、Lost Gate 上限、長時間失探の要求 inlier、Immediate Apply、確認フレームは一切変更していない**（§7–§9、§18）。

- **段階別診断（§1–§3、§5）**: `Relocalizer.relocalize()` が `RelocalizationResult.diagnostics`（`RelocalizationDiagnostics`）を返す。試行した Keyframe ごとに `RelocalizationKeyframeTrial { keyframeId, stage, coarseScore, lkRatio, inlierCount, meanReprojectionErrorPx, inlierRatio, spatialCells }` を記録し、`stage` は `coarse → landmarks → lk → pnp → error → ratio → spatial → ok`（`invalid` も）。集計: `candidatesTried / coarseTested / coarsePassed / lkTested / lkPassed / pnpTested / pnpPassed / validated / errorRejected / ratioRejected / spatialRejected / bestCoarseScore`。`best` は最も先の段階まで進んだ候補（同段階なら inlier 多い方）で、失敗時も必ず残す。拒否コードは best の段階から決める
- **失探中は直近試行を保持（§4）**: 試行は 3 フレームに 1 回なので、`RelocalizationOutput.diagnostics` は次の試行まで保持し `framesSinceAttempt` を添える（追跡復帰で消去）。HUD が 3 フレーム中 2 フレーム `none` だけになる問題の解消
- **Recovery 行（§10）**: `MapPoseOutput.observations / requiredInliers / recoveryMode`（`tracking` = `minPnPInliers` 12、`recovery` = `minRecoveryInliers` 24、`long` = `minRecoveryInliersLong` 40）。HUD `Need 14/24i obs 14 (recovery)` でどの規則が効いているか分かる
- **HUD をモバイル向けに（§11–§16）**: 1 行 1 値の縦長レイアウト（`Label  value`、ラベル 6 文字）。節は常時 `State`（14 px 太字）/ FPS / `TRACK`（Feat / PnP / Source + 履歴 / Lost / Rot / 2view / Reloc 1 行）/ `MAP`（Cand / Reject / Need / Gate / Plane 候補）/ `RELOC`（**失探中・試行中・再局所化直後のみ**: `KF 8 / try 2` / `NCC 2 best .31` / `LK 2` / `PnP 2 ran / 0 ok` / `VAL 0` / `Best KF3 14i 3.18px` / `Ratio .24 cells 4/9 ncc .31` / `Stage ratio` / `Fail ratio` / Cand・Apply・Jump・Post）/ `WORLD` / `OBJECT` / `TIMING`。CSS: `width: calc(100vw - 24px)`、`max-width: 420px`、`font-size: 12px / line-height 1.35`、`top: safe-area + 8px`、`max-height: calc(100dvh - safe-area - 24px)` でスクロール、`pre-wrap` + `overflow-wrap: anywhere`。警告値は橙（`hud-warn`）。`MAP` 行は現フレームの Map PnP 候補、`RELOC` 行は再局所化の候補と、ラベルで意味を分離（§16）
- **ログ**: `RELOC REJECT` に `keyframes = 8 tried = 2 / NCC 2/2 (best 0.31) LK 2/2 PnP 0/2 VAL 0 / best = KF3 stage ratio 14i 3.18px ratio 0.24 cells 4/9 / trials = KF3:ratio(14i) KF7:pnp(9i)` を追加（拒否コードまたは best の段階が変わったとき）
- テスト 156 件（+1）: 成功時に全カウンタが 1 / `ok`、一隅の inlier は `pnpPassed 1 → spatialRejected 1 → validated 0`・best の stage `spatial`、別シーンは `coarse` で脱落し `bestCoarseScore` が trial と一致、失探中の出力に diagnostics が保持され追跡復帰で消える、`recoveryMode` が `long` → `tracking` に戻る。ブラウザテスト全サンプル `PLANE_FOUND`
- **実機で読むべきもの**: `RELOC` 節の `NCC / LK / PnP / VAL` のどこで 0 になるか。`NCC 0` なら Keyframe 画像との粗一致が成立していない（視点変化が大きい、または Keyframe が別の場所）、`LK 0` なら粗一致は通るが特徴が追えない（ブラー / 露出差）、`PnP 0 ok` なら対応は取れるが Map と幾何が合わない（Landmark のずれ）、`VAL 0` なら inlier 数は足りるが error / ratio / spatial で落ちている（`Fail` 行にどれか）

### 修正指示書 v5 対応 — Relocalization の候補化・大域検証・確認（2026-10-05、実機確認待ち）

通常 Tracking = 時系列連続性重視（Temporal Gate）、Relocalization = 大域的な幾何整合性重視（Global Validation）の 2 本立て（§2、§34）。再局所化は通常の Jump Gate では拒否しないが、成功を無条件に信頼もしない。

- **P0: 候補 → 検証 → Apply（§3–§7、AC-1 / AC-2 / AC-3）**: `Relocalizer.relocalize()` は Keyframe マッチの結果を *候補*（`RelocalizationResult`: pose、inlier、誤差、`matchScore`（粗 NCC）、`lkRatio`、`inlierRatio`（PnP inlier / LK 追跡数）、`spatialCells`（3×3 グリッドの占有セル数）、`rejectCode`）として返し、`success` は大域検証を通過した意味。検証は inlier ≥ 25・誤差 ≤ 1.5 px（既存）に加え、`relocalization.minInlierRatio`（0.5）と `minSpatialCells`（4、AC-4: 一隅に固まった inlier は不採用）。拒否理由は `RelocalizationRejectCode`（`no_keyframes | low_match_score | insufficient_landmarks | lk_failed | insufficient_inliers | high_reprojection_error | low_inlier_ratio | poor_spatial_distribution | confirmation_failed | invalid_pose`、§19）
- **P2: 確認フレーム（§10–§11）**: 検証済み候補は、明らかに高品質（`immediateInliers` 60 以上かつ誤差 `immediateMaxErrorPx` 1.0 px 以下）なら即 Apply。それ以外は `pendingReloc` に保持し、次フレーム（スケジュールに関係なく毎フレーム試行）の候補が `confirmTranslationDepthRatio`（奥行きの 5%）/ `confirmRotationDeg`（5°）以内に再現されたとき Apply（`confirmationFrames` 1）。再現されなければ `confirmation_failed` で破棄（新しい候補は自身の確認を開始）。通常 PnP で先に復帰した場合は候補を捨てる
- **P0: 失探ゲート幅の上限（§13、AC-5）**: `lostGrowth = min(jumpRejectMaxLostGrowth(3), 1 + 0.1 × 失探フレーム)`。20 フレーム以降は 3 倍（机スケールで約 12 cm）で頭打ちし、それ以上のずれは再局所化だけが戻せる
- **長時間失探の通常 PnP を厳しく（§12、§14、AC-6）**: `longLostFrames`（30）を超えたら復帰に `minRecoveryInliersLong`（40）を要求（通常の復帰は 24）。古い Pose + 広いゲート + 少数 inlier での再シードを防ぐ
- **P1: Relocalization 後の整合性監視（§16–§17、AC-7）**: Apply 後 `postRelocMonitorFrames`（3）フレーム、Map PnP の Pose と再局所化 Pose の差の最大値（`postDeltaTranslation / RotationDeg`）を記録し、Map 候補がゲートで jump 拒否されたら `postInconsistent`。合成テスト: 95 フレームの失探から `candidate → success` で復帰、その後の Map Δ 0.145 map 単位（奥行きの約 0.6%）/ 0.01°
- **P1: 診断（§8、§18、§20–§21、AC-9）**: `RelocalizationOutput` に `attempt: none | candidate | success | fail`、`rejectCode`、品質値、`keyframeId`、保持姿勢からの `jumpTranslation / RotationDeg`、`post*`。`MapPoseOutput.relocalized`（Apply フレームと監視期間中 true）。Source 履歴に `R`（例 `MMMM··RMMM`）。HUD: `RELOC success kf 4 ok×1 RELOCALIZED` / `cand in 54 err 1.7px match 0.91 ratio 0.80 cells 6/9 kf 12` / `jump 82 cm / 18.4°` / `REJECT poor_spatial_distribution …` / `post map Δ 2 cm / 1.1° [INCONSISTENT]`。ログ: `RELOCALIZATION CANDIDATE|APPLIED … translation / rotation / inliers / error / match / inlier ratio / cells`、`RELOC REJECT reason = <code>`（コードが変わったとき）、`RELOC INCONSISTENT`（warn）
- **維持したもの**: `injectRelocalizedTracks`（§15）、PlaneTracker の `候補 → Gate → 一致 → 選択 → commit` 順（§23）、PlanePoint 生成条件（§24）、ARObject / WorldAnchor / FramePresenter / FAST / LK / RANSAC / PnP / FOV / スケール / One Euro（§22、§31）
- テスト 155 件（+7）: `spatialCellCount`、一隅に固まった inlier の `poor_spatial_distribution` 拒否（inlier・誤差は合格）、良い分布は採用、別シーンは拒否、確認フレーム（`candidate` の次フレームで `success`、`relocalized` と `R`、監視期間の整合）、高品質候補の即 Apply、ゲート上限（45 フレーム失探しても 1.0 unit の瞬間移動は `translation_jump` のまま、上限到達後に limit が一定）、長時間失探の `insufficient_inliers … < 40`。ブラウザテスト: カットを `RELOCALIZATION APPLIED`（181 inlier、即 Apply）で跨ぎ全サンプル `PLANE_FOUND`

### 修正指示書 v4 対応 — Trusted でも Jump Gate を通す（2026-10-05、実機確認待ち）

判定を 3 種類に分離（§8–§9）: A. PnP 品質（inlier / 誤差 = `trusted`）、B. 時系列連続性（前フレームの正準 Pose との並進・回転差 = Jump Gate）、C. Map / Plane 一致。**Trusted ≠ 連続** なので A は B を代替しない。

- **Trusted バイパス廃止（P0、AC-1 / AC-2）**: `MapTracker.update` の `if (!trusted(c)) validate(...)` を撤廃し、Map PnP・平面 PnP の両候補を無条件に `validatePoseCandidate(candidate, this._pose, limits)` に通す。`jumpRejectTrustedInliers` / `jumpRejectTrustedErrorPx` は候補の品質ラベル（HUD の `TRUSTED`）としてのみ残す。Map と Plane 両方あるときの `plane vs map` 一致判定（§6–§7）は維持
- **失探中のゲート幅**: 失探中の基準 Pose は予測 / 保持値で不確かさが増えるため、上限を `1 + jumpRejectLostGrowthPerFrame(0.1) × 失探フレーム数` 倍に広げる（10 フレームで 2 倍、再関連付けの上限 90 フレームで 10 倍 ≈ 机スケールで 40 cm）。品質によるバイパスではなく、常にゲートを評価して実測値と上限を出す。合成テスト: 全 Landmark が追えている 0.4 unit の瞬間移動（旧実装では trusted として即採用）は拒否され、同じ姿勢の証拠が続くと 10 フレーム以内に復帰する
- **再局所化はゲート対象外**（§19 の「再局所化の仕組みは維持」）: Keyframe PnP（inlier ≥ 25・誤差 ≤ 1.5 px・粗相関）で Map に対して大域的に検証済みの Pose を正準 Pose の再シードとし、次フレームから再びゲートの基準にする。保持姿勢との差は `RelocalizationOutput.jumpTranslation / jumpRotationDeg` としてログ・HUD に出す（最初は再局所化もゲートに通したが、ループ映像のカットで 1.5 s 失探し続けることが分かり撤回）
- **構造化した拒否理由（§11）**: `PoseRejection { code, reason, delta, limit }`、`code` は `translation_jump | rotation_jump | insufficient_observations | insufficient_inliers | high_reprojection_error | map_plane_disagreement | source_cooldown | invalid_pose`。`jumpRejected` は Map または Plane の jump コードで真（§10、文字列検索ではなく `isJumpRejection`）。`PoseSelection.map / plane`（`PoseCandidateReport`: inlier、誤差、前 Pose との Δ、trusted、reject）と `limits` を `MapPoseOutput.mapCandidate / planeCandidate / gateMax*` で出力
- **PlaneTracker の状態更新を分離（§12–§13、AC-4）**: `update()` は解くだけで Track を変更しない（候補、`accepted: false`）。候補がゲートと一致判定を通ったフレームでのみ `commit(tracks, canonicalPose)` が probation / off-plane / confidence を更新する（基準は常に正準 Pose。クールダウンだけで見送られた候補は有効扱い）。拒否された平面 Pose は平面状態を一切強化しない。PlanePoint の生成条件（Map source・inlier ≥ 20・誤差 ≤ 1.5 px・切替後 10 フレーム）は維持（§14、AC-5）
- **`MapTracker.setPose()` 削除**（§18 無条件 override の経路をコードから消す）
- **HUD（§25）**: TRACKING 節を `SOURCE MAP|PLANE|LOST + 履歴` / `MAP cand in 42 err 1.80px Δ 1.2 cm / 0.8° TRUSTED reject -` / `PLANE cand … reject translation_jump (31.0 > 8.0 cm)` / `Gate ≤ 8.0 cm / 20° map↔plane Δ` / `LOST N ms` に変更。Keyframes 行の reloc success に `jump X cm / Y°`
- **ログ（§26–§27）**: `SOURCE MAP -> PLANE … map/plane delta = 0.032m / rotation delta = 1.2deg`、`PLANE REJECT … reason = translation_jump / delta = 0.31m / limit = 0.08m / inliers = 60 error = 0.50px (trusted)`（コードが変わったときに 1 回）、`RELOCALIZED … jump from held pose = …`
- **確認（§34）**: ARCamera だけが `camera.position` を書き、ARObject は `place()` / 自身の移動 API のみ（grep で確認、AC-7）。WorldAnchor は mapFrameId ごとに 1 回生成。短時間 Lost での ARObject 削除なし（`holdPoseOnLostMs` 10 s、AC-6）
- **変更しなかったもの（§30）**: FOV、`assumedPlaneDistanceMeters`、One Euro、WorldAnchor、ARObject 配置、FramePresenter、FAST / LK / RANSAC / PnP
- テスト 148 件: trusted な Map 候補の瞬間移動が拒否され後で復帰 / trusted な Plane 候補の並進・回転ジャンプが拒否され `jumpRejected` が真 / PlaneTracker の拒否候補が状態を変えない / 合成机シーケンスで MAP → PLANE 切替フレームの Δ が 2 cm 未満・両候補とも reject なし（Test 4、AC-3）/ validatePoseCandidate の構造化理由。ブラウザテストはカットを再局所化で跨ぎ全サンプル `PLANE_FOUND`

### 修正指示書 v3 対応 — Camera Pose の 1 本化・Tracking Lost の扱い（2026-10-03、実機確認待ち）

- **Pose 候補 → 共通検証 → 正準 Pose**（§1–§6）: `src/vision/PoseValidation.ts` に `validatePoseCandidate(candidate, reference, limits)`（並進・回転の連続性、拒否理由つき）。`MapTracker.update` は Landmark PnP と平面 PnP（`external` 候補、品質つき）を両方「候補」として同じ上限（Landmark 中央奥行き 8% / 前フレーム移動量 3 倍 / 20°）で前フレームの採用姿勢と照合し、平面候補はさらに Map 候補との差でも照合する。`poseOverride` の無条件採用は廃止。Three.js に渡る Pose は `MapTracker.pose` の 1 本のみ
- **ヒステリシス**（§7）: `landmarks.sourceSwitchCooldownFrames`（15）。切替直後は Map 候補がある限り戻さない。拒否理由に `cooldown n/15`
- **PlanePoint の自己フィードバック遮断**（§8–§11）: 新規 PlanePoint は Pose source が `map` で inlier ≥ 20・誤差 ≤ 1.5 px、かつ source 切替から `planeTracking.liftCooldownFrames`（10）経過したフレームでのみ生成。平面 Pose からは生成しない
- **Tracking Lost と World Reset の分離**（§12–§16）: 短時間の Lost は最後の正常 Pose を保持（Three.js カメラは更新されない）、`world.holdPoseOnLostMs` 1.5 s → 10 s（デバッグ値）。Map / World のリセットは `lostResetFrames` 150 → 300（10 s）**かつ**再局所化の失敗 ≥ `relocalization.minAttemptsBeforeReset`（10）の両方を満たす場合のみ。WorldAnchor / ARObject は短時間 Lost では維持（従来どおり Map リセット時のみ破棄）
- **Pose smoothing 既定 OFF**（§22）: `world.smoothing: false`、`?smooth=1` で ON。One Euro のパラメータ自体は変更なし
- **HUD**（§19–§21）: `Pose source MAP|PLANE|PROPAGATED|LOST` と直近 24 フレームの履歴（M / P / ·）、`Sources map in N  plane in M  Δ cm / °`、`REJECTED <理由>`（例 `plane translation jump 0.42 > 0.08`、`cooldown 3/15`）、`Lost N ms`。`MapPoseOutput` に `mapInlierCount` / `planeInlierCount` / `rejectReason` / `sourceDelta*` / `sourceHistory`
- **実機結果（`32c6bce`）**: 失探中も特徴点 260 個すべて追跡できているのに `map in 0`・`REJECTED` なし → Landmark に紐づいた追跡点が 0 本。速い動きで Landmark 付きの追跡点が LK で切れ、FAST が同じ角を再検出しても新しい追跡点は Landmark に紐づかないため PnP が二度と成立せず、再局所化（粗相関 0.45 以上が必要）も失敗して 10 秒待ちになっていた。加えて失探中も「150 フレーム未観測の Landmark を削除」する枝刈りが走り Landmark が 170 → 106 に減っていた
- **Landmark の再関連付け（§15、Map を生かし続ける）**: 毎フレーム、追跡点を失った成熟 Landmark を現在の姿勢（短時間の失探中は伝播姿勢、`reassociateMaxLostFrames` 90 以内）で画像に投影し、半径 `reassociateRadiusPx`（4 px）内の未紐づけ追跡点に紐づける（空間ハッシュ。FAST は同じ角を約 1 px 以内に再検出する）。紐づけの正否は次フレームの PnP 分類が判定し、失探からの復帰には通常の 12 ではなく `minRecoveryInliers`（24）個の inlier を要求する（無関係なシーンでの偶然の一致 10 個程度を「復帰」と誤認しないため。合成テストで実際に起きた）。枝刈りは追跡できたフレームでのみ実行（失探で Map を削らない）。失探中の姿勢は回転を二視点事前値で、カメラ中心を直前の追跡速度で `velocityPropagationFrames`（10）フレームまで予測（以後は保持）し、投影位置を実際の特徴点に近づける。HUD の PnP 行に `relink N`、`t PRED` / `t HELD`
- **実機結果（`370c05a`）**: 床に戻しても `REJECTED map inliers 4 < 24` が続く。再関連付けは効いているが、保持姿勢のずれで 2.5 px 以内に入る Landmark が 4 個しかなく復帰条件に届かない。失探中の処理時間 67 ms（再局所化の試行が支配的）
- **誘導付き復帰（同一フレーム 2 段階）**: 失探中、少数の紐づけで解いた PnP が `recoverySeedInliers`（6）以上・誤差 `recoverySeedErrorPx`（2 px）以下なら、その仮姿勢で全未紐づけ Landmark を投影して半径 `recoveryReassociateRadiusPx`（4 px）で再関連付けし、拡大した観測で同一フレーム内に PnP をやり直す（ORB-SLAM の TrackLocalMap と同じ考え方）。拒否理由に `(guided from N)`。再局所化の試行間隔を 2 → 3 フレーム（`attemptEveryNFrames`）に広げて失探中のコストを下げる
- **姿勢に依存しない再関連付けの種**: 失探中は保持 / 予測姿勢での投影が数 px ずれて種がほとんど取れない（合成でも 2 個）。そこで各 Landmark に最後の画像位置（`lastX` / `lastY` / `imageAge`）を持たせ、追跡点を失った後は毎フレームの外れ値除去で既に求めているフレーム間 Homography（`VisionEngine.lastImageMotion`）で位置を伝播する。失探中の再関連付けはこの伝播位置を使う（追跡中は姿勢投影）。合成テスト: 全追跡点が入れ替わり、かつカメラが予測と違う動きをした直後でも、1 フレーム目で 131 本を再紐づけ、2 フレーム目に inlier 132 で復帰（回転誤差 0.5°）。失探中の最小観測数は `recoverySeedInliers`（4）
- **実機結果（`ca7afda`）**: 床から外して 12 秒後に戻すと `REJECTED map observations 2 < 4`・`reloc fail` のまま。画像位置の記憶は 90 フレームで切れるので長い離脱後は Keyframe 再局所化だけが頼りだが、候補が常に「直近 2 枚」固定で、床を見ていた古い Keyframe（原点ビューを含む）が一度も試されていなかった。粗相関 0.45 の門も視点が数度変わると通らない
- **再局所化の候補をラウンドロビンに**: 試行ごとにカーソルを進め、数回の試行で 8 枚すべてを試す。粗相関の門を 0.45 → 0.25（`coarseMinScore`。真の検証は PnP inlier 25 以上・誤差 1.5 px 以下）。失敗理由（`score 0.31 < 0.25 (kf 3)` / `lk 12/80 < 25` / `pnp 8/40 < 25` / `err 2.1 > 1.5 px`）を `RelocalizationOutput.reason` に出し、HUD に `Reloc fail <理由>` 行を追加
- **変更しなかったもの**（§29）: `assumedPlaneDistanceMeters`、`longSideFovDeg`、FAST / LK / RANSAC / One Euro の各パラメータ、Three.js 射影、ARObject のスケール
- テスト: PoseValidation（受理 / 並進ジャンプ / 回転ジャンプ / 対称性）、MapTracker の平面候補（Map と一致する候補はクールダウン後に採用、乖離する候補は拒否されカメラが飛ばない）

### 修正指示書 v2 対応 — Map 固定・診断強化（2026-10-03、実機 A/B 待ち）

「カメラを動かすと Cube が動く」症状への対処。アーキテクチャは維持（FAST → LK → RANSAC → 二視点 → 三角測量 → Landmark Map → PnP → Plane RANSAC → WorldAnchor → Three.js）。

- **Map を固定基準にする（最優先）**: `MapTracker.refineLandmark()`（視差が増えるたびに Landmark 位置を再三角測量で書き換え）を `landmarks.enableLandmarkDepthRefinement` で切替可能にし、**既定 OFF**。Pose → Landmark → Pose の正帰還を断つ。デモは `?refine=1` で ON（A/B 用）。注意: この精錬は実機で平面検出を成立させるために入れたものなので、OFF で `PLANE_FOUND` に到達しにくくなる可能性がある（その場合は初期化視差 `initMinParallaxPx` を上げる方向で対処）
- **PnP テレメトリ**: `MapTrackingResult` / `MapPoseOutput` に `cameraCenter`、`deltaTranslation`（前フレームからのカメラ中心移動、map 単位）、`deltaRotationDeg`、`translationHeld`（PnP 失敗で並進を保持し回転だけ伝播）、`source`（`map` / `plane` / `propagated`）。PnP 失敗・復帰は `[AR]` ログに 1 回ずつ出力。異常ジャンプの拒否はまだ行わず数値を出すだけ（§8）
- **HUD を §23 構成に**: TRACKING（features / tracked / inliers / PnP）、CAMERA（map C、world C、Δt cm、Δrot）、WORLD（scale、Landmark 数と plane / non-plane 内訳、平面 n・RMS・conf）、OBJECT（配置物の World 座標。カメラ移動で変化しないことを確認する）、TIMING（frame t / pose t / pose age、`debug.poseStaleMs` 超過で `POSE STALE`）
- **Camera / Object の分離**: 既存どおり `ARObject.place()` のみが位置を決め、`ARCamera` だけが `camera.position / quaternion` を書く（grep で確認）。`ARObject` に World 空間の移動 API（`setPosition` / `moveBy` / `setYaw` / `velocity` / `angularVelocityY` / `update(dt)`）、`ARWorld.update(dt)` / `planeToWorld`（World = 平面座標系なので恒等）。デモ `?walk=1` で Cube が X 方向に 5 cm/s で往復（Test D / E）
- **テスト**: CoordinateSystem round trip（恒等 → World 原点、X 移動、ランダム 200 姿勢で位置 1.5e-15 / 回転 3e-8 rad）、hitTest round trip（World → 投影 → hitTest で 1e-6 以内）、Camera / Object 分離（§24 Test 1–4: 右 / 左移動で Object World 不変・画面上は逆方向、回転で不変）、Object 移動（Test D / E）、PnP テレメトリ、精錬 ON/OFF の合成ドリフト比較（どちらも中央値 0.03 px: 合成平面では差が出ない）
- **FOV 感度テスト（§15）**: 同じ合成シーケンスを 60 / 63 / 66 / 69 / 72° で追跡。**平面シーンでは焦点距離の誤りはドリフトにならず**（全条件で中央値 0.03 px）、復元されるカメラ移動量だけがスケールする（Δx 0.151 → 0.163 m）。実機でのドリフトに FOV が効くとすれば非平面部分を通じてであり、合成テストでは順位付けできない
- **実験的: 平面アンカー姿勢（既定 OFF）**: 最初の指示書（Homography 中心）に基づき `PlaneTracker` を実装済み。確定した平面に特徴点を持ち上げ（画素 Ray ∩ 平面）、その 3D 点への PnP で姿勢を解く（= 平面誘導 Homography の n, d 既知分解と等価、奥行き非依存）。平面外の点は probation / 連続外れで除外。v2 §30 に従い主経路にはせず `planeTracking.enabled`（`?planetrack=1`）で比較用に残す。有効時は `VisionOutput.planeAnchor` / `planePose` を出力し、World はそのアンカー平面から生成
- **Pose smoothing A/B**: `?smooth=0`（§17）。**合格条件は smoothing OFF でも World 固定**
- **実機で確認すべきこと（§26, §34 Step 2）**: 既定（精錬 OFF）と `?refine=1` で、Cube 配置後に 5 / 10 / 20 / 50 cm 横移動したときの画面ドリフトを比較。HUD の `world C` が移動量相当（10 cm → ≈0.10）変化し、`Object 1` の X Y Z が不変であること
- **実機結果（Map 固定後）**: ゆっくり動かせば Cube は固定される（ユーザー確認）。少し速く動かすと失探、または別の場所へ飛ぶ
- **実機結果（速い動き対処後、`4d5375e`）**: 追跡点 257 個すべて追えているのに PnP inlier 0 で `RELOCALIZING`、Landmark が 157 → 649 に膨張（plane inlier 0）。精錬 OFF により、視差 6 px の小さな基線で三角測量した精度の悪い Landmark がそのまま固定され、カメラが動くほど PnP が合わなくなる → 失探・別の Landmark 群で復帰して飛ぶ、という構図
- **Landmark を「作るときに厳しく」（v2 §27–28、Map 固定は維持）**: 新規三角測量の必要視差 6 → 20 px、二視点残差 4 → 2.5 px、既存 Landmark の中央奥行きの 3 倍超 / 1/3 未満は棄却（`maxDepthRatio`）。新しい Landmark は候補扱いで、PnP inlier を 3 回重ねる（`minObservationsForPose`）まで姿勢の解には使わず（成熟 Landmark が 24 個以上あるとき。`minMaturePnPPoints`）、解いた姿勢で分類して成熟させる。若い Landmark が 2 フレーム外れたら即削除（`youngOutlierFrames`）
- **速い動きへの対処**: (1) LK ピラミッドを 3 → 4 段（`tracker.pyramidLevels`。1 フレームの追従範囲が約 28 → 56 px）、(2) 前フレームの変位を初期値にする等速予測（`tracker.predictMotion`）。合成の 32 px/frame パンで追跡維持率 0.76–0.80 → 0.82–0.95。(3) 姿勢ジャンプの拒否（§8、`landmarks.jumpReject*`）: inlier 40 未満または誤差 1.5 px 超の弱い PnP 解で、カメラ中心が max(Landmark 中央奥行きの 8%、前フレーム移動量の 3 倍) を超えて動くか 20° 超回転した場合は採用せず姿勢を保持（HUD の PnP 行に `JUMP`）。多数 inlier で誤差が小さい解は速い動きとして受け入れる

### 実機チューニング（iPhone Safari、2026-10-03）

実機の HUD スクリーンショットを元に調整した内容。合成データでは見えなかった実データ特有の問題への対処。

- **ビルド識別**: HUD 最下行に `Build phase5 <commit> <日時>`（`vite.config.ts` の `define`）。古いキャッシュの判別用
- **フレーム間外れ値除去**: Homography ゲートを 3 → 6 px にし、2 フレーム連続で外れた追跡点のみ削除。実際の部屋は平面ではないため、背景の正しい追跡点が毎フレーム約 8% 削られ、Landmark 化前に消えていた（63 lm → 134 lm、PnP inlier 0 → 127）
- **Landmark**: 初期化視差 20 → 30 px、PnP ゲート 4 → 6 px、三角測量誤差 3 → 4 px、必要視差 8 → 6 px、最低 PnP inlier 15 → 12。視差が 1.3 倍に増えるたびに再三角測量して奥行きを精錬。PnP 外れ値が続いた場合は Landmark を消さず追跡点との紐付けだけを外す（原因の多くは LK ドリフト）。三角測量アンカーは姿勢が確かなフレームでのみ設定
- **平面**: 重力あり → 法線を重力に固定して高さだけをロバスト推定（最密クラスタ）、inlier から最小二乗で法線を微調整（重力から 10° 以内）、inlier が 2 次元に広がることを要求（壁の水平スライスを除外）。閾値は距離の 2 → 5%、最低 inlier 30 → 20、使用 Landmark は PnP で確認済み（観測 ≥ 3）。重力なしは従来の RANSAC + 水平判定。重力 EMA を 0.2 → 0.06（手ぶれ加速度の影響低減）
- **再局所化**: 採用条件を inlier 25 以上・誤差 1.5 px 以下・粗相関 0.45 以上に厳格化、候補 2 枚を 2 フレームに 1 回（Vision 30 fps 維持）
- **結果**: Vision 30 fps（18 ms）、特徴点 263 全追跡、Landmark 134 / PnP inlier 127、再局所化なしで `AR_ACTIVE` 到達（Cube 配置）。スケールは机想定 0.5 m のため床では小さめに見える（`world.assumedPlaneDistanceMeters`）
- **Cube の流れ対策**: 焦点距離を長辺画角 66° から算出（`processing.longSideFovDeg`、`?fov=`。従来の fx ≈ 長辺は約 53° 相当で長すぎた）。表示をポーズに同期（処理したフレームをポーズ到着時に描画、`processing.syncVideoToPose`、`?sync=0`）。One Euro を軽く（minCutoff 1.5 → 4）
- **失探の頻発（同期表示導入後）**: 同期表示を `createImageBitmap(video)` から同期的なキャンバスのリングバッファ（`FramePresenter`、4 枚、表示解像度・長辺 1440 px 上限）に変更。iOS Safari では `createImageBitmap` が全解像度の読み戻しで数十 ms かかり、非同期コピーが溜まってメインスレッドを塞ぎ、キャプチャがフレーム落ち → LK が追えず失探していた。あわせて PnP 失敗 1 フレームで `RELOCALIZING` に落ちていた状態機械に猶予 3 フレーム（`state.mapLostFrameTolerance`。再局所化の試行自体は従来どおり 1 フレーム後から）。HUD の Vision FPS 行に `drop N`（バックエンド処理中に捨てたカメラフレーム数）

### Phase 5 — 実装済み（承認待ち）

- `src/vision/Keyframe.ts`: Keyframe（姿勢、画像ピラミッドのコピー、粗画像、Landmark 観測）
- `src/vision/Relocalizer.ts`: Keyframe 作成ポリシー（PnP inlier ≥ 30、前 Keyframe から回転 10° / 視差 40 px / 90 フレーム、最大 8 枚・最初の 1 枚は保持）と再局所化。手順: 直近の Keyframe 候補（毎フレーム 3 枚まで）→ 粗画像（1/8）でゼロ平均 NCC による整数シフト探索（±24 px → ±192 px 相当）→ シフトを初期値に Keyframe 画像から現フレームへ Pyramidal LK（Forward-Backward 付き）→ Keyframe 姿勢を事前値に PnP（LM + Huber）→ inlier ≥ 20・平均誤差 ≤ 2 px で採用。記述子不要（短時間の Lost・ブラー・一時的な遮蔽・同じ場所に戻る場合が対象。大きな視点変化や Loop Closure は非対象 §2）
- `FeatureTracker.track` に初期推定位置と変位ゲートの上書きを追加（再局所化用）
- `MapTracker.applyRelocalization`: 再局所化した姿勢を次の PnP の事前値にする。マップのリセット猶予を 30 → 150 フレーム（5 秒）に延長
- `VisionEngine`: マップ初期化フレームを最初の Keyframe にし、追跡中は Keyframe ポリシーで追加。カメラがマップ内で見失われた（PnP 失敗）フレームの次から毎フレーム再局所化を試行し、成功時は Keyframe 観測を生きた Track として注入（既存 Track とは Landmark / 近傍で統合）。`VisionOutput.relocalization`（Keyframe 数、試行結果、成功回数）
- 状態機械: マップありで PnP 失敗 → `RELOCALIZING`（特徴追跡が続いていても）。成功で PLANE_FOUND / PLANE_DETECTING へ復帰。猶予超過でマップリセット → TRACKING
- `ARSession` / `ARWorld`: RELOCALIZING 中は World を保持し、配置物は最後の姿勢で 1.5 s 表示後に非表示、復帰で再表示（§33）
- GLB 対応: `ARObject.fromModel`（フットプリント正規化、底面を Y = 0 に）、`ARSession.placeObject(model, hit, targetSize)` / `moveObject`、デモは `?model=URL&size=0.15` で Cube の代わりに GLB を配置
- テスト: 単体 116 件（粗シフト探索、**ブランク 8 フレーム後に同じマップへ復帰し配置点の再投影ずれ中央値 0.02 px**、別シーンを見てから戻る、Keyframe 数の上限、再局所化不能時のリセット）。ブラウザテストはループ映像のカット（約 180 px のジャンプ）を再局所化で跨ぎ、`mapFrameId` が全サンプルで不変であることを検証

### Phase 4 — 実装済み

- `src/math/CoordinateSystem.ts`: 座標変換を集約（マップ座標 = 初期化カメラの CV 座標、World = Three.js Y-up）。平面から World フレーム生成（原点 = 平面中心、+Y = 法線、−Z ≈ カメラ視線の平面射影）、map↔world 変換、CV カメラ姿勢 → Three.js カメラ姿勢（とその逆）、内部パラメータ → OpenGL 射影行列、`object-fit: cover` のビューポート内部パラメータ
- `src/math/Ray.ts`: ピクセル → カメラ Ray、Ray の剛体変換、Ray ∩ Plane（`t = −(n·o + d)/(n·r)`、平行・後方は null）
- `src/math/OneEuroFilter.ts`: One Euro フィルタ（スカラー / ベクトル）と速度適応 slerp による四元数平滑化（§34）
- `src/ar/WorldAnchor.ts`: 最初に `found` になった平面で World を一度だけ確定（以後の平面再推定では動かさない）。スケールは `assumedPlaneDistanceMeters`（既定 0.5 m、机想定）でカメラ–平面距離を固定。`hitTest`（処理画像ピクセル → World 平面上の点）、`cameraPose`、マップ再初期化で World 破棄
- `src/rendering/`: `ARCamera`（内部パラメータから射影、Pose smoothing）、`ARWorld`（World ルート、平面グリッド、Tracking Lost 時は 1.5 s 保持してから非表示）、`ARObject`（Cube ファクトリ、GLB 用の汎用 Object3D ラッパ）、`ARRenderer`（透過 WebGL、DPR 上限 2）
- `ARSession`: `threeCanvas`（セッションがレンダラ所有）または `threeScene` / `threeCamera`（既存シーンに接続）、`hitTest(x, y)`（CSS px → `ARHitResult { position: Vector3, normal, distance }`）、`placeCube` / `placeObject`、イベント `worldReady` / `worldLost`、`state` は配置後 `AR_ACTIVE`
- デモ: タップで Cube（10 cm）を配置・移動。HUD に World 行（scale, objects）
- GLB は `ARObject` 経由で `placeObject(object3d, hit)` に渡せる構造（ローダ導入は Cube 成立後）
- テスト: 単体 111 件（World フレームの直交性 / 往復変換 / Three.js カメラ姿勢の往復 / 射影行列とピンホールの一致 / cover マッピング、Ray-Plane、One Euro、WorldAnchor の hitTest 不変性・スケール・マップ破棄、**VisionEngine + WorldAnchor 統合: 合成の机シーケンスで hit test 配置点が 24 フレームのカメラ移動後も同じ机ピクセルへ再投影（中央値 0.01 px、最大 0.02 px）**）、ブラウザテストで hitTest → placeCube → `AR_ACTIVE`、Three.js カメラ高さ ≈ 0.5 m

### Phase 3 — 実装済み

- `src/math/Plane.ts`: 平面モデル、3 点平面、PCA 最小二乗、RANSAC 平面当てはめ（適応反復 + 再フィット）、水平度
- `src/math/PnP.ts`: 事前姿勢からの motion-only 最適化（LM + Huber、so(3) 更新）。平面上の点群でも安定
- `src/vision/LandmarkMap.ts`: Landmark（位置 / 観測数 / 最終観測 / 外れ値カウント）、上限・経過フレームによる削除
- `src/vision/MapTracker.ts`: Phase 2 の ref↔cur 姿勢から二視点初期化（マップ座標 = 参照カメラ、|t| = 1）、毎フレーム PnP でマップ座標系のカメラ姿勢、アンカー観測からの新規三角測量、外れ値 Landmark の除去。スケールはマップ内で一貫
- `src/vision/PlaneDetector.ts`: Landmark に RANSAC 平面 → 法線をカメラ側へ向ける → 水平判定（重力あり: |cos| ≥ 0.90、なし: カメラ −Y を上と仮定し 0.5）→ 時間安定性（法線角・法線方向の中心移動）→ 信頼度 → `found`（連続 5 フレーム安定、短時間の見失いは猶予）
- `src/sensors/GravityProvider.ts`: DeviceMotion の重力をカメラ座標へ変換（画面回転考慮、iOS の権限要求はボタン押下時）。水平判定のみに使用（IMU 融合は Phase 6）。`?gravity=x,y,z` でテスト用に上書き可
- 状態機械: TRACKING → PLANE_DETECTING（マップ初期化）→ PLANE_FOUND（安定水平面）。`ARSession` に `planeFound` / `planeLost` イベント
- `VisionOutput` に `mapPose`（マップ座標系の R, t、PnP inlier、再投影誤差）、`plane`、`landmarks`（デバッグ描画用）
- デバッグ: `PlaneRenderer`（Landmark と平面グリッド・法線を 2D オーバーレイに投影描画）、HUD に Map / Plane / Gravity 行
- テスト: 単体 93 件（RANSAC 平面、PnP 一般/平面/外れ値、PlaneDetector 床/壁/重力なし/見失い、MapTracker 初期化・スケール一貫・新規三角測量、VisionEngine 統合: 45° の机で PLANE_FOUND・壁は不採用・真下の床・重力なしフォールバック・静止）、ブラウザテストで `?gravity=0,0,1` を与え PLANE_FOUND を検証

### Phase 2 — 実装済み

- `src/math/Decomposition.ts`: 対称 Jacobi 固有値分解、3×3 SVD、小行列ユーティリティ
- `src/math/Pose.ts`: 回転・四元数・オイラー角、SO(3) 射影、剛体変換の合成（CV カメラ座標系）
- `src/math/EssentialMatrix.ts`: 正規化 8 点法、Sampson 距離、RANSAC、E 分解、recoverPose（cheirality）
- `src/math/HomographyDecomposition.ts`: Faugeras 法による H → (R, t/d, n) 分解、純回転判定、正深度サポート
- `src/math/Triangulation.ts`: 線形三角測量（cheirality 判定用。Landmark 化は Phase 3）
- `src/vision/PoseEstimator.ts`: H と E を両方 RANSAC で当てはめ、inlier 比でモデル選択（ORB-SLAM 流 0.45）。視差不足時は回転のみ。前フレームの法線で H の双対解を解消
- `VisionEngine`: 参照フレーム管理（追跡が参照フレームに結びつく点で ref→cur の二視点幾何を解き、点数不足または視差過大で参照を更新・回転を合成）。`VisionOutput.pose`（累積回転 / 四元数 / 単位並進方向 / モデル / 信頼度）
- HUD に Pose（yaw/pitch/roll、t 方向、モデル、視差、信頼度）を表示。`quality.poseDelta` を計算
- Pose smoothing（§34）は Three.js カメラへ反映する Phase 4 で導入する
- テスト: 単体 73 件（純回転 / 平面上の並進 / 参照更新 / 静止 / 非平面 E 経路 / 前進）、ブラウザテストで t 方向と回転を検証

### Phase 1 — 実装済み

- 純 TypeScript 実装（OpenCV.js 不使用、TypedArray ベース。後で WASM に置換可能な粒度でモジュール化）
- `src/camera/`: getUserMedia ラッパ（エラーコード対応）、処理解像度への縮小 + グレースケール、バッファプール
- `src/vision/`: ImagePyramid（3 段 + 勾配）、FAST-9 + NMS + Grid 分散 + 適応閾値、Pyramidal LK + Forward-Backward、Homography RANSAC、VisionEngine、TrackingQuality
- `src/worker/`: Transferable ベースの Worker プロトコル、Worker / Main-thread 両バックエンド
- `src/ar/`: ARConfig（全チューニング値）、ARState（状態機械）、ARSession（公開 API）
- `src/debug/`: HUD、特徴点・モーションベクトル描画、`[AR]` ロガー
- テスト: `npm test`（Vitest 単体 46 件）、`npm run test:browser`（headless Chromium + 合成カメラ映像）

### 実機検証の状況

- iPhone Safari（Vercel プレビュー）: 床で `PLANE_FOUND` → タップで Cube 配置 → `AR_ACTIVE` を確認。Vision 30 fps / 18 ms、特徴点 263、Landmark 134 / PnP inlier 127。ユーザー確認済み: 平面検出までの挙動と Cube の大きさに違和感なし（2026-10-03）
- 未確認: 端末を動かしたときの Cube の固定度（ドリフト / ジッタ）、Lost からの復帰の体感、Android Chrome

### 未実装（Phase 6 以降）

IMU 融合（Madgwick 等）/ Local BA / WASM SIMD 最適化。大きな視点変化からの再局所化（記述子マッチング）は対象外（§2）。
