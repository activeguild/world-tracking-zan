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
