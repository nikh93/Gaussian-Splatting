package io.github.nikh93.gsplat.capture

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.opengl.GLES30
import android.opengl.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.Choreographer
import android.view.SurfaceHolder
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import com.google.ar.core.Anchor
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import com.google.ar.core.exceptions.UnavailableException
import io.github.nikh93.gsplat.AppSettings
import io.github.nikh93.gsplat.ProjectActivity
import io.github.nikh93.gsplat.R
import io.github.nikh93.gsplat.data.Capture
import io.github.nikh93.gsplat.data.CaptureRepository
import io.github.nikh93.gsplat.databinding.ActivityCaptureBinding
import io.github.nikh93.gsplat.util.Insets
import java.util.EnumSet
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.min
import kotlin.math.sqrt

/**
 * AR capture screen: ARCore tracks the phone pose; photos are taken manually or
 * automatically whenever the phone was moved far enough while being held still.
 * For every photo the camera pose, intrinsics and (if available) depth points are
 * recorded - everything the Gaussian Splatting trainer needs.
 *
 * ARCore and the GL rendering run on the main thread, paced by the Choreographer (see
 * [EglCore] for why). The options are chosen beforehand in [CaptureSetupActivity].
 */
class CaptureActivity : AppCompatActivity(), SurfaceHolder.Callback, Choreographer.FrameCallback {

    private lateinit var binding: ActivityCaptureBinding
    private lateinit var settings: AppSettings
    private lateinit var repo: CaptureRepository
    private lateinit var capture: Capture
    private lateinit var recorder: CaptureRecorder
    private lateinit var rotationHelper: DisplayRotationHelper

    private var session: Session? = null
    private var installRequested = false
    private var depthSupported = false
    private lateinit var egl: EglCore
    private val background = BackgroundRenderer()
    private val photoMarkers = PhotoMarkerRenderer()
    private var running = false
    private var lastFrameTimestamp = 0L

    private var captureRequested = false
    private var autoCapture = false
    private var useDepth = false
    private var showPhotos = true
    private var finishing = false

    /* blur check: photos are compared with the recent sharpness of the scene */
    private val sharpnessRef = SharpnessReference(60)
    private var liveSharp: Boolean? = null
    private var captureDeadlineNs = 0L
    private var lastAttemptTimestamp = 0L

    /* which viewing directions around the object have a photo */
    private val coverage = CoverageTracker()

    /* a short message in the status pill, e.g. after a blurry photo */
    private var noticeText: String? = null
    private var noticeUntilMs = 0L

    private var lastCaptureTime = 0L
    private var prevPose: Pose? = null
    private var prevTimeNs = 0L
    private var trackingSinceNs = 0L
    private var lastReason: TrackingFailureReason? = null
    private var frameCounter = 0

    /* tracking statistics, logged once per second for diagnosis */
    private var statsSinceMs = 0L
    private var statsFrames = 0
    private var statsTracked = 0

    /*
     * Own IMU listener at the fastest rate. On the Galaxy A56 ARCore alone received only
     * ~6 IMU samples per second and tracking kept failing; with this listener the sensors
     * run fast and ARCore gets ~250 per second (tracking then held on every camera frame).
     * The counts go to the log once per second.
     */
    private lateinit var sensorManager: SensorManager
    private var imuGyro = 0
    private var imuAccel = 0
    private val imuListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {
            if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) imuAccel++ else imuGyro++
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private val viewMatrix = FloatArray(16)
    private val projMatrix = FloatArray(16)
    private val viewProj = FloatArray(16)

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (!granted) {
                Toast.makeText(this, R.string.camera_permission_needed, Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCaptureBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        Insets.pad(binding.topPanel, top = true)
        Insets.pad(binding.bottomPanel, bottom = true)
        // hide the system bars during capture (swipe from the edge shows them again)
        WindowCompat.getInsetsController(window, binding.root).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }

        settings = AppSettings(this)
        repo = CaptureRepository(this)
        capture = repo.create()
        recorder = CaptureRecorder(capture, repo)
        rotationHelper = DisplayRotationHelper(this)
        sensorManager = getSystemService(SensorManager::class.java)

        autoCapture = settings.autoCapture
        useDepth = settings.useDepth
        showPhotos = settings.showPhotos
        binding.coverageView.visibility = if (settings.showCoverage) View.VISIBLE else View.GONE

        binding.captureButton.setOnClickListener { captureRequested = true }
        binding.undoButton.setOnClickListener { undoLastPhoto() }
        binding.doneButton.setOnClickListener { finishCapture() }
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = finishCapture()
        })

        egl = EglCore()
        GLES30.glClearColor(0f, 0f, 0f, 1f)
        background.createOnGlThread()
        photoMarkers.createOnGlThread()
        binding.surface.holder.addCallback(this)
        updateUi(stable = false, state = TrackingState.PAUSED, reason = TrackingFailureReason.NONE, tooFast = false)
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }
        if (session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        installRequested = true
                        return
                    }
                    else -> {}
                }
                val s = Session(this)
                selectCameraConfig(s)
                configureSession(s)
                s.setCameraTextureName(background.textureId)
                session = s
            } catch (e: UnavailableException) {
                Toast.makeText(this, getString(R.string.arcore_not_supported, e.javaClass.simpleName), Toast.LENGTH_LONG)
                    .show()
                finish()
                return
            } catch (e: Exception) {
                Toast.makeText(this, getString(R.string.arcore_not_supported, e.message), Toast.LENGTH_LONG).show()
                finish()
                return
            }
        }
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            Toast.makeText(this, getString(R.string.arcore_not_supported, "camera"), Toast.LENGTH_LONG).show()
            session = null
            finish()
            return
        }
        rotationHelper.onResume()
        registerImu()
        running = true
        Choreographer.getInstance().postFrameCallback(this)
    }

    override fun onPause() {
        super.onPause()
        running = false
        Choreographer.getInstance().removeFrameCallback(this)
        rotationHelper.onPause()
        sensorManager.unregisterListener(imuListener)
        session?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        session?.close()
        session = null
        egl.release()
        if (!finishing) {
            // activity closed without "Fertig": still save what we have
            recorder.finish { if (capture.frames == 0) repo.delete(capture) }
        }
    }

    /** CPU image resolution for the photos (back camera, 30 fps), see [AppSettings.photoResolution]. */
    private fun selectCameraConfig(s: Session) {
        val res = settings.photoResolution
        if (res != AppSettings.RES_STANDARD) {
            val filter = CameraConfigFilter(s)
                .setFacingDirection(CameraConfig.FacingDirection.BACK)
                .setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30))
            val configs = s.getSupportedCameraConfigs(filter)
            val area = { c: CameraConfig -> c.imageSize.width * c.imageSize.height }
            val chosen = if (res == AppSettings.RES_MEDIUM) {
                configs.filter { area(it) >= 1280 * 720 }.minByOrNull(area) ?: configs.maxByOrNull(area)
            } else {
                configs.maxByOrNull(area)
            }
            chosen?.let { s.cameraConfig = it }
        }
        val c = s.cameraConfig
        Log.i(TAG, "camera config: cpu ${c.imageSize} gpu ${c.textureSize} fps ${c.fpsRange} resolution=$res")
    }

    private fun registerImu() {
        for (type in intArrayOf(Sensor.TYPE_GYROSCOPE_UNCALIBRATED, Sensor.TYPE_ACCELEROMETER)) {
            val sensor = sensorManager.getDefaultSensor(type) ?: continue
            val ok = sensorManager.registerListener(imuListener, sensor, SensorManager.SENSOR_DELAY_FASTEST)
            Log.i(TAG, "imu listener ${sensor.name} minDelay ${sensor.minDelay} us registered=$ok")
        }
    }

    private fun configureSession(s: Session) {
        depthSupported = s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        // Same configuration as SceneView (the AR test app that tracks fine on the Galaxy
        // A56). With plane finding and light estimation disabled ARCore got the IMU at only
        // ~6 samples per second there (160 ms apart instead of 8 ms) and tracking kept
        // dropping out; the planes themselves are not used.
        val config = Config(s).apply {
            focusMode = Config.FocusMode.AUTO
            updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
            lightEstimationMode = Config.LightEstimationMode.ENVIRONMENTAL_HDR
            depthMode = if (depthSupported && useDepth) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
        }
        s.configure(config)
        Log.i(
            TAG,
            "session config: depth ${config.depthMode} (supported=$depthSupported) focus ${config.focusMode} " +
                "planes ${config.planeFindingMode} light ${config.lightEstimationMode}, main thread"
        )
    }

    private fun finishCapture() {
        if (finishing) return
        finishing = true
        binding.captureButton.isEnabled = false
        binding.doneButton.isEnabled = false
        recorder.finish {
            runOnUiThread {
                if (capture.frames == 0) {
                    repo.delete(capture)
                    finish()
                } else {
                    Toast.makeText(this, getString(R.string.capture_saved, capture.frames), Toast.LENGTH_SHORT).show()
                    startActivity(Intent(this, ProjectActivity::class.java).putExtra(ProjectActivity.EXTRA_ID, capture.id))
                    finish()
                }
            }
        }
    }

    private fun undoLastPhoto() {
        if (finishing || recorder.frameCount == 0) return
        egl.makeCurrent()
        photoMarkers.removeLast()
        coverage.removeLast()
        recorder.removeLast()
        Log.i(TAG, "photo removed, ${recorder.frameCount} left")
        notice(getString(R.string.photo_removed))
    }

    // ------------------------------------------------------------------
    // surface and frame loop (main thread)

    override fun surfaceCreated(holder: SurfaceHolder) {
        egl.attach(holder.surface)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        egl.makeCurrent()
        GLES30.glViewport(0, 0, width, height)
        rotationHelper.onSurfaceChanged(width, height)
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        egl.detach()
    }

    override fun doFrame(frameTimeNanos: Long) {
        if (!running) return
        Choreographer.getInstance().postFrameCallback(this)
        try {
            drawFrame()
        } catch (e: Exception) {
            Log.w(TAG, "frame failed", e)
        }
    }

    private fun drawFrame() {
        val s = session ?: return
        if (!egl.hasWindow) return
        egl.makeCurrent()
        rotationHelper.updateSessionIfNeeded(s)
        val frame: Frame = s.update()
        // the display runs faster than the camera: nothing new to show
        if (frame.timestamp == lastFrameTimestamp) return
        lastFrameTimestamp = frame.timestamp

        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        background.draw(frame)

        val camera = frame.camera
        val tracking = camera.trackingState == TrackingState.TRACKING
        val pose = camera.pose
        val now = frame.timestamp

        val reason = camera.trackingFailureReason
        if (reason != lastReason) {
            Log.i(TAG, "tracking ${camera.trackingState} reason $reason")
            lastReason = reason
        }
        logTrackingStats(tracking)

        // motion estimate for blur avoidance; only from tracked poses, the pose
        // jumps when tracking (re)starts
        var speed = 0f
        var angSpeed = 0f
        if (tracking) {
            val p = prevPose
            if (p == null || now - prevTimeNs > MAX_TRACKING_GAP_NS) {
                trackingSinceNs = now
            } else {
                val dt = (now - prevTimeNs) / 1e9f
                if (dt > 0f) {
                    speed = distance(p, pose) / dt
                    angSpeed = angleDeg(p, pose) / dt
                }
            }
            prevPose = pose
            prevTimeNs = now
        }
        // ARCore may report TRACKING for single frames while it initialises, and on
        // some phones drops single frames to PAUSED while tracking; short gaps are tolerated
        val stable = prevPose != null && now - prevTimeNs <= MAX_TRACKING_GAP_NS &&
            prevTimeNs - trackingSinceNs > STABLE_TRACKING_NS

        var cameraPosition: FloatArray? = null
        if (tracking) {
            camera.getViewMatrix(viewMatrix, 0)
            camera.getProjectionMatrix(projMatrix, 0, 0.05f, 50f)
            Matrix.multiplyMM(viewProj, 0, projMatrix, 0, viewMatrix, 0)
            photoMarkers.updatePoses()

            if (stable && frameCounter % 15 == 7) sampleSharpness(frame)
            if (frameCounter % 15 == 0) submitFeaturePoints(frame)

            // auto mode: a new photo only when the view has moved on far enough that the
            // overlap with the last photo dropped below AUTO_OVERLAP (the first one when still)
            val ov = photoMarkers.overlapWithLast(viewProj)
            val still = speed < 0.35f && angSpeed < 45f
            val needPhoto = ov == null || ov < AUTO_OVERLAP
            val auto = autoCapture && still && needPhoto && now - lastCaptureTime > AUTO_MIN_INTERVAL_NS &&
                recorder.pendingJobs < 3
            // a tap waits up to MANUAL_WAIT_NS for a sharp frame, then takes the photo anyway
            if (captureRequested && captureDeadlineNs == 0L) captureDeadlineNs = now + MANUAL_WAIT_NS
            if (!finishing && stable && (captureRequested || auto) && now != lastAttemptTimestamp) {
                lastAttemptTimestamp = now
                val force = captureRequested && now > captureDeadlineNs
                val result = takePhoto(s, frame, pose, force)
                if (result == PhotoResult.TAKEN || result == PhotoResult.TAKEN_BLURRY) {
                    captureRequested = false
                    captureDeadlineNs = 0L
                    lastCaptureTime = now
                    flash()
                    if (result == PhotoResult.TAKEN_BLURRY) notice(getString(R.string.photo_blurry))
                }
            }
            cameraPosition = floatArrayOf(pose.tx(), pose.ty(), pose.tz())
            if (showPhotos) photoMarkers.draw(viewProj)
        }
        if (!stable) {
            captureRequested = false
            captureDeadlineNs = 0L
        }
        egl.swap()

        if (++frameCounter % 10 == 0 || !stable) {
            val tooFast = stable && (speed > 0.6f || angSpeed > 90f)
            updateUi(stable, camera.trackingState, reason, tooFast)
            updateCoverage(if (stable) cameraPosition else null)
        }
    }

    private fun logTrackingStats(tracking: Boolean) {
        val nowMs = SystemClock.elapsedRealtime()
        if (statsSinceMs == 0L) statsSinceMs = nowMs
        statsFrames++
        if (tracking) statsTracked++
        if (nowMs - statsSinceMs >= 1000L) {
            Log.i(
                TAG,
                "tracking stats: $statsTracked of $statsFrames camera frames tracked, " +
                    "app imu gyro $imuGyro accel $imuAccel per s"
            )
            imuGyro = 0
            imuAccel = 0
            statsSinceMs = nowMs
            statsFrames = 0
            statsTracked = 0
        }
    }

    private fun submitFeaturePoints(frame: Frame) {
        frame.acquirePointCloud().use { pc ->
            val ids = pc.ids
            val pts = pc.points
            val n = ids.remaining()
            if (n > 0) {
                val idArr = IntArray(n)
                ids.get(idArr)
                val ptArr = FloatArray(n * 4)
                pts.get(ptArr)
                recorder.submitFeaturePoints(idArr, ptArr, n)
            }
        }
    }

    private enum class PhotoResult { TAKEN, TAKEN_BLURRY, REJECTED_BLURRY, UNAVAILABLE }

    /** Sharpness of the current camera image for the live indicator and the reference. */
    private fun sampleSharpness(frame: Frame) {
        try {
            frame.acquireCameraImage().use { img ->
                val y = img.planes[0]
                val v = Sharpness.of(y.buffer, y.rowStride, img.width, img.height)
                sharpnessRef.add(v)
                liveSharp = sharpnessRef.isSharp(v)
            }
        } catch (e: NotYetAvailableException) {
            // no CPU image for this frame
        } catch (e: Exception) {
            Log.w(TAG, "sharpness sample failed", e)
        }
    }

    /**
     * Grabs CPU image + pose + intrinsics (+ depth) of the current frame. A blurry image is
     * rejected (the caller retries on the next camera frame) unless [force] is set.
     */
    private fun takePhoto(s: Session, frame: Frame, pose: Pose, force: Boolean): PhotoResult {
        val camera = frame.camera
        var sharpness = 0f
        var sharp = false
        // the blur check runs on the camera image itself, a rejected frame is never copied
        val yuv = try {
            frame.acquireCameraImage().use { img ->
                val y = img.planes[0]
                sharpness = Sharpness.of(y.buffer, y.rowStride, img.width, img.height)
                sharp = sharpnessRef.isSharp(sharpness)
                if (!sharp && !force) null else YuvFrame.copyOf(img)
            }
        } catch (e: NotYetAvailableException) {
            return PhotoResult.UNAVAILABLE
        } catch (e: Exception) {
            Log.w(TAG, "acquireCameraImage failed", e)
            return PhotoResult.UNAVAILABLE
        } ?: return PhotoResult.REJECTED_BLURRY
        val ii = camera.imageIntrinsics
        val focal = ii.focalLength
        val pp = ii.principalPoint
        val dims = ii.imageDimensions
        val intr = Intrinsics(focal[0], focal[1], pp[0], pp[1], dims[0], dims[1])
        val c2w = FloatArray(16)
        pose.toMatrix(c2w, 0)
        val depth = if (useDepth && depthSupported) sampleDepth(frame) else null
        val imgIntr = if (intr.width != yuv.width || intr.height != yuv.height) intr.scaledTo(yuv.width, yuv.height) else intr
        val distance = sceneDepth(frame, c2w, depth)
        // the marker hangs on an anchor so it stays where the photo was taken
        val anchor: Anchor? = try {
            s.createAnchor(pose)
        } catch (e: Exception) {
            Log.w(TAG, "createAnchor failed", e)
            null
        }
        addPhotoMarker(yuv, c2w, anchor, imgIntr, distance)
        coverage.addPhoto(c2w, distance)
        recorder.addFrame(yuv, c2w, intr, frame.timestamp, depth, sharpness)
        Log.i(TAG, "photo ${recorder.frameCount}: sharpness $sharpness (scene median ${sharpnessRef.median()}) forced=$force")
        return if (sharp) PhotoResult.TAKEN else PhotoResult.TAKEN_BLURRY
    }

    /** Thumbnail of the photo, shown in AR at the photo's pose. */
    private fun addPhotoMarker(yuv: YuvFrame, c2w: FloatArray, anchor: Anchor?, intr: Intrinsics, sceneDepth: Float) {
        val tw = 160
        val th = maxOf(1, tw * yuv.height / yuv.width)
        val buf = java.nio.ByteBuffer.allocateDirect(tw * th * 4)
        val rgb = IntArray(3)
        for (y in 0 until th) {
            val sy = y * yuv.height / th
            for (x in 0 until tw) {
                yuv.rgbAt(x * yuv.width / tw, sy, rgb)
                buf.put(rgb[0].toByte()).put(rgb[1].toByte()).put(rgb[2].toByte()).put(255.toByte())
            }
        }
        photoMarkers.add(c2w, anchor, intr, buf, tw, th, sceneDepth)
    }

    /** Typical distance to the scene for this photo: depth samples, else feature points, else 1 m. */
    private fun sceneDepth(frame: Frame, c2w: FloatArray, depth: DepthSamples?): Float {
        val values = ArrayList<Float>()
        if (depth != null && depth.count > 0) {
            for (i in 0 until depth.count) values.add(depth.depth[i])
        } else {
            try {
                frame.acquirePointCloud().use { pc ->
                    val pts = pc.points
                    val n = pts.remaining() / 4
                    for (i in 0 until n) {
                        val dx = pts.get(i * 4) - c2w[12]
                        val dy = pts.get(i * 4 + 1) - c2w[13]
                        val dz = pts.get(i * 4 + 2) - c2w[14]
                        // depth along the viewing direction (-z axis of the camera)
                        val d = -(c2w[8] * dx + c2w[9] * dy + c2w[10] * dz)
                        if (d in 0.1f..10f) values.add(d)
                    }
                }
            } catch (e: Exception) {
                // no point cloud this frame
            }
        }
        if (values.size < 5) return 1f
        values.sort()
        return values[values.size / 2]
    }

    private fun sampleDepth(frame: Frame): DepthSamples? {
        return try {
            frame.acquireDepthImage16Bits().use { img ->
                val w = img.width
                val h = img.height
                val plane = img.planes[0]
                val buf = plane.buffer.order(java.nio.ByteOrder.nativeOrder())
                val rowStride = plane.rowStride
                val pixStride = plane.pixelStride
                val ti = frame.camera.textureIntrinsics
                val tdim = ti.imageDimensions
                val tf = ti.focalLength
                val tp = ti.principalPoint
                val intr = Intrinsics(tf[0], tf[1], tp[0], tp[1], tdim[0], tdim[1]).scaledTo(w, h)
                val step = maxOf(1, min(w, h) / 60)
                val cap = (w / step + 1) * (h / step + 1)
                val us = FloatArray(cap)
                val vs = FloatArray(cap)
                val ds = FloatArray(cap)
                var n = 0
                var y = step / 2
                while (y < h) {
                    var x = step / 2
                    while (x < w) {
                        val mm = buf.getShort(y * rowStride + x * pixStride).toInt() and 0xFFFF
                        val d = mm / 1000f
                        if (d in 0.1f..4.0f) {
                            us[n] = x.toFloat()
                            vs[n] = y.toFloat()
                            ds[n] = d
                            n++
                        }
                        x += step
                    }
                    y += step
                }
                DepthSamples(us, vs, ds, n, intr)
            }
        } catch (e: NotYetAvailableException) {
            null
        } catch (e: Exception) {
            Log.w(TAG, "depth unavailable", e)
            null
        }
    }

    // ------------------------------------------------------------------
    // UI: one status pill, the shutter, undo and "Fertig"

    private fun notice(text: String) {
        noticeText = text
        noticeUntilMs = SystemClock.uptimeMillis() + NOTICE_MS
        binding.statusText.text = text
        binding.statusText.setTextColor(WARN_COLOR)
    }

    private fun updateUi(stable: Boolean, state: TrackingState, reason: TrackingFailureReason, tooFast: Boolean) {
        val photos = recorder.frameCount
        val notice = noticeText?.takeIf { SystemClock.uptimeMillis() < noticeUntilMs }
        val warning = when {
            notice != null -> notice
            !stable && (state == TrackingState.TRACKING || reason == TrackingFailureReason.NONE) ->
                getString(R.string.tracking_init)
            !stable -> getString(R.string.tracking_lost, reasonText(reason))
            tooFast -> getString(R.string.too_fast)
            liveSharp == false -> getString(R.string.sharpness_blurry)
            else -> null
        }
        val status = warning ?: when {
            photos == 0 && autoCapture -> getString(R.string.status_auto_start)
            photos == 0 -> getString(R.string.status_manual_start)
            autoCapture -> getString(R.string.status_auto_count, resources.getQuantityString(R.plurals.photos, photos, photos))
            else -> resources.getQuantityString(R.plurals.photos, photos, photos)
        }
        binding.statusText.text = status
        binding.statusText.setTextColor(if (warning != null) WARN_COLOR else 0xFFFFFFFF.toInt())
        binding.captureButton.isEnabled = stable && !finishing
        binding.undoButton.isEnabled = photos > 0 && !finishing
        binding.undoButton.alpha = if (photos > 0) 1f else 0.35f
    }

    /** Coverage dome in the corner and as progress ring around the shutter. */
    private fun updateCoverage(cameraPos: FloatArray?) {
        val s = coverage.snapshot(cameraPos)
        if (binding.coverageView.visibility == View.VISIBLE) binding.coverageView.update(s)
        binding.captureButton.progress = s.covered.toFloat() / s.total
    }

    private fun reasonText(r: TrackingFailureReason): String = when (r) {
        TrackingFailureReason.BAD_STATE -> "interner Fehler"
        TrackingFailureReason.INSUFFICIENT_LIGHT -> "zu dunkel"
        TrackingFailureReason.EXCESSIVE_MOTION -> "zu schnelle Bewegung"
        TrackingFailureReason.INSUFFICIENT_FEATURES -> "zu wenig Struktur im Bild"
        TrackingFailureReason.CAMERA_UNAVAILABLE -> "Kamera nicht verfügbar"
        else -> r.name
    }

    private fun flash() {
        binding.flash.alpha = 0.6f
        binding.flash.animate().alpha(0f).setDuration(180).start()
    }

    companion object {
        private const val TAG = "CaptureActivity"
        /** automatic capture also triggers when the overlap with the last photo drops below this */
        private const val AUTO_OVERLAP = 0.7f
        /** at most one automatic photo per second */
        private const val AUTO_MIN_INTERVAL_NS = 1_000_000_000L
        private const val STABLE_TRACKING_NS = 500_000_000L
        private const val MAX_TRACKING_GAP_NS = 300_000_000L
        private const val MANUAL_WAIT_NS = 1_500_000_000L
        private const val NOTICE_MS = 2500L
        private const val WARN_COLOR = 0xFFFFD740.toInt()

        fun distance(a: Pose, b: Pose): Float {
            val dx = a.tx() - b.tx()
            val dy = a.ty() - b.ty()
            val dz = a.tz() - b.tz()
            return sqrt(dx * dx + dy * dy + dz * dz)
        }

        /** Rotation angle between two poses in degrees. */
        fun angleDeg(a: Pose, b: Pose): Float {
            val qa = a.rotationQuaternion
            val qb = b.rotationQuaternion
            val dot = abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]).coerceAtMost(1f)
            return Math.toDegrees(2.0 * acos(dot.toDouble())).toFloat()
        }
    }
}
