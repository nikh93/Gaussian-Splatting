package io.github.nikh93.gsplat.capture

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
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
import java.util.EnumSet
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.min
import kotlin.math.sqrt

/**
 * AR capture screen: ARCore tracks the phone pose; photos are taken manually or
 * automatically whenever the phone was moved far enough while being held still.
 * For every photo the camera pose, intrinsics and (if available) depth points are
 * recorded - everything the Gaussian Splatting trainer needs.
 */
class CaptureActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var binding: ActivityCaptureBinding
    private lateinit var settings: AppSettings
    private lateinit var repo: CaptureRepository
    private lateinit var capture: Capture
    private lateinit var recorder: CaptureRecorder
    private lateinit var rotationHelper: DisplayRotationHelper

    private var session: Session? = null
    private var installRequested = false
    private var depthSupported = false
    private val background = BackgroundRenderer()
    private val pointRenderer = PointCloudRenderer()
    private val photoMarkers = PhotoMarkerRenderer()
    private var shownPreviewVersion = -1

    @Volatile private var captureRequested = false
    @Volatile private var autoCapture = true
    @Volatile private var useDepth = true
    @Volatile private var finishing = false
    @Volatile private var showPhotos = true
    private var lastOverlap: Float? = null

    private var lastCapturePose: Pose? = null
    private var lastCaptureTime = 0L
    private var prevPose: Pose? = null
    private var prevTimeNs = 0L
    private var frameCounter = 0

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

        settings = AppSettings(this)
        repo = CaptureRepository(this)
        capture = repo.create()
        recorder = CaptureRecorder(capture, repo)
        rotationHelper = DisplayRotationHelper(this)

        autoCapture = settings.autoCapture
        useDepth = settings.useDepth
        binding.autoSwitch.isChecked = autoCapture
        binding.depthSwitch.isChecked = useDepth
        binding.autoSwitch.setOnCheckedChangeListener { _, checked ->
            autoCapture = checked
            settings.autoCapture = checked
        }
        binding.depthSwitch.setOnCheckedChangeListener { _, checked ->
            useDepth = checked
            settings.useDepth = checked
            configureSession()
        }
        binding.photosSwitch.setOnCheckedChangeListener { _, checked -> showPhotos = checked }
        binding.captureButton.setOnClickListener { captureRequested = true }
        binding.doneButton.setOnClickListener { finishCapture() }
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = finishCapture()
        })

        binding.surface.preserveEGLContextOnPause = true
        binding.surface.setEGLContextClientVersion(3)
        binding.surface.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        binding.surface.setRenderer(this)
        binding.surface.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        updateStats()
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
                session = s
                configureSession()
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
        binding.surface.onResume()
        rotationHelper.onResume()
    }

    override fun onPause() {
        super.onPause()
        rotationHelper.onPause()
        binding.surface.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        session?.close()
        session = null
        if (!finishing) {
            // activity closed without "Fertig": still save what we have
            recorder.finish { if (capture.frames == 0) repo.delete(capture) }
        }
    }

    /** Use the highest CPU image resolution for the photos (back camera, 30 fps). */
    private fun selectCameraConfig(s: Session) {
        val filter = CameraConfigFilter(s)
            .setFacingDirection(CameraConfig.FacingDirection.BACK)
            .setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30))
        val configs = s.getSupportedCameraConfigs(filter)
        val best = configs.maxByOrNull { it.imageSize.width * it.imageSize.height }
        if (best != null) {
            s.cameraConfig = best
            Log.i(TAG, "camera config: cpu ${best.imageSize} gpu ${best.textureSize}")
        }
    }

    private fun configureSession() {
        val s = session ?: return
        depthSupported = s.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
        val config = Config(s).apply {
            focusMode = Config.FocusMode.AUTO
            updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            planeFindingMode = Config.PlaneFindingMode.DISABLED
            lightEstimationMode = Config.LightEstimationMode.DISABLED
            depthMode = if (depthSupported && useDepth) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
        }
        s.configure(config)
        runOnUiThread { binding.depthSwitch.isEnabled = depthSupported }
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

    // ------------------------------------------------------------------
    // GL renderer

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES30.glClearColor(0f, 0f, 0f, 1f)
        background.createOnGlThread()
        pointRenderer.createOnGlThread()
        photoMarkers.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES30.glViewport(0, 0, width, height)
        rotationHelper.onSurfaceChanged(width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        rotationHelper.updateSessionIfNeeded(s)
        val frame: Frame = try {
            s.setCameraTextureName(background.textureId)
            s.update()
        } catch (e: Exception) {
            Log.w(TAG, "session.update failed", e)
            return
        }
        background.draw(frame)

        val camera = frame.camera
        val tracking = camera.trackingState == TrackingState.TRACKING
        val pose = camera.pose
        val now = frame.timestamp

        // motion estimate for blur avoidance
        var speed = 0f
        var angSpeed = 0f
        prevPose?.let { p ->
            val dt = (now - prevTimeNs) / 1e9f
            if (dt > 0f) {
                speed = distance(p, pose) / dt
                angSpeed = angleDeg(p, pose) / dt
            }
        }
        prevPose = pose
        prevTimeNs = now

        if (tracking) {
            camera.getViewMatrix(viewMatrix, 0)
            camera.getProjectionMatrix(projMatrix, 0, 0.05f, 50f)
            Matrix.multiplyMM(viewProj, 0, projMatrix, 0, viewMatrix, 0)

            if (++frameCounter % 15 == 0) {
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

            // overlap of the current view with the last photo (projected image rectangle)
            lastOverlap = photoMarkers.overlapWithLast(viewProj)
            val ov = lastOverlap
            val still = speed < 0.35f && angSpeed < 45f
            val needPhoto = movedEnough(pose) || (ov != null && ov < AUTO_OVERLAP)
            val auto = autoCapture && still && needPhoto && now - lastCaptureTime > 400_000_000L &&
                recorder.pendingJobs < 4
            if (!finishing && (captureRequested || auto)) {
                captureRequested = false
                if (takePhoto(frame, pose)) {
                    lastCapturePose = pose
                    lastCaptureTime = now
                    runOnUiThread { flash() }
                }
            }

            if (recorder.previewVersion != shownPreviewVersion) {
                shownPreviewVersion = recorder.previewVersion
                val p = recorder.preview
                pointRenderer.update(p, p.size / 6)
            }
            pointRenderer.draw(viewProj)
            if (showPhotos) photoMarkers.draw(viewProj)
        } else {
            captureRequested = false
            lastOverlap = null
        }

        if (frameCounter % 10 == 0 || !tracking) {
            val reason = camera.trackingFailureReason
            val tooFast = tracking && (speed > 0.6f || angSpeed > 90f)
            val ov = lastOverlap
            runOnUiThread {
                updateTracking(camera.trackingState, reason, tooFast)
                updateOverlap(ov)
            }
        }
    }

    private fun movedEnough(pose: Pose): Boolean {
        val last = lastCapturePose ?: return true
        return distance(last, pose) > 0.07f || angleDeg(last, pose) > 10f
    }

    /** Grabs CPU image + pose + intrinsics (+ depth) of the current frame. */
    private fun takePhoto(frame: Frame, pose: Pose): Boolean {
        val camera = frame.camera
        val yuv = try {
            frame.acquireCameraImage().use { YuvFrame.copyOf(it) }
        } catch (e: NotYetAvailableException) {
            return false
        } catch (e: Exception) {
            Log.w(TAG, "acquireCameraImage failed", e)
            return false
        }
        val ii = camera.imageIntrinsics
        val focal = ii.focalLength
        val pp = ii.principalPoint
        val dims = ii.imageDimensions
        val intr = Intrinsics(focal[0], focal[1], pp[0], pp[1], dims[0], dims[1])
        val c2w = FloatArray(16)
        pose.toMatrix(c2w, 0)
        val depth = if (useDepth && depthSupported) sampleDepth(frame) else null
        val imgIntr = if (intr.width != yuv.width || intr.height != yuv.height) intr.scaledTo(yuv.width, yuv.height) else intr
        addPhotoMarker(yuv, c2w, imgIntr, sceneDepth(frame, c2w, depth))
        recorder.addFrame(yuv, c2w, intr, frame.timestamp, depth)
        runOnUiThread { updateStats() }
        return true
    }

    /** Thumbnail of the photo, shown in AR at the photo's pose (GL thread). */
    private fun addPhotoMarker(yuv: YuvFrame, c2w: FloatArray, intr: Intrinsics, sceneDepth: Float) {
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
        photoMarkers.add(c2w, intr, buf, tw, th, sceneDepth)
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
    // UI

    private fun updateStats() {
        binding.statsText.text = getString(R.string.capture_stats, recorder.frameCount, recorder.pointCount)
    }

    private fun updateTracking(state: TrackingState, reason: TrackingFailureReason, tooFast: Boolean) {
        binding.trackingText.text = when {
            state == TrackingState.TRACKING && tooFast -> getString(R.string.too_fast)
            state == TrackingState.TRACKING -> getString(R.string.tracking_ok)
            reason == TrackingFailureReason.NONE -> getString(R.string.tracking_init)
            else -> getString(R.string.tracking_lost, reasonText(reason))
        }
        binding.captureButton.isEnabled = state == TrackingState.TRACKING && !finishing
        updateStats()
    }

    private fun updateOverlap(ov: Float?) {
        val v = binding.overlapText
        if (ov == null || !showPhotos) {
            v.visibility = View.GONE
            return
        }
        val pct = (ov * 100f).toInt()
        v.visibility = View.VISIBLE
        v.text = getString(if (ov < 0.4f) R.string.overlap_low else R.string.overlap_text, pct)
        v.setTextColor(
            when {
                ov < 0.4f -> 0xFFFF5252.toInt()  // gap ahead: take a photo
                ov <= 0.85f -> 0xFF69F0AE.toInt() // recommended range
                else -> 0xFFFFD740.toInt()        // very similar to the last photo
            }
        )
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
