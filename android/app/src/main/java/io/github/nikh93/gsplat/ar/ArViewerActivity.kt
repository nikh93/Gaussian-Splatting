package io.github.nikh93.gsplat.ar

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.opengl.GLES30
import android.opengl.Matrix
import android.os.Bundle
import android.util.Log
import android.view.Choreographer
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.SurfaceHolder
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.lifecycleScope
import com.google.ar.core.Anchor
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.cardboard.sdk.Initialize
import com.google.cardboard.sdk.QrCode
import io.github.nikh93.gsplat.R
import io.github.nikh93.gsplat.capture.BackgroundRenderer
import io.github.nikh93.gsplat.capture.DisplayRotationHelper
import io.github.nikh93.gsplat.capture.EglCore
import io.github.nikh93.gsplat.databinding.ActivityArViewerBinding
import io.github.nikh93.gsplat.util.Insets
import io.github.nikh93.gsplat.viewer.OrbitCamera
import io.github.nikh93.gsplat.viewer.SplatDrawer
import io.github.nikh93.gsplat.viewer.SplatLoader
import io.github.nikh93.gsplat.viewer.SplatScene
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale
import kotlin.math.atan2
import kotlin.math.log10
import kotlin.math.pow

/**
 * Shows a Gaussian splat model in the room.
 *
 *  1. ARCore looks for the floor; a target circle marks the point under the screen centre.
 *  2. "Platzieren" fixes that floor point; swiping up/down raises a vertical line.
 *  3. "Platzieren" again anchors the model centre at the top of the line (ARCore anchor).
 *     From then on only the model is shown (no camera image), while ARCore keeps tracking
 *     the phone, so the model stays in place when walking around. A slider scales the model
 *     (1:1 = real size: captures from this app are metric), a horizontal swipe turns it.
 *  4. "Cardboard" shows the model in stereo for a Google Cardboard viewer (Cardboard SDK,
 *     head pose from ARCore); the gear button scans the viewer's QR code.
 *
 * ARCore and GL run on the main thread with [EglCore], like the capture screen (see there).
 */
class ArViewerActivity : AppCompatActivity(), SurfaceHolder.Callback, Choreographer.FrameCallback {

    private enum class Stage { SEARCH, HEIGHT, PLACED }

    private lateinit var binding: ActivityArViewerBinding
    private lateinit var rotationHelper: DisplayRotationHelper
    private lateinit var egl: EglCore
    private lateinit var sensorManager: SensorManager
    private val background = BackgroundRenderer()
    private val guides = ArGuideRenderer()
    private val drawer = SplatDrawer()
    private var stereo: CardboardStereo? = null
    private var cardboardError: String? = null

    private var session: Session? = null
    private var installRequested = false
    private var running = false
    private var lastFrameTimestamp = 0L
    private var surfaceWidth = 1
    private var surfaceHeight = 1

    private var scene: SplatScene? = null
    private var sceneUploaded = false
    private var upAxis = 0

    private var stage = Stage.SEARCH
    private var placeRequested = false
    private var floorAnchor: Anchor? = null
    private var modelAnchor: Anchor? = null
    private var floorHit: Pose? = null
    private var height = 0f
    private var yawDeg = 0f
    private var modelScale = 1f
    private var vr = false
    private var pendingVrAfterScan = false
    private var trackedRecently = false
    private var lastTrackedNs = 0L

    private val viewMatrix = FloatArray(16)
    private val projMatrix = FloatArray(16)
    private val viewProj = FloatArray(16)
    private val modelMatrix = FloatArray(16)
    private val viewModel = FloatArray(16)
    private val upMatrix = FloatArray(16)
    private val tmp = FloatArray(16)
    private var frameCounter = 0

    /* own IMU listener at the fastest rate, see CaptureActivity (Galaxy A56 tracking) */
    private val imuListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent) {}
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (!granted) {
                Toast.makeText(this, R.string.camera_permission_needed, Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityArViewerBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        Insets.pad(binding.topPanel, top = true)
        Insets.pad(binding.bottomPanel, bottom = true)
        WindowCompat.getInsetsController(window, binding.root).apply {
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            hide(WindowInsetsCompat.Type.systemBars())
        }
        rotationHelper = DisplayRotationHelper(this)
        sensorManager = getSystemService(SensorManager::class.java)
        upAxis = intent.getIntExtra(EXTRA_UP_AXIS, 0)

        try {
            Initialize.initialize(applicationContext)
        } catch (e: Throwable) {
            cardboardError = e.message ?: e.javaClass.simpleName
            Log.e(TAG, "Cardboard SDK init failed", e)
        }

        binding.placeButton.setOnClickListener { placeRequested = true }
        binding.resetButton.setOnClickListener { resetPlacement() }
        binding.backButton.setOnClickListener { finish() }
        binding.realSizeButton.setOnClickListener { binding.scaleSlider.value = 0f }
        binding.scaleSlider.addOnChangeListener { _, value, _ ->
            modelScale = 10f.pow(value)
            updateSizeText()
        }
        binding.cardboardButton.setOnClickListener { startCardboard() }
        binding.vrCloseButton.setOnClickListener { setVr(false) }
        binding.vrSettingsButton.setOnClickListener { scanViewerQr() }
        setupGestures()

        egl = EglCore()
        background.createOnGlThread()
        guides.createOnGlThread()
        drawer.createOnGlThread()
        binding.surface.holder.addCallback(this)
        loadModel()
        updateUi()
    }

    private fun loadModel() {
        val path = intent.getStringExtra(EXTRA_PATH) ?: run { finish(); return }
        lifecycleScope.launch {
            try {
                val s = withContext(Dispatchers.IO) { SplatLoader.load(File(path)) }
                scene = s
                sceneUploaded = false
                updateUi()
            } catch (e: Exception) {
                Toast.makeText(
                    this@ArViewerActivity,
                    getString(R.string.viewer_load_failed, e.message ?: e.javaClass.simpleName),
                    Toast.LENGTH_LONG
                ).show()
                finish()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }
        if (session == null) {
            try {
                if (ArCoreApk.getInstance().requestInstall(this, !installRequested) ==
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED
                ) {
                    installRequested = true
                    return
                }
                val s = Session(this)
                // SceneView's configuration, which tracks reliably (see CaptureActivity)
                s.configure(Config(s).apply {
                    focusMode = Config.FocusMode.AUTO
                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                    lightEstimationMode = Config.LightEstimationMode.ENVIRONMENTAL_HDR
                })
                egl.makeCurrent()
                s.setCameraTextureName(background.textureId)
                session = s
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
        // back from the QR code scanner: rebuild the lens distortion for the new viewer
        stereo?.paramsChanged = true
        if (pendingVrAfterScan) {
            pendingVrAfterScan = false
            setVr(true)
        }
        rotationHelper.onResume()
        for (type in intArrayOf(Sensor.TYPE_GYROSCOPE_UNCALIBRATED, Sensor.TYPE_ACCELEROMETER)) {
            sensorManager.getDefaultSensor(type)?.let {
                sensorManager.registerListener(imuListener, it, SensorManager.SENSOR_DELAY_FASTEST)
            }
        }
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
        egl.makeCurrent()
        stereo?.release()
        drawer.release()
        session?.close()
        session = null
        egl.release()
    }

    // ------------------------------------------------------------------
    // placement

    private fun resetPlacement() {
        floorAnchor?.detach()
        modelAnchor?.detach()
        floorAnchor = null
        modelAnchor = null
        height = 0f
        setVr(false)
        stage = Stage.SEARCH
        updateUi()
    }

    /** Floor hit under the screen centre (upward facing plane), or null. */
    private fun centerHit(frame: Frame): HitResult? {
        for (hit in frame.hitTest(surfaceWidth / 2f, surfaceHeight / 2f)) {
            val t = hit.trackable
            if (t is Plane && t.type == Plane.Type.HORIZONTAL_UPWARD_FACING && t.isPoseInPolygon(hit.hitPose)) {
                return hit
            }
        }
        return null
    }

    private fun handlePlacement(s: Session, frame: Frame) {
        when (stage) {
            Stage.SEARCH -> {
                val hit = centerHit(frame)
                floorHit = hit?.hitPose
                if (placeRequested && hit != null) {
                    floorAnchor = hit.createAnchor()
                    height = 0f
                    stage = Stage.HEIGHT
                }
            }
            Stage.HEIGHT -> if (placeRequested) {
                val fp = floorAnchor?.pose ?: return
                modelAnchor = s.createAnchor(Pose.makeTranslation(fp.tx(), fp.ty() + height, fp.tz()))
                floorAnchor?.detach()
                floorAnchor = null
                // turn the model so that its +Z side faces the user
                val cam = frame.camera.pose
                yawDeg = Math.toDegrees(atan2((cam.tx() - fp.tx()).toDouble(), (cam.tz() - fp.tz()).toDouble())).toFloat()
                stage = Stage.PLACED
            }
            Stage.PLACED -> {}
        }
        if (placeRequested) {
            placeRequested = false
            runOnUiThread { updateUi() }
        }
    }

    /** model = anchor * yaw * scale * upAxis * translate(-centre); the model centre sits at the anchor */
    private fun computeModelMatrix(x: Float, y: Float, z: Float) {
        val s = scene ?: return
        Matrix.setIdentityM(modelMatrix, 0)
        Matrix.translateM(modelMatrix, 0, x, y, z)
        Matrix.rotateM(modelMatrix, 0, yawDeg, 0f, 1f, 0f)
        Matrix.scaleM(modelMatrix, 0, modelScale, modelScale, modelScale)
        OrbitCamera.upAxisMatrix(upAxis, upMatrix)
        Matrix.multiplyMM(tmp, 0, modelMatrix, 0, upMatrix, 0)
        Matrix.translateM(tmp, 0, -s.center[0], -s.center[1], -s.center[2])
        tmp.copyInto(modelMatrix)
    }

    // ------------------------------------------------------------------
    // Cardboard

    private fun startCardboard() {
        cardboardError?.let {
            Toast.makeText(this, getString(R.string.ar_cardboard_unavailable, it), Toast.LENGTH_LONG).show()
            return
        }
        if (QrCode.getSavedDeviceParams() == null) {
            // first use: scan the viewer's QR code (the official Cardboard flow)
            pendingVrAfterScan = true
            QrCode.scanQrCodeAndSaveDeviceParams()
        } else {
            setVr(true)
        }
    }

    private fun scanViewerQr() {
        pendingVrAfterScan = true
        QrCode.scanQrCodeAndSaveDeviceParams()
    }

    private fun setVr(on: Boolean) {
        if (on && stage != Stage.PLACED) return
        vr = on
        if (on && stereo == null) stereo = CardboardStereo()
        binding.vrOverlay.visibility = if (on) View.VISIBLE else View.GONE
        binding.topPanel.visibility = if (on) View.GONE else View.VISIBLE
        binding.bottomPanel.visibility = if (on) View.GONE else View.VISIBLE
    }

    // ------------------------------------------------------------------
    // surface and frame loop (main thread)

    override fun surfaceCreated(holder: SurfaceHolder) {
        egl.attach(holder.surface)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        egl.makeCurrent()
        surfaceWidth = maxOf(width, 1)
        surfaceHeight = maxOf(height, 1)
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
        val frame = s.update()
        if (frame.timestamp == lastFrameTimestamp) return // no new camera frame / pose
        lastFrameTimestamp = frame.timestamp

        scene?.let { if (!sceneUploaded) { drawer.upload(it); sceneUploaded = true } }

        val camera = frame.camera
        val tracking = camera.trackingState == TrackingState.TRACKING
        if (tracking) {
            lastTrackedNs = frame.timestamp
            camera.getViewMatrix(viewMatrix, 0)
            camera.getProjectionMatrix(projMatrix, 0, 0.02f, 100f)
            Matrix.multiplyMM(viewProj, 0, projMatrix, 0, viewMatrix, 0)
            if (sceneUploaded) handlePlacement(s, frame)
        }
        // single dropped frames are normal on some phones (see CaptureActivity)
        trackedRecently = frame.timestamp - lastTrackedNs < 300_000_000L

        if (stage == Stage.PLACED) {
            val anchor = modelAnchor
            if (anchor != null && anchor.trackingState != TrackingState.STOPPED) {
                val p = anchor.pose
                computeModelMatrix(p.tx(), p.ty(), p.tz())
            }
            if (vr) {
                stereo?.render(surfaceWidth, surfaceHeight, viewMatrix, modelMatrix, drawer, 1f)
            } else {
                GLES30.glViewport(0, 0, surfaceWidth, surfaceHeight)
                GLES30.glClearColor(0f, 0f, 0f, 1f)
                GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
                Matrix.multiplyMM(viewModel, 0, viewMatrix, 0, modelMatrix, 0)
                drawer.update(viewModel)
                drawer.draw(viewModel, projMatrix, surfaceWidth, surfaceHeight, 1f)
            }
        } else {
            GLES30.glViewport(0, 0, surfaceWidth, surfaceHeight)
            GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
            GLES30.glBindVertexArray(0)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
            background.draw(frame)
            if (tracking) drawGuides()
        }
        egl.swap()

        if (++frameCounter % 10 == 0) updateUi()
    }

    private fun drawGuides() {
        when (stage) {
            Stage.SEARCH -> floorHit?.let {
                guides.drawRing(viewProj, it.tx(), it.ty(), it.tz(), 0.08f, 0.1f, TARGET_COLOR)
                guides.drawRing(viewProj, it.tx(), it.ty(), it.tz(), 0f, 0.012f, TARGET_COLOR)
            }
            Stage.HEIGHT -> floorAnchor?.pose?.let {
                val top = it.ty() + height
                guides.drawRing(viewProj, it.tx(), it.ty(), it.tz(), 0.08f, 0.1f, FIXED_COLOR)
                guides.drawVertical(viewProj, it.tx(), it.ty(), top, it.tz(), 0.004f, LINE_COLOR)
                guides.drawRing(viewProj, it.tx(), top, it.tz(), 0.03f, 0.045f, LINE_COLOR)
                // preview of the model at the chosen centre
                computeModelMatrix(it.tx(), top, it.tz())
                Matrix.multiplyMM(viewModel, 0, viewMatrix, 0, modelMatrix, 0)
                drawer.update(viewModel)
                drawer.draw(viewModel, projMatrix, surfaceWidth, surfaceHeight, 1f)
            }
            Stage.PLACED -> {}
        }
    }

    // ------------------------------------------------------------------
    // UI

    private fun updateUi() {
        val loaded = scene != null
        binding.statusText.text = when {
            !loaded -> getString(R.string.ar_loading)
            !trackedRecently && stage != Stage.PLACED -> getString(R.string.ar_search)
            !trackedRecently -> getString(R.string.ar_lost)
            stage == Stage.SEARCH && floorHit == null -> getString(R.string.ar_search)
            stage == Stage.SEARCH -> getString(R.string.ar_aim)
            stage == Stage.HEIGHT -> getString(R.string.ar_height, height)
            else -> getString(R.string.ar_placed)
        }
        val placed = stage == Stage.PLACED
        binding.placeButton.visibility = if (placed) View.GONE else View.VISIBLE
        binding.placeButton.isEnabled = loaded && trackedRecently && (stage == Stage.HEIGHT || floorHit != null)
        binding.resetButton.visibility = if (stage == Stage.SEARCH) View.GONE else View.VISIBLE
        binding.scaleGroup.visibility = if (stage == Stage.SEARCH) View.INVISIBLE else View.VISIBLE
        binding.cardboardButton.visibility = if (placed) View.VISIBLE else View.GONE
        binding.sizeText.visibility = if (stage == Stage.SEARCH) View.GONE else View.VISIBLE
        updateSizeText()
    }

    private fun updateSizeText() {
        val s = scene ?: return
        val ratio = if (modelScale >= 1f) String.format(Locale.GERMANY, "%.2f : 1", modelScale)
        else String.format(Locale.GERMANY, "1 : %.2f", 1f / modelScale)
        binding.sizeText.text = getString(R.string.ar_size, ratio, 2f * s.radius * modelScale)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupGestures() {
        val scaleDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean {
                if (stage == Stage.SEARCH || vr) return false
                val v = (log10(modelScale * d.scaleFactor)).coerceIn(
                    binding.scaleSlider.valueFrom, binding.scaleSlider.valueTo
                )
                binding.scaleSlider.value = v
                return true
            }
        })
        var lastX = 0f
        var lastY = 0f
        binding.surface.setOnTouchListener { v, ev ->
            scaleDetector.onTouchEvent(ev)
            if (ev.pointerCount == 1) {
                when (ev.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        lastX = ev.x
                        lastY = ev.y
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val dx = ev.x - lastX
                        val dy = ev.y - lastY
                        lastX = ev.x
                        lastY = ev.y
                        when (stage) {
                            // swipe up = higher; full screen height = 2 m
                            Stage.HEIGHT -> {
                                height = (height - dy / v.height * 2f).coerceIn(0f, 3f)
                                binding.statusText.text = getString(R.string.ar_height, height)
                            }
                            Stage.PLACED -> if (!vr) yawDeg += dx / v.width * 360f
                            Stage.SEARCH -> {}
                        }
                    }
                }
            } else {
                lastX = ev.getX(0)
                lastY = ev.getY(0)
            }
            true
        }
    }

    companion object {
        private const val TAG = "ArViewer"
        const val EXTRA_PATH = "path"
        const val EXTRA_UP_AXIS = "up_axis"
        private val TARGET_COLOR = floatArrayOf(1f, 1f, 1f, 0.9f)
        private val FIXED_COLOR = floatArrayOf(0.3f, 0.85f, 1f, 0.9f)
        private val LINE_COLOR = floatArrayOf(1f, 0.85f, 0.1f, 0.95f)
    }
}
