package io.github.nikh93.gsplat.viewer

import android.annotation.SuppressLint
import android.content.Intent
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import io.github.nikh93.gsplat.R
import io.github.nikh93.gsplat.ar.ArViewerActivity
import io.github.nikh93.gsplat.databinding.ActivityViewerBinding
import io.github.nikh93.gsplat.util.Insets
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 3D viewer for Gaussian Splatting models (.ply from the trainer / other tools, .splat).
 *
 * Controls: one finger orbits, two fingers pan, pinch zooms, double tap resets.
 * The panel offers reset, auto rotation, background colour, up axis, splat size and FOV.
 *
 * Every change (camera, centre, toolbar) is stored per model in [ModelViewStore] and
 * restored when the model is opened again, until "Ansicht zurücksetzen" brings back the
 * original view.
 */
class ViewerActivity : AppCompatActivity() {

    private lateinit var binding: ActivityViewerBinding
    private val camera = OrbitCamera()
    private lateinit var renderer: SplatRenderer
    private var splatCount = 0
    @Volatile private var modelFile: File? = null
    private lateinit var store: ModelViewStore
    private var stateReady = false // the stored state was applied; changes are saved from now on
    private var backgroundIndex = 0
    private val backgrounds = arrayOf(
        floatArrayOf(0f, 0f, 0f),
        floatArrayOf(0.18f, 0.18f, 0.2f),
        floatArrayOf(1f, 1f, 1f),
    )
    private val upAxisNames = arrayOf("+Y", "−Y", "+Z", "−Z")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityViewerBinding.inflate(layoutInflater)
        setContentView(binding.root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        Insets.pad(binding.topBar, top = true)
        Insets.pad(binding.controlPanel, bottom = true)

        store = ModelViewStore(this)
        renderer = SplatRenderer(camera)
        renderer.onFirstFrame = { runOnUiThread { binding.loadingPanel.visibility = View.GONE } }
        binding.surface.setEGLContextClientVersion(3)
        binding.surface.setEGLConfigChooser(8, 8, 8, 8, 0, 0)
        binding.surface.preserveEGLContextOnPause = true
        binding.surface.setRenderer(renderer)
        binding.surface.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        setupControls()
        setupGestures()
        loadModel()
    }

    override fun onResume() {
        super.onResume()
        binding.surface.onResume()
    }

    override fun onPause() {
        saveState()
        binding.surface.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        renderer.release()
        super.onDestroy()
    }

    private fun loadModel() {
        val path = intent.getStringExtra(EXTRA_PATH)
        val uri: Uri? = intent.data
        binding.loadingPanel.visibility = View.VISIBLE
        lifecycleScope.launch {
            try {
                val scene = withContext(Dispatchers.IO) {
                    val file = when {
                        path != null -> File(path)
                        uri != null -> copyToCache(uri)
                        else -> throw IllegalArgumentException("keine Datei angegeben")
                    }
                    modelFile = file
                    SplatLoader.load(file) { p ->
                        runOnUiThread { binding.loadingText.text = getString(R.string.viewer_loading) + " ${(p * 100).toInt()}%" }
                    }
                }
                splatCount = scene.count
                camera.setHome(scene.center, scene.radius)
                modelFile?.let { f -> store.load(f)?.let { applyState(it) } }
                stateReady = true
                renderer.setScene(scene)
                binding.arButton.isEnabled = true
                startInfoUpdates()
            } catch (e: Exception) {
                binding.loadingText.text = getString(R.string.viewer_load_failed, e.message ?: e.javaClass.simpleName)
                binding.loadingProgress.visibility = View.GONE
            }
        }
    }

    /** Files opened from other apps arrive as content:// URIs. */
    private fun copyToCache(uri: Uri): File {
        val name = uri.lastPathSegment?.substringAfterLast('/')?.takeIf { it.endsWith(".ply") || it.endsWith(".splat") }
            ?: "shared.ply"
        val out = File(cacheDir, name)
        contentResolver.openInputStream(uri)!!.use { input -> out.outputStream().use { input.copyTo(it) } }
        return out
    }

    private fun startInfoUpdates() {
        lifecycleScope.launch {
            while (isActive) {
                binding.infoText.text = getString(R.string.viewer_info, splatCount, renderer.fps)
                delay(500)
            }
        }
    }

    private fun setupControls() {
        binding.toggleControls.setOnClickListener {
            binding.controlPanel.visibility =
                if (binding.controlPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        binding.resetButton.setOnClickListener { resetView() }
        binding.arButton.setOnClickListener {
            val f = modelFile ?: return@setOnClickListener
            startActivity(
                Intent(this, ArViewerActivity::class.java)
                    .putExtra(ArViewerActivity.EXTRA_PATH, f.absolutePath)
                    .putExtra(ArViewerActivity.EXTRA_UP_AXIS, camera.upAxis)
            )
        }
        binding.autoRotateButton.addOnCheckedChangeListener { _, checked ->
            renderer.autoRotate = checked
            saveState()
        }
        binding.backgroundButton.setOnClickListener {
            backgroundIndex = (backgroundIndex + 1) % backgrounds.size
            renderer.background = backgrounds[backgroundIndex]
            saveState()
        }
        updateUpAxisLabel()
        binding.upAxisButton.setOnClickListener {
            camera.upAxis = (camera.upAxis + 1) % 4
            camera.reset()
            updateUpAxisLabel()
            saveState()
        }
        binding.scaleSlider.addOnChangeListener { _, value, fromUser ->
            renderer.splatScale = value
            if (fromUser) saveState()
        }
        binding.fovSlider.addOnChangeListener { _, value, fromUser ->
            camera.fovDeg = value
            if (fromUser) saveState()
        }
    }

    // ------------------------------------------------------------------
    // per model view state

    private fun applyState(st: ModelViewState) {
        camera.upAxis = st.upAxis.coerceIn(0, 3)
        updateUpAxisLabel()
        st.target?.takeIf { it.size == 3 && st.distance > 0f }?.let {
            camera.restoreOrbit(it, st.distance, st.yaw, st.pitch)
        }
        binding.fovSlider.value = st.fovDeg.coerceIn(binding.fovSlider.valueFrom, binding.fovSlider.valueTo)
        binding.scaleSlider.value = st.splatScale.coerceIn(binding.scaleSlider.valueFrom, binding.scaleSlider.valueTo)
        backgroundIndex = st.backgroundIndex.coerceIn(0, backgrounds.size - 1)
        renderer.background = backgrounds[backgroundIndex]
        binding.autoRotateButton.isChecked = st.autoRotate
        renderer.autoRotate = st.autoRotate
    }

    /** Stores the current viewer parameters (the AR parameters of the model are kept). */
    private fun saveState() {
        val f = modelFile ?: return
        if (!stateReady) return
        val orbit = camera.orbitState()
        store.update(f) { st ->
            st.target = floatArrayOf(orbit[0], orbit[1], orbit[2])
            st.distance = orbit[3]
            st.yaw = orbit[4]
            st.pitch = orbit[5]
            st.upAxis = camera.upAxis
            st.fovDeg = camera.fovDeg
            st.splatScale = renderer.splatScale
            st.backgroundIndex = backgroundIndex
            st.autoRotate = renderer.autoRotate
        }
    }

    /** "Ansicht zurücksetzen": forget everything stored for this model, back to the original. */
    private fun resetView() {
        stateReady = false
        modelFile?.let { store.clear(it) }
        applyState(ModelViewState())
        camera.reset()
        stateReady = true
    }

    private fun updateUpAxisLabel() {
        binding.upAxisButton.text = getString(R.string.viewer_up_axis) + ": " + upAxisNames[camera.upAxis]
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupGestures() {
        val scaleDetector = ScaleGestureDetector(this, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
            override fun onScale(d: ScaleGestureDetector): Boolean {
                camera.zoom(d.scaleFactor)
                return true
            }
        })
        val tapDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDoubleTap(e: MotionEvent): Boolean {
                resetView()
                return true
            }
        })

        var lastX = 0f
        var lastY = 0f
        var lastPointers = 0
        binding.surface.setOnTouchListener { v, ev ->
            scaleDetector.onTouchEvent(ev)
            tapDetector.onTouchEvent(ev)
            // centroid of all pointers
            var cx = 0f
            var cy = 0f
            for (i in 0 until ev.pointerCount) {
                cx += ev.getX(i)
                cy += ev.getY(i)
            }
            cx /= ev.pointerCount
            cy /= ev.pointerCount
            val pointers = when (ev.actionMasked) {
                MotionEvent.ACTION_POINTER_UP -> ev.pointerCount - 1
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> 0
                else -> ev.pointerCount
            }
            if (ev.actionMasked == MotionEvent.ACTION_MOVE && pointers == lastPointers) {
                val dx = cx - lastX
                val dy = cy - lastY
                if (pointers == 1) camera.rotate(dx, dy, v.height)
                else if (pointers >= 2) camera.pan(dx, dy, v.height)
            }
            if (ev.actionMasked == MotionEvent.ACTION_POINTER_UP) {
                // recompute the centroid without the lifted pointer to avoid jumps
                var sx = 0f
                var sy = 0f
                for (i in 0 until ev.pointerCount) {
                    if (i == ev.actionIndex) continue
                    sx += ev.getX(i)
                    sy += ev.getY(i)
                }
                if (pointers > 0) {
                    cx = sx / pointers
                    cy = sy / pointers
                }
            }
            lastX = cx
            lastY = cy
            lastPointers = pointers
            // camera moved: remember it for the next time the model is opened
            if (ev.actionMasked == MotionEvent.ACTION_UP) saveState()
            true
        }
    }

    companion object {
        const val EXTRA_PATH = "path"
    }
}
