package io.github.nikh93.gsplat.viewer

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.SystemClock
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/** Orbit viewer: renders the scene with [SplatDrawer] from an [OrbitCamera]. */
class SplatRenderer(private val camera: OrbitCamera) : GLSurfaceView.Renderer {

    @Volatile var splatScale = 1f
    @Volatile var autoRotate = false
    @Volatile var background = floatArrayOf(0f, 0f, 0f)
    @Volatile var fps = 0
        private set
    var onFirstFrame: (() -> Unit)? = null

    private val drawer = SplatDrawer()
    private var shown: SplatScene? = null
    @Volatile private var pendingScene: SplatScene? = null

    private var width = 1
    private var height = 1
    private val viewMat = FloatArray(16)
    private val projMat = FloatArray(16)
    private var frames = 0
    private var fpsTime = 0L
    private var lastFrameTime = 0L
    private var firstFrameDone = false

    fun setScene(s: SplatScene) {
        pendingScene = s
    }

    fun release() = drawer.release()

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        drawer.createOnGlThread()
        // the context may have been recreated: upload the scene again
        shown?.let { pendingScene = pendingScene ?: it }
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        width = maxOf(w, 1)
        height = maxOf(h, 1)
        GLES30.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        pendingScene?.let {
            pendingScene = null
            drawer.upload(it)
            shown = it
        }
        val now = SystemClock.elapsedRealtime()
        val dt = if (lastFrameTime == 0L) 0f else (now - lastFrameTime) / 1000f
        lastFrameTime = now
        if (autoRotate) camera.autoRotate(dt * 0.4f)

        val bg = background
        GLES30.glClearColor(bg[0], bg[1], bg[2], 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        if (drawer.scene == null) return

        camera.viewModel(viewMat)
        val (near, far) = camera.nearFar()
        Matrix.perspectiveM(projMat, 0, camera.fovDeg, width.toFloat() / height, near, far)
        drawer.update(viewMat)
        drawer.draw(viewMat, projMat, width, height, splatScale)

        frames++
        if (now - fpsTime >= 1000) {
            fps = (frames * 1000 / maxOf(now - fpsTime, 1)).toInt()
            frames = 0
            fpsTime = now
        }
        if (!firstFrameDone) {
            firstFrameDone = true
            onFirstFrame?.invoke()
        }
    }
}
