package io.github.nikh93.gsplat.viewer

import android.opengl.Matrix
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Orbit camera around a target point. All methods are synchronised because
 * gestures arrive on the UI thread while the GL thread reads the matrices.
 */
class OrbitCamera {
    private val target = FloatArray(3)
    private var distance = 2f
    private var yaw = 0.6f
    private var pitch = 0.35f
    private var minDistance = 0.01f
    private var maxDistance = 100f
    private var homeTarget = FloatArray(3)
    private var homeDistance = 2f
    var fovDeg = 60f
        @Synchronized get
        @Synchronized set

    /** 0:+Y up, 1:-Y up, 2:+Z up, 3:-Z up (models from other tools are often not Y-up). */
    var upAxis = 0
        @Synchronized get
        @Synchronized set

    private val model = FloatArray(16)
    private val view = FloatArray(16)

    @Synchronized
    fun setHome(center: FloatArray, radius: Float) {
        homeTarget = center.copyOf()
        homeDistance = radius * 2.2f
        minDistance = radius * 0.01f
        maxDistance = radius * 50f
        reset()
    }

    @Synchronized
    fun reset() {
        homeTarget.copyInto(target)
        distance = homeDistance
        yaw = 0.6f
        pitch = 0.35f
    }

    /** Current orbit: target (3), distance, yaw, pitch. */
    @Synchronized
    fun orbitState(): FloatArray = floatArrayOf(target[0], target[1], target[2], distance, yaw, pitch)

    @Synchronized
    fun restoreOrbit(t: FloatArray, dist: Float, yawRad: Float, pitchRad: Float) {
        t.copyInto(target, 0, 0, 3)
        distance = dist.coerceIn(minDistance, maxDistance)
        yaw = yawRad
        pitch = pitchRad.coerceIn(-1.55f, 1.55f)
    }

    @Synchronized
    fun rotate(dxPixels: Float, dyPixels: Float, viewHeight: Int) {
        val k = 3.5f / max(viewHeight, 1)
        yaw -= dxPixels * k
        pitch = (pitch + dyPixels * k).coerceIn(-1.55f, 1.55f)
    }

    @Synchronized
    fun autoRotate(radians: Float) {
        yaw += radians
    }

    @Synchronized
    fun zoom(factor: Float) {
        distance = (distance / factor).coerceIn(minDistance, maxDistance)
    }

    /** Moves the target in the camera plane. */
    @Synchronized
    fun pan(dxPixels: Float, dyPixels: Float, viewHeight: Int) {
        val scale = 2f * distance * kotlin.math.tan(Math.toRadians(fovDeg / 2.0)).toFloat() / max(viewHeight, 1)
        // camera basis in the (up-axis adjusted) model frame
        val cy = cos(yaw)
        val sy = sin(yaw)
        val cp = cos(pitch)
        val sp = sin(pitch)
        val right = floatArrayOf(cy, 0f, -sy)
        val up = floatArrayOf(-sp * sy, cp, -sp * cy)
        val m = upModel()
        for (k in 0 until 3) {
            // model direction = M^T * frame direction (m is column major)
            val r = right[0] * m[k * 4] + right[1] * m[k * 4 + 1] + right[2] * m[k * 4 + 2]
            val u = up[0] * m[k * 4] + up[1] * m[k * 4 + 1] + up[2] * m[k * 4 + 2]
            target[k] += -dxPixels * scale * r + dyPixels * scale * u
        }
    }

    /** Rotation that maps the selected up axis of the model onto +Y. */
    private fun upModel(): FloatArray {
        upAxisMatrix(upAxis, model)
        return model
    }

    /**
     * Writes the view*model matrix into [out]. The orbit happens in a frame where the
     * chosen up axis is +Y; the target is expressed in model coordinates.
     */
    @Synchronized
    fun viewModel(out: FloatArray) {
        val m = upModel()
        // target in the rotated frame
        val t = FloatArray(4)
        Matrix.multiplyMV(t, 0, m, 0, floatArrayOf(target[0], target[1], target[2], 1f), 0)
        val ex = t[0] + distance * cos(pitch) * sin(yaw)
        val ey = t[1] + distance * sin(pitch)
        val ez = t[2] + distance * cos(pitch) * cos(yaw)
        Matrix.setLookAtM(view, 0, ex, ey, ez, t[0], t[1], t[2], 0f, 1f, 0f)
        Matrix.multiplyMM(out, 0, view, 0, m, 0)
    }

    companion object {
        /** Rotation (column major) mapping up axis 0:+Y, 1:-Y, 2:+Z, 3:-Z of a model onto +Y. */
        fun upAxisMatrix(upAxis: Int, out: FloatArray) {
            Matrix.setIdentityM(out, 0)
            when (upAxis) {
                1 -> Matrix.rotateM(out, 0, 180f, 0f, 0f, 1f)
                2 -> Matrix.rotateM(out, 0, -90f, 1f, 0f, 0f)
                3 -> Matrix.rotateM(out, 0, 90f, 1f, 0f, 0f)
            }
        }
    }

    @Synchronized
    fun nearFar(): Pair<Float, Float> = Pair(max(distance * 0.005f, 0.001f), min(distance * 200f, 10000f))
}
