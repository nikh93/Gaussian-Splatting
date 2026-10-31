package io.github.nikh93.gsplat.viewer

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import kotlin.math.sqrt

/**
 * Background worker for view dependent work:
 *  - sorts splats back to front for alpha blending (16 bit counting sort over
 *    view space depth, O(n))
 *  - evaluates the spherical harmonics colour of every splat for the current
 *    camera position (if the model has SH bands)
 * Requests are conflated: only the newest view matrix is processed.
 */
class SplatSorter(private val scene: SplatScene, private val count: Int) {

    class Result(val indices: IntBuffer, val colors: ByteBuffer?)

    private val lock = Object()
    private var pendingView: FloatArray? = null
    private var running = true
    private val results = arrayOf(alloc(), alloc())
    private var writeIndex = 0
    @Volatile private var ready: Result? = null
    private val keys = IntArray(count)
    private val zs = FloatArray(count)
    private val counts = IntArray(65536)
    private val basis = FloatArray(16)

    private val thread = Thread({ loop() }, "splat-sorter").apply {
        priority = Thread.NORM_PRIORITY - 1
        start()
    }

    private fun alloc(): Result = Result(
        ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()).asIntBuffer(),
        if (scene.shDegree > 0) ByteBuffer.allocateDirect(count * 4).order(ByteOrder.nativeOrder()) else null,
    )

    /** [view] = column major view*model matrix. */
    fun request(view: FloatArray) {
        synchronized(lock) {
            pendingView = view.copyOf()
            lock.notifyAll()
        }
    }

    /** Returns a newly computed result once, or null. */
    fun poll(): Result? {
        val r = ready
        if (r != null) ready = null
        return r
    }

    fun shutdown() {
        synchronized(lock) {
            running = false
            lock.notifyAll()
        }
    }

    private fun loop() {
        while (true) {
            val view: FloatArray
            synchronized(lock) {
                while (running && pendingView == null) lock.wait()
                if (!running) return
                view = pendingView!!
                pendingView = null
            }
            val out = results[writeIndex]
            sort(view, out.indices)
            out.colors?.let { evalColors(view, it) }
            synchronized(lock) {
                ready = out
                writeIndex = 1 - writeIndex
            }
        }
    }

    private fun sort(v: FloatArray, out: IntBuffer) {
        // view space z (camera looks down -z): farthest = most negative, drawn first
        val m2 = v[2]
        val m6 = v[6]
        val m10 = v[10]
        val m14 = v[14]
        var minZ = Float.MAX_VALUE
        var maxZ = -Float.MAX_VALUE
        val p = scene.positions
        for (i in 0 until count) {
            val z = m2 * p[i * 3] + m6 * p[i * 3 + 1] + m10 * p[i * 3 + 2] + m14
            zs[i] = z
            if (z < minZ) minZ = z
            if (z > maxZ) maxZ = z
        }
        val scale = if (maxZ > minZ) 65535f / (maxZ - minZ) else 0f
        counts.fill(0)
        for (i in 0 until count) {
            val k = ((zs[i] - minZ) * scale).toInt().coerceIn(0, 65535)
            keys[i] = k
            counts[k]++
        }
        var acc = 0
        for (k in 0 until 65536) {
            val c = counts[k]
            counts[k] = acc
            acc += c
        }
        for (i in 0 until count) {
            out.put(counts[keys[i]]++, i)
        }
        out.position(0)
    }

    /** RGBA8 colour per splat for the camera position of [v] (same basis as the trainer). */
    private fun evalColors(v: FloatArray, out: ByteBuffer) {
        // camera centre in model space: -(sR)^T t / s^2 for v = [sR | t] (uniform model scale s)
        val s2 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).let { if (it > 0f) it else 1f }
        val cx = -(v[0] * v[12] + v[1] * v[13] + v[2] * v[14]) / s2
        val cy = -(v[4] * v[12] + v[5] * v[13] + v[6] * v[14]) / s2
        val cz = -(v[8] * v[12] + v[9] * v[13] + v[10] * v[14]) / s2
        val deg = scene.shDegree
        val bands = (deg + 1) * (deg + 1) - 1
        val base = scene.shBase!!
        val rest = scene.shRest!!
        val opacity = scene.opacity!!
        val p = scene.positions
        for (i in 0 until count) {
            var dx = p[i * 3] - cx
            var dy = p[i * 3 + 1] - cy
            var dz = p[i * 3 + 2] - cz
            val l = sqrt(dx * dx + dy * dy + dz * dz)
            if (l > 1e-12f) {
                dx /= l; dy /= l; dz /= l
            }
            shBasis(deg, dx, dy, dz, basis)
            var r = base[i * 3]
            var g = base[i * 3 + 1]
            var b = base[i * 3 + 2]
            val o = i * bands * 3
            for (k in 0 until bands) {
                val w = basis[k + 1]
                r += w * rest[o + k * 3]
                g += w * rest[o + k * 3 + 1]
                b += w * rest[o + k * 3 + 2]
            }
            val q = i * 4
            out.put(q, to8(r))
            out.put(q + 1, to8(g))
            out.put(q + 2, to8(b))
            out.put(q + 3, opacity[i])
        }
        out.position(0)
    }

    private fun to8(v: Float): Byte = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt().toByte()

    companion object {
        private const val C1 = 0.4886025119029199f
        private val C2 = floatArrayOf(1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
            -1.0925484305920792f, 0.5462742152960396f)
        private val C3 = floatArrayOf(-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
            0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f)

        /** Real SH basis in the ordering of the 3DGS reference implementation (b[0] = DC, unused here). */
        fun shBasis(deg: Int, x: Float, y: Float, z: Float, b: FloatArray) {
            b[0] = 0.28209479177387814f
            if (deg < 1) return
            b[1] = -C1 * y
            b[2] = C1 * z
            b[3] = -C1 * x
            if (deg < 2) return
            val xx = x * x
            val yy = y * y
            val zz = z * z
            b[4] = C2[0] * x * y
            b[5] = C2[1] * y * z
            b[6] = C2[2] * (2f * zz - xx - yy)
            b[7] = C2[3] * x * z
            b[8] = C2[4] * (xx - yy)
            if (deg < 3) return
            b[9] = C3[0] * y * (3f * xx - yy)
            b[10] = C3[1] * x * y * z
            b[11] = C3[2] * y * (4f * zz - xx - yy)
            b[12] = C3[3] * z * (2f * zz - 3f * xx - 3f * yy)
            b[13] = C3[4] * x * (4f * zz - xx - yy)
            b[14] = C3[5] * z * (xx - yy)
            b[15] = C3[6] * x * (xx - 3f * yy)
        }
    }
}
