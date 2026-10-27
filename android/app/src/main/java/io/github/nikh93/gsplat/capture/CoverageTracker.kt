package io.github.nikh93.gsplat.capture

import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Which viewing directions around the object already have a photo. The object centre is
 * the point closest to all photo viewing axes (people point the phone at the object);
 * every photo then falls into one cell of a dome: [RING_SECTORS] azimuth sectors per
 * elevation ring (low, middle, top). Splatting needs the object from all sides and from
 * several heights, so the cells show what is still missing.
 *
 * [addPhoto] runs on the GL thread, [snapshot] on the UI thread.
 */
class CoverageTracker {

    class Snapshot(
        /** per ring: covered flag per sector */
        val cells: Array<BooleanArray>,
        val covered: Int,
        val total: Int,
        /** current phone direction (azimuth rad, ring) or null without centre */
        val currentAzimuth: Float?,
        val currentElevationDeg: Float?,
    )

    private val positions = ArrayList<FloatArray>()
    private val axes = ArrayList<FloatArray>()
    private var fallbackCenter: FloatArray? = null
    private var center: FloatArray? = null
    /** azimuth of the first photo, drawn at the bottom of the dome */
    private var azimuthOrigin = 0f

    @Synchronized
    fun addPhoto(c2w: FloatArray, sceneDepth: Float) {
        val pos = floatArrayOf(c2w[12], c2w[13], c2w[14])
        val fwd = floatArrayOf(-c2w[8], -c2w[9], -c2w[10]) // camera looks along -z
        positions.add(pos)
        axes.add(fwd)
        if (fallbackCenter == null) {
            fallbackCenter = floatArrayOf(pos[0] + fwd[0] * sceneDepth, pos[1] + fwd[1] * sceneDepth, pos[2] + fwd[2] * sceneDepth)
        }
        val c = estimateCenter() ?: fallbackCenter!!
        center = c
        azimuthOrigin = azimuth(positions[0], c)
    }

    @Synchronized
    fun snapshot(cameraPos: FloatArray?): Snapshot {
        val cells = Array(RING_SECTORS.size) { BooleanArray(RING_SECTORS[it]) }
        val c = center
        var curAz: Float? = null
        var curEl: Float? = null
        if (c != null) {
            for (p in positions) {
                val ring = ring(elevationDeg(p, c))
                cells[ring][sector(azimuth(p, c), RING_SECTORS[ring])] = true
            }
            if (cameraPos != null) {
                curAz = azimuth(cameraPos, c) - azimuthOrigin
                curEl = elevationDeg(cameraPos, c)
            }
        }
        val covered = cells.sumOf { r -> r.count { it } }
        return Snapshot(cells, covered, RING_SECTORS.sum(), curAz, curEl)
    }

    /** Least squares point nearest to all viewing axes; null while they are (nearly) parallel. */
    private fun estimateCenter(): FloatArray? {
        if (positions.size < 3) return null
        val a = DoubleArray(9)
        val b = DoubleArray(3)
        for (i in positions.indices) {
            val p = positions[i]
            val d = axes[i]
            // (I - d d^T)
            for (r in 0 until 3) for (k in 0 until 3) {
                val m = (if (r == k) 1.0 else 0.0) - d[r] * d[k]
                a[r * 3 + k] += m
                b[r] += m * p[k]
            }
        }
        val det = det3(a)
        // the axes must converge from clearly different directions
        if (det < 1e-2 * positions.size * positions.size * positions.size) return null
        val x = solve3(a, b, det)
        // the centre has to lie in front of the cameras, not behind the user
        var inFront = 0
        for (i in positions.indices) {
            val p = positions[i]
            val d = axes[i]
            if ((x[0] - p[0]) * d[0] + (x[1] - p[1]) * d[1] + (x[2] - p[2]) * d[2] > 0f) inFront++
        }
        return if (inFront * 2 > positions.size) x else null
    }

    private fun azimuth(p: FloatArray, c: FloatArray): Float = atan2(p[2] - c[2], p[0] - c[0]) // Y is up

    private fun elevationDeg(p: FloatArray, c: FloatArray): Float {
        val dx = p[0] - c[0]
        val dy = p[1] - c[1]
        val dz = p[2] - c[2]
        val len = sqrt(dx * dx + dy * dy + dz * dz)
        if (len < 1e-4f) return 0f
        return Math.toDegrees(asin((dy / len).coerceIn(-1f, 1f)).toDouble()).toFloat()
    }

    private fun ring(elevationDeg: Float): Int = when {
        elevationDeg < RING_LIMITS[0] -> 0
        elevationDeg < RING_LIMITS[1] -> 1
        else -> 2
    }

    private fun sector(azimuth: Float, sectors: Int): Int {
        val twoPi = (2 * Math.PI).toFloat()
        var a = (azimuth - azimuthOrigin) % twoPi
        if (a < 0f) a += twoPi
        // sector 0 is centred on the first photo
        return ((a / twoPi) * sectors + 0.5f).toInt() % sectors
    }

    private fun det3(m: DoubleArray): Double =
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6])

    /** Cramer's rule */
    private fun solve3(m: DoubleArray, b: DoubleArray, det: Double): FloatArray {
        fun withColumn(col: Int): DoubleArray = DoubleArray(9) { i -> if (i % 3 == col) b[i / 3] else m[i] }
        return FloatArray(3) { (det3(withColumn(it)) / det).toFloat() }
    }

    companion object {
        /** azimuth sectors of the low, middle and top ring */
        val RING_SECTORS = intArrayOf(12, 12, 6)
        /** elevation limits between the rings in degrees (seen from the object centre) */
        val RING_LIMITS = floatArrayOf(15f, 45f)
    }
}
