package io.github.nikh93.gsplat.capture

import java.nio.ByteBuffer

/**
 * Image sharpness as the variance of the Laplacian of the luminance, the usual blur
 * measure for photogrammetry frame selection. The value has no absolute meaning (it
 * depends on the scene's texture), so photos are compared with the recent frames of
 * the same capture in [SharpnessReference].
 */
object Sharpness {

    /** Evaluated on a grid of about [SAMPLES_X] columns, with full resolution neighbours. */
    fun of(y: ByteBuffer, rowStride: Int, width: Int, height: Int): Float {
        val step = maxOf(2, width / SAMPLES_X)
        var n = 0
        var sum = 0.0
        var sumSq = 0.0
        var row = 1
        while (row < height - 1) {
            val base = row * rowStride
            var x = 1
            while (x < width - 1) {
                val c = y.get(base + x).toInt() and 0xFF
                val l = y.get(base + x - 1).toInt() and 0xFF
                val r = y.get(base + x + 1).toInt() and 0xFF
                val u = y.get(base - rowStride + x).toInt() and 0xFF
                val d = y.get(base + rowStride + x).toInt() and 0xFF
                val lap = (4 * c - l - r - u - d).toDouble()
                sum += lap
                sumSq += lap * lap
                n++
                x += step
            }
            row += step
        }
        if (n == 0) return 0f
        val mean = sum / n
        return (sumSq / n - mean * mean).toFloat()
    }

    private const val SAMPLES_X = 320
}

/**
 * Rolling median of the sharpness of recent camera frames. A photo counts as blurry when
 * it is clearly below what this scene normally gives (adaptive instead of a fixed cutoff,
 * which would reject sharp photos of weakly textured scenes).
 */
class SharpnessReference(private val size: Int = 30) {
    private val values = ArrayDeque<Float>()

    fun add(v: Float) {
        values.addLast(v)
        while (values.size > size) values.removeFirst()
    }

    /** null until enough frames were seen. */
    fun median(): Float? {
        if (values.size < 5) return null
        val sorted = values.sorted()
        return sorted[sorted.size / 2]
    }

    fun isSharp(v: Float): Boolean {
        val m = median() ?: return true
        return v >= m * MIN_RATIO
    }

    companion object {
        /** below 60 % of the scene's typical sharpness a frame is visibly blurred */
        const val MIN_RATIO = 0.6f
    }
}
