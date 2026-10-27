package io.github.nikh93.gsplat.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class CaptureQualityTest {

    /** Column major camera-to-world matrix of a camera at [p] looking at [t] (OpenGL axes, Y up). */
    private fun lookAt(p: FloatArray, t: FloatArray): FloatArray {
        fun norm(v: FloatArray): FloatArray {
            val l = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
            return floatArrayOf(v[0] / l, v[1] / l, v[2] / l)
        }
        fun cross(a: FloatArray, b: FloatArray) =
            floatArrayOf(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        val back = norm(floatArrayOf(p[0] - t[0], p[1] - t[1], p[2] - t[2]))
        val right = norm(cross(floatArrayOf(0f, 1f, 0f), back))
        val up = cross(back, right)
        return floatArrayOf(
            right[0], right[1], right[2], 0f,
            up[0], up[1], up[2], 0f,
            back[0], back[1], back[2], 0f,
            p[0], p[1], p[2], 1f,
        )
    }

    private val obj = floatArrayOf(0.3f, 0.1f, -1.2f)

    private fun cameraAround(deg: Float, height: Float, dist: Float = 0.8f): FloatArray {
        val a = Math.toRadians(deg.toDouble())
        return floatArrayOf(obj[0] + dist * cos(a).toFloat(), obj[1] + height, obj[2] + dist * sin(a).toFloat())
    }

    @Test
    fun fullCircleAtTableHeightCoversTheLowRing() {
        val tracker = CoverageTracker()
        for (i in 0 until 24) tracker.addPhoto(lookAt(cameraAround(i * 15f, 0.05f), obj), 0.8f)
        val s = tracker.snapshot(null)
        assertTrue(s.cells[0].all { it })
        assertFalse(s.cells[1].any { it })
        assertFalse(s.cells[2].any { it })
        assertEquals(12, s.covered)
        assertEquals(30, s.total)
    }

    @Test
    fun ringsFollowTheHeightAboveTheObject() {
        val tracker = CoverageTracker()
        for (i in 0 until 12) tracker.addPhoto(lookAt(cameraAround(i * 30f, 0f), obj), 0.8f)
        for (i in 0 until 12) tracker.addPhoto(lookAt(cameraAround(i * 30f, 0.5f), obj), 0.9f) // ~32 degrees
        for (i in 0 until 6) tracker.addPhoto(lookAt(cameraAround(i * 60f, 1.5f, 0.3f), obj), 1.5f) // ~79 degrees
        val s = tracker.snapshot(null)
        assertEquals(30, s.covered)
    }

    @Test
    fun halfCircleLeavesTheOtherSideOpen() {
        val tracker = CoverageTracker()
        for (i in 0..6) tracker.addPhoto(lookAt(cameraAround(i * 30f, 0f), obj), 0.8f)
        val s = tracker.snapshot(cameraAround(0f, 0f))
        assertEquals(7, s.covered)
        // the phone stands where the first photo was taken: drawn at the bottom (azimuth 0)
        assertNotNull(s.currentAzimuth)
        assertEquals(0f, s.currentAzimuth!!, 0.05f)
    }

    @Test
    fun parallelAxesFallBackToTheFirstPhotosDepth() {
        val tracker = CoverageTracker()
        // walking sideways while looking straight ahead: no convergence, centre from depth
        for (i in 0 until 5) {
            val p = floatArrayOf(i * 0.1f, 0f, 0f)
            tracker.addPhoto(lookAt(p, floatArrayOf(p[0], 0f, -1f)), 1f)
        }
        val s = tracker.snapshot(null)
        assertTrue(s.covered in 1..5)
    }

    private fun image(w: Int, h: Int, f: (Int, Int) -> Int): ByteBuffer {
        val b = ByteBuffer.allocate(w * h)
        for (y in 0 until h) for (x in 0 until w) b.put(y * w + x, f(x, y).coerceIn(0, 255).toByte())
        return b
    }

    @Test
    fun blurLowersSharpness() {
        val w = 640
        val h = 480
        val sharp = image(w, h) { x, y -> if (((x / 3) + (y / 3)) % 2 == 0) 220 else 30 }
        // the same pattern, box-blurred over 7 pixels
        val blurred = image(w, h) { x, y ->
            var s = 0
            for (dy in -3..3) for (dx in -3..3) {
                val xx = (x + dx).coerceIn(0, w - 1)
                val yy = (y + dy).coerceIn(0, h - 1)
                s += if (((xx / 3) + (yy / 3)) % 2 == 0) 220 else 30
            }
            s / 49
        }
        val a = Sharpness.of(sharp, w, w, h)
        val b = Sharpness.of(blurred, w, w, h)
        assertTrue("sharp $a blurred $b", a > 5 * b)

        val ref = SharpnessReference()
        repeat(10) { ref.add(a) }
        assertTrue(ref.isSharp(a * 0.9f))
        assertFalse(ref.isSharp(b))
    }

    @Test
    fun referenceAcceptsEverythingUntilItHasSeenTheScene() {
        val ref = SharpnessReference()
        ref.add(100f)
        assertTrue(ref.isSharp(1f))
    }
}
