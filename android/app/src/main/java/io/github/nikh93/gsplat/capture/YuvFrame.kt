package io.github.nikh93.gsplat.capture

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import java.io.ByteArrayOutputStream

/**
 * Copy of an ARCore CPU camera image (YUV_420_888). Copying the planes is fast,
 * so the [Image] can be closed immediately on the GL thread; JPEG encoding and
 * colour sampling happen later on a worker thread.
 */
class YuvFrame private constructor(
    val width: Int,
    val height: Int,
    private val y: ByteArray,
    private val u: ByteArray,
    private val v: ByteArray,
    private val yRowStride: Int,
    private val uvRowStride: Int,
    private val uvPixelStride: Int,
) {
    companion object {
        fun copyOf(image: Image): YuvFrame {
            require(image.format == ImageFormat.YUV_420_888) { "unexpected image format ${image.format}" }
            val planes = image.planes
            fun bytes(i: Int): ByteArray {
                val buf = planes[i].buffer
                buf.rewind()
                val arr = ByteArray(buf.remaining())
                buf.get(arr)
                return arr
            }
            return YuvFrame(
                image.width, image.height, bytes(0), bytes(1), bytes(2),
                planes[0].rowStride, planes[1].rowStride, planes[1].pixelStride
            )
        }
    }

    fun toNv21(): ByteArray {
        val out = ByteArray(width * height * 3 / 2)
        var o = 0
        for (row in 0 until height) {
            System.arraycopy(y, row * yRowStride, out, o, width)
            o += width
        }
        val cw = width / 2
        val ch = height / 2
        for (row in 0 until ch) {
            val base = row * uvRowStride
            for (col in 0 until cw) {
                val idx = base + col * uvPixelStride
                out[o++] = v[idx]
                out[o++] = u[idx]
            }
        }
        return out
    }

    fun toJpeg(quality: Int): ByteArray {
        val img = YuvImage(toNv21(), ImageFormat.NV21, width, height, null)
        val bos = ByteArrayOutputStream(width * height / 2)
        img.compressToJpeg(Rect(0, 0, width, height), quality, bos)
        return bos.toByteArray()
    }

    /** RGB (0..255) at pixel (x, y) or null when outside the image. */
    fun rgbAt(px: Int, py: Int, out: IntArray): Boolean {
        if (px < 0 || py < 0 || px >= width || py >= height) return false
        val yy = (y[py * yRowStride + px].toInt() and 0xFF).toFloat()
        val ci = (py / 2) * uvRowStride + (px / 2) * uvPixelStride
        val uu = (u[ci].toInt() and 0xFF) - 128f
        val vv = (v[ci].toInt() and 0xFF) - 128f
        out[0] = (yy + 1.402f * vv).toInt().coerceIn(0, 255)
        out[1] = (yy - 0.344136f * uu - 0.714136f * vv).toInt().coerceIn(0, 255)
        out[2] = (yy + 1.772f * uu).toInt().coerceIn(0, 255)
        return true
    }
}
