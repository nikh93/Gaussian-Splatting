package io.github.nikh93.gsplat.viewer

import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Gaussian splat scene ready for the GPU.
 *
 * [gpuData] holds 32 bytes per splat (two RGBA32UI texels):
 *   texel 0: x, y, z (float bits), colour RGBA8
 *   texel 1: half2(cov_xx, cov_xy), half2(cov_xz, cov_yy), half2(cov_yz, cov_zz), s (float bits)
 * The covariance is stored divided by its largest diagonal entry s, so every half
 * value lies in [-1, 1] regardless of the splat size (no overflow for huge background
 * splats, full precision for tiny ones).
 */
class SplatScene(
    val count: Int,
    val positions: FloatArray,
    val gpuData: ByteBuffer,
    val center: FloatArray,
    val radius: Float,
    /** View dependent colour: degree (0 = none), base colour 0.5 + C0*dc (3n), higher bands [n][k][rgb]. */
    val shDegree: Int = 0,
    val shBase: FloatArray? = null,
    val shRest: FloatArray? = null,
    val opacity: ByteArray? = null,
)

object SplatLoader {
    private const val SH_C0 = 0.28209479177387814f
    const val MAX_SPLATS = 4_000_000
    /** Memory budget for higher SH bands; the degree is reduced for large models. */
    private const val SH_BUDGET_BYTES = 64L * 1024 * 1024

    fun load(file: File, onProgress: (Float) -> Unit = {}): SplatScene {
        val name = file.name.lowercase()
        RandomAccessFile(file, "r").use { raf ->
            val ch = raf.channel
            val map = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size()).order(ByteOrder.LITTLE_ENDIAN)
            val head = ByteArray(min(3L, ch.size()).toInt())
            map.get(head)
            map.position(0)
            return if (String(head) == "ply") loadPly(map, onProgress)
            else if (name.endsWith(".splat")) loadSplat(map, onProgress)
            else throw IOException("Unbekanntes Format (erwartet .ply oder .splat)")
        }
    }

    // ------------------------------------------------------------------

    private class Raw(n: Int) {
        var shDegree = 0
        var shBase: FloatArray? = null
        var shRest: FloatArray? = null
        val pos = FloatArray(n * 3)
        val scale = FloatArray(n * 3) // activated (metres)
        val rot = FloatArray(n * 4)   // w x y z, normalised
        val rgba = IntArray(n)
    }

    private fun loadSplat(map: ByteBuffer, onProgress: (Float) -> Unit): SplatScene {
        val total = map.capacity() / 32
        val n = min(total, MAX_SPLATS)
        val raw = Raw(n)
        for (i in 0 until n) {
            val o = i * 32
            for (k in 0 until 3) {
                raw.pos[i * 3 + k] = map.getFloat(o + k * 4)
                raw.scale[i * 3 + k] = map.getFloat(o + 12 + k * 4)
            }
            val r = map.get(o + 24).toInt() and 0xFF
            val g = map.get(o + 25).toInt() and 0xFF
            val b = map.get(o + 26).toInt() and 0xFF
            val a = map.get(o + 27).toInt() and 0xFF
            raw.rgba[i] = r or (g shl 8) or (b shl 16) or (a shl 24)
            var qw = ((map.get(o + 28).toInt() and 0xFF) - 128) / 128f
            var qx = ((map.get(o + 29).toInt() and 0xFF) - 128) / 128f
            var qy = ((map.get(o + 30).toInt() and 0xFF) - 128) / 128f
            var qz = ((map.get(o + 31).toInt() and 0xFF) - 128) / 128f
            val l = sqrt(qw * qw + qx * qx + qy * qy + qz * qz).let { if (it < 1e-6f) 1f else it }
            qw /= l; qx /= l; qy /= l; qz /= l
            raw.rot[i * 4] = qw; raw.rot[i * 4 + 1] = qx; raw.rot[i * 4 + 2] = qy; raw.rot[i * 4 + 3] = qz
            if (i % 50000 == 0) onProgress(0.5f * i / n)
        }
        return build(raw, n, onProgress)
    }

    private class Prop(val name: String, val type: String, val offset: Int)

    private fun typeSize(t: String): Int = when (t) {
        "char", "uchar", "int8", "uint8" -> 1
        "short", "ushort", "int16", "uint16" -> 2
        "int", "uint", "float", "int32", "uint32", "float32" -> 4
        "double", "float64" -> 8
        else -> throw IOException("PLY-Typ nicht unterstützt: $t")
    }

    private fun read(map: ByteBuffer, off: Int, t: String): Float = when (t) {
        "float", "float32" -> map.getFloat(off)
        "double", "float64" -> map.getDouble(off).toFloat()
        "uchar", "uint8" -> (map.get(off).toInt() and 0xFF).toFloat()
        "char", "int8" -> map.get(off).toFloat()
        "short", "int16" -> map.getShort(off).toFloat()
        "ushort", "uint16" -> (map.getShort(off).toInt() and 0xFFFF).toFloat()
        "int", "int32" -> map.getInt(off).toFloat()
        "uint", "uint32" -> (map.getInt(off).toLong() and 0xFFFFFFFFL).toFloat()
        else -> 0f
    }

    private fun loadPly(map: ByteBuffer, onProgress: (Float) -> Unit): SplatScene {
        // --- header
        val sb = StringBuilder()
        var pos = 0
        val limit = min(map.capacity(), 1 shl 20)
        while (pos < limit) {
            sb.append((map.get(pos).toInt() and 0xFF).toChar())
            pos++
            if (sb.endsWith("end_header\n")) break
        }
        if (!sb.endsWith("end_header\n")) throw IOException("PLY-Header ungültig")
        val dataStart = pos
        var count = -1
        var format = ""
        var inVertex = false
        val props = ArrayList<Prop>()
        var stride = 0
        for (line in sb.lines()) {
            val t = line.trim().split(Regex("\\s+"))
            when {
                t[0] == "format" -> format = t.getOrElse(1) { "" }
                t[0] == "element" -> {
                    inVertex = t.getOrNull(1) == "vertex"
                    if (inVertex) count = t[2].toInt()
                    else if (count >= 0) break
                }
                t[0] == "property" && inVertex -> {
                    if (t[1] == "list") throw IOException("PLY-Listen werden nicht unterstützt")
                    props.add(Prop(t[2], t[1], stride))
                    stride += typeSize(t[1])
                }
            }
        }
        if (format != "binary_little_endian") throw IOException("Nur binäre PLY-Dateien werden unterstützt ($format)")
        if (count <= 0) throw IOException("PLY enthält keine Punkte")
        val byName = props.associateBy { it.name }
        fun p(name: String) = byName[name]
        val px = p("x") ?: throw IOException("PLY ohne x/y/z")
        val py = p("y")!!
        val pz = p("z")!!
        val dc = arrayOf(p("f_dc_0"), p("f_dc_1"), p("f_dc_2"))
        val rgb = arrayOf(p("red"), p("green"), p("blue"))
        val op = p("opacity")
        val sc = arrayOf(p("scale_0"), p("scale_1"), p("scale_2"))
        val rt = arrayOf(p("rot_0"), p("rot_1"), p("rot_2"), p("rot_3"))
        val isGaussian = sc.all { it != null } && rt.all { it != null }
        val restCount = props.count { it.name.startsWith("f_rest_") }
        val fileCoeffs = restCount / 3 + 1 // per channel, including DC
        var fileDegree = 0
        while ((fileDegree + 2) * (fileDegree + 2) <= fileCoeffs && fileDegree < 3) fileDegree++
        val rest = Array(restCount) { p("f_rest_$it") }

        val total = min(count, (map.capacity() - dataStart) / stride)
        val step = if (total > MAX_SPLATS) (total + MAX_SPLATS - 1) / MAX_SPLATS else 1
        val n = total / step
        val raw = Raw(n)
        var deg = if (dc[0] != null && rest.all { it != null }) fileDegree else 0
        while (deg > 0 && n.toLong() * ((deg + 1) * (deg + 1) - 1) * 3 * 4 > SH_BUDGET_BYTES) deg--
        val bands = (deg + 1) * (deg + 1) - 1
        if (deg > 0) {
            raw.shDegree = deg
            raw.shBase = FloatArray(n * 3)
            raw.shRest = FloatArray(n * bands * 3)
        }
        for (i in 0 until n) {
            val o = dataStart + i * step * stride
            raw.pos[i * 3] = read(map, o + px.offset, px.type)
            raw.pos[i * 3 + 1] = read(map, o + py.offset, py.type)
            raw.pos[i * 3 + 2] = read(map, o + pz.offset, pz.type)
            var r: Float
            var g: Float
            var b: Float
            if (dc[0] != null) {
                r = 0.5f + SH_C0 * read(map, o + dc[0]!!.offset, dc[0]!!.type)
                g = 0.5f + SH_C0 * read(map, o + dc[1]!!.offset, dc[1]!!.type)
                b = 0.5f + SH_C0 * read(map, o + dc[2]!!.offset, dc[2]!!.type)
            } else if (rgb[0] != null) {
                val div = if (rgb[0]!!.type.startsWith("float")) 1f else 255f
                r = read(map, o + rgb[0]!!.offset, rgb[0]!!.type) / div
                g = read(map, o + rgb[1]!!.offset, rgb[1]!!.type) / div
                b = read(map, o + rgb[2]!!.offset, rgb[2]!!.type) / div
            } else {
                r = 0.8f; g = 0.8f; b = 0.8f
            }
            val a = if (op != null) 1f / (1f + exp(-read(map, o + op.offset, op.type))) else 1f
            if (deg > 0) {
                raw.shBase!![i * 3] = r
                raw.shBase!![i * 3 + 1] = g
                raw.shBase!![i * 3 + 2] = b
                // file layout: f_rest_{c * (fileCoeffs - 1) + (k - 1)}
                val dst = raw.shRest!!
                for (k in 1..bands) for (c in 0 until 3) {
                    val pr = rest[c * (fileCoeffs - 1) + (k - 1)]!!
                    dst[(i * bands + (k - 1)) * 3 + c] = read(map, o + pr.offset, pr.type)
                }
            }
            raw.rgba[i] = to8(r) or (to8(g) shl 8) or (to8(b) shl 16) or (to8(a) shl 24)
            if (isGaussian) {
                for (k in 0 until 3) raw.scale[i * 3 + k] = exp(read(map, o + sc[k]!!.offset, sc[k]!!.type))
                var qw = read(map, o + rt[0]!!.offset, rt[0]!!.type)
                var qx = read(map, o + rt[1]!!.offset, rt[1]!!.type)
                var qy = read(map, o + rt[2]!!.offset, rt[2]!!.type)
                var qz = read(map, o + rt[3]!!.offset, rt[3]!!.type)
                val l = sqrt(qw * qw + qx * qx + qy * qy + qz * qz).let { if (it < 1e-12f) 1f else it }
                qw /= l; qx /= l; qy /= l; qz /= l
                raw.rot[i * 4] = qw; raw.rot[i * 4 + 1] = qx; raw.rot[i * 4 + 2] = qy; raw.rot[i * 4 + 3] = qz
            } else {
                raw.rot[i * 4] = 1f
            }
            if (i % 50000 == 0) onProgress(0.5f * i / n)
        }
        if (!isGaussian) {
            // plain point cloud: render as small round splats
            val (_, radius) = bounds(raw.pos, n)
            val s = radius / 400f
            for (i in 0 until n * 3) raw.scale[i] = s
        }
        return build(raw, n, onProgress)
    }

    private fun to8(v: Float): Int = (v.coerceIn(0f, 1f) * 255f + 0.5f).toInt()

    /** Robust centre (median) and radius (90th percentile distance). */
    private fun bounds(pos: FloatArray, n: Int): Pair<FloatArray, Float> {
        if (n == 0) return Pair(floatArrayOf(0f, 0f, 0f), 1f)
        val step = max(1, n / 20000)
        val m = (n + step - 1) / step
        val center = FloatArray(3)
        for (k in 0 until 3) {
            val vals = FloatArray(m) { pos[(it * step) * 3 + k] }
            vals.sort()
            center[k] = vals[m / 2]
        }
        val d = FloatArray(m) {
            val i = it * step
            val dx = pos[i * 3] - center[0]
            val dy = pos[i * 3 + 1] - center[1]
            val dz = pos[i * 3 + 2] - center[2]
            sqrt(dx * dx + dy * dy + dz * dz)
        }
        d.sort()
        return Pair(center, max(d[(m * 9) / 10], 1e-3f))
    }

    private fun build(raw: Raw, n: Int, onProgress: (Float) -> Unit): SplatScene {
        val (center, radius) = bounds(raw.pos, n)
        val buf = ByteBuffer.allocateDirect(n * 32).order(ByteOrder.nativeOrder())
        val ib = buf.asIntBuffer()
        val R = FloatArray(9)
        for (i in 0 until n) {
            val w = raw.rot[i * 4]
            val x = raw.rot[i * 4 + 1]
            val y = raw.rot[i * 4 + 2]
            val z = raw.rot[i * 4 + 3]
            R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y)
            R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x)
            R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y)
            val s0 = raw.scale[i * 3]
            val s1 = raw.scale[i * 3 + 1]
            val s2 = raw.scale[i * 3 + 2]
            // M = R * diag(s), Sigma = M M^T
            val m00 = R[0] * s0; val m01 = R[1] * s1; val m02 = R[2] * s2
            val m10 = R[3] * s0; val m11 = R[4] * s1; val m12 = R[5] * s2
            val m20 = R[6] * s0; val m21 = R[7] * s1; val m22 = R[8] * s2
            var xx = m00 * m00 + m01 * m01 + m02 * m02
            var xy = m00 * m10 + m01 * m11 + m02 * m12
            var xz = m00 * m20 + m01 * m21 + m02 * m22
            var yy = m10 * m10 + m11 * m11 + m12 * m12
            var yz = m10 * m20 + m11 * m21 + m12 * m22
            var zz = m20 * m20 + m21 * m21 + m22 * m22
            val norm = max(max(xx, yy), max(zz, 1e-30f))
            xx /= norm; xy /= norm; xz /= norm; yy /= norm; yz /= norm; zz /= norm
            val o = i * 8
            ib.put(o, java.lang.Float.floatToRawIntBits(raw.pos[i * 3]))
            ib.put(o + 1, java.lang.Float.floatToRawIntBits(raw.pos[i * 3 + 1]))
            ib.put(o + 2, java.lang.Float.floatToRawIntBits(raw.pos[i * 3 + 2]))
            ib.put(o + 3, raw.rgba[i])
            ib.put(o + 4, packHalf2(xx, xy))
            ib.put(o + 5, packHalf2(xz, yy))
            ib.put(o + 6, packHalf2(yz, zz))
            ib.put(o + 7, java.lang.Float.floatToRawIntBits(norm))
            if (i % 50000 == 0) onProgress(0.5f + 0.5f * i / n)
        }
        onProgress(1f)
        val opacity = if (raw.shDegree > 0) ByteArray(n) { (raw.rgba[it] ushr 24).toByte() } else null
        return SplatScene(n, raw.pos, buf, center, radius, raw.shDegree, raw.shBase, raw.shRest, opacity)
    }

    /** Two floats as IEEE half precision, first in the low 16 bits (GLSL unpackHalf2x16). */
    private fun packHalf2(a: Float, b: Float): Int = (toHalf(a) and 0xFFFF) or (toHalf(b) shl 16)

    fun toHalf(f: Float): Int {
        val bits = java.lang.Float.floatToRawIntBits(f)
        val sign = (bits ushr 16) and 0x8000
        val exp = (bits ushr 23) and 0xFF
        var mant = bits and 0x7FFFFF
        if (exp == 0xFF) return sign or 0x7C00 or (if (mant != 0) 0x200 else 0) // inf / nan
        val e = exp - 127 + 15
        if (e >= 0x1F) return sign or 0x7C00 // overflow -> inf
        if (e <= 0) {
            if (e < -10) return sign // underflow -> 0
            mant = mant or 0x800000
            val shift = 14 - e
            var h = mant shr shift
            if ((mant shr (shift - 1)) and 1 != 0) h += 1 // round
            return sign or h
        }
        var h = sign or (e shl 10) or (mant shr 13)
        if (mant and 0x1000 != 0) h += 1 // round half up (may carry into exponent: still correct)
        return h
    }
}
