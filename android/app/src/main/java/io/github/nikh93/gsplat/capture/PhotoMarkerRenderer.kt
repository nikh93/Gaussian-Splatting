package io.github.nikh93.gsplat.capture

import android.opengl.GLES30
import android.opengl.Matrix
import com.google.ar.core.Anchor
import com.google.ar.core.TrackingState
import io.github.nikh93.gsplat.util.GlUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs

/**
 * Shows every captured photo in AR at the pose it was taken from: a camera frustum
 * (wire pyramid) with the photo's thumbnail on its image plane. This makes gaps and
 * the overlap between neighbouring photos visible while capturing.
 *
 * Every marker hangs on an ARCore [Anchor] created at the photo's camera pose. When
 * ARCore corrects its map (drift, loop closure) the anchor moves with the real scene,
 * so the marker stays where the photo was taken instead of floating away with the
 * world origin. The geometry is therefore stored in camera space.
 *
 * All methods run on the thread that owns the GL context.
 */
class PhotoMarkerRenderer {

    private class Marker(
        val lines: FloatBuffer,  // 8 line segments, xyz in camera space
        val quad: FloatBuffer,   // triangle strip, xyz in camera space + uv
        val texture: Int,
        val anchor: Anchor?,
        /** current camera-to-world pose: the anchor's while it is tracked */
        val c2w: FloatArray,
        val intr: Intrinsics,
        val sceneDepth: Float,
    ) {
        /** an anchor ARCore does not track right now has no valid pose: not drawn */
        var visible = true
    }

    private val markers = ArrayList<Marker>()
    private var lineProgram = 0
    private var texProgram = 0
    private var lPos = 0
    private var lMvp = 0
    private var lColor = 0
    private var tPos = 0
    private var tUv = 0
    private var tMvp = 0
    private var tAlpha = 0
    private var tTex = 0
    private val mvp = FloatArray(16)

    val count: Int get() = markers.size

    fun createOnGlThread() {
        lineProgram = GlUtil.createProgram(LINE_VS, LINE_FS)
        lPos = GLES30.glGetAttribLocation(lineProgram, "aPos")
        lMvp = GLES30.glGetUniformLocation(lineProgram, "uMvp")
        lColor = GLES30.glGetUniformLocation(lineProgram, "uColor")
        texProgram = GlUtil.createProgram(TEX_VS, TEX_FS)
        tPos = GLES30.glGetAttribLocation(texProgram, "aPos")
        tUv = GLES30.glGetAttribLocation(texProgram, "aUv")
        tMvp = GLES30.glGetUniformLocation(texProgram, "uMvp")
        tAlpha = GLES30.glGetUniformLocation(texProgram, "uAlpha")
        tTex = GLES30.glGetUniformLocation(texProgram, "uTex")
    }

    /**
     * Adds a photo. [c2w] column major camera pose (OpenGL axes, sensor aligned), [anchor]
     * an anchor at that pose (null if ARCore could not create one), [thumb] RGBA pixels of
     * size tw x th, [sceneDepth] typical distance to the scene.
     */
    fun add(c2w: FloatArray, anchor: Anchor?, intr: Intrinsics, thumb: ByteBuffer, tw: Int, th: Int, sceneDepth: Float) {
        val size = MARKER_DEPTH
        // image plane corners in camera space (row 0 of the image = top = +y)
        val hx = intr.width * 0.5f / intr.fx * size
        val hy = intr.height * 0.5f / intr.fy * size
        val cx = (intr.width * 0.5f - intr.cx) / intr.fx * size
        val cy = -(intr.height * 0.5f - intr.cy) / intr.fy * size
        val tl = floatArrayOf(cx - hx, cy + hy, -size)
        val tr = floatArrayOf(cx + hx, cy + hy, -size)
        val bl = floatArrayOf(cx - hx, cy - hy, -size)
        val br = floatArrayOf(cx + hx, cy - hy, -size)
        val apex = floatArrayOf(0f, 0f, 0f)
        val lines = GlUtil.floatBuffer(
            floatArrayOf(
                *apex, *tl, *apex, *tr, *apex, *bl, *apex, *br,
                *tl, *tr, *tr, *br, *br, *bl, *bl, *tl,
            )
        )
        val quad = GlUtil.floatBuffer(
            floatArrayOf(
                tl[0], tl[1], tl[2], 0f, 0f,
                tr[0], tr[1], tr[2], 1f, 0f,
                bl[0], bl[1], bl[2], 0f, 1f,
                br[0], br[1], br[2], 1f, 1f,
            )
        )
        val tex = IntArray(1)
        GLES30.glGenTextures(1, tex, 0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, tex[0])
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 1)
        thumb.position(0)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA, tw, th, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, thumb
        )
        markers.add(Marker(lines, quad, tex[0], anchor, c2w.copyOf(), intr, sceneDepth))
    }

    /** Removes the latest photo (undo). */
    fun removeLast() {
        val m = markers.removeLastOrNull() ?: return
        m.anchor?.detach()
        GLES30.glDeleteTextures(1, intArrayOf(m.texture), 0)
    }

    fun clear() {
        while (markers.isNotEmpty()) removeLast()
    }

    /** Follows ARCore's corrections: copies the anchor poses (call once per frame). */
    fun updatePoses() {
        for (m in markers) {
            val a = m.anchor ?: continue
            m.visible = a.trackingState == TrackingState.TRACKING
            if (m.visible) a.pose.toMatrix(m.c2w, 0)
        }
    }

    fun draw(viewProj: FloatArray) {
        if (markers.isEmpty()) return
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        GLES30.glDepthMask(false)

        // thumbnails
        GLES30.glUseProgram(texProgram)
        GLES30.glUniform1i(tTex, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glEnableVertexAttribArray(tPos)
        GLES30.glEnableVertexAttribArray(tUv)
        for ((i, m) in markers.withIndex()) {
            if (!m.visible) continue
            Matrix.multiplyMM(mvp, 0, viewProj, 0, m.c2w, 0)
            GLES30.glUniformMatrix4fv(tMvp, 1, false, mvp, 0)
            // see-through, so the subject behind a photo stays recognisable
            GLES30.glUniform1f(tAlpha, if (i == markers.size - 1) 0.4f else 0.25f)
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, m.texture)
            m.quad.position(0)
            GLES30.glVertexAttribPointer(tPos, 3, GLES30.GL_FLOAT, false, 20, m.quad)
            m.quad.position(3)
            GLES30.glVertexAttribPointer(tUv, 2, GLES30.GL_FLOAT, false, 20, m.quad)
            GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        }
        GLES30.glDisableVertexAttribArray(tPos)
        GLES30.glDisableVertexAttribArray(tUv)

        // frustums: older photos cyan, the latest one yellow
        GLES30.glUseProgram(lineProgram)
        GLES30.glLineWidth(3f)
        GLES30.glEnableVertexAttribArray(lPos)
        for ((i, m) in markers.withIndex()) {
            if (!m.visible) continue
            Matrix.multiplyMM(mvp, 0, viewProj, 0, m.c2w, 0)
            GLES30.glUniformMatrix4fv(lMvp, 1, false, mvp, 0)
            if (i == markers.size - 1) GLES30.glUniform4f(lColor, 1f, 0.85f, 0.1f, 1f)
            else GLES30.glUniform4f(lColor, 0.2f, 0.85f, 1f, 0.6f)
            m.lines.position(0)
            GLES30.glVertexAttribPointer(lPos, 3, GLES30.GL_FLOAT, false, 12, m.lines)
            GLES30.glDrawArrays(GLES30.GL_LINES, 0, 16)
        }
        GLES30.glDisableVertexAttribArray(lPos)
        GLES30.glDepthMask(true)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    /** Fraction (0..1) of the current view covered by the last photo, null without photos. */
    fun overlapWithLast(viewProj: FloatArray): Float? {
        val m = markers.lastOrNull() ?: return null
        return overlap(m.c2w, m.intr, m.sceneDepth, viewProj)
    }

    companion object {
        /**
         * Overlap estimate: the photo's image rectangle at its scene depth [depth] is projected
         * into the current view ([viewProj], column major) and clipped to the screen square.
         * Returns the covered fraction of the screen (0..1).
         */
        fun overlap(c2w: FloatArray, intr: Intrinsics, depth: Float, viewProj: FloatArray): Float {
            val d = depth
            val corners = arrayOf(
                floatArrayOf(-intr.cx / intr.fx * d, intr.cy / intr.fy * d),
                floatArrayOf((intr.width - intr.cx) / intr.fx * d, intr.cy / intr.fy * d),
                floatArrayOf((intr.width - intr.cx) / intr.fx * d, -(intr.height - intr.cy) / intr.fy * d),
                floatArrayOf(-intr.cx / intr.fx * d, -(intr.height - intr.cy) / intr.fy * d),
            )
            var poly = ArrayList<FloatArray>(4)
            for (c in corners) {
                val w = toWorld(c2w, c[0], c[1], -d)
                val x = viewProj[0] * w[0] + viewProj[4] * w[1] + viewProj[8] * w[2] + viewProj[12]
                val y = viewProj[1] * w[0] + viewProj[5] * w[1] + viewProj[9] * w[2] + viewProj[13]
                val ww = viewProj[3] * w[0] + viewProj[7] * w[1] + viewProj[11] * w[2] + viewProj[15]
                if (ww <= 1e-4f) return 0f // the photographed region is behind the camera
                poly.add(floatArrayOf(x / ww, y / ww))
            }
            // clip against the screen square [-1, 1]^2 (Sutherland-Hodgman)
            for (edge in 0 until 4) {
                val out = ArrayList<FloatArray>(poly.size + 4)
                for (i in poly.indices) {
                    val a = poly[i]
                    val b = poly[(i + 1) % poly.size]
                    val ina = inside(a, edge)
                    val inb = inside(b, edge)
                    if (ina) out.add(a)
                    if (ina != inb) out.add(intersect(a, b, edge))
                }
                poly = out
                if (poly.isEmpty()) return 0f
            }
            var area = 0f
            for (i in poly.indices) {
                val a = poly[i]
                val b = poly[(i + 1) % poly.size]
                area += a[0] * b[1] - b[0] * a[1]
            }
            return (abs(area) * 0.5f / 4f).coerceIn(0f, 1f)
        }

        private fun inside(p: FloatArray, edge: Int) = when (edge) {
            0 -> p[0] >= -1f
            1 -> p[0] <= 1f
            2 -> p[1] >= -1f
            else -> p[1] <= 1f
        }

        private fun intersect(a: FloatArray, b: FloatArray, edge: Int): FloatArray {
            val (axis, bound) = when (edge) {
                0 -> 0 to -1f
                1 -> 0 to 1f
                2 -> 1 to -1f
                else -> 1 to 1f
            }
            val t = (bound - a[axis]) / (b[axis] - a[axis])
            return floatArrayOf(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        }

        fun toWorld(c2w: FloatArray, x: Float, y: Float, z: Float) = floatArrayOf(
            c2w[0] * x + c2w[4] * y + c2w[8] * z + c2w[12],
            c2w[1] * x + c2w[5] * y + c2w[9] * z + c2w[13],
            c2w[2] * x + c2w[6] * y + c2w[10] * z + c2w[14],
        )

        /** distance of the drawn image plane from the camera centre (metres) */
        private const val MARKER_DEPTH = 0.06f

        private const val LINE_VS = """
            uniform mat4 uMvp;
            attribute vec3 aPos;
            void main() { gl_Position = uMvp * vec4(aPos, 1.0); }
        """
        private const val LINE_FS = """
            precision mediump float;
            uniform vec4 uColor;
            void main() { gl_FragColor = uColor; }
        """
        private const val TEX_VS = """
            uniform mat4 uMvp;
            attribute vec3 aPos;
            attribute vec2 aUv;
            varying vec2 vUv;
            void main() { gl_Position = uMvp * vec4(aPos, 1.0); vUv = aUv; }
        """
        private const val TEX_FS = """
            precision mediump float;
            uniform sampler2D uTex;
            uniform float uAlpha;
            varying vec2 vUv;
            void main() { gl_FragColor = vec4(texture2D(uTex, vUv).rgb, uAlpha); }
        """
    }
}
