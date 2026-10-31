package io.github.nikh93.gsplat.ar

import android.opengl.GLES30
import io.github.nikh93.gsplat.util.GlUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.sin

/**
 * Placement guides drawn over the camera image: the target circle on the floor and
 * the vertical line from the floor up to the chosen model centre.
 */
class ArGuideRenderer {
    private var program = 0
    private var aPos = 0
    private var uMvp = 0
    private var uColor = 0
    private val buf: FloatBuffer =
        ByteBuffer.allocateDirect(4 * 3 * 2 * (SEGMENTS + 1)).order(ByteOrder.nativeOrder()).asFloatBuffer()

    fun createOnGlThread() {
        program = GlUtil.createProgram(VS, FS)
        aPos = GLES30.glGetAttribLocation(program, "aPos")
        uMvp = GLES30.glGetUniformLocation(program, "uMvp")
        uColor = GLES30.glGetUniformLocation(program, "uColor")
    }

    private fun begin(viewProj: FloatArray, color: FloatArray) {
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(uMvp, 1, false, viewProj, 0)
        GLES30.glUniform4fv(uColor, 1, color, 0)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_SRC_ALPHA, GLES30.GL_ONE_MINUS_SRC_ALPHA)
        // client side vertex arrays need the default VAO and no bound buffer
        GLES30.glBindVertexArray(0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glEnableVertexAttribArray(aPos)
    }

    private fun end() {
        GLES30.glDisableVertexAttribArray(aPos)
        GLES30.glDisable(GLES30.GL_BLEND)
    }

    /** Horizontal ring around (x, y, z), radii in metres. */
    fun drawRing(viewProj: FloatArray, x: Float, y: Float, z: Float, inner: Float, outer: Float, color: FloatArray) {
        buf.clear()
        for (i in 0..SEGMENTS) {
            val a = (2.0 * Math.PI * i / SEGMENTS).toFloat()
            val c = cos(a)
            val s = sin(a)
            buf.put(x + outer * c).put(y).put(z + outer * s)
            buf.put(x + inner * c).put(y).put(z + inner * s)
        }
        buf.flip()
        begin(viewProj, color)
        GLES30.glVertexAttribPointer(aPos, 3, GLES30.GL_FLOAT, false, 12, buf)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 2 * (SEGMENTS + 1))
        end()
    }

    /** Vertical bar from (x, y0, z) to (x, y1, z), drawn as a thin cross of two quads. */
    fun drawVertical(viewProj: FloatArray, x: Float, y0: Float, y1: Float, z: Float, halfWidth: Float, color: FloatArray) {
        buf.clear()
        buf.put(x - halfWidth).put(y0).put(z)
        buf.put(x + halfWidth).put(y0).put(z)
        buf.put(x - halfWidth).put(y1).put(z)
        buf.put(x + halfWidth).put(y1).put(z)
        buf.put(x).put(y0).put(z - halfWidth)
        buf.put(x).put(y0).put(z + halfWidth)
        buf.put(x).put(y1).put(z - halfWidth)
        buf.put(x).put(y1).put(z + halfWidth)
        buf.flip()
        begin(viewProj, color)
        GLES30.glVertexAttribPointer(aPos, 3, GLES30.GL_FLOAT, false, 12, buf)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 4, 4)
        end()
    }

    companion object {
        private const val SEGMENTS = 48
        private const val VS = """
            uniform mat4 uMvp;
            attribute vec3 aPos;
            void main() { gl_Position = uMvp * vec4(aPos, 1.0); }
        """
        private const val FS = """
            precision mediump float;
            uniform vec4 uColor;
            void main() { gl_FragColor = uColor; }
        """
    }
}
