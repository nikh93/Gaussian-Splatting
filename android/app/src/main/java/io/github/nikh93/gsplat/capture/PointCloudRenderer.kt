package io.github.nikh93.gsplat.capture

import android.opengl.GLES30
import io.github.nikh93.gsplat.util.GlUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** Draws the collected 3D points as coloured dots so the user sees the coverage. */
class PointCloudRenderer {
    private var program = 0
    private var aPos = 0
    private var aColor = 0
    private var uMvp = 0
    private var uSize = 0
    private var buffer: FloatBuffer = alloc(4096)
    private var count = 0

    fun createOnGlThread() {
        program = GlUtil.createProgram(VERTEX, FRAGMENT)
        aPos = GLES30.glGetAttribLocation(program, "aPos")
        aColor = GLES30.glGetAttribLocation(program, "aColor")
        uMvp = GLES30.glGetUniformLocation(program, "uMvp")
        uSize = GLES30.glGetUniformLocation(program, "uSize")
    }

    /** [xyzrgb] interleaved: x, y, z, r, g, b (colour in 0..1). */
    fun update(xyzrgb: FloatArray, n: Int) {
        if (buffer.capacity() < n * 6) buffer = alloc(n * 6 * 2)
        buffer.position(0)
        buffer.put(xyzrgb, 0, n * 6)
        buffer.position(0)
        count = n
    }

    fun draw(viewProj: FloatArray) {
        if (count == 0) return
        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(uMvp, 1, false, viewProj, 0)
        GLES30.glUniform1f(uSize, 6f)
        buffer.position(0)
        GLES30.glVertexAttribPointer(aPos, 3, GLES30.GL_FLOAT, false, 24, buffer)
        buffer.position(3)
        GLES30.glVertexAttribPointer(aColor, 3, GLES30.GL_FLOAT, false, 24, buffer)
        GLES30.glEnableVertexAttribArray(aPos)
        GLES30.glEnableVertexAttribArray(aColor)
        GLES30.glDrawArrays(GLES30.GL_POINTS, 0, count)
        GLES30.glDisableVertexAttribArray(aPos)
        GLES30.glDisableVertexAttribArray(aColor)
        buffer.position(0)
    }

    private fun alloc(floats: Int): FloatBuffer =
        ByteBuffer.allocateDirect(floats * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

    companion object {
        private const val VERTEX = """
            uniform mat4 uMvp;
            uniform float uSize;
            attribute vec3 aPos;
            attribute vec3 aColor;
            varying vec3 vColor;
            void main() {
                gl_Position = uMvp * vec4(aPos, 1.0);
                gl_PointSize = uSize;
                vColor = aColor;
            }
        """
        private const val FRAGMENT = """
            precision mediump float;
            varying vec3 vColor;
            void main() {
                vec2 d = gl_PointCoord - vec2(0.5);
                if (dot(d, d) > 0.25) discard;
                gl_FragColor = vec4(vColor, 1.0);
            }
        """
    }
}
