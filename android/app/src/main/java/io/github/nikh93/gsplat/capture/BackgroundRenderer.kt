package io.github.nikh93.gsplat.capture

import android.opengl.GLES11Ext
import android.opengl.GLES30
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import io.github.nikh93.gsplat.util.GlUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** Draws the ARCore camera image (external OES texture) as a full screen quad. */
class BackgroundRenderer {
    var textureId = -1
        private set
    private var program = 0
    private var aPos = 0
    private var aTex = 0
    private val quadCoords: FloatBuffer = GlUtil.floatBuffer(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
    private val texCoords: FloatBuffer =
        ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

    fun createOnGlThread() {
        val tex = IntArray(1)
        GLES30.glGenTextures(1, tex, 0)
        textureId = tex[0]
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)

        program = GlUtil.createProgram(VERTEX, FRAGMENT)
        aPos = GLES30.glGetAttribLocation(program, "aPos")
        aTex = GLES30.glGetAttribLocation(program, "aTex")
    }

    fun draw(frame: Frame) {
        if (frame.hasDisplayGeometryChanged()) {
            quadCoords.position(0)
            texCoords.position(0)
            frame.transformCoordinates2d(
                Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quadCoords,
                Coordinates2d.TEXTURE_NORMALIZED, texCoords
            )
        }
        if (frame.timestamp == 0L) return // camera has not produced a frame yet

        quadCoords.position(0)
        texCoords.position(0)
        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthMask(false)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES30.glUseProgram(program)
        GLES30.glVertexAttribPointer(aPos, 2, GLES30.GL_FLOAT, false, 0, quadCoords)
        GLES30.glVertexAttribPointer(aTex, 2, GLES30.GL_FLOAT, false, 0, texCoords)
        GLES30.glEnableVertexAttribArray(aPos)
        GLES30.glEnableVertexAttribArray(aTex)
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4)
        GLES30.glDisableVertexAttribArray(aPos)
        GLES30.glDisableVertexAttribArray(aTex)
        GLES30.glDepthMask(true)
        GLES30.glEnable(GLES30.GL_DEPTH_TEST)
    }

    companion object {
        private const val VERTEX = """
            attribute vec4 aPos;
            attribute vec2 aTex;
            varying vec2 vTex;
            void main() {
                gl_Position = aPos;
                vTex = aTex;
            }
        """
        private const val FRAGMENT = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            varying vec2 vTex;
            uniform samplerExternalOES sTexture;
            void main() {
                gl_FragColor = texture2D(sTexture, vTex);
            }
        """
    }
}
