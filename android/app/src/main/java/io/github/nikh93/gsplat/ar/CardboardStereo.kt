package io.github.nikh93.gsplat.ar

import android.opengl.GLES30
import android.opengl.Matrix
import com.google.cardboard.sdk.DistortionRenderer
import com.google.cardboard.sdk.LensDistortion
import com.google.cardboard.sdk.QrCode
import com.google.cardboard.sdk.nativetypes.EyeTextureDescription
import com.google.cardboard.sdk.nativetypes.EyeType
import io.github.nikh93.gsplat.viewer.SplatDrawer

/**
 * Google Cardboard stereo rendering with the official Cardboard SDK.
 *
 * Head tracking is not done by the Cardboard head tracker but by ARCore (6 DoF, the
 * user can walk around the anchored model): the ARCore view matrix is the head pose,
 * the Cardboard SDK supplies the per eye offsets and projections of the scanned viewer
 * (QR code) and renders the lens distortion. Both eyes are drawn side by side into one
 * texture, which the distortion renderer warps onto the display.
 *
 * All methods run on the thread that owns the GL context.
 */
class CardboardStereo {
    /** set after a new viewer was scanned: the distortion is rebuilt before the next frame */
    var paramsChanged = true

    private var lens: LensDistortion? = null
    private var distortion: DistortionRenderer? = null
    private var width = 0
    private var height = 0
    private var fbo = 0
    private var texture = 0
    private val eyeFromHead = Array(2) { FloatArray(16) }
    private val eyeProj = Array(2) { FloatArray(16) }
    private val left = EyeTextureDescription()
    private val right = EyeTextureDescription()
    private val headModel = FloatArray(16)
    private val eyeModel = FloatArray(16)

    private fun setup(w: Int, h: Int) {
        if (!paramsChanged && w == width && h == height && lens != null) return
        paramsChanged = false
        width = w
        height = h
        // the viewer scanned with the QR code, else the original Cardboard V1
        val params = QrCode.getSavedDeviceParams() ?: QrCode.getCardboardV1DeviceParams()
        lens?.close()
        val l = LensDistortion(params, w, h)
        lens = l
        val d = distortion ?: DistortionRenderer().also { distortion = it }
        for (eye in intArrayOf(EyeType.LEFT, EyeType.RIGHT)) {
            l.getEyeFromHeadMatrix(eye, eyeFromHead[eye])
            l.getEyeProjectionMatrix(eye, NEAR, FAR, eyeProj[eye])
            l.getDistortionMesh(eye)?.let { d.setMesh(it, eye) }
        }

        // render target: both eyes side by side
        if (texture == 0) {
            val ids = IntArray(1)
            GLES30.glGenTextures(1, ids, 0)
            texture = ids[0]
            GLES30.glGenFramebuffers(1, ids, 0)
            fbo = ids[0]
        }
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, w, h, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
        )
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glFramebufferTexture2D(
            GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0, GLES30.GL_TEXTURE_2D, texture, 0
        )
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)

        left.texture = texture.toLong()
        left.leftU = 0f
        left.rightU = 0.5f
        left.topV = 1f
        left.bottomV = 0f
        right.texture = texture.toLong()
        right.leftU = 0.5f
        right.rightU = 1f
        right.topV = 1f
        right.bottomV = 0f
    }

    /**
     * Renders the model for both eyes. [headView] is the ARCore view matrix (world to
     * display-oriented camera), [model] the model matrix, [w] x [h] the display size.
     */
    fun render(w: Int, h: Int, headView: FloatArray, model: FloatArray, drawer: SplatDrawer, splatScale: Float) {
        setup(w, h)
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo)
        GLES30.glViewport(0, 0, w, h)
        GLES30.glClearColor(0f, 0f, 0f, 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        Matrix.multiplyMM(headModel, 0, headView, 0, model, 0)
        drawer.update(headModel) // one depth order for both eyes
        val eyeW = w / 2
        for (eye in intArrayOf(EyeType.LEFT, EyeType.RIGHT)) {
            Matrix.multiplyMM(eyeModel, 0, eyeFromHead[eye], 0, headModel, 0)
            GLES30.glViewport(eye * eyeW, 0, eyeW, h)
            drawer.draw(eyeModel, eyeProj[eye], eyeW, h, splatScale)
        }
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, 0)
        GLES30.glViewport(0, 0, w, h)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT)
        distortion?.renderEyeToDisplay(0, 0, 0, w, h, left, right)
        // the distortion renderer has its own GL state; leave a clean one for the next draws
        GLES30.glDisable(GLES30.GL_SCISSOR_TEST)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
        GLES30.glBindBuffer(GLES30.GL_ELEMENT_ARRAY_BUFFER, 0)
        GLES30.glBindVertexArray(0)
    }

    fun release() {
        lens?.close()
        lens = null
        distortion?.close()
        distortion = null
        if (texture != 0) {
            GLES30.glDeleteTextures(1, intArrayOf(texture), 0)
            GLES30.glDeleteFramebuffers(1, intArrayOf(fbo), 0)
            texture = 0
            fbo = 0
        }
    }

    companion object {
        private const val NEAR = 0.02f
        private const val FAR = 100f
    }
}
