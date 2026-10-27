package io.github.nikh93.gsplat.capture

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.view.Surface

/**
 * A GLES 3 context for rendering on the thread that owns it (here: the main thread).
 *
 * ARCore is driven from the main thread like in SceneView: on the Galaxy A56 ARCore
 * received the IMU samples only in 160 ms batches while Session.update() ran on a
 * GLSurfaceView render thread, and tracking dropped to PAUSED several times a second.
 * Without a window surface the context is made current on a 1x1 pbuffer, so the camera
 * texture stays valid while the SurfaceView is recreated.
 */
class EglCore {
    private val display: EGLDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
    private val config: EGLConfig
    private val context: EGLContext
    private val pbuffer: EGLSurface
    private var window: EGLSurface = EGL14.EGL_NO_SURFACE

    init {
        val version = IntArray(2)
        check(EGL14.eglInitialize(display, version, 0, version, 1)) { "eglInitialize failed" }
        val attribs = intArrayOf(
            EGL14.EGL_RED_SIZE, 8, EGL14.EGL_GREEN_SIZE, 8, EGL14.EGL_BLUE_SIZE, 8, EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_DEPTH_SIZE, 16,
            EGL14.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT or EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_NONE,
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val num = IntArray(1)
        check(EGL14.eglChooseConfig(display, attribs, 0, configs, 0, 1, num, 0) && num[0] > 0) { "no EGL config" }
        config = configs[0]!!
        context = EGL14.eglCreateContext(
            display, config, EGL14.EGL_NO_CONTEXT, intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE), 0
        )
        check(context != EGL14.EGL_NO_CONTEXT) { "eglCreateContext failed" }
        pbuffer = EGL14.eglCreatePbufferSurface(
            display, config, intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE), 0
        )
        makeCurrent()
    }

    val hasWindow: Boolean get() = window != EGL14.EGL_NO_SURFACE

    fun attach(surface: Surface) {
        detach()
        window = EGL14.eglCreateWindowSurface(display, config, surface, intArrayOf(EGL14.EGL_NONE), 0)
        makeCurrent()
    }

    fun detach() {
        if (window == EGL14.EGL_NO_SURFACE) return
        EGL14.eglMakeCurrent(display, pbuffer, pbuffer, context)
        EGL14.eglDestroySurface(display, window)
        window = EGL14.EGL_NO_SURFACE
    }

    fun makeCurrent() {
        val s = if (hasWindow) window else pbuffer
        EGL14.eglMakeCurrent(display, s, s, context)
    }

    fun swap() {
        if (hasWindow) EGL14.eglSwapBuffers(display, window)
    }

    fun release() {
        detach()
        EGL14.eglMakeCurrent(display, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        EGL14.eglDestroySurface(display, pbuffer)
        EGL14.eglDestroyContext(display, context)
        EGL14.eglTerminate(display)
    }

    private companion object {
        const val EGL_OPENGL_ES3_BIT = 0x40
    }
}
