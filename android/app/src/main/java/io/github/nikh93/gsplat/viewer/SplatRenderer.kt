package io.github.nikh93.gsplat.viewer

import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import android.os.SystemClock
import io.github.nikh93.gsplat.util.GlUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.tan

/**
 * Real-time Gaussian splat renderer (OpenGL ES 3.0).
 *
 * Every splat is an instanced screen aligned quad. The vertex shader fetches the
 * splat from an integer texture, projects its 3D covariance to a 2D ellipse
 * (EWA splatting, as in the 3DGS paper) and sizes the quad to 2 sigma per axis
 * (scaled by sqrt 2); the fragment shader evaluates the gaussian. Splats are
 * blended back to front using indices from [SplatSorter].
 */
class SplatRenderer(private val camera: OrbitCamera) : GLSurfaceView.Renderer {

    @Volatile var splatScale = 1f
    @Volatile var autoRotate = false
    @Volatile var background = floatArrayOf(0f, 0f, 0f)
    @Volatile var fps = 0
        private set
    var onFirstFrame: (() -> Unit)? = null

    private var scene: SplatScene? = null
    @Volatile private var pendingScene: SplatScene? = null
    private var sorter: SplatSorter? = null
    private var drawCount = 0

    private var program = 0
    private var vao = 0
    private var quadVbo = 0
    private var indexVbo = 0
    private var texture = 0
    private var colorTexture = 0
    private var rowSplats = 1
    private var colorRow = 1
    private var useColorTex = false
    private var uProj = 0
    private var uView = 0
    private var uFocal = 0
    private var uViewport = 0
    private var uScale = 0
    private var uRow = 0
    private var uData = 0
    private var uColor = 0
    private var uUseColor = 0
    private var uColorRow = 0

    private var width = 1
    private var height = 1
    private val viewMat = FloatArray(16)
    private val projMat = FloatArray(16)
    private val lastSorted = FloatArray(16)
    private var hasSorted = false
    private var frames = 0
    private var fpsTime = 0L
    private var lastFrameTime = 0L
    private var firstFrameDone = false

    fun setScene(s: SplatScene) {
        pendingScene = s
    }

    fun release() {
        sorter?.shutdown()
        sorter = null
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        program = GlUtil.createProgram(VERTEX, FRAGMENT)
        uProj = GLES30.glGetUniformLocation(program, "uProj")
        uView = GLES30.glGetUniformLocation(program, "uView")
        uFocal = GLES30.glGetUniformLocation(program, "uFocal")
        uViewport = GLES30.glGetUniformLocation(program, "uViewport")
        uScale = GLES30.glGetUniformLocation(program, "uSplatScale")
        uRow = GLES30.glGetUniformLocation(program, "uRowSplats")
        uData = GLES30.glGetUniformLocation(program, "uData")
        uColor = GLES30.glGetUniformLocation(program, "uColor")
        uUseColor = GLES30.glGetUniformLocation(program, "uUseColor")
        uColorRow = GLES30.glGetUniformLocation(program, "uColorRow")

        val ids = IntArray(2)
        GLES30.glGenVertexArrays(1, ids, 0)
        vao = ids[0]
        GLES30.glGenBuffers(2, ids, 0)
        quadVbo = ids[0]
        indexVbo = ids[1]
        GLES30.glBindVertexArray(vao)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, quadVbo)
        val quad = GlUtil.floatBuffer(floatArrayOf(-2f, -2f, 2f, -2f, -2f, 2f, 2f, 2f))
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, 8 * 4, quad, GLES30.GL_STATIC_DRAW)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 2, GLES30.GL_FLOAT, false, 8, 0)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, indexVbo)
        GLES30.glEnableVertexAttribArray(1)
        GLES30.glVertexAttribIPointer(1, 1, GLES30.GL_UNSIGNED_INT, 4, 0)
        GLES30.glVertexAttribDivisor(1, 1)
        GLES30.glBindVertexArray(0)

        val tex = IntArray(2)
        GLES30.glGenTextures(2, tex, 0)
        texture = tex[0]
        colorTexture = tex[1]
        // the context may have been recreated: upload the scene again
        scene?.let { pendingScene = pendingScene ?: it }
        scene = null
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        width = maxOf(w, 1)
        height = maxOf(h, 1)
        GLES30.glViewport(0, 0, width, height)
    }

    private fun upload(s: SplatScene) {
        val maxSize = IntArray(1)
        GLES30.glGetIntegerv(GLES30.GL_MAX_TEXTURE_SIZE, maxSize, 0)
        val texW = (min(maxSize[0], 4096) / 2) * 2
        rowSplats = texW / 2
        val maxRows = maxSize[0]
        val n = min(s.count, rowSplats * maxRows)
        val rows = (n + rowSplats - 1) / rowSplats

        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 4)
        GLES30.glTexImage2D(
            GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA32UI, texW, rows, 0,
            GLES30.GL_RGBA_INTEGER, GLES30.GL_UNSIGNED_INT, null
        )
        val fullRows = n / rowSplats
        val data = s.gpuData
        if (fullRows > 0) {
            data.position(0)
            GLES30.glTexSubImage2D(
                GLES30.GL_TEXTURE_2D, 0, 0, 0, texW, fullRows,
                GLES30.GL_RGBA_INTEGER, GLES30.GL_UNSIGNED_INT, data
            )
        }
        val rest = n - fullRows * rowSplats
        if (rest > 0) {
            data.position(fullRows * rowSplats * 32)
            GLES30.glTexSubImage2D(
                GLES30.GL_TEXTURE_2D, 0, 0, fullRows, rest * 2, 1,
                GLES30.GL_RGBA_INTEGER, GLES30.GL_UNSIGNED_INT, data.slice()
            )
        }
        data.position(0)
        GlUtil.checkError("texture upload")

        // view dependent colours (spherical harmonics) live in an RGBA8 texture
        useColorTex = s.shDegree > 0
        if (useColorTex) {
            colorRow = min(maxSize[0], 4096)
            val colorRows = (n + colorRow - 1) / colorRow
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, colorTexture)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST)
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST)
            GLES30.glTexImage2D(
                GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8, colorRow, colorRows, 0,
                GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null
            )
            // start with the base colour stored in the splat texture
            val initial = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder())
            val ints = data.asIntBuffer()
            for (i in 0 until n) initial.putInt(i * 4, ints.get(i * 8 + 3))
            uploadColors(initial, n)
        }

        // identity order until the first sort arrives
        val idx = ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asIntBuffer()
        for (i in 0 until n) idx.put(i, i)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, indexVbo)
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, n * 4, idx, GLES30.GL_DYNAMIC_DRAW)
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)

        drawCount = n
        sorter?.shutdown()
        sorter = SplatSorter(s, n)
        hasSorted = false
        scene = s
    }

    private fun uploadColors(colors: ByteBuffer, n: Int) {
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, colorTexture)
        GLES30.glPixelStorei(GLES30.GL_UNPACK_ALIGNMENT, 4)
        val full = n / colorRow
        if (full > 0) {
            colors.position(0)
            GLES30.glTexSubImage2D(
                GLES30.GL_TEXTURE_2D, 0, 0, 0, colorRow, full, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, colors
            )
        }
        val rest = n - full * colorRow
        if (rest > 0) {
            colors.position(full * colorRow * 4)
            GLES30.glTexSubImage2D(
                GLES30.GL_TEXTURE_2D, 0, 0, full, rest, 1, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, colors.slice()
            )
        }
        colors.position(0)
    }

    override fun onDrawFrame(gl: GL10?) {
        pendingScene?.let {
            pendingScene = null
            upload(it)
        }
        val now = SystemClock.elapsedRealtime()
        val dt = if (lastFrameTime == 0L) 0f else (now - lastFrameTime) / 1000f
        lastFrameTime = now
        if (autoRotate) camera.autoRotate(dt * 0.4f)

        val bg = background
        GLES30.glClearColor(bg[0], bg[1], bg[2], 1f)
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT or GLES30.GL_DEPTH_BUFFER_BIT)
        val s = scene ?: return

        camera.viewModel(viewMat)
        val (near, far) = camera.nearFar()
        val fov = camera.fovDeg
        Matrix.perspectiveM(projMat, 0, fov, width.toFloat() / height, near, far)
        val fy = height / (2f * tan(Math.toRadians(fov / 2.0)).toFloat())
        val fx = fy // square pixels

        // resort when the view changed noticeably
        val srt = sorter
        if (srt != null && (!hasSorted || viewChanged())) {
            srt.request(viewMat)
            viewMat.copyInto(lastSorted)
            hasSorted = true
        }
        srt?.poll()?.let { res ->
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, indexVbo)
            GLES30.glBufferSubData(GLES30.GL_ARRAY_BUFFER, 0, drawCount * 4, res.indices)
            GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, 0)
            if (useColorTex) res.colors?.let { uploadColors(it, drawCount) }
        }

        GLES30.glDisable(GLES30.GL_DEPTH_TEST)
        GLES30.glDepthMask(false)
        GLES30.glEnable(GLES30.GL_BLEND)
        GLES30.glBlendFunc(GLES30.GL_ONE, GLES30.GL_ONE_MINUS_SRC_ALPHA)

        GLES30.glUseProgram(program)
        GLES30.glUniformMatrix4fv(uProj, 1, false, projMat, 0)
        GLES30.glUniformMatrix4fv(uView, 1, false, viewMat, 0)
        GLES30.glUniform2f(uFocal, fx, fy)
        GLES30.glUniform2f(uViewport, width.toFloat(), height.toFloat())
        GLES30.glUniform1f(uScale, splatScale)
        GLES30.glUniform1i(uRow, rowSplats)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, texture)
        GLES30.glUniform1i(uData, 0)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1)
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, colorTexture)
        GLES30.glUniform1i(uColor, 1)
        GLES30.glUniform1i(uUseColor, if (useColorTex) 1 else 0)
        GLES30.glUniform1i(uColorRow, colorRow)
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0)
        GLES30.glBindVertexArray(vao)
        GLES30.glDrawArraysInstanced(GLES30.GL_TRIANGLE_STRIP, 0, 4, drawCount)
        GLES30.glBindVertexArray(0)
        GLES30.glDisable(GLES30.GL_BLEND)
        GLES30.glDepthMask(true)

        frames++
        if (now - fpsTime >= 1000) {
            fps = (frames * 1000 / maxOf(now - fpsTime, 1)).toInt()
            frames = 0
            fpsTime = now
        }
        if (!firstFrameDone) {
            firstFrameDone = true
            onFirstFrame?.invoke()
        }
    }

    private fun viewChanged(): Boolean {
        // compare the view direction row (z axis) and translation
        var d = 0f
        for (i in intArrayOf(2, 6, 10)) d += abs(viewMat[i] - lastSorted[i])
        val t = abs(viewMat[14] - lastSorted[14])
        return d > 0.01f || t > 0.01f * (1f + abs(lastSorted[14]))
    }

    companion object {
        private const val VERTEX = """#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp sampler2D;

uniform usampler2D uData;
uniform sampler2D uColor;
uniform int uUseColor;
uniform int uColorRow;
uniform mat4 uProj;
uniform mat4 uView;
uniform vec2 uFocal;
uniform vec2 uViewport;
uniform float uSplatScale;
uniform int uRowSplats;

layout(location = 0) in vec2 aQuad;
layout(location = 1) in uint aIndex;

out vec4 vColor;
out vec2 vPos;

void main() {
    int idx = int(aIndex);
    ivec2 tc = ivec2((idx % uRowSplats) * 2, idx / uRowSplats);
    uvec4 t0 = texelFetch(uData, tc, 0);
    uvec4 t1 = texelFetch(uData, tc + ivec2(1, 0), 0);
    vec3 center = uintBitsToFloat(t0.xyz);
    vec4 cam = uView * vec4(center, 1.0);
    vec4 clip = uProj * cam;
    // generous bound: large splats centred outside the screen can still cover it
    float bound = 3.0 * clip.w;
    if (cam.z > -0.0001 || clip.x < -bound || clip.x > bound || clip.y < -bound || clip.y > bound) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec2 a = unpackHalf2x16(t1.x);
    vec2 b = unpackHalf2x16(t1.y);
    vec2 c = unpackHalf2x16(t1.z);
    float s = uintBitsToFloat(t1.w) * uSplatScale * uSplatScale;
    mat3 V = mat3(a.x, a.y, b.x,
                  a.y, b.y, c.x,
                  b.x, c.x, c.y) * s;

    // Jacobian of the perspective projection (pixels, y up), d = distance along the view axis.
    // Like the trainer, x/d and y/d are clamped to 1.3x the half FOV so splats far
    // outside the screen do not smear across it.
    float d = -cam.z;
    vec2 lim = 1.3 * 0.5 * uViewport / uFocal;
    vec2 txy = clamp(cam.xy / d, -lim, lim) * d;
    vec3 j0 = vec3(uFocal.x / d, 0.0, uFocal.x * txy.x / (d * d));
    vec3 j1 = vec3(0.0, uFocal.y / d, uFocal.y * txy.y / (d * d));
    mat3 W = mat3(uView);
    vec3 r0 = transpose(W) * j0;  // rows of T = J W
    vec3 r1 = transpose(W) * j1;
    float ca = dot(r0, V * r0) + 0.3;
    float cb = dot(r0, V * r1);
    float cc = dot(r1, V * r1) + 0.3;

    float mid = 0.5 * (ca + cc);
    float rad = length(vec2(0.5 * (ca - cc), cb));
    float l1 = mid + rad;
    float l2 = mid - rad;
    if (l2 <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }
    vec2 dir = abs(cb) < 1e-12 ? (ca >= cc ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(cb, l1 - ca));
    vec2 major = min(sqrt(2.0 * l1), 2048.0) * dir;
    vec2 minor = min(sqrt(2.0 * l2), 2048.0) * vec2(dir.y, -dir.x);

    uint col = t0.w;
    vColor = vec4(float(col & 255u), float((col >> 8) & 255u), float((col >> 16) & 255u), float(col >> 24)) / 255.0;
    if (uUseColor == 1) {
        vColor.rgb = texelFetch(uColor, ivec2(idx % uColorRow, idx / uColorRow), 0).rgb;
    }
    vPos = aQuad;
    vec2 ndc = clip.xy / clip.w;
    vec2 offset = (aQuad.x * major + aQuad.y * minor) * 2.0 / uViewport;
    gl_Position = vec4(ndc + offset, 0.0, 1.0);
}
"""

        private const val FRAGMENT = """#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vPos;
out vec4 fragColor;
void main() {
    float r2 = dot(vPos, vPos);
    if (r2 > 4.0) discard;
    float alpha = exp(-r2) * vColor.a;
    if (alpha < 1.0 / 255.0) discard;
    fragColor = vec4(vColor.rgb * alpha, alpha);
}
"""
    }
}
