package io.github.nikh93.gsplat.capture

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * Round camera shutter: a white disc inside a white ring. While the button is disabled
 * (no stable tracking) it is dimmed. The ring doubles as a progress bar: the green arc
 * is the share of viewing directions around the object that already have a photo.
 */
class ShutterButton @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private val density = resources.displayMetrics.density
    private val disc = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFFFFFFF.toInt() }
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFFFFFF.toInt(); style = Paint.Style.STROKE; strokeWidth = 4 * density
    }
    private val progressPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF4CAF50.toInt(); style = Paint.Style.STROKE; strokeWidth = 4 * density
        strokeCap = Paint.Cap.ROUND
    }
    private val oval = RectF()

    /** 0..1, drawn as green arc on the ring */
    var progress = 0f
        set(v) {
            if (field != v) {
                field = v
                invalidate()
            }
        }

    init {
        isClickable = true
        isFocusable = true
    }

    override fun setEnabled(enabled: Boolean) {
        super.setEnabled(enabled)
        alpha = if (enabled) 1f else 0.4f
    }

    override fun setPressed(pressed: Boolean) {
        super.setPressed(pressed)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val cx = width / 2f
        val cy = height / 2f
        val r = min(width, height) / 2f - ring.strokeWidth / 2
        canvas.drawCircle(cx, cy, r, ring)
        if (progress > 0f) {
            oval.set(cx - r, cy - r, cx + r, cy + r)
            canvas.drawArc(oval, -90f, 360f * progress.coerceAtMost(1f), false, progressPaint)
        }
        val inner = r - ring.strokeWidth / 2 - 4 * density
        canvas.drawCircle(cx, cy, if (isPressed) inner * 0.9f else inner, disc)
    }
}
