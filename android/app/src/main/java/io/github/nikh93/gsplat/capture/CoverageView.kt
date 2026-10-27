package io.github.nikh93.gsplat.capture

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

/**
 * Top view of the coverage dome: the outer ring holds the low photos, the middle ring the
 * photos from above, the centre the photos from high above. Covered cells are green, the
 * dot is where the phone is now. The first photo is at the bottom.
 */
class CoverageView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    private var snapshot: CoverageTracker.Snapshot? = null

    private val density = resources.displayMetrics.density
    private val filled = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xCC4CAF50.toInt(); style = Paint.Style.STROKE }
    private val empty = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0x55FFFFFF; style = Paint.Style.STROKE }
    private val dot = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFFFD740.toInt(); style = Paint.Style.FILL }
    private val dotOutline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF000000.toInt(); style = Paint.Style.STROKE; strokeWidth = 1.5f * density
    }
    private val oval = RectF()

    fun update(s: CoverageTracker.Snapshot) {
        snapshot = s
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val s = snapshot ?: return
        val cx = width / 2f
        val cy = height / 2f
        val radius = min(width, height) / 2f - 2 * density
        val ringWidth = radius / s.cells.size
        val gap = 2f // degrees between cells
        for (ring in s.cells.indices) {
            // ring 0 (low) outside, last ring (top) inside
            val r = radius - ringWidth * (ring + 0.5f)
            val sectors = s.cells[ring].size
            val sweep = 360f / sectors
            oval.set(cx - r, cy - r, cx + r, cy + r)
            for (k in 0 until sectors) {
                val p = if (s.cells[ring][k]) filled else empty
                p.strokeWidth = ringWidth * 0.8f
                // sector 0 is centred at the bottom (90 degrees in canvas coordinates)
                canvas.drawArc(oval, 90f - sweep / 2 + k * sweep + gap / 2, sweep - gap, false, p)
            }
        }
        val az = s.currentAzimuth ?: return
        val el = s.currentElevationDeg ?: return
        val rr = radiusFor(el, radius, ringWidth)
        val x = cx + rr * cos(Math.PI / 2 + az).toFloat()
        val y = cy + rr * sin(Math.PI / 2 + az).toFloat()
        canvas.drawCircle(x, y, 5 * density, dot)
        canvas.drawCircle(x, y, 5 * density, dotOutline)
    }

    /** Elevation to radius, matching the ring bands (low ring from -30 degrees, top ring up to 90). */
    private fun radiusFor(el: Float, radius: Float, ringWidth: Float): Float {
        val limits = CoverageTracker.RING_LIMITS
        val bands = floatArrayOf(-30f, limits[0], limits[1], 90f)
        val e = el.coerceIn(bands[0], bands[3])
        for (ring in 0 until 3) {
            if (e <= bands[ring + 1]) {
                val t = (e - bands[ring]) / (bands[ring + 1] - bands[ring])
                return radius - ringWidth * (ring + t)
            }
        }
        return 0f
    }
}
