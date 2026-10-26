package io.github.nikh93.gsplat.util

import android.view.View
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

/** Keeps overlay controls clear of the status bar, navigation bar and display cutouts. */
object Insets {
    fun pad(view: View, top: Boolean = false, bottom: Boolean = false) {
        val l = view.paddingLeft
        val t = view.paddingTop
        val r = view.paddingRight
        val b = view.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(view) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.setPadding(
                l + bars.left,
                t + if (top) bars.top else 0,
                r + bars.right,
                b + if (bottom) bars.bottom else 0,
            )
            insets
        }
    }
}
