package io.github.nikh93.gsplat.capture

import android.app.Activity
import android.hardware.display.DisplayManager
import android.os.Build
import android.view.Display
import android.view.WindowManager
import com.google.ar.core.Session

/** Tracks display rotation / size changes and forwards them to the ARCore session. */
class DisplayRotationHelper(private val activity: Activity) : DisplayManager.DisplayListener {
    private var viewportChanged = false
    private var viewportWidth = 0
    private var viewportHeight = 0
    private val displayManager = activity.getSystemService(DisplayManager::class.java)

    private val display: Display
        get() = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            activity.display!!
        } else {
            @Suppress("DEPRECATION")
            (activity.getSystemService(Activity.WINDOW_SERVICE) as WindowManager).defaultDisplay
        }

    fun onResume() = displayManager.registerDisplayListener(this, null)

    fun onPause() = displayManager.unregisterDisplayListener(this)

    fun onSurfaceChanged(width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        viewportChanged = true
    }

    fun updateSessionIfNeeded(session: Session) {
        if (viewportChanged) {
            session.setDisplayGeometry(display.rotation, viewportWidth, viewportHeight)
            viewportChanged = false
        }
    }

    override fun onDisplayAdded(displayId: Int) {}
    override fun onDisplayRemoved(displayId: Int) {}
    override fun onDisplayChanged(displayId: Int) {
        viewportChanged = true
    }
}
