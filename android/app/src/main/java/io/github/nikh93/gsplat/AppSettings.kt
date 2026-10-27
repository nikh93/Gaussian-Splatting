package io.github.nikh93.gsplat

import android.content.Context
import io.github.nikh93.gsplat.net.GsplatClient

/** Persistent user settings (server address, training parameters). */
class AppSettings(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var host: String
        get() = prefs.getString("host", "") ?: ""
        set(v) = prefs.edit().putString("host", v.trim()).apply()

    var port: Int
        get() = prefs.getInt("port", 8080)
        set(v) = prefs.edit().putInt("port", v).apply()

    var token: String
        get() = prefs.getString("token", "") ?: ""
        set(v) = prefs.edit().putString("token", v.trim()).apply()

    var iterations: Int
        get() = prefs.getInt("iterations", 30000)
        set(v) = prefs.edit().putInt("iterations", v).apply()

    var maxDim: Int
        get() = prefs.getInt("max_dim", 1600)
        set(v) = prefs.edit().putInt("max_dim", v).apply()

    /** automatic capture is opt-in (new key: earlier versions defaulted to on) */
    var autoCapture: Boolean
        get() = prefs.getBoolean("auto_capture_v2", false)
        set(v) = prefs.edit().putBoolean("auto_capture_v2", v).apply()

    /** depth is opt-in (new key): ARCore's depth fails on some phones and slows tracking down */
    var useDepth: Boolean
        get() = prefs.getBoolean("use_depth_v2", false)
        set(v) = prefs.edit().putBoolean("use_depth_v2", v).apply()

    /**
     * Photo resolution: [RES_STANDARD] is ARCore's default camera config (640x480 on most
     * phones, too coarse for splatting), [RES_MEDIUM] the smallest config of at least 720p,
     * [RES_MAX] the largest. The tracking problems on the Galaxy A56 turned out to be
     * independent of the resolution, so the largest is the default again.
     */
    var photoResolution: Int
        get() = prefs.getInt("photo_resolution", RES_MAX)
        set(v) = prefs.edit().putInt("photo_resolution", v).apply()

    val isServerConfigured: Boolean get() = host.isNotBlank()

    companion object {
        const val RES_STANDARD = 0
        const val RES_MEDIUM = 1
        const val RES_MAX = 2
    }

    fun client(): GsplatClient = GsplatClient(GsplatClient.baseUrl(host, port), token.ifBlank { null })
}
