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

    var useDepth: Boolean
        get() = prefs.getBoolean("use_depth", true)
        set(v) = prefs.edit().putBoolean("use_depth", v).apply()

    val isServerConfigured: Boolean get() = host.isNotBlank()

    fun client(): GsplatClient = GsplatClient(GsplatClient.baseUrl(host, port), token.ifBlank { null })
}
