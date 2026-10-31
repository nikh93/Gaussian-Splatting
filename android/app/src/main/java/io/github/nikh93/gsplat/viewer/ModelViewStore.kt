package io.github.nikh93.gsplat.viewer

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Viewer and AR parameters of one model, kept until "Ansicht zurücksetzen".
 * Values that are null / negative were never changed and use the defaults.
 */
data class ModelViewState(
    // orbit viewer
    var target: FloatArray? = null,
    var distance: Float = -1f,
    var yaw: Float = 0f,
    var pitch: Float = 0f,
    var upAxis: Int = 0,
    var fovDeg: Float = DEFAULT_FOV,
    var splatScale: Float = 1f,
    var backgroundIndex: Int = 0,
    var autoRotate: Boolean = false,
    // AR view
    var arScale: Float = 1f,
    var arHeight: Float = -1f,      // height of the model centre above the floor (m)
    var arYawOffset: Float = 0f,    // turn relative to "facing the user" at placement (deg)
) {
    fun toJson(): JSONObject = JSONObject().apply {
        target?.let { put("target", JSONArray(it.map { v -> v.toDouble() })) }
        put("distance", distance.toDouble())
        put("yaw", yaw.toDouble())
        put("pitch", pitch.toDouble())
        put("upAxis", upAxis)
        put("fov", fovDeg.toDouble())
        put("splatScale", splatScale.toDouble())
        put("background", backgroundIndex)
        put("autoRotate", autoRotate)
        put("arScale", arScale.toDouble())
        put("arHeight", arHeight.toDouble())
        put("arYawOffset", arYawOffset.toDouble())
    }

    companion object {
        const val DEFAULT_FOV = 60f

        fun fromJson(j: JSONObject) = ModelViewState(
            target = j.optJSONArray("target")?.let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } },
            distance = j.optDouble("distance", -1.0).toFloat(),
            yaw = j.optDouble("yaw", 0.0).toFloat(),
            pitch = j.optDouble("pitch", 0.0).toFloat(),
            upAxis = j.optInt("upAxis", 0),
            fovDeg = j.optDouble("fov", DEFAULT_FOV.toDouble()).toFloat(),
            splatScale = j.optDouble("splatScale", 1.0).toFloat(),
            backgroundIndex = j.optInt("background", 0),
            autoRotate = j.optBoolean("autoRotate", false),
            arScale = j.optDouble("arScale", 1.0).toFloat(),
            arHeight = j.optDouble("arHeight", -1.0).toFloat(),
            arYawOffset = j.optDouble("arYawOffset", 0.0).toFloat(),
        )
    }
}

/** Persists a [ModelViewState] per model file (path + size, so a replaced model starts fresh). */
class ModelViewStore(context: Context) {
    private val prefs = context.getSharedPreferences("model_view_state", Context.MODE_PRIVATE)

    private fun key(file: File) = file.absolutePath + "|" + file.length()

    fun load(file: File): ModelViewState? =
        prefs.getString(key(file), null)?.let {
            try {
                ModelViewState.fromJson(JSONObject(it))
            } catch (e: Exception) {
                null
            }
        }

    fun save(file: File, state: ModelViewState) {
        prefs.edit().putString(key(file), state.toJson().toString()).apply()
    }

    /** Applies [change] to the stored state (or the defaults) and saves it. */
    fun update(file: File, change: (ModelViewState) -> Unit) {
        val s = load(file) ?: ModelViewState()
        change(s)
        save(file, s)
    }

    fun clear(file: File) {
        prefs.edit().remove(key(file)).apply()
    }
}
