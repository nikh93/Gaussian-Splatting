package io.github.nikh93.gsplat.data

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * A capture on the device:
 *
 *   files/captures/<id>/
 *       meta.json          name, counts, server job id + last known state
 *       transforms.json    camera poses + intrinsics (nerfstudio format)
 *       points3d.ply       initial point cloud for training
 *       images/            photos (frame_00000.jpg, ...)
 *       model.ply          downloaded result (optional)
 */
data class Capture(
    val id: String,
    val dir: File,
    var name: String,
    var created: Long,
    var frames: Int,
    var points: Int,
    var jobId: String?,
    var serverState: String?,
    var lastStatus: String?,
) {
    val transforms: File get() = File(dir, "transforms.json")
    val pointsFile: File get() = File(dir, "points3d.ply")
    val imagesDir: File get() = File(dir, "images")
    val modelFile: File get() = File(dir, "model.ply")
    val hasModel: Boolean get() = modelFile.isFile && modelFile.length() > 0

    fun firstImage(): File? = imagesDir.listFiles()?.filter { it.extension == "jpg" }?.minByOrNull { it.name }

    fun sizeBytes(): Long = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }

    fun toJson(): JSONObject = JSONObject().apply {
        put("name", name)
        put("created", created)
        put("frames", frames)
        put("points", points)
        put("job_id", jobId ?: JSONObject.NULL)
        put("server_state", serverState ?: JSONObject.NULL)
        put("last_status", lastStatus ?: JSONObject.NULL)
    }
}

class CaptureRepository(context: Context) {
    private val root = File(context.filesDir, "captures").apply { mkdirs() }
    val modelsDir = File(context.filesDir, "models").apply { mkdirs() }

    fun list(): List<Capture> =
        root.listFiles()?.filter { it.isDirectory }?.mapNotNull { load(it.name) }?.sortedByDescending { it.created }
            ?: emptyList()

    fun load(id: String): Capture? {
        val dir = File(root, id)
        val meta = File(dir, "meta.json")
        if (!dir.isDirectory) return null
        return try {
            val j = if (meta.isFile) JSONObject(meta.readText()) else JSONObject()
            Capture(
                id = id,
                dir = dir,
                name = j.optString("name", id),
                created = j.optLong("created", dir.lastModified()),
                frames = j.optInt("frames", 0),
                points = j.optInt("points", 0),
                jobId = j.optStringOrNull("job_id"),
                serverState = j.optStringOrNull("server_state"),
                lastStatus = j.optStringOrNull("last_status"),
            )
        } catch (e: Exception) {
            null
        }
    }

    fun create(): Capture {
        val now = System.currentTimeMillis()
        val id = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date(now))
        val dir = File(root, id)
        File(dir, "images").mkdirs()
        val c = Capture(
            id = id, dir = dir,
            name = "Aufnahme " + SimpleDateFormat("dd.MM.yyyy HH:mm", Locale.GERMANY).format(Date(now)),
            created = now, frames = 0, points = 0, jobId = null, serverState = null, lastStatus = null,
        )
        save(c)
        return c
    }

    fun save(c: Capture) {
        val tmp = File(c.dir, "meta.json.tmp")
        tmp.writeText(c.toJson().toString(2))
        tmp.renameTo(File(c.dir, "meta.json"))
    }

    fun delete(c: Capture) {
        c.dir.deleteRecursively()
    }

    private fun JSONObject.optStringOrNull(key: String): String? =
        if (isNull(key) || !has(key)) null else optString(key)
}
