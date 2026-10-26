package io.github.nikh93.gsplat.net

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

data class ServerInfo(val name: String, val version: String, val device: String, val gpuMemoryMb: Long)

data class JobStatus(
    val id: String,
    val name: String,
    val state: String,
    val iteration: Int,
    val totalIterations: Int,
    val progress: Double,
    val loss: Double,
    val psnr: Double,
    val gaussians: Int,
    val modelAvailable: Boolean,
    val modelIteration: Int,
    val error: String,
    val created: Long,
) {
    val isActive: Boolean get() = state == "queued" || state == "training"

    companion object {
        fun fromJson(j: JSONObject) = JobStatus(
            id = j.getString("id"),
            name = j.optString("name"),
            state = j.optString("state"),
            iteration = j.optInt("iteration"),
            totalIterations = j.optInt("total_iterations"),
            progress = j.optDouble("progress", 0.0),
            loss = j.optDouble("loss", 0.0),
            psnr = j.optDouble("psnr", 0.0),
            gaussians = j.optInt("gaussians"),
            modelAvailable = j.optBoolean("model_available"),
            modelIteration = j.optInt("model_iteration"),
            error = j.optString("error"),
            created = j.optLong("created"),
        )
    }
}

class HttpException(val code: Int, message: String) : IOException(message)

/**
 * Client for the desktop gsplat-server REST API (see desktop/src/server.c).
 * The server is reached through Tailscale, e.g. http://100.101.102.103:8080.
 */
class GsplatClient(private val baseUrl: String, private val token: String?) {

    companion object {
        /** Builds the base URL from a host (IPv4, IPv6, MagicDNS name or full URL) and port. */
        fun baseUrl(host: String, port: Int): String {
            val h = host.trim().trimEnd('/')
            if (h.startsWith("http://") || h.startsWith("https://")) return h
            val bracketed = if (h.contains(':') && !h.startsWith("[")) "[$h]" else h
            return "http://$bracketed:$port"
        }

        private const val CONNECT_TIMEOUT = 10_000
        private const val READ_TIMEOUT = 120_000
    }

    private fun open(path: String, method: String): HttpURLConnection {
        val c = URL(baseUrl + path).openConnection() as HttpURLConnection
        c.requestMethod = method
        c.connectTimeout = CONNECT_TIMEOUT
        c.readTimeout = READ_TIMEOUT
        c.useCaches = false
        if (token != null) c.setRequestProperty("Authorization", "Bearer $token")
        return c
    }

    private fun readResponse(c: HttpURLConnection): String {
        val code = c.responseCode
        val stream = if (code in 200..299) c.inputStream else c.errorStream
        val text = stream?.bufferedReader()?.use { it.readText() } ?: ""
        if (code !in 200..299) {
            val msg = try {
                JSONObject(text).optString("error", text)
            } catch (e: Exception) {
                text.ifBlank { "HTTP $code" }
            }
            throw HttpException(code, msg)
        }
        return text
    }

    private suspend fun requestJson(path: String, method: String, body: JSONObject? = null): JSONObject =
        withContext(Dispatchers.IO) {
            val c = open(path, method)
            try {
                if (body != null) {
                    val bytes = body.toString().toByteArray()
                    c.doOutput = true
                    c.setRequestProperty("Content-Type", "application/json")
                    c.setFixedLengthStreamingMode(bytes.size)
                    c.outputStream.use { it.write(bytes) }
                }
                JSONObject(readResponse(c).ifBlank { "{}" })
            } finally {
                c.disconnect()
            }
        }

    suspend fun info(): ServerInfo {
        val j = requestJson("/api/info", "GET")
        return ServerInfo(j.optString("name"), j.optString("version"), j.optString("device"), j.optLong("gpu_memory_mb"))
    }

    suspend fun jobs(): List<JobStatus> {
        val arr = requestJson("/api/jobs", "GET").optJSONArray("jobs") ?: return emptyList()
        return (0 until arr.length()).map { JobStatus.fromJson(arr.getJSONObject(it)) }
    }

    suspend fun createJob(name: String, iterations: Int, maxDim: Int): JobStatus =
        JobStatus.fromJson(
            requestJson("/api/jobs", "POST", JSONObject().apply {
                put("name", name)
                put("iterations", iterations)
                put("max_dim", maxDim)
            })
        )

    suspend fun status(jobId: String): JobStatus = JobStatus.fromJson(requestJson("/api/jobs/$jobId", "GET"))

    suspend fun start(jobId: String, iterations: Int? = null, maxDim: Int? = null): JobStatus =
        JobStatus.fromJson(requestJson("/api/jobs/$jobId/start", "POST", JSONObject().apply {
            iterations?.let { put("iterations", it) }
            maxDim?.let { put("max_dim", it) }
        }))

    suspend fun cancel(jobId: String): JobStatus = JobStatus.fromJson(requestJson("/api/jobs/$jobId/cancel", "POST"))

    suspend fun delete(jobId: String) {
        requestJson("/api/jobs/$jobId", "DELETE")
    }

    /** Uploads one file; [relPath] like "images/frame_00001.jpg". */
    suspend fun upload(jobId: String, relPath: String, file: File, onBytes: (Long) -> Unit) =
        withContext(Dispatchers.IO) {
            val encoded = relPath.split('/').joinToString("/") { URLEncoder.encode(it, "UTF-8") }
            val c = open("/api/jobs/$jobId/files/$encoded", "PUT")
            try {
                c.doOutput = true
                c.setRequestProperty("Content-Type", "application/octet-stream")
                c.setFixedLengthStreamingMode(file.length())
                c.outputStream.use { out ->
                    file.inputStream().use { input ->
                        val buf = ByteArray(64 * 1024)
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            onBytes(n.toLong())
                        }
                    }
                }
                readResponse(c)
            } finally {
                c.disconnect()
            }
        }

    /** Downloads model.ply / model.splat to [dest] (written atomically). */
    suspend fun download(jobId: String, fileName: String, dest: File, onProgress: (Long, Long) -> Unit) =
        withContext(Dispatchers.IO) {
            val c = open("/api/jobs/$jobId/$fileName", "GET")
            val tmp = File(dest.parentFile, dest.name + ".part")
            try {
                if (c.responseCode !in 200..299) readResponse(c)
                val total = c.contentLengthLong
                var done = 0L
                c.inputStream.use { input ->
                    tmp.outputStream().use { out ->
                        val buf = ByteArray(256 * 1024)
                        while (true) {
                            coroutineContext.ensureActive()
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            done += n
                            onProgress(done, total)
                        }
                    }
                }
                if (total > 0 && done != total) throw IOException("Download unvollständig ($done / $total Bytes)")
                if (dest.exists()) dest.delete()
                if (!tmp.renameTo(dest)) throw IOException("Konnte ${dest.name} nicht speichern")
            } finally {
                tmp.delete()
                c.disconnect()
            }
        }
}
