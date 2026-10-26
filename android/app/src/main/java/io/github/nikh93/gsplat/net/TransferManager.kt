package io.github.nikh93.gsplat.net

import io.github.nikh93.gsplat.AppSettings
import io.github.nikh93.gsplat.data.Capture
import io.github.nikh93.gsplat.data.CaptureRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File

sealed class Transfer {
    data object Idle : Transfer()
    data class Uploading(val fraction: Float) : Transfer()
    data class Downloading(val fraction: Float) : Transfer()
    data class Failed(val message: String) : Transfer()
    data object Finished : Transfer()
}

/**
 * Runs uploads/downloads outside of any Activity so they survive screen changes.
 * State is published per key (capture id or job id).
 */
object TransferManager {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val jobs = HashMap<String, Job>()
    private val _state = MutableStateFlow<Map<String, Transfer>>(emptyMap())
    val state: StateFlow<Map<String, Transfer>> = _state.asStateFlow()

    fun stateOf(key: String): Transfer = _state.value[key] ?: Transfer.Idle

    fun isRunning(key: String): Boolean = synchronized(jobs) { jobs[key]?.isActive == true }

    private fun set(key: String, t: Transfer) = _state.update { it + (key to t) }

    private fun enqueue(key: String, block: suspend () -> Unit) {
        synchronized(jobs) {
            if (jobs[key]?.isActive == true) return
            jobs[key] = scope.launch {
                try {
                    block()
                    set(key, Transfer.Finished)
                } catch (e: Exception) {
                    set(key, Transfer.Failed(e.message ?: e.javaClass.simpleName))
                }
            }
        }
    }

    /** Creates a server job, uploads all capture files and starts training. */
    fun upload(capture: Capture, repo: CaptureRepository, settings: AppSettings) = enqueue(capture.id) {
        val client = settings.client()
        set(capture.id, Transfer.Uploading(0f))
        val files = buildList {
            add("transforms.json" to capture.transforms)
            if (capture.pointsFile.isFile) add("points3d.ply" to capture.pointsFile)
            capture.imagesDir.listFiles()?.filter { it.isFile && it.extension == "jpg" }?.sortedBy { it.name }
                ?.forEach { add("images/${it.name}" to it) }
        }
        if (!capture.transforms.isFile) throw IllegalStateException("transforms.json fehlt")
        val total = files.sumOf { it.second.length() }.coerceAtLeast(1)
        var sent = 0L
        val job = client.createJob(capture.name, settings.iterations, settings.maxDim)
        capture.jobId = job.id
        capture.serverState = "uploading"
        repo.save(capture)
        for ((rel, file) in files) {
            client.upload(job.id, rel, file) { n ->
                sent += n
                set(capture.id, Transfer.Uploading(sent.toFloat() / total))
            }
        }
        val started = client.start(job.id)
        capture.serverState = started.state
        repo.save(capture)
    }

    /** Restarts training of an already uploaded job. */
    fun restart(capture: Capture, repo: CaptureRepository, settings: AppSettings) = enqueue(capture.id) {
        val id = capture.jobId ?: throw IllegalStateException("kein Server-Job")
        val st = settings.client().start(id, settings.iterations, settings.maxDim)
        capture.serverState = st.state
        repo.save(capture)
    }

    /** Downloads model.ply of [jobId] into [dest]; [key] identifies the transfer. */
    fun download(key: String, jobId: String, dest: File, settings: AppSettings, onDone: (() -> Unit)? = null) =
        enqueue(key) {
            set(key, Transfer.Downloading(0f))
            settings.client().download(jobId, "model.ply", dest) { done, total ->
                set(key, Transfer.Downloading(if (total > 0) done.toFloat() / total else 0f))
            }
            onDone?.invoke()
        }

    fun clear(key: String) = _state.update { it - key }
}
