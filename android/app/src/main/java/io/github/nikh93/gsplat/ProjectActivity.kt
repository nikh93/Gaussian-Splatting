package io.github.nikh93.gsplat

import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import io.github.nikh93.gsplat.data.Capture
import io.github.nikh93.gsplat.data.CaptureRepository
import io.github.nikh93.gsplat.databinding.ActivityProjectBinding
import io.github.nikh93.gsplat.net.HttpException
import io.github.nikh93.gsplat.net.JobStatus
import io.github.nikh93.gsplat.net.Transfer
import io.github.nikh93.gsplat.net.TransferManager
import io.github.nikh93.gsplat.viewer.ViewerActivity
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** Details of one capture: upload + training on the desktop, download and viewing of the model. */
class ProjectActivity : AppCompatActivity() {

    private lateinit var binding: ActivityProjectBinding
    private lateinit var repo: CaptureRepository
    private lateinit var settings: AppSettings
    private lateinit var capture: Capture
    private var status: JobStatus? = null
    private var jobMissing = false

    private val downloadKey get() = "download-" + capture.id

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityProjectBinding.inflate(layoutInflater)
        setContentView(binding.root)
        repo = CaptureRepository(this)
        settings = AppSettings(this)
        val id = intent.getStringExtra(EXTRA_ID)
        val c = id?.let { repo.load(it) }
        if (c == null) {
            finish()
            return
        }
        capture = c
        title = capture.name

        capture.firstImage()?.let { f ->
            val opts = BitmapFactory.Options().apply { inSampleSize = 4 }
            binding.preview.setImageBitmap(BitmapFactory.decodeFile(f.absolutePath, opts))
        }

        binding.uploadButton.setOnClickListener { uploadOrRestart() }
        binding.cancelButton.setOnClickListener { cancelTraining() }
        binding.downloadButton.setOnClickListener { downloadModel() }
        binding.viewButton.setOnClickListener { view() }
        binding.shareButton.setOnClickListener { share() }
        binding.renameButton.setOnClickListener { rename() }
        binding.deleteButton.setOnClickListener { delete() }

        // transfer progress
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                TransferManager.state.collect { updateUi() }
            }
        }
        // poll the server while the screen is visible
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                while (true) {
                    pollStatus()
                    delay(if (status?.isActive == true) 2000 else 6000)
                }
            }
        }
    }

    private suspend fun pollStatus() {
        val jobId = capture.jobId ?: run { updateUi(); return }
        if (!settings.isServerConfigured || TransferManager.stateOf(capture.id) is Transfer.Uploading) {
            updateUi()
            return
        }
        try {
            val st = settings.client().status(jobId)
            status = st
            jobMissing = false
            capture.serverState = st.state
            capture.lastStatus = describe(st)
            repo.save(capture)
        } catch (e: HttpException) {
            if (e.code == 404) {
                jobMissing = true
                status = null
            }
        } catch (e: Exception) {
            // server unreachable: keep the last known state
        }
        updateUi()
    }

    private fun describe(st: JobStatus): String = when (st.state) {
        "uploading" -> getString(R.string.state_uploading, 0)
        "queued" -> getString(R.string.state_queued)
        "training" -> getString(R.string.state_training, st.iteration, st.totalIterations, st.psnr, st.gaussians)
        "done" -> getString(R.string.state_done, st.psnr, st.gaussians)
        "failed" -> getString(R.string.state_failed, st.error)
        "cancelled" -> getString(R.string.state_cancelled)
        else -> getString(R.string.state_unknown, st.state)
    }

    private fun updateUi() {
        val st = status
        val transfer = TransferManager.stateOf(capture.id)
        val download = TransferManager.stateOf(downloadKey)
        binding.nameText.text = capture.name
        binding.infoText.text = getString(
            R.string.frames_count, capture.frames, capture.points, MainActivity.formatSize(capture.sizeBytes())
        )

        var progress: Float? = null
        val text = when {
            transfer is Transfer.Uploading -> {
                progress = transfer.fraction
                getString(R.string.state_uploading, (transfer.fraction * 100).toInt())
            }
            transfer is Transfer.Failed -> getString(R.string.error_generic, transfer.message)
            download is Transfer.Downloading -> {
                progress = download.fraction
                getString(R.string.downloading, (download.fraction * 100).toInt())
            }
            download is Transfer.Failed -> getString(R.string.error_generic, download.message)
            jobMissing -> getString(R.string.job_missing)
            st != null -> {
                if (st.state == "training") progress = st.progress.toFloat()
                describe(st)
            }
            capture.lastStatus != null -> capture.lastStatus!!
            else -> getString(R.string.state_local)
        }
        binding.stateText.text = if (capture.hasModel)
            text + "\n" + getString(R.string.model_ready, MainActivity.formatSize(capture.modelFile.length()))
        else text

        binding.progress.visibility = if (progress != null) View.VISIBLE else View.GONE
        progress?.let { binding.progress.setProgressCompat((it * 1000).toInt(), true) }

        val busy = TransferManager.isRunning(capture.id)
        val active = st?.isActive == true
        binding.uploadButton.isEnabled = !busy && !active && capture.frames > 0
        binding.uploadButton.text = getString(
            if (capture.jobId != null && !jobMissing && st != null) R.string.restart_training else R.string.upload_and_train
        )
        binding.cancelButton.visibility = if (active) View.VISIBLE else View.GONE
        binding.downloadButton.isEnabled = st?.modelAvailable == true && !TransferManager.isRunning(downloadKey)
        binding.downloadButton.text =
            getString(if (st?.state == "training") R.string.download_snapshot else R.string.download_model)
        binding.viewButton.isEnabled = capture.hasModel || capture.pointsFile.isFile
        binding.shareButton.isEnabled = capture.hasModel

        if (busy || active) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    private fun uploadOrRestart() {
        if (!settings.isServerConfigured) {
            toast(getString(R.string.enter_host))
            return
        }
        TransferManager.clear(capture.id)
        val st = status
        if (capture.jobId != null && st != null && !jobMissing && st.state != "uploading") {
            TransferManager.restart(capture, repo, settings)
        } else {
            jobMissing = false
            TransferManager.upload(capture, repo, settings)
        }
        updateUi()
    }

    private fun cancelTraining() {
        val id = capture.jobId ?: return
        lifecycleScope.launch {
            try {
                status = settings.client().cancel(id)
            } catch (e: Exception) {
                toast(getString(R.string.error_generic, e.message))
            }
            updateUi()
        }
    }

    private fun downloadModel() {
        val id = capture.jobId ?: return
        TransferManager.clear(downloadKey)
        TransferManager.download(downloadKey, id, capture.modelFile, settings) {
            runOnUiThread {
                updateUi()
                view()
            }
        }
        updateUi()
    }

    private fun view() {
        val file = if (capture.hasModel) capture.modelFile else capture.pointsFile
        if (!file.isFile) return
        startActivity(Intent(this, ViewerActivity::class.java).putExtra(ViewerActivity.EXTRA_PATH, file.absolutePath))
    }

    private fun share() {
        if (!capture.hasModel) return
        val uri = FileProvider.getUriForFile(this, "$packageName.files", capture.modelFile)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, capture.name + ".ply")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(send, getString(R.string.share_model)))
    }

    private fun rename() {
        val input = EditText(this).apply { setText(capture.name) }
        AlertDialog.Builder(this)
            .setTitle(R.string.rename)
            .setView(input)
            .setPositiveButton(android.R.string.ok) { _, _ ->
                capture.name = input.text.toString().ifBlank { capture.name }
                repo.save(capture)
                title = capture.name
                updateUi()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun delete() {
        AlertDialog.Builder(this)
            .setMessage(R.string.delete_confirm)
            .setPositiveButton(R.string.delete) { _, _ ->
                repo.delete(capture)
                finish()
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    companion object {
        const val EXTRA_ID = "capture_id"
    }
}
