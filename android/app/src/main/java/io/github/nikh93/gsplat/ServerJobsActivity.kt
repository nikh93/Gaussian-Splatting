package io.github.nikh93.gsplat

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import io.github.nikh93.gsplat.data.CaptureRepository
import io.github.nikh93.gsplat.databinding.ActivityServerJobsBinding
import io.github.nikh93.gsplat.databinding.ItemCaptureBinding
import io.github.nikh93.gsplat.net.JobStatus
import io.github.nikh93.gsplat.net.Transfer
import io.github.nikh93.gsplat.net.TransferManager
import io.github.nikh93.gsplat.viewer.ViewerActivity
import kotlinx.coroutines.launch
import java.io.File

/** Lists all jobs on the desktop server; tapping a finished job downloads and shows its model. */
class ServerJobsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityServerJobsBinding
    private lateinit var settings: AppSettings
    private lateinit var repo: CaptureRepository
    private var jobs: List<JobStatus> = emptyList()
    private val adapter = JobAdapter()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityServerJobsBinding.inflate(layoutInflater)
        setContentView(binding.root)
        title = getString(R.string.server_jobs)
        settings = AppSettings(this)
        repo = CaptureRepository(this)
        binding.jobList.layoutManager = LinearLayoutManager(this)
        binding.jobList.adapter = adapter
        binding.refresh.setOnRefreshListener { load() }

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                TransferManager.state.collect { states ->
                    val active = states.entries.firstOrNull { it.key.startsWith("server-") && it.value is Transfer.Downloading }
                    val d = active?.value as? Transfer.Downloading
                    binding.downloadProgress.visibility = if (d != null) View.VISIBLE else View.GONE
                    d?.let { binding.downloadProgress.setProgressCompat((it.fraction * 1000).toInt(), true) }
                    states.entries.firstOrNull { it.key.startsWith("server-") && it.value is Transfer.Failed }?.let {
                        toast(getString(R.string.error_generic, (it.value as Transfer.Failed).message))
                        TransferManager.clear(it.key)
                    }
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        load()
    }

    private fun load() {
        binding.refresh.isRefreshing = true
        lifecycleScope.launch {
            try {
                jobs = settings.client().jobs()
            } catch (e: Exception) {
                toast(getString(R.string.connection_failed, e.message ?: e.javaClass.simpleName))
            }
            binding.refresh.isRefreshing = false
            binding.emptyText.visibility = if (jobs.isEmpty()) View.VISIBLE else View.GONE
            adapter.notifyDataSetChanged()
        }
    }

    private fun modelFile(job: JobStatus) = File(repo.modelsDir, job.id + ".ply")

    private fun openJob(job: JobStatus) {
        val f = modelFile(job)
        if (f.isFile && job.state == "done" && f.lastModified() / 1000 >= job.created) {
            show(f)
            return
        }
        if (!job.modelAvailable) {
            toast(describe(job))
            return
        }
        TransferManager.clear("server-" + job.id)
        TransferManager.download("server-" + job.id, job.id, f, settings) { runOnUiThread { show(f) } }
    }

    private fun show(f: File) {
        startActivity(Intent(this, ViewerActivity::class.java).putExtra(ViewerActivity.EXTRA_PATH, f.absolutePath))
    }

    private fun confirmDelete(job: JobStatus) {
        AlertDialog.Builder(this)
            .setTitle(job.name)
            .setMessage(R.string.delete_confirm)
            .setPositiveButton(R.string.delete) { _, _ ->
                lifecycleScope.launch {
                    try {
                        settings.client().delete(job.id)
                        modelFile(job).delete()
                    } catch (e: Exception) {
                        toast(getString(R.string.error_generic, e.message))
                    }
                    load()
                }
            }
            .setNegativeButton(R.string.cancel, null)
            .show()
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

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    private inner class JobAdapter : RecyclerView.Adapter<JobAdapter.VH>() {
        inner class VH(val b: ItemCaptureBinding) : RecyclerView.ViewHolder(b.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            VH(ItemCaptureBinding.inflate(LayoutInflater.from(parent.context), parent, false))

        override fun getItemCount() = jobs.size

        override fun onBindViewHolder(holder: VH, position: Int) {
            val job = jobs[position]
            holder.b.thumb.visibility = View.GONE
            holder.b.title.text = job.name
            holder.b.subtitle.text = job.id
            holder.b.state.text = describe(job) + if (modelFile(job).isFile) " · lokal" else ""
            holder.itemView.setOnClickListener { openJob(job) }
            holder.itemView.setOnLongClickListener {
                confirmDelete(job)
                true
            }
        }
    }
}
