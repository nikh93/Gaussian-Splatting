package io.github.nikh93.gsplat

import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import io.github.nikh93.gsplat.capture.CaptureActivity
import io.github.nikh93.gsplat.data.Capture
import io.github.nikh93.gsplat.data.CaptureRepository
import io.github.nikh93.gsplat.databinding.ActivityMainBinding
import io.github.nikh93.gsplat.databinding.ItemCaptureBinding
import io.github.nikh93.gsplat.util.LogExport
import io.github.nikh93.gsplat.viewer.ViewerActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var settings: AppSettings
    private lateinit var repo: CaptureRepository
    private val adapter = CaptureAdapter { open(it) }

    private val openDocument = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            startActivity(
                Intent(this, ViewerActivity::class.java).setData(uri).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            )
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        settings = AppSettings(this)
        repo = CaptureRepository(this)

        binding.hostInput.setText(settings.host)
        binding.portInput.setText(settings.port.toString())
        binding.tokenInput.setText(settings.token)
        binding.iterationsInput.setText(settings.iterations.toString())
        binding.maxDimInput.setText(settings.maxDim.toString())

        binding.captureList.layoutManager = LinearLayoutManager(this)
        binding.captureList.adapter = adapter

        binding.testButton.setOnClickListener { testConnection() }
        binding.serverJobsButton.setOnClickListener {
            saveSettings()
            if (!settings.isServerConfigured) toast(getString(R.string.enter_host))
            else startActivity(Intent(this, ServerJobsActivity::class.java))
        }
        binding.newCaptureButton.setOnClickListener {
            saveSettings()
            startActivity(Intent(this, CaptureActivity::class.java))
        }
        binding.openFileButton.setOnClickListener { openDocument.launch(arrayOf("*/*")) }
    }

    override fun onResume() {
        super.onResume()
        refresh()
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onPrepareOptionsMenu(menu: Menu): Boolean {
        menu.findItem(R.id.action_high_res)?.isChecked = settings.highResCamera
        return super.onPrepareOptionsMenu(menu)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean = when (item.itemId) {
        R.id.action_export_logs -> {
            exportLogs()
            true
        }
        R.id.action_high_res -> {
            settings.highResCamera = !settings.highResCamera
            item.isChecked = settings.highResCamera
            true
        }
        else -> super.onOptionsItemSelected(item)
    }

    private fun exportLogs() {
        lifecycleScope.launch {
            try {
                LogExport.share(this@MainActivity)
            } catch (e: Exception) {
                toast(getString(R.string.export_logs_failed, e.message ?: e.javaClass.simpleName))
            }
        }
    }

    override fun onPause() {
        super.onPause()
        saveSettings()
    }

    private fun saveSettings() {
        settings.host = binding.hostInput.text?.toString().orEmpty()
        settings.port = binding.portInput.text?.toString()?.toIntOrNull() ?: 8080
        settings.token = binding.tokenInput.text?.toString().orEmpty()
        settings.iterations = (binding.iterationsInput.text?.toString()?.toIntOrNull() ?: 30000).coerceIn(100, 200000)
        settings.maxDim = (binding.maxDimInput.text?.toString()?.toIntOrNull() ?: 1600).coerceIn(128, 4096)
    }

    private fun refresh() {
        lifecycleScope.launch {
            val list = withContext(Dispatchers.IO) { repo.list() }
            adapter.submit(list)
            binding.emptyText.visibility = if (list.isEmpty()) View.VISIBLE else View.GONE
        }
    }

    private fun testConnection() {
        saveSettings()
        if (!settings.isServerConfigured) {
            toast(getString(R.string.enter_host))
            return
        }
        binding.connectionStatus.text = "…"
        lifecycleScope.launch {
            binding.connectionStatus.text = try {
                val info = settings.client().info()
                getString(R.string.connection_ok, "${info.device} (${info.backend}, ${info.gpuMemoryMb} MB)")
            } catch (e: Exception) {
                getString(R.string.connection_failed, e.message ?: e.javaClass.simpleName)
            }
        }
    }

    private fun open(c: Capture) {
        startActivity(Intent(this, ProjectActivity::class.java).putExtra(ProjectActivity.EXTRA_ID, c.id))
    }

    private fun toast(msg: String) = Toast.makeText(this, msg, Toast.LENGTH_LONG).show()

    // ------------------------------------------------------------------

    private class CaptureAdapter(private val onClick: (Capture) -> Unit) : RecyclerView.Adapter<CaptureAdapter.VH>() {
        private var items: List<Capture> = emptyList()

        class VH(val b: ItemCaptureBinding) : RecyclerView.ViewHolder(b.root)

        fun submit(list: List<Capture>) {
            items = list
            @Suppress("NotifyDataSetChanged")
            notifyDataSetChanged()
        }

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            VH(ItemCaptureBinding.inflate(LayoutInflater.from(parent.context), parent, false))

        override fun getItemCount() = items.size

        override fun onBindViewHolder(holder: VH, position: Int) {
            val c = items[position]
            val ctx = holder.itemView.context
            holder.b.title.text = c.name
            holder.b.subtitle.text = ctx.getString(R.string.frames_count, c.frames, c.points, formatSize(c.sizeBytes()))
            holder.b.state.text = when {
                c.hasModel -> ctx.getString(R.string.model_ready, formatSize(c.modelFile.length()))
                c.lastStatus != null -> c.lastStatus
                else -> ctx.getString(R.string.state_local)
            }
            holder.b.thumb.setImageBitmap(null)
            c.firstImage()?.let { f ->
                val opts = BitmapFactory.Options().apply { inSampleSize = 8 }
                holder.b.thumb.setImageBitmap(BitmapFactory.decodeFile(f.absolutePath, opts))
            }
            holder.itemView.setOnClickListener { onClick(c) }
        }
    }

    companion object {
        fun formatSize(bytes: Long): String = when {
            bytes >= 1L shl 30 -> String.format("%.1f GB", bytes / (1024.0 * 1024 * 1024))
            bytes >= 1L shl 20 -> String.format("%.1f MB", bytes / (1024.0 * 1024))
            else -> String.format("%d KB", bytes / 1024)
        }
    }
}
