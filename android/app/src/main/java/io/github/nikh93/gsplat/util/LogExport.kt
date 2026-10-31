package io.github.nikh93.gsplat.util

import android.app.Activity
import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.content.FileProvider
import io.github.nikh93.gsplat.R
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Saves the app's logcat (incl. ARCore, which logs from our process) and opens the share sheet. */
object LogExport {

    suspend fun share(activity: Activity) {
        val file = withContext(Dispatchers.IO) { write(activity) }
        val uri = FileProvider.getUriForFile(activity, "${activity.packageName}.files", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, file.name)
            clipData = ClipData.newRawUri(file.name, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        activity.startActivity(Intent.createChooser(send, activity.getString(R.string.export_logs)))
    }

    private fun write(context: Context): File {
        val dir = File(context.cacheDir, "logs").apply { mkdirs() }
        dir.listFiles()?.forEach { it.delete() }
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val file = File(dir, "gsplat-log-$stamp.txt")
        file.bufferedWriter().use { out ->
            out.write(header(context))
            out.write("\n")
            out.flush()
            // An app may only read its own entries; this covers earlier runs that are still in the buffer.
            val process = ProcessBuilder("logcat", "-d", "-v", "threadtime", "-b", "main,system,crash")
                .redirectErrorStream(true)
                .start()
            process.inputStream.bufferedReader().use { it.copyTo(out) }
            process.waitFor()
        }
        return file
    }

    private fun header(context: Context): String {
        val pm = context.packageManager
        val app = pm.getPackageInfo(context.packageName, 0)
        val arcore = try {
            pm.getPackageInfo("com.google.ar.core", 0).versionName
        } catch (e: Exception) {
            "nicht installiert"
        }
        return buildString {
            appendLine("Gaussian Splatting ${app.versionName}")
            appendLine("Exportiert: ${Date()}")
            appendLine("Gerät: ${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE})")
            appendLine("Android: ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
            appendLine("ARCore: $arcore")
        }
    }
}
