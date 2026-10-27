package io.github.nikh93.gsplat.capture

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import io.github.nikh93.gsplat.AppSettings
import io.github.nikh93.gsplat.R
import io.github.nikh93.gsplat.databinding.ActivityCaptureSetupBinding

/**
 * Asks for all options of a capture before the photo mode starts, so the photo mode
 * itself shows nothing but the camera image, the shutter and "Fertig".
 */
class CaptureSetupActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCaptureSetupBinding
    private lateinit var settings: AppSettings

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCaptureSetupBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setTitle(R.string.new_capture)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        settings = AppSettings(this)

        binding.triggerGroup.check(if (settings.autoCapture) R.id.triggerAuto else R.id.triggerManual)
        binding.resolutionGroup.check(
            when (settings.photoResolution) {
                AppSettings.RES_STANDARD -> R.id.resStandard
                AppSettings.RES_MEDIUM -> R.id.resMedium
                else -> R.id.resMax
            }
        )
        binding.photosSwitch.isChecked = settings.showPhotos
        binding.coverageSwitch.isChecked = settings.showCoverage
        binding.depthSwitch.isChecked = settings.useDepth

        binding.startButton.setOnClickListener {
            save()
            startActivity(Intent(this, CaptureActivity::class.java))
            finish()
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    override fun onPause() {
        super.onPause()
        save()
    }

    private fun save() {
        settings.autoCapture = binding.triggerGroup.checkedRadioButtonId == R.id.triggerAuto
        settings.photoResolution = when (binding.resolutionGroup.checkedRadioButtonId) {
            R.id.resStandard -> AppSettings.RES_STANDARD
            R.id.resMedium -> AppSettings.RES_MEDIUM
            else -> AppSettings.RES_MAX
        }
        settings.showPhotos = binding.photosSwitch.isChecked
        settings.showCoverage = binding.coverageSwitch.isChecked
        settings.useDepth = binding.depthSwitch.isChecked
    }
}
