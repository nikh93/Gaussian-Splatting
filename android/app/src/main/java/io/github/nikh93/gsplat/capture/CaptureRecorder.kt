package io.github.nikh93.gsplat.capture

import android.util.Log
import io.github.nikh93.gsplat.data.Capture
import io.github.nikh93.gsplat.data.CaptureRepository
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.floor

/** Pinhole intrinsics in pixels of a given image size. */
data class Intrinsics(val fx: Float, val fy: Float, val cx: Float, val cy: Float, val width: Int, val height: Int) {
    fun scaledTo(w: Int, h: Int): Intrinsics {
        val sx = w.toFloat() / width
        val sy = h.toFloat() / height
        return Intrinsics(fx * sx, fy * sy, cx * sx, cy * sy, w, h)
    }
}

/** Depth samples of one frame: pixel coordinates in the depth image and metric depth. */
class DepthSamples(val u: FloatArray, val v: FloatArray, val depth: FloatArray, val count: Int, val intr: Intrinsics)

/**
 * Collects photos, camera poses and 3D points of a capture and writes them in the
 * nerfstudio format the desktop server expects. All file IO and point processing
 * runs on one worker thread; public methods may be called from the GL thread.
 */
class CaptureRecorder(private val capture: Capture, private val repo: CaptureRepository) {

    private data class FrameRecord(
        val file: String,
        val c2w: FloatArray, // column major 4x4 (ARCore Pose.toMatrix), OpenGL camera convention
        val intr: Intrinsics,
        val timestampNs: Long,
        val sharpness: Float,
    )

    private val worker = Executors.newSingleThreadExecutor()
    private val frames = ArrayList<FrameRecord>()
    private val pending = AtomicInteger(0)

    /* points: voxel hash -> index into the point arrays (worker thread only) */
    private val voxel = 0.008f
    private val voxelIndex = HashMap<Long, Int>()
    private var px = FloatArray(3 * 65536)
    private var pc = IntArray(65536) // packed rgb
    private var numPoints = 0
    private val maxPoints = 600_000

    /* ARCore feature points (id -> x,y,z), coloured once seen in a photo */
    private val featurePoints = HashMap<Int, FloatArray>()
    private val featureColored = HashSet<Int>()

    @Volatile var frameCount = 0
        private set
    @Volatile var pointCount = 0
        private set

    val pendingJobs: Int get() = pending.get()

    init {
        capture.imagesDir.mkdirs()
    }

    /** Called from the GL thread with the latest ARCore feature points (x,y,z,conf per point). */
    fun submitFeaturePoints(ids: IntArray, xyzc: FloatArray, n: Int) {
        pending.incrementAndGet()
        worker.execute {
            try {
                for (i in 0 until n) {
                    if (xyzc[i * 4 + 3] < 0.3f) continue
                    featurePoints[ids[i]] = floatArrayOf(xyzc[i * 4], xyzc[i * 4 + 1], xyzc[i * 4 + 2])
                }
            } finally {
                pending.decrementAndGet()
            }
        }
    }

    /**
     * Stores one photo. [c2w] is the column major camera pose (sensor aligned, Pose.toMatrix),
     * [intr] the CPU image intrinsics, [depth] optional depth samples of the same frame,
     * [sharpness] the photo's blur measure (see [Sharpness]).
     */
    fun addFrame(image: YuvFrame, c2w: FloatArray, intr: Intrinsics, timestampNs: Long, depth: DepthSamples?, sharpness: Float) {
        val index = frameCount
        frameCount++
        pending.incrementAndGet()
        worker.execute {
            try {
                val name = String.format(Locale.US, "images/frame_%05d.jpg", index)
                val file = File(capture.dir, name)
                file.writeBytes(image.toJpeg(92))
                val imgIntr = if (intr.width != image.width || intr.height != image.height)
                    intr.scaledTo(image.width, image.height) else intr
                frames.add(FrameRecord(name, c2w, imgIntr, timestampNs, sharpness))
                if (depth != null) addDepthPoints(depth, c2w, image, imgIntr)
                colorFeaturePoints(c2w, image, imgIntr)
                pointCount = numPoints
                if (frames.size % 10 == 1) writeTransforms()
                capture.frames = frames.size
                capture.points = numPoints
                repo.save(capture)
            } catch (e: Exception) {
                Log.e(TAG, "failed to store frame", e)
            } finally {
                pending.decrementAndGet()
            }
        }
    }

    /**
     * Drops the latest photo (undo): its image file and its entry in transforms.json.
     * Points it already contributed stay; they only seed the training.
     */
    fun removeLast() {
        if (frameCount == 0) return
        frameCount--
        pending.incrementAndGet()
        worker.execute {
            try {
                val f = frames.removeLastOrNull() ?: return@execute
                File(capture.dir, f.file).delete()
                writeTransforms()
                capture.frames = frames.size
                repo.save(capture)
            } catch (e: Exception) {
                Log.e(TAG, "failed to remove frame", e)
            } finally {
                pending.decrementAndGet()
            }
        }
    }

    /** Writes transforms.json and points3d.ply; [done] runs on the worker thread. */
    fun finish(done: () -> Unit) {
        worker.execute {
            try {
                writeTransforms()
                writePoints()
                capture.frames = frames.size
                capture.points = numPoints
                repo.save(capture)
            } catch (e: Exception) {
                Log.e(TAG, "failed to finish capture", e)
            }
            done()
        }
        worker.shutdown()
    }

    fun awaitIdle(timeoutMs: Long) {
        worker.awaitTermination(timeoutMs, TimeUnit.MILLISECONDS)
    }

    // ------------------------------------------------------------------
    // points

    private fun addPoint(x: Float, y: Float, z: Float, rgb: Int) {
        if (numPoints >= maxPoints) return
        val key = voxelKey(x, y, z)
        if (voxelIndex.containsKey(key)) return
        if (numPoints * 3 + 3 > px.size) {
            px = px.copyOf(px.size * 2)
            pc = pc.copyOf(pc.size * 2)
        }
        px[numPoints * 3] = x
        px[numPoints * 3 + 1] = y
        px[numPoints * 3 + 2] = z
        pc[numPoints] = rgb
        voxelIndex[key] = numPoints
        numPoints++
    }

    private fun voxelKey(x: Float, y: Float, z: Float): Long {
        val ix = floor(x / voxel).toLong() and 0x1FFFFF
        val iy = floor(y / voxel).toLong() and 0x1FFFFF
        val iz = floor(z / voxel).toLong() and 0x1FFFFF
        return (ix shl 42) or (iy shl 21) or iz
    }

    /** Unprojects depth samples into world space and colours them from the photo. */
    private fun addDepthPoints(d: DepthSamples, c2w: FloatArray, image: YuvFrame, ci: Intrinsics) {
        val rgb = IntArray(3)
        val di = d.intr
        for (i in 0 until d.count) {
            val z = d.depth[i]
            val xn = (d.u[i] - di.cx) / di.fx
            val yn = (d.v[i] - di.cy) / di.fy
            // colour: same normalised coordinates in the CPU image
            val cu = (xn * ci.fx + ci.cx).toInt()
            val cv = (yn * ci.fy + ci.cy).toInt()
            if (!image.rgbAt(cu, cv, rgb)) continue
            // OpenGL camera space: x right, y up, looking down -z
            val lx = xn * z
            val ly = -yn * z
            val lz = -z
            val wx = c2w[0] * lx + c2w[4] * ly + c2w[8] * lz + c2w[12]
            val wy = c2w[1] * lx + c2w[5] * ly + c2w[9] * lz + c2w[13]
            val wz = c2w[2] * lx + c2w[6] * ly + c2w[10] * lz + c2w[14]
            addPoint(wx, wy, wz, (rgb[0] shl 16) or (rgb[1] shl 8) or rgb[2])
        }
    }

    /** Assigns colours to feature points that project into the new photo. */
    private fun colorFeaturePoints(c2w: FloatArray, image: YuvFrame, ci: Intrinsics) {
        val rgb = IntArray(3)
        // world -> camera: R^T (p - t)
        val tx = c2w[12]
        val ty = c2w[13]
        val tz = c2w[14]
        for ((id, p) in featurePoints) {
            if (id in featureColored) continue
            val dx = p[0] - tx
            val dy = p[1] - ty
            val dz = p[2] - tz
            val lx = c2w[0] * dx + c2w[1] * dy + c2w[2] * dz
            val ly = c2w[4] * dx + c2w[5] * dy + c2w[6] * dz
            val lz = c2w[8] * dx + c2w[9] * dy + c2w[10] * dz
            val depth = -lz
            if (depth < 0.05f) continue
            val u = (ci.fx * lx / depth + ci.cx).toInt()
            val v = (ci.fy * (-ly) / depth + ci.cy).toInt()
            if (!image.rgbAt(u, v, rgb)) continue
            featureColored.add(id)
            addPoint(p[0], p[1], p[2], (rgb[0] shl 16) or (rgb[1] shl 8) or rgb[2])
        }
    }

    // ------------------------------------------------------------------
    // files

    private fun writeTransforms() {
        val arr = JSONArray()
        for (f in frames) {
            val m = JSONArray()
            for (r in 0 until 4) {
                val row = JSONArray()
                for (c in 0 until 4) row.put(f.c2w[c * 4 + r].toDouble())
                m.put(row)
            }
            arr.put(JSONObject().apply {
                put("file_path", f.file)
                put("transform_matrix", m)
                put("fl_x", f.intr.fx.toDouble())
                put("fl_y", f.intr.fy.toDouble())
                put("cx", f.intr.cx.toDouble())
                put("cy", f.intr.cy.toDouble())
                put("w", f.intr.width)
                put("h", f.intr.height)
                put("timestamp_ns", f.timestampNs)
                put("sharpness", f.sharpness.toDouble())
            })
        }
        val root = JSONObject().apply {
            put("camera_model", "OPENCV")
            put("source", "gsplat-android (ARCore)")
            put("coordinate_system", "ARCore world, Y up, metres; transform_matrix = camera-to-world, OpenGL camera axes")
            put("ply_file_path", "points3d.ply")
            put("frames", arr)
        }
        val tmp = File(capture.dir, "transforms.json.tmp")
        tmp.writeText(root.toString(1))
        tmp.renameTo(capture.transforms)
    }

    private fun writePoints() {
        val tmp = File(capture.dir, "points3d.ply.tmp")
        tmp.bufferedWriter().use { w ->
            w.write("ply\nformat ascii 1.0\nelement vertex $numPoints\n")
            w.write("property float x\nproperty float y\nproperty float z\n")
            w.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for (i in 0 until numPoints) {
                val c = pc[i]
                w.write(
                    String.format(
                        Locale.US, "%.5f %.5f %.5f %d %d %d\n",
                        px[i * 3], px[i * 3 + 1], px[i * 3 + 2], (c shr 16) and 0xFF, (c shr 8) and 0xFF, c and 0xFF
                    )
                )
            }
        }
        tmp.renameTo(capture.pointsFile)
    }

    companion object {
        private const val TAG = "CaptureRecorder"
    }
}
