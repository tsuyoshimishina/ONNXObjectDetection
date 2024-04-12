package com.cellgraphics.onnxobjectdetection

import android.content.res.AssetManager
import android.graphics.*
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import com.otaliastudios.cameraview.CameraView
import com.otaliastudios.cameraview.frame.Frame
import java.io.BufferedReader
import java.io.InputStreamReader

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private var detectorAddr = 0L
    private var frameWidth = 0
    private var frameHeight = 0
    private val paint = Paint()
    private val labels = arrayListOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val cameraView = findViewById<CameraView>(R.id.camera)
        cameraView.setLifecycleOwner(this)
        cameraView.addFrameProcessor {
            frame -> detectObjectNative(frame)
        }

        // init the paint for drawing the detections
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 3f
        paint.textSize = 50f
        paint.textAlign = Paint.Align.LEFT

        // Set the detections drawings surface transparent
        val surfaceView = findViewById<SurfaceView>(R.id.surfaceView)
        surfaceView.setZOrderOnTop(true)
        surfaceView.holder.setFormat(PixelFormat.TRANSPARENT)

        loadLabels()
    }

    private fun loadLabels() {
        val labelsInput = this.assets.open("labels.txt")
        val br = BufferedReader(InputStreamReader(labelsInput))
        var line = br.readLine()
        while (line != null) {
            labels.add(line)
            line = br.readLine()
        }
        br.close()
    }

    private fun detectObjectNative(frame: Frame) {
        if (this.detectorAddr == 0L) {
            this.detectorAddr = initDetector(this.assets)
            this.frameWidth = frame.size.width
            this.frameHeight = frame.size.height
        }

        val res = detect(
            this.detectorAddr,
            frame.getData(),
            frame.size.width,
            frame.size.height,
            frame.rotationToUser
        )

        val surfaceView = findViewById<SurfaceView>(R.id.surfaceView)
        val canvas = surfaceView.holder.lockCanvas()
        if (canvas != null) {
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY)
            // Draw the detections, in our case there are only 3
            this.drawDetection(canvas, frame.rotationToUser, res, 0)
            this.drawDetection(canvas, frame.rotationToUser, res, 1)
            this.drawDetection(canvas, frame.rotationToUser, res, 2)
            surfaceView.holder.unlockCanvasAndPost(canvas)
        }
    }

    private fun drawDetection(
        canvas: Canvas,
        rotation: Int,
        detectionsArr: FloatArray,
        detectionIdx: Int
    ) {
        // Filter by score
        val score = detectionsArr[detectionIdx * 6 + 1]
        if (score < 0.6) return

        // Get the frame dimensions
        val w = if (rotation == 0 || rotation == 180) this.frameWidth else this.frameHeight
        val h = if (rotation == 0 || rotation == 180) this.frameHeight else this.frameWidth

        val camera = findViewById<CameraView>(R.id.camera)

        // detection coords are in frame coord system, convert to screen coords
        val scaleX = camera.width.toFloat() / w
        val scaleY = camera.height.toFloat() / h

        // The camera view offset on screen
        val xoff = camera.left.toFloat()
        val yoff = camera.top.toFloat()

        val classId = detectionsArr[detectionIdx * 6 + 0].toInt()
        val xmin = xoff + detectionsArr[detectionIdx * 6 + 2] * scaleX
        val xmax = xoff + detectionsArr[detectionIdx * 6 + 3] * scaleX
        val ymin = yoff + detectionsArr[detectionIdx * 6 + 4] * scaleY
        val ymax = yoff + detectionsArr[detectionIdx * 6 + 5] * scaleY

        // Draw the rect
        val p = Path()
        p.moveTo(xmin, ymin)
        p.lineTo(xmax, ymin)
        p.lineTo(xmax, ymax)
        p.lineTo(xmin, ymax)
        p.lineTo(xmin, ymin)
        canvas.drawPath(p, paint)

        // Draw the label and score
        val label = labels[classId]
        val txt = "%s (%.2f)".format(label, score)
        canvas.drawText(txt, xmin, ymin, paint)
    }

    private external fun initDetector(assetManager: AssetManager): Long

    private external fun detect(
        detectorAddr: Long,
        srcAddr: ByteArray,
        width: Int,
        height: Int,
        rotation: Int
    ): FloatArray

    companion object {
        // Used to load the 'onnxobjectdetection' library on application startup.
        init {
            System.loadLibrary("onnxobjectdetection")
        }
    }
}