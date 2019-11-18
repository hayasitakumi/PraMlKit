// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.firebase.codelab.mlkit

import android.app.Activity
import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.util.Pair
import android.view.View
import android.widget.*
import android.widget.AdapterView.OnItemSelectedListener
import com.google.firebase.codelab.mlkit.GraphicOverlay.Graphic
import com.google.firebase.codelab.mlkit.R.id
import com.google.firebase.codelab.mlkit.R.layout
import com.google.firebase.ml.common.FirebaseMLException
import com.google.firebase.ml.common.modeldownload.FirebaseLocalModel
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.common.modeldownload.FirebaseRemoteModel
import com.google.firebase.ml.custom.*
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentText
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentTextRecognizer
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions.Builder
import com.google.firebase.ml.vision.text.FirebaseVisionText
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import java.util.AbstractMap.SimpleEntry
import kotlin.collections.Map.Entry
import kotlin.experimental.and
import kotlin.math.max

class MainActivity : AppCompatActivity(), OnItemSelectedListener {
    /**
     * An instance of the driver class to run model inference with Firebase.
     */
    private var mInterpreter: FirebaseModelInterpreter? = null
    /**
     * Data configuration of input & output data of model.
     */
    private var mDataOptions: FirebaseModelInputOutputOptions? = null
    private var mImageView: ImageView? = null
    private var mTextButton: Button? = null
    private var mFaceButton: Button? = null
    private var mCloudButton: Button? = null
    private var mRunCustomModelButton: Button? = null
    private var mSelectedImage: Bitmap? = null
    private var mGraphicOverlay: GraphicOverlay? = null
    // Max width (portrait mode)
    private var mImageMaxWidth: Int? = null
    // Max height (portrait mode)
    private var mImageMaxHeight: Int? = null
    /**
     * Labels corresponding to the output of the vision model.
     */
    private var mLabelList: List<String>? = null
    private val sortedLabels = PriorityQueue<Entry<String, Float>>(RESULTS_TO_SHOW,
            Comparator<Entry<String?, Float?>?> { o1, o2 -> o1!!.value!!.compareTo(o2!!.value!!) })
    /* Preallocated buffers for storing image data. */
    private val intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(layout.activity_main)
        mImageView = findViewById<ImageView?>(id.image_view)
        mTextButton = findViewById<Button?>(id.button_text)
        mFaceButton = findViewById<Button?>(id.button_face)
        mCloudButton = findViewById<Button?>(id.button_cloud_text)
        mRunCustomModelButton = findViewById<Button?>(id.button_run_custom_model)
        mGraphicOverlay = findViewById<GraphicOverlay?>(id.graphic_overlay)
        mTextButton!!.setOnClickListener { runTextRecognition() }
        mFaceButton!!.setOnClickListener { runFaceContourDetection() }
        mCloudButton!!.setOnClickListener { runCloudTextRecognition() }
        mRunCustomModelButton!!.setOnClickListener { runModelInference() }
        val dropdown: Spinner = findViewById(id.spinner)
        val items = arrayOf("Test Image 1 (Text)", "Test Image 2 (Text)", "Test Image 3" +
                " (Face)", "Test Image 4 (Object)", "Test Image 5 (Object)")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, items)
        dropdown.adapter = adapter
        dropdown.onItemSelectedListener = this
        initCustomModel()
    }

    private fun runTextRecognition() {
        val image: FirebaseVisionImage? = FirebaseVisionImage.fromBitmap(mSelectedImage!!)
        val recognizer: FirebaseVisionTextRecognizer = FirebaseVision.getInstance()
                .onDeviceTextRecognizer
        mTextButton!!.isEnabled = false
        recognizer.processImage(image!!)
                .addOnSuccessListener { texts ->
                    mTextButton!!.isEnabled = true
                    processTextRecognitionResult(texts!!)
                }
                .addOnFailureListener { e ->
                    // Task failed with an exception

                    mTextButton!!.isEnabled = true
                    e.printStackTrace()
                }
    }

    private fun processTextRecognitionResult(texts: FirebaseVisionText) {
        val blocks: List<FirebaseVisionText.TextBlock> = texts.textBlocks
        if (blocks.isEmpty()) {
            showToast("No text found")
            return
        }
        mGraphicOverlay!!.clear()
        for (i in blocks.indices) {
            val lines: List<FirebaseVisionText.Line> = blocks[i].lines
            for (j in lines.indices) {
                val elements: List<FirebaseVisionText.Element> = lines[j].elements
                for (k in elements.indices) {
                    val textGraphic: Graphic = TextGraphic(mGraphicOverlay, elements[k])
                    mGraphicOverlay!!.add(textGraphic)
                }
            }
        }
    }

    private fun runFaceContourDetection() {
        val image: FirebaseVisionImage? = FirebaseVisionImage.fromBitmap(mSelectedImage!!)
        val options: FirebaseVisionFaceDetectorOptions? = Builder()
                .setPerformanceMode(FirebaseVisionFaceDetectorOptions.FAST)
                .setContourMode(FirebaseVisionFaceDetectorOptions.ALL_CONTOURS)
                .build()
        mFaceButton!!.isEnabled = false
        val detector: FirebaseVisionFaceDetector = FirebaseVision.getInstance().getVisionFaceDetector(options!!)
        detector.detectInImage(image!!)
                .addOnSuccessListener { faces ->
                    mFaceButton!!.isEnabled = true
                    processFaceContourDetectionResult(faces)
                }
                .addOnFailureListener { e ->
                    // Task failed with an exception

                    mFaceButton!!.isEnabled = true
                    e.printStackTrace()
                }
    }

    private fun processFaceContourDetectionResult(faces: List<FirebaseVisionFace>) {
        // Task completed successfully

        if (faces.isEmpty()) {
            showToast("No face found")
            return
        }
        mGraphicOverlay!!.clear()
        for (i in faces.indices) {
            val face = faces[i]
            val faceGraphic = FaceContourGraphic(mGraphicOverlay)
            mGraphicOverlay!!.add(faceGraphic)
            faceGraphic.updateFace(face)
        }
    }

    private fun initCustomModel() {
        mLabelList = loadLabelList(this)
        val inputDims = intArrayOf(DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE)
        val outputDims = intArrayOf(DIM_BATCH_SIZE, mLabelList!!.size)
        try {
            mDataOptions = FirebaseModelInputOutputOptions.Builder()
                    .setInputFormat(0, FirebaseModelDataType.BYTE, inputDims)
                    .setOutputFormat(0, FirebaseModelDataType.BYTE, outputDims)
                    .build()
            val conditions: FirebaseModelDownloadConditions? = FirebaseModelDownloadConditions.Builder()
                    .requireWifi()
                    .build()
            val remoteModel: FirebaseRemoteModel? = FirebaseRemoteModel.Builder(HOSTED_MODEL_NAME)
                    .enableModelUpdates(true)
                    .setInitialDownloadConditions(conditions!!)
                    .setUpdatesDownloadConditions(conditions)  // You could also specify
                    // different conditions
                    // for updates
                    .build()
            val localModel: FirebaseLocalModel? = FirebaseLocalModel.Builder("asset")
                    .setAssetFilePath(LOCAL_MODEL_ASSET).build()
            val manager: FirebaseModelManager = FirebaseModelManager.getInstance()
            manager.registerRemoteModel(remoteModel!!)
            manager.registerLocalModel(localModel!!)
            val modelOptions: FirebaseModelOptions? = FirebaseModelOptions.Builder()
                    .setRemoteModelName(HOSTED_MODEL_NAME)
                    .setLocalModelName("asset")
                    .build()
            mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions!!)
        } catch (e: FirebaseMLException) {
            showToast("Error while setting up the model")
            e.printStackTrace()
        }
    }

    private fun runModelInference() {
        if (mInterpreter == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.")
            return
        }
        // Create input data.


        val imgData = convertBitmapToByteBuffer(mSelectedImage)
        try {
            val inputs: FirebaseModelInputs? = FirebaseModelInputs.Builder().add(imgData).build()
            // Here's where the magic happens!!


            mInterpreter!!
                    .run(inputs!!, mDataOptions!!)
                    .addOnFailureListener { e ->
                        e.printStackTrace()
                        showToast("Error running model inference")
                    }
                    .continueWith { task ->
                        val labelProbArray: Array<ByteArray> = task.result!!.getOutput(0)
                        val topLabels = getTopLabels(labelProbArray)
                        mGraphicOverlay!!.clear()
                        val labelGraphic: Graphic = LabelGraphic(mGraphicOverlay!!, topLabels)
                        mGraphicOverlay!!.add(labelGraphic)
                        topLabels
                    }
        } catch (e: FirebaseMLException) {
            e.printStackTrace()
            showToast("Error running model inference")
        }
    }

    private fun runCloudTextRecognition() {
        mCloudButton!!.isEnabled = false
        val image: FirebaseVisionImage? = FirebaseVisionImage.fromBitmap(mSelectedImage!!)
        val recognizer: FirebaseVisionDocumentTextRecognizer = FirebaseVision.getInstance()
                .cloudDocumentTextRecognizer
        recognizer.processImage(image!!)
                .addOnSuccessListener { texts ->
                    mCloudButton!!.isEnabled = true
                    processCloudTextRecognitionResult(texts)
                }
                .addOnFailureListener { e ->
                    // Task failed with an exception

                    mCloudButton!!.isEnabled = true
                    e.printStackTrace()
                }
    }

    private fun processCloudTextRecognitionResult(text: FirebaseVisionDocumentText?) {
        // Task completed successfully

        if (text == null) {
            showToast("No text found")
            return
        }
        mGraphicOverlay!!.clear()
        val blocks: List<FirebaseVisionDocumentText.Block> = text.blocks
        for (i in blocks.indices) {
            val paragraphs: List<FirebaseVisionDocumentText.Paragraph> = blocks[i].paragraphs
            for (j in paragraphs.indices) {
                val words: List<FirebaseVisionDocumentText.Word> = paragraphs[j].words
                for (l in words.indices) {
                    val cloudDocumentTextGraphic = CloudTextGraphic(mGraphicOverlay!!,
                            words[l])
                    mGraphicOverlay!!.add(cloudDocumentTextGraphic)
                }
            }
        }
    }

    /**
     * Gets the top labels in the results.
     */
    @Synchronized
    private fun getTopLabels(labelProbArray: Array<ByteArray>): List<String> {
        for (i in mLabelList!!.indices) {
            sortedLabels.add(SimpleEntry(mLabelList!![i], (labelProbArray[0][i] and 0xff.toByte()) / 255.0f))

            if (sortedLabels.size > RESULTS_TO_SHOW) {
                sortedLabels.poll()
            }
        }
        val result: MutableList<String> = ArrayList()
        val size = sortedLabels.size
        for (i in 0 until size) {
            val label: Entry<String, Float> = sortedLabels.poll()
            result.add(label.key + ":" + label.value)
        }
        Log.d(TAG, "labels: $result")
        return result
    }

    /**
     * Reads label list from Assets.
     */
    private fun loadLabelList(activity: Activity): List<String> {
        val labelList: MutableList<String> = ArrayList()
        try {
            BufferedReader(InputStreamReader(activity.assets.open(LABEL_PATH))).use { reader ->
                var line: String?
                while (reader.readLine().also { line = it } != null) {
                    labelList.add(line!!)
                }
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to read label list.", e)
        }
        return labelList
    }

    /**
     * Writes Image data into a `ByteBuffer`.
     */
    @Synchronized
    private fun convertBitmapToByteBuffer(bitmap: Bitmap?): ByteBuffer {
        val imgData: ByteBuffer = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE)
        imgData.order(ByteOrder.nativeOrder())
        val scaledBitmap: Bitmap = Bitmap.createScaledBitmap(bitmap!!, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y,
                true)
        imgData.rewind()
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0,
                scaledBitmap.width, scaledBitmap.height)
        // Convert the image to int points.


        var pixel = 0
        for (i in 0 until DIM_IMG_SIZE_X) {
            for (j in 0 until DIM_IMG_SIZE_Y) {
                val `val` = intValues[pixel++]
                imgData.put((`val` shr 16 and 0xFF).toByte())
                imgData.put((`val` shr 8 and 0xFF).toByte())
                imgData.put((`val` and 0xFF).toByte())
            }
        }
        return imgData
    }

    private fun showToast(message: String) {
        Toast.makeText(applicationContext, message, Toast.LENGTH_SHORT).show()
    }

    // Functions for loading images from app assets.// Calculate the max width in portrait mode. This is done lazily since we need to
    // wait for
    // a UI layout pass to get the right values. So delay it to first time image
    // rendering time.

    // Returns max image width, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private val imageMaxWidth: Int
        get() {
            if (mImageMaxWidth == null) {
                // Calculate the max width in portrait mode. This is done lazily since we need to
                // wait for
                // a UI layout pass to get the right values. So delay it to first time image
                // rendering time.
                mImageMaxWidth = mImageView!!.width
            }
            return mImageMaxWidth!!
        }// Calculate the max width in portrait mode. This is done lazily since we need to
    // wait for
    // a UI layout pass to get the right values. So delay it to first time image
    // rendering time.

    // Returns max image height, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private val imageMaxHeight: Int
        get() {
            if (mImageMaxHeight == null) {
                // Calculate the max width in portrait mode. This is done lazily since we need to
                // wait for
                // a UI layout pass to get the right values. So delay it to first time image
                // rendering time.
                mImageMaxHeight = mImageView!!.height
            }
            return mImageMaxHeight!!
        }

    // Gets the targeted width / height.
    private val targetedWidthHeight: Pair<Int, Int>
        get() {
            val targetWidth: Int
            val targetHeight: Int
            val maxWidthForPortraitMode = imageMaxWidth
            val maxHeightForPortraitMode = imageMaxHeight
            targetWidth = maxWidthForPortraitMode
            targetHeight = maxHeightForPortraitMode
            return Pair(targetWidth, targetHeight)
        }

    override fun onItemSelected(parent: AdapterView<*>?, v: View, position: Int, id: Long) {
        mGraphicOverlay!!.clear()
        when (position) {
            0 -> mSelectedImage = getBitmapFromAsset(this, "Please_walk_on_the_grass.jpg")
            1 -> mSelectedImage = getBitmapFromAsset(this, "nl2.jpg")
            2 -> mSelectedImage = getBitmapFromAsset(this, "grace_hopper.jpg")
            3 -> mSelectedImage = getBitmapFromAsset(this, "tennis.jpg")
            4 -> mSelectedImage = getBitmapFromAsset(this, "mountain.jpg")
        }
        if (mSelectedImage != null) {
            // Get the dimensions of the View

            val targetedSize = targetedWidthHeight
            val targetWidth: Int = targetedSize.first
            val maxHeight: Int = targetedSize.second

            // Determine how much to scale down the image


            val scaleFactor = max(
                    mSelectedImage!!.width.toFloat() / targetWidth.toFloat(),
                    mSelectedImage!!.height.toFloat() / maxHeight.toFloat())
            val resizedBitmap: Bitmap = Bitmap.createScaledBitmap(
                    mSelectedImage!!,
                    (mSelectedImage!!.width / scaleFactor).toInt(),
                    (mSelectedImage!!.height / scaleFactor).toInt(),
                    true)
            mImageView!!.setImageBitmap(resizedBitmap)
            mSelectedImage = resizedBitmap
        }
    }

    override fun onNothingSelected(parent: AdapterView<*>?) {
        // Do nothing

    }

    companion object {
        private const val TAG = "MainActivity"
        /**
         * Name of the model file hosted with Firebase.
         */
        private const val HOSTED_MODEL_NAME = "cloud_model_1"
        private const val LOCAL_MODEL_ASSET = "mobilenet_v1_1.0_224_quant.tflite"
        /**
         * Name of the label file stored in Assets.
         */
        private const val LABEL_PATH = "labels.txt"
        /**
         * Number of results to show in the UI.
         */
        private const val RESULTS_TO_SHOW = 3
        /**
         * Dimensions of inputs.
         */
        private const val DIM_BATCH_SIZE = 1
        private const val DIM_PIXEL_SIZE = 3
        private const val DIM_IMG_SIZE_X = 224
        private const val DIM_IMG_SIZE_Y = 224
        fun getBitmapFromAsset(context: Context, filePath: String): Bitmap? {
            val assetManager: AssetManager = context.assets
            val `is`: InputStream
            var bitmap: Bitmap? = null
            try {
                `is` = assetManager.open(filePath)
                bitmap = BitmapFactory.decodeStream(`is`)
            } catch (e: IOException) {
                e.printStackTrace()
            }
            return bitmap
        }
    }
}