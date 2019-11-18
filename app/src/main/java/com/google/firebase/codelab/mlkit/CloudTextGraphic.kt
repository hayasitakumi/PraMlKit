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

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Paint.Style
import com.google.firebase.codelab.mlkit.GraphicOverlay.Graphic
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentText
import com.google.firebase.ml.vision.document.FirebaseVisionDocumentText.Word

/**
 * Graphic instance for rendering TextBlock position, size, and ID within an associated graphic
 * overlay view.
 */
class CloudTextGraphic internal constructor(overlay: GraphicOverlay, private val word: Word?) : Graphic(overlay) {
    private val rectPaint: Paint = Paint()
    private val textPaint: Paint
    /**
     * Draws the text block annotations for position, size, and raw value on the supplied canvas.
     */
    override fun draw(canvas: Canvas) {
        checkNotNull(word) { "Attempting to draw a null text." }

        val wordStr = StringBuilder()
        val wordRect = word.boundingBox
        canvas.drawRect(wordRect!!, rectPaint)
        val symbols: List<FirebaseVisionDocumentText.Symbol> = word.symbols
        for (m in symbols.indices) {
            wordStr.append(symbols[m].text)
        }
        canvas.drawText(wordStr.toString(), wordRect.left.toFloat(), wordRect.bottom.toFloat(), textPaint)
    }

    companion object {
        private const val TEXT_COLOR = Color.GREEN
        private const val TEXT_SIZE = 60.0f
        private const val STROKE_WIDTH = 5.0f
    }

    init {
        rectPaint.color = TEXT_COLOR
        rectPaint.style = Style.STROKE
        rectPaint.strokeWidth = STROKE_WIDTH
        textPaint = Paint()
        textPaint.color = TEXT_COLOR
        textPaint.textSize = TEXT_SIZE
        // Redraw the overlay, as this graphic has been added.


        postInvalidate()
    }
}