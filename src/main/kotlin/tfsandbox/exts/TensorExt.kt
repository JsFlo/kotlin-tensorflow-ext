package tfsandbox.exts

import org.tensorflow.Tensor

fun createTensor(obj: Any): Tensor = when (obj) {
    is String -> obj.createStringTensor()
    else -> Tensor.create(obj)
}


// assuming all utf-8
private fun String.createStringTensor() = Tensor.create(this.toByteArray(charset("UTF-8")))

