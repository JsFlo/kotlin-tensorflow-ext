package tfsandbox.exts

import org.tensorflow.Session
import org.tensorflow.Tensor

fun Session.Runner.run(funToRun: (t: Tensor) -> Unit) {
    run()[0].use { output ->
        funToRun(output)
    }
}