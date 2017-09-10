package tfsandbox


import org.junit.Assert
import org.junit.Test
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.TensorFlow
import tfsandbox.exts.addConstant
import tfsandbox.exts.run
import kotlin.use


class ConstantTest {

    fun ByteArray.toUTF8String() = java.lang.String(this, "UTF-8")

    @Test
    fun sessionExecutesConstant() {
        Graph().use { graph ->

            val value = "Hello from  ${TensorFlow.version()}"
            val myConst = graph.addConstant("aConstant", value)

            Session(graph).use { sess ->

                sess.runner()
                        .fetch(myConst)
                        .run({ output ->
                            Assert.assertEquals(value, output.bytesValue().toUTF8String())
                        })

            }
        }
    }

}