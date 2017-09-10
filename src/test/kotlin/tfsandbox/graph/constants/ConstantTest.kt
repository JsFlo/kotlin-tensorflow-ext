package tfsandbox.graph.constants

import org.junit.Assert
import org.junit.Test
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.TensorFlow
import tfsandbox.exts.addConstant
import tfsandbox.exts.run

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

    @Test
    fun operationInGraphExistsByName() {
        Graph().use { graph ->

            val myConst1 = graph.addConstant("aConstant1", "val")

            val operationOutput = graph.operation("aConstant1")
            Assert.assertEquals(operationOutput, myConst1.op())
        }
    }

}