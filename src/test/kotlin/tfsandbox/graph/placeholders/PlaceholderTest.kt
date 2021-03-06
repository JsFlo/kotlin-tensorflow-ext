package tfsandbox.graph.placeholders

import org.junit.Assert
import org.junit.Test
import org.tensorflow.DataType
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import tfsandbox.exts.Operation
import tfsandbox.exts.OperationType
import tfsandbox.exts.addPlaceholder
import tfsandbox.exts.runFirstTensor

class PlaceholderTest {

    @Test
    public fun sessionExecutesPlaceholder() {
        Graph().use { graph ->

            // Creates a graph for y = a (placeholder) + b (placeholder)
            val a = graph.addPlaceholder("a", DataType.FLOAT)
            val b = graph.addPlaceholder("b", DataType.FLOAT)
            val y = graph.Operation("y", OperationType.ADD, a, b)

            Session(graph).use { sess ->

                val ta = Tensor.create(10f)
                val tb = Tensor.create(10f)
                sess.runner()
                        .feed(a, ta)
                        .feed(b, tb)
                        .fetch(y)
                        .runFirstTensor {
                            println("${ta.floatValue()} + ${tb.floatValue()} = ${it.floatValue()}")
                            Assert.assertEquals(ta.floatValue() + tb.floatValue(), it.floatValue())
                        }
                ta.close()
                tb.close()
            }
        }
    }


}