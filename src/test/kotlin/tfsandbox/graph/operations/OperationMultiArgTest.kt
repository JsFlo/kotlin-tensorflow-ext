package tfsandbox.graph.operations

import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import org.tensorflow.*
import tfsandbox.exts.Operation
import tfsandbox.exts.OperationType
import tfsandbox.exts.addPlaceholder
import tfsandbox.exts.runFirstTensor
import java.util.*

@RunWith(Parameterized::class)
public class OperationMultiArgTest(val listOfFloats: List<Float>, val operationType: OperationType) {


    companion object {
        // Could abstract more to re run same test cases over all types
        // could use random generators to come up with random number of test cases
        @Parameterized.Parameters(name = "Testing operation {2}") @JvmStatic
        public fun getData(): Collection<*> {
            return Arrays.asList(*arrayOf(
                    arrayOf<Any>(listOf(2f, 2f, 5f, 29f, 84f), OperationType.ADD),
                    arrayOf<Any>(listOf(3f, 45f), OperationType.ADD),
                    arrayOf<Any>(listOf(4f, 2f, 6f), OperationType.ADD),
                    arrayOf<Any>(listOf(5f, 23f, 50f, 2f), OperationType.ADD),
                    arrayOf<Any>(listOf(1f, 2f, 5f, 29f, 84f), OperationType.ADD),

                    arrayOf<Any>(listOf(2f, 2f, 5f, 29f, 84f), OperationType.SUB),
                    arrayOf<Any>(listOf(3f, 45f), OperationType.SUB),
                    arrayOf<Any>(listOf(4f, 2f, 6f), OperationType.SUB),
                    arrayOf<Any>(listOf(5f, 23f, 50f, 2f), OperationType.SUB),
                    arrayOf<Any>(listOf(1f, 2f, 5f, 29f, 84f), OperationType.SUB),

                    arrayOf<Any>(listOf(2f, 2f, 5f, 29f, 84f), OperationType.MUL),
                    arrayOf<Any>(listOf(3f, 45f), OperationType.MUL),
                    arrayOf<Any>(listOf(4f, 2f, 6f), OperationType.MUL),
                    arrayOf<Any>(listOf(5f, 23f, 50f, 2f), OperationType.MUL),
                    arrayOf<Any>(listOf(1f, 2f, 5f, 29f, 84f), OperationType.MUL),

                    arrayOf<Any>(listOf(2f, 2f, 5f, 29f, 84f), OperationType.DIV),
                    arrayOf<Any>(listOf(3f, 45f), OperationType.DIV),
                    arrayOf<Any>(listOf(4f, 2f, 6f), OperationType.DIV),
                    arrayOf<Any>(listOf(5f, 23f, 50f, 2f), OperationType.DIV),
                    arrayOf<Any>(listOf(1f, 2f, 5f, 29f, 84f), OperationType.DIV)
            ))

        }
    }


    @Test
    public fun executeOperation() {
        Graph().use { graph ->

            var opName = "opName"

            // List<Float> -> List<Pair<Output, Tensor<Float>>>
            val listOfData = listOfFloats.mapIndexed { index, fl ->
                Pair<Output, Tensor>(graph.addPlaceholder(opName + index, DataType.FLOAT),
                        Tensor.create(fl))
            }

            val y = graph.Operation("y", operationType, listOfData.map { it.first }.toList())

            Session(graph).use { sess ->
                val runner = sess.runner()

                listOfData.forEach { (output, tensor) ->
                    runner.feed(output, tensor)
                }

                runner.fetch(y)
                        .runFirstTensor {
                            val expectedTotal = calculateTotal(operationType, listOfData)
                            Assert.assertEquals(expectedTotal, it.floatValue())
                        }

                // might be able to close after feeding
                listOfData.forEach { (_, tensor) ->
                    tensor.close()
                }
            }
        }
    }

    // expectedTotal will be calculated and tested against the tensor that tensorflow throws back
    fun calculateTotal(operationType: OperationType, listOfData: List<Pair<Output, Tensor>>): Float {
        println("Operation: ${operationType.opName}")
        var expectedTotal = 0f

        // calculating our own to test against
        listOfData.map { it.second }.forEachIndexed { index, tensor ->
            println(tensor.floatValue())
            if (index == 0) {
                expectedTotal = tensor.floatValue()
            } else {
                expectedTotal = when (operationType) {
                    OperationType.ADD -> expectedTotal + tensor.floatValue()
                    OperationType.SUB -> expectedTotal - tensor.floatValue()
                    OperationType.MUL -> expectedTotal * tensor.floatValue()
                    OperationType.DIV -> expectedTotal / tensor.floatValue()
                }
            }
        }
        println("---------------------")
        println(expectedTotal)

        return expectedTotal
    }
}