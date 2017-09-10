package tfsandbox.graph.constants

import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.Parameterized
import org.tensorflow.Graph
import org.tensorflow.Session
import tfsandbox.exts.addConstant
import tfsandbox.exts.runFirstTensor
import java.util.*

@RunWith(Parameterized::class)
public class ConstantFetchAndCheckValue(val constants: List<Data>, val constantNameToFetch: String,
                                        val expectedValue: Any) {

    companion object {
        data class Data(val name: String, val value: Any)

        val listOfStringConstants = listOf<Data>(
                Data("constant1", "value1"),
                Data("constant2", "value2"),
                Data("constant3", "value3"),
                Data("constant4", "value4"),
                Data("constant5", "value5"))

        val listOfFloatConstants = listOf<Data>(
                Data("constant1", 1f),
                Data("constant2", 2f),
                Data("constant3", 3f),
                Data("constant4", 4f),
                Data("constant5", 5f))

        val listOfIntConstants = listOf<Data>(
                Data("constant1", 1),
                Data("constant2", 2),
                Data("constant3", 3),
                Data("constant4", 4),
                Data("constant5", 5))

        // Could abstract more to re run same test cases over all types
        @Parameterized.Parameters(name = "Fetching {1}: Expecting: {2}") @JvmStatic
        public fun getData(): Collection<*> {
            return Arrays.asList(*arrayOf(
                    arrayOf<Any>(listOfStringConstants, "constant1", "value1"),
                    arrayOf<Any>(listOfStringConstants, "constant2", "value2"),
                    arrayOf<Any>(listOfStringConstants, "constant3", "value3"),
                    arrayOf<Any>(listOfStringConstants, "constant4", "value4"),
                    arrayOf<Any>(listOfStringConstants, "constant5", "value5"),

                    arrayOf<Any>(listOfFloatConstants, "constant1", 1f),
                    arrayOf<Any>(listOfFloatConstants, "constant2", 2f),
                    arrayOf<Any>(listOfFloatConstants, "constant3", 3f),
                    arrayOf<Any>(listOfFloatConstants, "constant4", 4f),
                    arrayOf<Any>(listOfFloatConstants, "constant5", 5f),

                    arrayOf<Any>(listOfIntConstants, "constant1", 1),
                    arrayOf<Any>(listOfIntConstants, "constant2", 2),
                    arrayOf<Any>(listOfIntConstants, "constant3", 3),
                    arrayOf<Any>(listOfIntConstants, "constant4", 4),
                    arrayOf<Any>(listOfIntConstants, "constant5", 5)))

        }
    }

    @Test
    public fun fetchAndCheck() {
        Graph().use { graph ->

            for ((name, value) in constants) {
                graph.addConstant(name, value)
            }

            Session(graph).use { sess ->

                sess.runner()
                        .fetch(constantNameToFetch)
                        .runFirstTensor({ output ->
                            when (expectedValue) {
                                is String -> Assert.assertEquals(expectedValue, output.bytesValue().toUTF8String())
                                is Float -> Assert.assertEquals(expectedValue, output.floatValue())
                                is Int -> Assert.assertEquals(expectedValue, output.intValue())
                                else -> Assert.assertEquals(expectedValue, output)
                            }

                        })

            }
        }
    }

    fun ByteArray.toUTF8String() = java.lang.String(this, "UTF-8")
}