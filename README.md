# kotlin-tensorflow-ext

I've been using the Python API of [Tensorflow](https://www.tensorflow.org/)
recently for creating some simple models. I also have been trying to learn a new
programming language called Kotlin since [Google announced that Kotlin is an officially
                                          supported language for Android Developement](https://www.youtube.com/watch?v=d8ALcQiuPWs).

I noticed that the Java API of Tensorflow is in its early stages and had an ["experimental API"](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
which felt like the perfect opportunity to try to use and learn
Kotlin's extension functions to provide a better API and to learn the current limitations, compared to the python api, of the Java API.

## Constants
```kotlin
Graph().use { graph ->

val value = "Hello from  ${TensorFlow.version()}"
val myConst = graph.addConstant("aConstant", value)

Session(graph).use { sess ->

sess.runner()
.fetch(myConst)
.runFirstTensor({ output ->
Assert.assertEquals(value, output.bytesValue().toUTF8String())
})

}
}
```

## Placeholders
```kotlin
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
```
## Operations
```kotlin
val a = graph.addPlaceholder("a", DataType.FLOAT)
val b = graph.addPlaceholder("b", DataType.FLOAT)

val y = graph.Operation("y", OperationType.ADD, a, b)
val y = graph.Operation("y", OperationType.SUB, a, b)
val y = graph.Operation("y", OperationType.MUL, a, b)
val y = graph.Operation("y", OperationType.DIV, a, b)

val y = graph.Operation("multiArg", OperationType.Add, a, b, c, d, e, f)
```
