# WIP: Differentiable F#, executing using TensorFlow

The repo contains 

1.	TensorFlow API for the FSharp Programming Language

2.	A DSL implementation for `tf { ... }` that supports first-pass shape-checking/inference and other nice things

It allows differentiation of F# code as follows:

    // Define a function which will be executed using TensorFlow
    let f x = tf { return x * x + v 4.0 * x }

    // Get the derivative of the function
    let df = DT.Diff f

    // Run the derivative 
    df (v 3.0) |> DT.Run

As well as complete expression of deep neural networks.

The design is very much being done to allow alternative execution with Torch or DiffSharp (we may try DiffSharp if/when we get Gunes Baydin on board to implement Tensors in that)

# Roadmap

* Port gradient and training/optimization code (from the Scala-Tensorflow)

* Add more differentiation options (grad, hessian etc.)

* Generate larger TF surface area in DSL

* Add proper testing for DSL 

* Port testing for core API

* many more things

