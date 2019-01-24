# **work in progress some serious wet paint here**

# TensorFlow.FSharp - Differentiable F#, executing using TensorFlow

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

# Roadmap - Core API

* Port gradient and training/optimization code (from the Scala-Tensorflow)

* Port testing for core API

* Add docs

# Roadmap - DSL

* Add more differentiation options (grad, hessian etc.)

* Switch to using ported gradient code when it is available in core API

* Generate larger TF surface area in `tf { ... }` DSL

* Add proper testing for DSL 

* Consider control flow translation in DSL

* Add docs

* Add examples of how to do static graph building and analysis based on FCS, quotations and/or interpretation, e.g. for visualization

* Many more things

