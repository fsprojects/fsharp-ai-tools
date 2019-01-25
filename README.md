# **work in progress some serious wet paint here**

# TensorFlow.FSharp - Differentiable F#, executing using TensorFlow

The repo contains:

1.	TensorFlow API for the F# Programming Language

2.	A DSL implementation for `tf { ... }` that supports first-pass shape-checking/inference and other nice things

The most important result here is preliminary proof-of-concept that it is possible to build real, full-speed
TF graphs using a fairly thin differentiable-DSL whose intended semantics are clear and relatively independent
of TF, while still achieving some degree of cohabitation with the TF ecosystem (graphs are graphs, and we believe they
can be saved/visualized using standard TF tools, and even made to interoperate with TF-Python).

# The TensorFlow API for F# 

See `TensorFlow.FSharp`.  This API is designed in a similar way to `TensorFlowSharp`, but is implemented directly in F# and
contains additional functionality.

# The Differentiable F# DSL

See `TensorFlow.FSharp.DSL`.  This DSL allows differentiation of F# code as follows:

```fsharp
    // Define a function which will be executed using TensorFlow
    let f x = tf { return x * x + v 4.0 * x }

    // Get the derivative of the function
    let df x = DT.diff f x  // computes x*2 + 4.0

    // Run the derivative 
    df (v 3.0) |> DT.Run // returns 6.0 + 4.0 = 10.0
```
For a scalar function with respect to a single input variables:
```fsharp
    // Define a function which will be executed using TensorFlow
    let f x = tf { return x * x + v 4.0 * x }

    // Get the derivative of the function
    let df x = DT.diff f x  // computes x*2 + 4.0

    // Run the derivative 
    df (v 3.0) |> DT.RunScalar // returns 6.0 + 4.0 = 10.0
```
For a scalar function with multiple input variables:
```fsharp
    // Define a function which will be executed using TensorFlow
    // computes [ x1*x1*x3 + x2*x2*x2 + x3*x3*x1 + x1*x1 ]
    let f (xs: DT<'T>) = tf { return DT.ReduceSum (xs * xs * DT.ReverseV2 xs) } 

    // Get the partial derivatives of the scalar function
    // computes [ 2*x1*x3 + x3*x3; 3*x2*x2; 2*x3*x1 + x1*x1 ]
    let df xs = DT.diff f xs   

    // Run the derivative 
    df (vec [ 3.0; 4.0; 5.0 ]) |> DT.RunArray // returns [ 55.0; 48.0; 39.0 ]
```

More examples/tests are in [dsl-tests.fsx](https://github.com/fsprojects/TensorFlow.FSharp/blob/master/tests/dsl-tests.fsx).

* `tf { ... }` indicates a block of code expressed using the DSL and intended to be executed as a TensorFlow graph.  The
  use of `tf { ... }` is actually optional but recommended for clarity for all significant chunks of differentiable code.

* `DT` stands for `differentiable tensor` and the one type of `DT<_>` values are used to represent differentiable scalars, vectors, matrices and tensors.
  If you are familiar with the design of `DiffSharp` there are similarities here: DiffSharp defines `D` (differentiable scalar), `DV` (differentiable
  vector), `DM` (differentiable matrix).

* `DT.diff` is used to differentiate of `R^n -> R` scalar-valued functions (loss functions) w.r.t. multiple input variables. If 
  a scalar input is used, a single total deriative is returned. If a vector of inputs is used, a vector of
  partial derivatives are returned.

* `DT.jacobian` is used to differentiate `R^n -> R^2` vector-valued functions w.r.t. multiple input variables. A vector or
  matrix of partial derivatives is returned.

* Other gradient-based functions include `DT.grad`, `DT.curl`, `DT.hessian` and `DT.divergence`.

* In the prototype, all gradient-based functions are implemented using TensorFlow's `AddGradients`, i.e. the C++ implementation of
  gradients. Not all gradient-based functions are implemented efficiently for all inputs.

* `DT.*` is a DSL for expressing differentiable tensors using the TensorFlow fundamental building blocks.  The naming
  of operators in this DSL are currently TensorFLow specific and may change.

* A preliminary pass of shape inference is performed _before_ any TensorFlow operations are performed.  This
  allows you to check the shapes of your differentiable code indepentently of TensorFlow's shape computations.
  A shape inference system is used which allows for many shapes to be inferred and is akin to F# type inference.
  Not all TensorFlow automatic shape transformations are applied during shape inference.

  It is possible that at some future point this shape inference may be performed statically for F# code, or via a
  secondary tool.

While the above are toy examples, the approach scales (at least in principle) to the complete expression of deep neural networks
and full TensorFlow computation graphs. The links below show the implementation of a common DNN sample (the samples may not
yet run, this is wet paint):

* [NeuralStyleTransfer using F# TensorFlow API](https://github.com/fsprojects/TensorFlow.FSharp/blob/master/examples/NeuralStyleTransfer.fsx)

* [NeuralStyleTransfer in DSL form](https://github.com/fsprojects/TensorFlow.FSharp/blob/master/examples/NeuralStyleTransfer-dsl.fsx)

The design is intended to allow alternative execution with Torch or DiffSharp.
DiffSharp may be used once Tensors are available in that library.


# Roadmap - Core API

* Port gradient and training/optimization code (from the Scala-Tensorflow)

* Port testing for core API

* Add docs

# Roadmap - DSL

* Switch to using ported gradient code when it is available in core API

* Generate larger TF surface area in `tf { ... }` DSL

* Add proper testing for DSL 

* Consider control flow translation in DSL

* Add docs

* Add examples of how to do static graph building and analysis based on FCS, quotations and/or interpretation, e.g. for visualization

* Performance testing
