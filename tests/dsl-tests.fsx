// Build the Debug 'TensorFlow.FSharp.Tests' before using this

#I __SOURCE_DIRECTORY__
#r "netstandard"

#r "bin/Debug/net472/TensorFlow.FSharp.dll"

//open Argu
open System
open TensorFlow.FSharp
open TensorFlow.FSharp.DSL

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

module PlayWithTF = 
    tf { return v 1.0 }
    |> DT.RunScalar

    tf { return (vec [1.0; 2.0]) }
    |> DT.RunArray

    tf { return (DT.Stack [v 1.0; v 2.0]) }
    |> DT.RunArray

    // Gives a shape error of course
    // tf { return (vec [1.0; 2.0] + matrix [ [1.0; 2.0] ]) }
    // |> DT.RunArray

    // Test indexer notation
    tf { return (vec [1.0]).[0] }
    |> DT.RunScalar

    // Test indexer notation 
    tf { return (vec [1.0; 2.0]).[1] }
    |> DT.RunScalar
   
    // Test slicing notation
    tf { return (vec [1.0; 2.0]).[0..0] }
    |> DT.RunArray

    tf { return DT.Reverse (vec [1.0; 2.0]) }
    |> DT.RunArray

    let f x = tf { return x * x + v 4.0 * x }
    let df x = DT.diff f x

    df (v 3.0)
    |> DT.RunScalar
    |> (=) (2.0 * 3.0 + 4.0)

    tf { return vec [1.0; 2.0] + v 4.0 }
    |> DT.RunArray

    tf { return sum (vec [1.0; 2.0] + v 4.0) }
    |> DT.RunScalar

    let f2 x = tf { return sum (vec [1.0; 2.0] * x * x) }
    let df2 x = DT.grad f2 x

    f2 (vec [1.0; 2.0])
    |> DT.RunScalar

    df2 (vec [1.0; 2.0])
    |> DT.RunArray

    //let x = vec [1.0; 2.0]
    //(DT.Stack [| x.[1]; x.[0] |]).Shape
    //|> DT.RunArray

    let f3 (x: DT<_>) = tf { return vec [1.0; 2.0] * x * DT.Reverse x } //[ x1*x2; 2*x2*x1 ] 
    let df3 x = DT.jacobian f3 x // [ [ x2; x1 ]; [2*x2; 2*x1 ] ]  
    let expected (x1, x2) = array2D [| [| x2; x1 |]; [| 2.0*x2; 2.0*x1 |] |]  

    f3 (vec [1.0; 2.0])
    |> DT.RunArray

    df3 (vec [1.0; 2.0])
    |> DT.RunArray2D
    |> (=) (expected (1.0, 2.0))
    // expect 

    tf { use _ = DT.WithScope("foo")
         return vec [1.0; 2.0] + v 4.0 }
   // |> DT.Diff
    |> DT.RunArray

    tf { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] }
    |> DT.RunArray2D

    tf { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] + v 4.0 }
    |> DT.RunArray2D

    tf { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> DT.RunArray

    let var v nm = DT.Variable (v, name=nm)
    
    // Specifying values for variables in the graph
    tf { return var (vec [ 1.0 ]) "x" + v 4.0 }
    |> fun dt -> DT.RunArray(dt, ["x", (vec [2.0] :> _)] )

    tf { return var (vec [ 1.0 ]) "x" * var (vec [ 1.0 ]) "x" + v 4.0 }
    |> DT.RunArray

    tf { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> fun dt -> DT.RunArray(dt, ["hey", upcast (vec [2.0])])
       // Gives 6.0





module GradientAscentWithParameters =

    type DParams = 
        { xs: DT<double>
          y: DT<double> } 

    // "marked up" differentiability, eDSL (embedded DSL, value algebra DSL)
    // Define a function which will be executed using TensorFlow
    // ReqKnowledge: DT. But they get shape checking.
    let f (ps: DParams) = 
        tf { return sin (v 0.5 * ps.xs.[0] * ps.xs.[0] - v 0.25 * ps.y * ps.y + v 3.0) * cos (v 2.0 * ps.xs.[1] + v 1.0 - exp ps.y) }

    // Get the partial derivatives of the scalar function
    // computes [ 2*x1*x3 + x3*x3; 3*x2*x2; 2*x3*x1 + x1*x1 ]
    let df (ps: DParams) = 
        // TODO: ideally this would be
        //    DT.gradients (f ps) ps
        // returning D<DParams>, a type derived structurally from DParams.
        DT.gradients (f ps) [| ps.xs; ps.y |] 

    let rate = 0.1

    type Params = 
        { xs: double[]
          y: double } 

    // Gradient ascent
    let step (ps: Params) =
        
        // DSL construction
        let nodes = df { xs = vec ps.xs; y = v ps.y }

        // DSL elimination 
        let dzxs, dzy = nodes |> DT.RunArrayAndScalar
        
        printfn "size = %f" (sqrt (dzxs.[0]*dzxs.[0] + dzxs.[1]*dzxs.[1] + dzy*dzy))
        { xs = [| ps.xs.[0] + rate * dzxs.[0]; ps.xs.[1] + rate * dzxs.[1] |]; y = rate * dzy }

    let initialParams = { xs = [| -0.3;  -0.3 |]; y = 0.3 }
    initialParams |> Seq.unfold (fun ps -> Some (ps, step ps)) |> Seq.truncate 200 |> Seq.toArray


    
    (*
       // Implicit differentiability
       type Params2 = 
           { xs: double[]
             y: double } 
           
       // Implicit differentiability, quotation-based DSL
       // Define a function which will be executed using TensorFlow
       //
       // Pro: slightly more readable
       // Con: no idea what we can use in the code (esp. library operators)
       // ReqKnowledge: Array, Array2D, FsAlg.Vec, FsAlg.Matrix, ..... 10-20 
       //   Why are they?....  well, some shape is done in regular F# through type distinctions

       let f2 (ps: Params2) = 
           // Oh no! I have no idea whether I was allowed Array.map or not!
           let xs = ps.xs |> Array.map (fun x -> x + 1.0) 
           // Oh no! people tell me to us SonOfFsAlg for math code, but the autodiff framework doesn't understand it! Disaster!
           //let xs = ps.x |> MyMatrix.map (fun x -> x + 1.0) 
           // Crap there is no extend function for Array --> Array2D
           // Oh no, F# array + array2D programming is kind of half-complete w.r.t task
           sin (0.5 * xs.[0] * xs.[1] - 0.25 * ps.y * ps.y + 3.0) * cos (2.0 * xs.[0] + 1.0 - exp ps.y) 
           *)

(*
module GradientDescentWithVariables =
    // Define a function which will be executed using TensorFlow
    let f (x: DT<double>, y: DT<double>) = 
        tf { return sin (v 0.5 * x * x - v 0.25 * y * y + v 3.0) * cos (v 2.0 * x + v 1.0 - exp y) }

    // Get the partial derivatives of the scalar function
    // computes [ 2*x1*x3 + x3*x3; 3*x2*x2; 2*x3*x1 + x1*x1 ]
    let df (x, y) = 
        let dfs = DT.gradients (f (x,y))  [| x; y |] 
        (dfs.[0], dfs.[1])

    let rate = 0.1
    let step (x, y) = 
        // Repeatedly run the derivative 
        let dzx, dzy = df (v x, v y) 
        let dzx = dzx |> DT.RunScalar 
        let dzy = dzy |> DT.RunScalar 
        printfn "size = %f" (sqrt (dzx*dzx + dzy*dzy))
        (x + rate * dzx, y + rate * dzy)

    (-0.3, 0.3) |> Seq.unfold (fun pos -> Some (pos, step pos)) |> Seq.truncate 200 |> Seq.toArray
*)

module ModelExample =
    let modelSize = 10
    let checkSize = 3
    let trainSize = 500
    let validationSize = 100
    let rnd = Random()
    let noise eps = (rnd.NextDouble() - 0.5) * eps 

    /// The true function we use to generate the training data (also a linear model plus some noise)
    let trueCoeffs = [| for i in 1 .. modelSize -> double i |]
    let trueFunction (xs: double[]) = Array.sum [| for i in 0 .. modelSize - 1 -> trueCoeffs.[i] * xs.[i]  |] + noise 0.5

    let makeData size = 
        [| for i in 1 .. size -> 
            let xs = [| for i in 0 .. modelSize - 1 -> rnd.NextDouble() |]
            xs, trueFunction xs |]
        |> Array.unzip

    /// Make the data used to symbolically check the model
    let checkData = makeData checkSize

    /// Make the training data
    let trainData = makeData trainSize

    /// Make the validation data
    let validationInputs, validationOutputs = makeData validationSize

    /// Evaluate the model for input and coefficients
    let model (xs: DT<double>, coeffs: DT<double>) = 
        tf { return DT.Sum (xs * coeffs,axis= [| 1 |]) }

    /// Evaluate the loss function for the model w.r.t. a true output
    let loss (z: DT<double>) tgt = 
        tf { let dz = z - tgt
             return DT.Sum (dz * dz) / v (double modelSize) / v (double z.Shape.[0].Value) }

    // Gradient of the loss function w.r.t. the coefficients
    let objective (xs, y) coeffs = 
        let xnodes = batchOfVecs xs
        let ynode = batchOfScalars y
        let coeffnodes = vec coeffs
        let coffnodesBatch = batchExtend coeffnodes
        coeffnodes, loss (model (xnodes, coffnodesBatch)) ynode

    // Gradient of the objective function w.r.t. the coefficients
    let dobjective_dcoeffs (xs, y) coeffs = 
        let coeffnodes, z = objective (xs, y) coeffs
        DT.gradient z coeffnodes 
 
    let validation coeffs = 
        let _coeffnodes, z = objective (validationInputs, validationOutputs) coeffs 
        z |> DT.RunScalar

    // Note, the rate in this example is constant. Many practical optimizers use variable
    // update (rate) - often reducing - which makes them more robust to poor convergence.
    let rate = 2.0
    let step inputs (coeffs: double[]) = 
        let dz = dobjective_dcoeffs inputs coeffs 
        let coeffs = (vec coeffs - v rate * dz) |> DT.RunArray
        printfn "coeffs = %A, dz = %A, validation = %A" coeffs dz (validation coeffs)
        coeffs

    let initialCoeffs = [| for i in 0 .. modelSize - 1 -> rnd.NextDouble()  * double modelSize|]

    // Train the inputs in one batch
    let train inputs nsteps =
        initialCoeffs |> Seq.unfold (fun coeffs -> Some (coeffs, step inputs coeffs)) |> Seq.truncate nsteps |> Seq.last

    [<LiveCheck>]
    let check1 = train checkData checkSize

    [<LiveTest>]
    let test1 = train trainData 10

    let learnedCoeffs = train trainData 200
     // [|1.017181246; 2.039034327; 2.968580146; 3.99544071; 4.935430581;
     //   5.988228378; 7.030374908; 8.013975714; 9.020138699; 9.98575733|]

    validation trueCoeffs
    validation learnedCoeffs
    //   [|0.007351991009; 1.004220712; 2.002591797; 3.018333918; 3.996983572; 4.981999364; 5.986054734; 7.005387338; 8.005461854; 8.991150034|]

module NeuralTransferFragments =
    let input = matrix4 [ for i in 0 .. 9 -> [ for j in 1 .. 40 -> [ for k in 1 .. 40 -> [ for m in 0 .. 2 -> double (i+j+k+m) ]]]]
    let name = "a"
    let instance_norm (input, name) =
        tf { use _ = DT.WithScope(name + "/instance_norm")
             let mu, sigma_sq = DT.Moments (input, axes=[0;1])
             let shift = DT.Variable (v 0.0, name + "/shift")
             let scale = DT.Variable (v 1.0, name + "/scale")
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift }

    let friendly4D (d : 'T[,,,]) =
        [| for i in 0..Array4D.length1 d - 1 -> [| for j in 0..Array4D.length2 d - 1 -> [| for k in 0..Array4D.length3 d - 1 -> [| for m in 0..Array4D.length4 d - 1 -> d.[i,j,k,m]  |]|]|]|]
        |> array2D |> Array2D.map array2D

    instance_norm (input, name) |> DT.RunArray4D |> friendly4D

    let out_channels = 128
    let filter_size = 7
    let conv_init_vars (out_channels:int, filter_size:int, is_transpose: bool, name) =
        let weights_shape = 
            if is_transpose then
                shape [ filter_size; filter_size; out_channels; -1 ]
            else
                shape [ filter_size; filter_size; -1; out_channels ]
        tf { let truncatedNormal = DT.TruncatedNormal(weights_shape)
             return DT.Variable (truncatedNormal * v 0.1, name + "/weights") }

    let is_relu = 1
    let stride = 1
    let conv_layer (input, out_channels, filter_size, stride, is_relu, name) = 
        tf { let filters = conv_init_vars (out_channels, filter_size, false, name)
             let x = DT.Conv2D (input, filters, stride=stride)
             let x = instance_norm (x, name)
             if is_relu then 
                 return DT.Relu x 
             else 
                 return x }

    conv_layer (input, out_channels, filter_size, 1, true, "layer")  |> DT.RunArray4D |> friendly4D
    //(fun input -> conv_layer (input, out_channels, filter_size, 1, true, "layer")) |> DT.gradient |> apply input |> DT.RunArray4D |> friendly4D

    let residual_block (input, filter_size, name) = 
        tf { let tmp = conv_layer(input, 128, filter_size, 1, true, name + "_c1")
             return input + conv_layer(tmp, 128, filter_size, 1, false, name + "_c2") }

    let conv2D_transpose (input, filter, stride) = 
        tf { return DT.Conv2DBackpropInput(filter, input, stride, padding = "SAME") }
  
    let conv_transpose_layer (input: DT<double>, num_filters, filter_size, stride, name) =
        tf { let filters = conv_init_vars (num_filters, filter_size, true, name)
             return DT.Relu (instance_norm (conv2D_transpose (input, filters, stride), name))
           }

    let to_pixel_value (input: DT<double>) = 
        tf { return tanh input * v 150.0 + (v 255.0 / v 2.0) }

    // The style-transfer tf

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         return x }
    |> DT.RunArray4D |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         return x }
    |> DT.RunArray4D |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         return x }
    |> DT.RunArray4D |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         let x = residual_block (x, 3, "resid1")
         return x }
    |> DT.RunArray4D |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         let x = residual_block (x, 3, "resid1")
         let x = residual_block (x, 3, "resid2")
         let x = residual_block (x, 3, "resid3")
         let x = residual_block (x, 3, "resid4")
         let x = residual_block (x, 3, "resid5")
         return x }
    |> DT.RunArray4D |> friendly4D


    let t1 = 
        tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
             let x = conv_layer (x, 64, 3, 2, true, "conv2")
             let x = conv_layer (x, 128, 3, 2, true, "conv3")
             let x = residual_block (x, 3, "resid1")
             let x = residual_block (x, 3, "resid2")
             let x = residual_block (x, 3, "resid3")
             let x = residual_block (x, 3, "resid4")
             let x = residual_block (x, 3, "resid5")
             return x }

    let t2 = 
        tf { return conv_transpose_layer (t1, 64, 3, 2, "conv_t1") }
        |> DT.RunArray4D |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         let x = residual_block (x, 3, "resid1")
         let x = residual_block (x, 3, "resid2")
         let x = residual_block (x, 3, "resid3")
         let x = residual_block (x, 3, "resid4")
         let x = residual_block (x, 3, "resid5")
         let x = conv_transpose_layer (x, 64, 3, 2, "conv_t1") // TODO: check fails
         return x }
    |> DT.RunArray4D |> friendly4D

(*
module TensorFlow_PDE_Example = 
    u_init = np.zeros([N, N], dtype=np.float32)
    ut_init = np.zeros([N, N], dtype=np.float32)

    # Some rain drops hit a pond at random points
    for n in range(40):
      a,b = np.random.randint(0, N, 2)
      u_init[a,b] = np.random.uniform()

    DisplayArray(u_init, rng=[-0.1, 0.1])

    # Parameters:
    # eps -- time resolution
    # damping -- wave damping
    eps = tf.placeholder(tf.float32, shape=())
    damping = tf.placeholder(tf.float32, shape=())

    # Create variables for simulation state
    U  = tf.Variable(u_init)
    Ut = tf.Variable(ut_init)

    # Discretized PDE update rules
    U_ = U + eps * Ut
    Ut_ = Ut + eps * (laplace(U) - damping * Ut)

    # Operation to update the state
    step = tf.group(
      U.assign(U_),
      Ut.assign(Ut_))
    # Initialize state to initial conditions
    tf.global_variables_initializer().run()

    # Run 1000 steps of PDE
    for i in range(1000):
      # Step simulation
      step.run({eps: 0.03, damping: 0.04})
      DisplayArray(U.eval(), rng=[-0.1, 0.1])
*)

(*
(fun input -> 
        tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
             let x = conv_layer (x, 64, 3, 2, true, "conv2")
             let x = conv_layer (x, 128, 3, 2, true, "conv3")
             let x = residual_block (x, 3, "resid1")
             let x = residual_block (x, 3, "resid2")
             let x = residual_block (x, 3, "resid3")
             let x = residual_block (x, 3, "resid4")
             let x = residual_block (x, 3, "resid5")
             let x = conv_transpose_layer (x, 64, 3, 2, "conv_t1") // TODO: check fails
             return x })
    |> DT.Diff |> apply input |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

//2019-01-24 17:21:17.095544: F tensorflow/core/grappler/costs/op_level_cost_estimator.cc:577] Check failed: iz == filter_shape.dim(in_channel_index).size() (64 vs. 128)
    *)


(*
let x = conv_layer (x, 3, 9, 1, false, "conv_t3")
             let x = to_pixel_value x
             let x = DT.ClipByValue (x, v 0.0, v 255.0)
             return x }
*)

 (*
    let d = array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |]
    let shape = shape [ d.GetLength(0); d.GetLength(1)  ]
    let graph = new TFGraph()
    let session= new TFSession(graph)
    let n1 = graph.Const(new TFTensor(d))
    let res1 = session.Run( [|  |], [| |], [| n1 |])
*)

    //graph.Const("a", "b")
    //graph.Variable( .MakeName("a", "b")

    //TFTensor(TFDataType.Double, [| 2;2|], )
    //(TFDataType.Double, [| 2;2|])
    //TFOutput
    //graph.Inpu

