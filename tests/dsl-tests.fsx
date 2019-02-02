// Build the Debug 'TensorFlow.FSharp.Tests' before using this

#I __SOURCE_DIRECTORY__
#r "netstandard"

#r "bin/Debug/net472/TensorFlow.FSharp.dll"
#nowarn "49"

//open Argu
open System
open System.IO
open TensorFlow.FSharp
open TensorFlow.FSharp.DSL

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

let apply x f = f x

module PlayWithTF = 
    tf { return v 1.0 }
    |> DT.RunScalar

    tf { return (vec [1.0; 2.0]) }
    |> DT.RunArray

    tf { return (vec [1.0]).[0] }
    |> DT.RunScalar

    tf { return (vec [1.0; 2.0]).[1] }
    |> DT.RunScalar
   
    tf { return (vec [1.0; 2.0]).[0..0] }
    |> DT.RunArray

    tf { return v 1.0 + v 4.0 }
    |> DT.Run

    tf { return DT.ReverseV2 (vec [1.0; 2.0]) }
    |> DT.RunArray

    let f x = tf { return x *. x + v 4.0 *. x }
    let df x = DT.diff f x

    df (v 3.0)
    |> DT.RunScalar
    |> (=) (2.0 * 3.0 + 4.0)

    tf { return vec [1.0; 2.0] + v 4.0 }
    |> DT.RunArray

    tf { return sum (vec [1.0; 2.0] + v 4.0) }
    |> DT.RunScalar

    let f2 x = tf { return sum (vec [1.0; 2.0] *. x *. x) }
    let df2 x = DT.grad f2 x

    f2 (vec [1.0; 2.0])
    |> DT.RunScalar

    df2 (vec [1.0; 2.0])
    |> DT.RunArray

    //let x = vec [1.0; 2.0]
    //(DT.Stack [| x.[1]; x.[0] |]).Shape
    //|> DT.RunArray

    let f3 (x: DT<_>) = tf { return vec [1.0; 2.0] *. x *. DT.ReverseV2 x } //[ x1*x2; 2*x2*x1 ] 
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

    tf { return var (vec [ 1.0 ]) "x" *. var (vec [ 1.0 ]) "x" + v 4.0 }
    |> DT.RunArray

    tf { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> fun dt -> DT.RunArray(dt, ["hey", upcast (vec [2.0])])
       // Gives 6.0

module GradientAscentWithoutVariables =
    // Define a function which will be executed using TensorFlow
    let f (x: DT<double>, y: DT<double>) = 
        tf { return sin (v 0.5 *. x *. x - v 0.25 *. y *. y + v 3.0) *. cos (v 2.0 *. x + v 1.0 - exp y) }

    // Get the partial derivatives of the scalar function
    // computes [ 2*x1*x3 + x3*x3; 3*x2*x2; 2*x3*x1 + x1*x1 ]
    let df (x, y) = 
        let dfs = DT.gradients (f (x,y))  [| x; y |] 
        (dfs.[0], dfs.[1])

    let rate = 0.1

    // Gradient ascent
    let step (x, y) = 
        let nodes = df (v x, v y) 
        let dzx, dzy = nodes |> DT.RunScalarPair 
        printfn "size = %f" (sqrt (dzx*dzx + dzy*dzy))
        (x + rate * dzx, y + rate * dzy)

    (-0.3, 0.3) |> Seq.unfold (fun pos -> Some (pos, step pos)) |> Seq.truncate 200 |> Seq.toArray

(*
module GradientDescentWithVariables =
    // Define a function which will be executed using TensorFlow
    let f (x: DT<double>, y: DT<double>) = 
        tf { return sin (v 0.5 *. x *. x - v 0.25 *. y *. y + v 3.0) *. cos (v 2.0 *. x + v 1.0 - exp y) }

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
    let trainSize = 200
    let rnd = Random()

    /// The true function we use to generate the training data (also a linear model)
    let trueCoeffs = [| for i in 0 .. modelSize - 1 -> double i |]
    let trueFunction (xs: double[]) = Array.sum [| for i in 0 .. modelSize - 1 -> trueCoeffs.[i] * xs.[i] |]

    /// Make the training data
    let trainingInputs, trainingOutputs = 
        [| for i in 1 .. trainSize -> 
            let xs = [| for i in 0 .. modelSize - 1 -> rnd.NextDouble() |]
            xs, trueFunction xs |]
        |> Array.unzip

    /// Evaluate the model for input and coefficients
    let model (xs: DT<double>, coeffs: DT<double>) = 
        tf { return DT.Sum (xs *. coeffs,axis= [| 1 |]) }

    /// Evaluate the loss function for the model w.r.t. a true output
    let loss (z: DT<double>) tgt = 
        tf { let dz = z - tgt in return DT.Sum (dz *. dz) }

    // Gradient of the loss function w.r.t. the coefficients
    let dloss_dcoeffs (xs, y) coeffs = 
        let xnodes = batchVec xs
        let ynode = batchScalar y
        let coeffnodes = vec coeffs
        let coffnodesBatch = batchExtend coeffnodes
        let z = loss (model (xnodes, coffnodesBatch)) ynode
        DT.gradient z coeffnodes 

    let rate = 0.001
    let step inputs (coeffs: double[]) = 
        let dz = dloss_dcoeffs inputs coeffs 
        let coeffs = (vec coeffs - v rate *. dz) |> DT.RunArray
        printfn "coeffs = %A, dz = %A" coeffs dz
        coeffs

    let initialCoeffs = [| for i in 0 .. modelSize - 1 -> rnd.NextDouble()  * double modelSize|]

    // Train the inputs in one batch
    let train inputs =
        initialCoeffs |> Seq.unfold (fun coeffs -> Some (coeffs, step inputs coeffs)) |> Seq.truncate 200 |> Seq.last

    train (trainingInputs, trainingOutputs)
    //   [|0.007351991009; 1.004220712; 2.002591797; 3.018333918; 3.996983572; 4.981999364; 5.986054734; 7.005387338; 8.005461854; 8.991150034|]

(*
    let mutable weights1 = vec [ 3.0; 4.0; 5.0 ]
	let mutable weights2 = vec [ 0.2; -0.3; 0.4 ]

    let model (input: string) = 
	    tf { weights1 * weights2 * v (double input.Length) } 

    let loss input = 
	    tf { return DT.Sum (model input) }
		
    let df input = DT.gradients (loss input) [ weights1; weights2 ]   

	// An unsophisticated optimization loop
	for input in [ "hello"; "goodbye" ] do
	    let [| dweights1; dweights2 |] = df input
	    if dweights1.Length + dweights2.Length < 0.
		weights1 <- weights1 + dweights1
	    weights2 <- weights2 + dweights2
*)

module NeuralTransferFragments =
    let input = matrix4 [ for i in 0 .. 9 -> [ for j in 1 .. 40 -> [ for k in 1 .. 40 -> [ for m in 0 .. 2 -> double (i+j+k+m) ]]]]
    let name = "a"
    let instance_norm (input, name) =
        tf { use _ = TF.WithScope(name + "/instance_norm")
             let mu, sigma_sq = DT.Moments (input, axes=[0;1])
             let shift = DT.Variable (v 0.0, name + "/shift")
             let scale = DT.Variable (v 1.0, name + "/scale")
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale *. normalized + shift }

    let friendly4D (d : 'T[,,,]) =
        [| for i in 0..Array4D.length1 d - 1 -> [| for j in 0..Array4D.length2 d - 1 -> [| for k in 0..Array4D.length3 d - 1 -> [| for m in 0..Array4D.length4 d - 1 -> d.[i,j,k,m]  |]|]|]|]
        |> array2D |> Array2D.map array2D

    instance_norm (input, name) |> DT.RunArray4D |> friendly4D

    let out_channels = 128
    let filter_size = 7
    let conv_init_vars (out_channels:int, filter_size:int, is_transpose: bool, name) =
        let weights_shape = 
            if is_transpose then
                Shape [| Dim filter_size; Dim filter_size; Dim out_channels; Dim.Inferred |]
            else
                Shape [| Dim filter_size; Dim filter_size; Dim.Inferred; Dim out_channels |]
        tf { let truncatedNormal = DT.TruncatedNormal(weights_shape)
             return DT.Variable (truncatedNormal *. v 0.1, name + "/weights") }

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
        tf { return tanh input *. v 150.0 + (v 255.0 / v 2.0) }

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
    (TFDataType.Double, [| 2;2|])
    TFOutput
    //graph.Inpu

