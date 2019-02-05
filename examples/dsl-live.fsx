// Build the Debug 'TensorFlow.FSharp.Tests' before using this

#I __SOURCE_DIRECTORY__
#r "netstandard"
#r "../tests/bin/Debug/net472/TensorFlow.FSharp.dll"

#nowarn "49"

open System
open TensorFlow.FSharp
open TensorFlow.FSharp.DSL

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

#if !LIVECHECKING
fsi.AddPrintTransformer(DT.PrintTransform)
#endif

module PlayWithTF = 
    
    let f x = 
       x * x + v 4.0 * x 

    // Get the derivative of the function. This computes "x*2 + 4.0"
    let df x = DT.diff f x  

    // Run the function
    f (v 3.0) 

    // Most operators need explicit evaluation even in interactive mode
    df (v 3.0) 

    // returns 6.0 + 4.0 = 10.0
    df (v 3.0) |> DT.Eval

    v 2.0 + v 1.0 

    // You can wrap in "tf { return ... }" if you like
    tf { return v 2.0 + v 1.0 }
          
    // This can be handy for locals
    tf { let x = (vec [1.0; 2.0] + vec [1.0;2.0] ) 
         return x + x }   
  
    // Now show that adding vectors of the wrong sizes gives a static error
    // Adding a vector and a matrix gives an error
    vec [1.0; 2.0] + matrix [ [1.0; 2.0] ]
         
    // Math-multiplying matrices of the wrong sizes gives an error
    tf { let matrix1 = 
             matrix [ [1.0; 2.0]
                      [1.0; 2.0]
                      [1.0; 2.0] ]
         let matrix2 =  
             matrix [ [1.0; 2.0]
                      [1.0; 2.0]
                      [1.0; 2.0] ]
         return matrix1 *! matrix2 }
    
    // What about functions?  
    let f3 (x:int) = 
        let m = matrix [ [1.0; 2.0; 4.0]; [1.0; 2.0; 6.0] ]
        m
         
    // Functions are only checked when there is a [<LiveCheck>] exercising the code path
    
    [<LiveCheck>]
    let _ = f3 4

    //let f19 = 
    //    tf { return (vec [1.0; 2.0 ] + vec [ 1.0; 4.0 ]  )  }  
    //      
    //[<LiveCheck>] 
    // let _ = f19 3   
      

module GradientAscentWithoutVariables =

    let inline sqr x = x * x

    // Define a function which will be executed using TensorFlow
    let f (xs: DT<double>) = 
        sin (v 0.5 * sqr xs.[0] - v 0.25 * sqr xs.[1] + v 3.0) * cos (v 2.0 * xs.[0] + v 1.0 - exp xs.[1])
           
    // Get the partial derivatives of the function
    let df xs =  DT.diff f xs  
        
    let rate = 0.1

    // Gradient ascent
    let step xs =   
        printfn "xs = %A" xs
        let dzx = df xs 
        // Evaluate to output values and prevent graph explosion 
        xs + v rate * dzx |> DT.Eval

    let train steps = 
        vec [ -0.3; 0.3 ] |> Seq.unfold (fun pos -> Some (pos, step pos)) |> Seq.truncate steps |> Seq.toArray

    [<LiveCheck>] 
    let _ = train 4 |> ignore 
    
#if !LIVECHECKING
    train 100 |> ignore 

#endif



module ModelExample =

    let modelSize = 10

    let checkSize = 5

#if !LIVECHECKING
    let trainSize = 500

    let validationSize = 100
#endif

    let rnd = Random()

    let noise eps = (rnd.NextDouble() - 0.5) * eps 

    /// The true function we use to generate the training data (also a linear model plus some noise)
    let trueCoeffs = [| for i in 1 .. modelSize -> double i |]

    let trueFunction (xs: double[]) = 
        Array.sum [| for i in 0 .. modelSize - 1 -> trueCoeffs.[i] * xs.[i]  |] + noise 0.5

    let makeData size = 
        [| for i in 1 .. size -> 
            let xs = [| for i in 0 .. modelSize - 1 -> rnd.NextDouble() |]
            xs, trueFunction xs |]
         
    /// Make the data used to symbolically check the model
    let checkData = makeData checkSize

#if !LIVECHECKING 
    /// Make the training data
    let trainData = makeData trainSize

    /// Make the validation data
    let validationData = makeData validationSize
#endif
 
    let prepare data = 
        let xs, y = Array.unzip data
        let xs = batchOfVecs xs
        let y = batchOfScalars y
        (xs, y)

    /// Evaluate the model for input and coefficients
    let model (xs: DT<double>, coeffs: DT<double>) = 
        DT.Sum (xs * coeffs,axis= [| 1|])
           
    let meanSquareError (z: DT<double>) tgt = 
        let dz = z - tgt 
        DT.Sum (dz * dz) / v (double modelSize) / v (double z.Shape.[0].Value) 

    /// The loss function for the model w.r.t. a true output
    let loss (xs, y) coeffs = 
        let coeffsBatch = batchExtend coeffs
        let y2 = model (xs, coeffsBatch)
        meanSquareError y y2
          
    // Gradient of the objective function w.r.t. the coefficients
    let dloss_dcoeffs inputs coeffs = 
        let z = loss inputs coeffs
        DT.gradient z coeffs 

#if !LIVECHECKING
    let validation coeffs = 
        let z = loss (prepare validationData) (vec coeffs)
        z |> DT.Eval
#endif

    // Note, the rate in this example is constant. Many practical optimizers use variable
    // update (rate) - often reducing - which makes them more robust to poor convergence.
    let rate = 2.0
     
    let step inputs (coeffs: DT<double>) = 
        let dz = dloss_dcoeffs inputs coeffs 

        // Note, force the evaluation at this step
        let coeffs = (coeffs - v rate * dz) |> DT.Eval
        printfn "coeffs = %A, dz = %A" coeffs (dz |> DT.Eval) //(validation coeffs)
        coeffs 
         
    let initialCoeffs = vec [ for i in 0 .. modelSize - 1 -> rnd.NextDouble()  * double modelSize ]
     
    // Train the inputs in one batch
    let train inputs nsteps =
        let inputs2 = prepare inputs
        initialCoeffs |> Seq.unfold (fun coeffs -> Some (coeffs, step inputs2 coeffs)) |> Seq.truncate nsteps |> Seq.last
           
    [<LiveCheck>]
    let check1 = train checkData 1 

    // What happens if we LiveCheck twice?
    //[<LiveCheck>]
    //let check2 = train checkInputs2 checkOutputs2 1 
 
#if !LIVECHECKING
    [<LiveTest>]
    let test1 = train trainData 10

    let learnedCoeffs = train trainData 200 |> DT.toArray
     // [|1.017181246; 2.039034327; 2.968580146; 3.99544071; 4.935430581;
     //   5.988228378; 7.030374908; 8.013975714; 9.020138699; 9.98575733|]

    validation trueCoeffs

    validation learnedCoeffs
#endif

module ODEs = 
    let lotka_volterra(du, u: DT<double>, p: DT<double>, t) = 
        let x, y = u.[0], u.[1]
        let α, β, δ, γ = p.[0], p.[1], p.[2], p.[3]
        let dx = α*x - β*x*y
        let dy = -δ*y + γ*x*y 
        DT.Stack [dx; dy]

    let u0 = vec [1.0; 1.0]
    let tspan = (0.0, 10.0)
    let p = vec [ 1.5; 1.0; 3.0; 1.0]
    //let prob = ODEProblem(lotka_volterra, u0, tspan, p)
    //let sol = solve(prob, Tsit5())
    //sol |> Chart.Lines 




(*
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
                Shape [| Dim filter_size; Dim filter_size; Dim out_channels; Dim.Inferred |]
            else
                Shape [| Dim filter_size; Dim filter_size; Dim.Inferred; Dim out_channels |]
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
module SimpleCases = 
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

    let f x = tf { return x * x + v 4.0 * x }
    let df x = DT.diff f x

    //df (v 3.0)
    //|> DT.RunScalar
    //|> (=) (2.0 * 3.0 + 4.0)

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

    let f3 (x: DT<_>) = tf { return vec [1.0; 2.0] * x * DT.ReverseV2 x } //[ x1*x2; 2*x2*x1 ] 
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

    let f100 (x: DT<double>) y = 
        tf { return x + y } 
    
    [<LiveCheck>]
    //let _ = f100 (vec [ 1.0; 2.0]) (vec [ 2.0; 3.0])
    let g = f100 (vec [ 1.0; 2.0]) 
    
    g (vec [ 2.0; 3.0])
     
    [<LiveCheck>]
    let _ = f100 (matrix [ [ 1.0; 2.0; 3.0 ]; [ 2.0; 3.0; 4.5] ]) (matrix [ [ 2.0; 3.0; 4.5]; [ 2.0; 3.0; 4.5] ])  
      

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

    *)