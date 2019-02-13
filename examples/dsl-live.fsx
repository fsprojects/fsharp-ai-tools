  // Build the Debug 'TensorFlow.FSharp.Tests' before using this

#I __SOURCE_DIRECTORY__
#r "netstandard"
#r "../tests/bin/Debug/net461/TensorFlow.FSharp.Proto.dll"
#r "../tests/bin/Debug/net461/TensorFlow.FSharp.dll"
#r "FSharp.Compiler.Interactive.Settings.dll"
#nowarn "49"

open System
open TensorFlow.FSharp
open TensorFlow.FSharp.DSL
open Microsoft.FSharp.Compiler.Interactive.Settings

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

fsi.AddPrintTransformer(DT.PrintTransform)

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

    // You can wrap in "fm { return ... }" if you like
    fm { return v 2.0 + v 1.0 }
     
    let test1() = 
        vec [1.0; 2.0] + vec [1.0; 2.0]
    
    // Math-multiplying matrices of the wrong sizes gives an error
    let test2() = 
        let matrix1 = 
            matrix [ [1.0; 2.0]
                     [1.0; 2.0]
                     [1.0; 2.0] ]
        let matrix2 =  
            matrix [ [1.0; 2.0]
                     [1.0; 2.0] ] 
        matrix1 *! matrix2 
    
    // Things are only checked when there is a [<LiveCheck>] exercising the code path
    
    [<LiveCheck>]
    let check1 = test1() |> ignore

    [<LiveCheck>]
    let check2 = test2() |> ignore

module GradientDescent =

    // Note, the rate in this example is constant. Many practical optimizers use variable
    // update (rate) - often reducing - which makes them more robust to poor convergence.
    let rate = 0.005

    // Gradient descent
    let step f xs =   
        // Get the partial derivatives of the function
        let df xs =  DT.diff f xs  
        printfn "xs = %A" xs
        let dzx = df xs 
        // Evaluate to output values 
        xs - v rate * dzx |> DT.Eval

    let train f initial steps = 
        initial |> Seq.unfold (fun pos -> Some (pos, step f pos)) |> Seq.truncate steps 

module GradientDescentExample =

    // A numeric function of two parameters, returning a scalar, see
    // https://en.wikipedia.org/wiki/Gradient_descent
    let f (xs: Vec) = 
        sin (v 0.5 * sqr xs.[0] - v 0.25 * sqr xs.[1] + v 3.0) * -cos (v 2.0 * xs.[0] + v 1.0 - exp xs.[1])

    // Pass this Define a numeric function of two parameters, returning a scalar
    let train numSteps = GradientDescent.train f (vec [ -0.3; 0.3 ]) numSteps

    [<LiveCheck>] 
    let check1 = train 4 |> Seq.last 
    
    let results = train 200 |> Seq.last

module ModelExample =

    let modelSize = 10

    let checkSize = 5

    let trainSize = 500

    let validationSize = 100

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

    /// Make the training data
    let trainData = makeData trainSize

    /// Make the validation data
    let validationData = makeData validationSize
 
    let prepare data = 
        let xs, y = Array.unzip data
        let xs = batchOfVecs xs
        let y = batchOfScalars y
        (xs, y)

    /// Evaluate the model for input and coefficients
    let model (xs: Vec, coeffs: DT<double>) = 
        DT.Sum (xs * coeffs,axis= [| 1 |])
           
    let meanSquareError (z: DT<double>) tgt = 
        let dz = z - tgt 
        DT.Sum (dz * dz) / v (double modelSize) / v (double z.Shape.[0].Value) 

    /// The loss function for the model w.r.t. a true output
    let loss (xs, y) coeffs = 
        let y2 = model (xs, batchExtend coeffs)
        meanSquareError y y2
          
    let validation coeffs = 
        let z = loss (prepare validationData) (vec coeffs)
        z |> DT.Eval

    let train inputs steps =
        let initialCoeffs = vec [ for i in 0 .. modelSize - 1 -> rnd.NextDouble()  * double modelSize ]
        let inputs = prepare inputs
        GradientDescent.train (loss inputs) initialCoeffs steps
           
    [<LiveCheck>]
    let check1 = train checkData 1  |> Seq.last

    let learnedCoeffs = train trainData 200 |> Seq.last |> DT.toArray
         // [|1.017181246; 2.039034327; 2.968580146; 3.99544071; 4.935430581;
         //   5.988228378; 7.030374908; 8.013975714; 9.020138699; 9.98575733|]

    validation trueCoeffs

    validation learnedCoeffs

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


module NeuralTransferFragments =

    let name = "a"

    let instance_norm (input, name) =
        fm { use _ = DT.WithScope(name + "/instance_norm")
             let mu, sigma_sq = DT.Moments (input, axes=[0;1])
             let shift = DT.Variable (v 0.0, name + "/shift")
             let scale = DT.Variable (v 1.0, name + "/scale")
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift }

    //instance_norm (input, name) |> DT.Eval |> DT.toArray4D |> friendly4D

    let out_channels = 128
    let filter_size = 7
    let conv_init_vars (out_channels:int, filter_size:int, is_transpose: bool, name) =
        let weights_shape = 
            if is_transpose then
                Shape.NoFlex [| Dim.Known filter_size; Dim.Known filter_size; Dim.Known out_channels; Dim.Inferred |]
            else
                Shape.NoFlex [| Dim.Known filter_size; Dim.Known filter_size; Dim.Inferred; Dim.Known out_channels |]
        fm { let truncatedNormal = DT.TruncatedNormal(weights_shape)
             return DT.Variable (truncatedNormal * v 0.1, name + "/weights") }

    let is_relu = 1
    let stride = 1
    let conv_layer (input, out_channels, filter_size, stride, is_relu, name) = 
        fm { let filters = conv_init_vars (out_channels, filter_size, false, name)
             let x = DT.Conv2D (input, filters, stride=stride)
             let x = instance_norm (x, name)
             if is_relu then 
                 return DT.Relu x 
             else 
                 return x }

    //conv_layer (input, out_channels, filter_size, 1, true, "layer")  |> DT.Eval |> DT.toArray4D |> friendly4D
    //(fun input -> conv_layer (input, out_channels, filter_size, 1, true, "layer")) |> DT.gradient |> apply input |> DT.Eval |> DT.toArray4D |> friendly4D

    let conv2D_transpose (input, filter, stride) = 
        fm { return DT.Conv2DBackpropInput(filter, input, stride, padding = "SAME") }
  
    let conv_transpose_layer (input: DT<double>, num_filters, filter_size, stride, name) =
        fm { let filters = conv_init_vars (num_filters, filter_size, true, name)
             return DT.Relu (instance_norm (conv2D_transpose (input, filters, stride), name))}

    let to_pixel_value (input: DT<double>) = 
        tanh input * v 150.0 + (v 255.0 / v 2.0) 

    let residual_block (input, filter_size, name) = 
        fm { let tmp = conv_layer(input, 128, filter_size, 1, true, name + "_c1")
             return input + conv_layer(tmp, 128, filter_size, 1, false, name + "_c2") }

    // The style-transfer neural network
    let style_transfer input = 
        fm { let x = conv_layer (input, 32, 9, 1, true, "conv1")
             let x = conv_layer (x, 64, 3, 2, true, "conv2")
             let x = conv_layer (x, 128, 3, 2, true, "conv3")
             let x = residual_block (x, 3, "resid1")
             let x = residual_block (x, 3, "resid2")
             let x = residual_block (x, 3, "resid3")
             let x = residual_block (x, 3, "resid4")
             let x = residual_block (x, 3, "resid5")
             let x = conv_transpose_layer (x, 64, 3, 2, "conv_t1") 
             let x = conv_transpose_layer (x, 32, 3, 2, "conv_t2")
             let x = conv_layer (x, 3, 9, 1, false, "conv_t3")
             let x = to_pixel_value x
             let x = DT.ClipByValue (x, v 0.0, v 255.0)
             return x }
        |> DT.Eval 




    let dummyImages() = DT.Stack [ for i in 1 .. 10 -> DT.Dummy [ Dim.Known 474;  Dim.Known 712; Dim.Known 3] ]

    [<LiveCheck>]
    let test() = style_transfer (dummyImages())





(*

let to_pixel_value (input: DT<double>) = 
    fm { return tanh input * v 150.0 + (v 255.0 / v 2.0) }

fm { let x = conv_layer (input, 32, 9, 1, true, "conv1")
     return x }
|> DT.Eval |> DT.toArray4D |> friendly4D

fm { let x = conv_layer (input, 32, 9, 1, true, "conv1")
     let x = conv_layer (x, 64, 3, 2, true, "conv2")
     return x }
|> DT.Eval |> DT.toArray4D |> friendly4D

fm { let x = conv_layer (input, 32, 9, 1, true, "conv1")
     let x = conv_layer (x, 64, 3, 2, true, "conv2")
     let x = conv_layer (x, 128, 3, 2, true, "conv3")
     return x }
|> DT.Eval |> DT.toArray4D |> friendly4D

fm { let x = conv_layer (input, 32, 9, 1, true, "conv1")
     let x = conv_layer (x, 64, 3, 2, true, "conv2")
     let x = conv_layer (x, 128, 3, 2, true, "conv3")
     let x = residual_block (x, 3, "resid1")
     return x }
|> DT.Eval |> DT.toArray4D |> friendly4D

fm { let x = conv_layer (input, 32, 9, 1, true, "conv1")
     let x = conv_layer (x, 64, 3, 2, true, "conv2")
     let x = conv_layer (x, 128, 3, 2, true, "conv3")
     let x = residual_block (x, 3, "resid1")
     let x = residual_block (x, 3, "resid2")
     let x = residual_block (x, 3, "resid3")
     let x = residual_block (x, 3, "resid4")
     let x = residual_block (x, 3, "resid5")
     return x }
|> DT.Eval |> DT.toArray4D |> friendly4D


let t1 = 
    fm { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         let x = residual_block (x, 3, "resid1")
         let x = residual_block (x, 3, "resid2")
         let x = residual_block (x, 3, "resid3")
         let x = residual_block (x, 3, "resid4")
         let x = residual_block (x, 3, "resid5")
         return x }

let t2 = 
    fm { return conv_transpose_layer (t1, 64, 3, 2, "conv_t1") }
    |> DT.Eval |> DT.toArray4D |> friendly4D



module SimpleCases = 
    fm { return (vec [1.0; 2.0]) }
    |> DT.RunArray

    fm { return (vec [1.0]).[0] }
    |> DT.RunScalar

    fm { return (vec [1.0; 2.0]).[1] }
    |> DT.RunScalar
   
    fm { return (vec [1.0; 2.0]).[0..0] }
    |> DT.RunArray

    fm { return v 1.0 + v 4.0 }
    |> DT.Run

    fm { return DT.ReverseV2 (vec [1.0; 2.0]) }
    |> DT.RunArray

    let f x = fm { return x * x + v 4.0 * x }
    let df x = DT.diff f x

    //df (v 3.0)
    //|> DT.RunScalar
    //|> (=) (2.0 * 3.0 + 4.0)

    fm { return vec [1.0; 2.0] + v 4.0 }
    |> DT.RunArray

    fm { return sum (vec [1.0; 2.0] + v 4.0) }
    |> DT.RunScalar

    let f2 x = fm { return sum (vec [1.0; 2.0] * x * x) }
    let df2 x = DT.grad f2 x
       
    f2 (vec [1.0; 2.0])
    |> DT.RunScalar

    df2 (vec [1.0; 2.0])
    |> DT.RunArray
      
    //let x = vec [1.0; 2.0]
    //(DT.Stack [| x.[1]; x.[0] |]).Shape
    //|> DT.RunArray

    let f3 (x: DT<_>) = fm { return vec [1.0; 2.0] * x * DT.ReverseV2 x } //[ x1*x2; 2*x2*x1 ] 
    let df3 x = DT.jacobian f3 x // [ [ x2; x1 ]; [2*x2; 2*x1 ] ]  
    let expected (x1, x2) = array2D [| [| x2; x1 |]; [| 2.0*x2; 2.0*x1 |] |]  

    f3 (vec [1.0; 2.0])
    |> DT.RunArray

    df3 (vec [1.0; 2.0])
    |> DT.RunArray2D
    |> (=) (expected (1.0, 2.0))
    // expect 

    fm { use _ = DT.WithScope("foo")
         return vec [1.0; 2.0] + v 4.0 }
   // |> DT.Diff
    |> DT.RunArray

    fm { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] }
    |> DT.RunArray2D

    let f100 (x: DT<double>) y = 
        fm { return x + y } 
    
    [<LiveCheck>]
    //let _ = f100 (vec [ 1.0; 2.0]) (vec [ 2.0; 3.0])
    let g = f100 (vec [ 1.0; 2.0]) 
    
    g (vec [ 2.0; 3.0])
     
    [<LiveCheck>]
    let _ = f100 (matrix [ [ 1.0; 2.0; 3.0 ]; [ 2.0; 3.0; 4.5] ]) (matrix [ [ 2.0; 3.0; 4.5]; [ 2.0; 3.0; 4.5] ])  
      

    fm { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] + v 4.0 }
    |> DT.RunArray2D

    fm { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> DT.RunArray

    let var v nm = DT.Variable (v, name=nm)
    
    // Specifying values for variables in the graph
    fm { return var (vec [ 1.0 ]) "x" + v 4.0 }
    |> fun dt -> DT.RunArray(dt, ["x", (vec [2.0] :> _)] )
     
    fm { return var (vec [ 1.0 ]) "x" * var (vec [ 1.0 ]) "x" + v 4.0 }
    |> DT.RunArray

    fm { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> fun dt -> DT.RunArray(dt, ["hey", upcast (vec [2.0])])
       // Gives 6.0

module GradientDescentWithVariables =
    // Define a function which will be executed using TensorFlow
    let f (x: DT<double>, y: DT<double>) = 
        fm { return sin (v 0.5 * x * x - v 0.25 * y * y + v 3.0) * cos (v 2.0 * x + v 1.0 - exp y) }

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
        fm { let x = conv_layer (input, 32, 9, 1, true, "conv1")
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

    