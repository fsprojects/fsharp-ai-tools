// Build the Debug 'FSAI.Tools.Tests' before using this

module FM.Tests

open NUnit.Framework

open System
open FSAI.Tools
open FSAI.Tools.FM

[<Test>]
let check64() = 
    if not System.Environment.Is64BitProcess then exit 100

let shouldEqual (msg: string) (v1: 'T) (v2: 'T) = 
    if v1 <> v2 then 
        Assert.Fail(sprintf "fail %s: expected %A, got %A" msg v1 v2)

[<Test>]
let ``basic checks 1``() = 
    v 1.0f |> fm.to_scalar |> shouldEqual "wcevwo1" 1.0f

    v 1.0 |> fm.to_scalar |> shouldEqual "wcevwo2" 1.0

    (v 1.0 + v 3.0)  |> fm.to_scalar |> shouldEqual "wcevwo3" 4.0

[<Test>]
let ``basic checks 2``() = 

    node { return v 1.0 }
    |> fm.to_scalar |> shouldEqual "wcevwo4" 1.0
    
    node { return v 1.0 }
    |> fm.eval
    |> fm.to_scalar |> shouldEqual "wcevwo5" 1.0

[<Test>]
let ``basic checks vec``() = 
    node { return (vec [1.0; 2.0]) }
    |> fm.eval
    |> fm.toArray 
    |> shouldEqual "wcevwo6" [| 1.0; 2.0 |]

[<Test>]
let ``basic checks Pack``() = 
    node { return (fm.pack [v 1.0; v 2.0]) }
    |> fm.eval
    |> fm.toArray 
    |> shouldEqual "wcevwo7" [| 1.0; 2.0 |]

    // Gives a shape error of course
    // node { return (vec [1.0; 2.0] + matrix [ [1.0; 2.0] ]) }
    // |> fm.eval

[<Test>]
let ``basic checks indexers``() = 
    // Test indexer notation
    node { return (vec [1.0]).[0] }
    |> fm.eval
    |> fm.to_scalar 
    |> shouldEqual "wcevwo8" 1.0

    // Test indexer notation 
    node { return (vec [1.0; 2.0]).[1] }
    |> fm.eval
    |> fm.to_scalar
    |> shouldEqual "wcevwo9" 2.0
   
[<Test>]
let ``basic checks slicing``() = 
    // Test slicing notation
    node { return (vec [1.0; 2.0]).[0..0] }
    |> fm.eval
    |> fm.toArray 
    |> shouldEqual "wcevwo10" [| 1.0 |]

[<Test>]
let ``basic checks Reverse``() = 
    node { return fm.reverse (vec [1.0; 2.0]) }
    |> fm.eval
    |> fm.toArray 
    |> shouldEqual "wcevwo11" [| 2.0; 1.0 |]

[<Test>]
let ``basic checks Zero``() = 
    node { return fm.Zero }
    |> fm.eval
    |> fm.to_scalar 
    |> shouldEqual "wcevwo11" 0.0

[<Test>]
let ``basic checks Zero assert_shape``() = 
    node { return fm.Zero |> fm.assert_shape (shape [ 2; 2 ]) }
    |> fm.eval
    |> fm.toArray2D
    |> shouldEqual "wcevwo11" (array2D [ [ 0.0; 0.0 ]; [0.0; 0.0]])

[<Test>]
let ``basic checks diff``() = 
    let f x = node { return x * x + v 4.0 * x }
    let df x = fm.diff f x

    df (v 3.0)
    |> fm.eval
    |> fm.to_scalar
    |> shouldEqual "wcevwo12" (2.0 * 3.0 + 4.0)


[<Test>]
let ``basic checks scalar addition, multiplication, subtraction, division``() = 
    let f (x: DT<_>) = v 1.0 * x * x * v 1.0 / v 1.0 + v 4.0 * x + v 5.0 - v 1.0 + v 1.0
    let df x = fm.diff f x

    df (v 3.0)
    |> fm.eval
    |> fm.to_scalar
    |> shouldEqual "wcevwo12" (2.0 * 3.0 + 4.0)


[<Test>]
let ``basic checks broadcast``() = 
    node { return vec [1.0; 2.0] + v 4.0 }
    |> fm.eval
    |> fm.toArray
    |> shouldEqual "wcevwo13" [| 5.0; 6.0 |]

[<Test>]
let ``basic checks sum plus broadcast``() = 
    node { return sum (vec [1.0; 2.0] + v 4.0) }
    |> fm.eval
    |> fm.to_scalar
    |> shouldEqual "wcevwo14" 11.0

[<Test>]
let ``basic checks grad``() = 
    let f2 x = node { return sum (vec [1.0; 2.0] * x * x) }
    let df2 x = fm.grad f2 x

    f2 (vec [1.0; 2.0])
    |> fm.eval
    |> fm.to_scalar
    |> shouldEqual "wcevwo15" 9.0

//[<Test>]
//let ``basic checks jacobian``() = 
//    let f3e (x: double[]) = [| x.[0]*x.[1]; 2.0*x.[1]*x.[0] |] 
//    let f3 (x: DT<_>) = vec [1.0; 2.0] * x * fm.reverse x //[ x1*x2; 2*x2*x1 ] 
//    let df3 x = fm.jacobian f3 x // [ [ x2; x1 ]; [2*x2; 2*x1 ] ]  
//    let expected (x1, x2) = array2D [| [| x2; x1 |]; [| 2.0*x2; 2.0*x1 |] |]  

//    f3 (vec [1.0; 2.0])
//    |> fm.eval
//    |> fm.toArray
//    |> shouldEqual "wcevwo16" (f3e [| 1.0; 2.0 |])

//    df3 (vec [1.0; 2.0])
//    |> fm.eval
//    |> fm.toArray2D
//    |> shouldEqual "wcevwo16" (expected (1.0, 2.0))
//    // expect 
   
#if SLICE_GRAD
[<Test>]
let ``basic checks grad 2``() = 
    let f3e (x: double[]) = x.[0]*x.[1] + 2.0*x.[1]*x.[0]
    let f3 (x: DT<_>) = x.[0]*x.[1] + v 2.0*x.[1]*x.[0]
    let g3 x = fm.grad f3 x // [ [ x2; x1 ]; [2*x2; 2*x1 ] ]  
    let expected (x0, x1) = [| 3.0*x1; 3.0*x0  |]

    f3 (vec [1.0; 2.0])
    |> fm.eval
    |> fm.to_scalar
    |> shouldEqual "wcevwo17" (f3e [| 1.0; 2.0 |])

    g3 (vec [1.0; 2.0])
    |> fm.eval
    |> fm.toArray
    |> shouldEqual "wcevwo18" (expected (1.0, 2.0))
#endif

(*

[<Test>]
let ``basic checks hessian``() = 
    let f3e (x: double[]) = x.[0]*x.[1] + 2.0*x.[1]*x.[0]
    let f3 (x: DT<_>) = x.[0]*x.[1] + v 2.0*x.[1]*x.[0]
    let hf3 = fm.hessian f3 
    let expected (x0, x1) = array2D [| [| 3.0*x1; 3.0*x0  |] |]

    f3 (vec [1.0; 2.0])
    |> fm.eval
    |> fm.to_scalar
    |> shouldEqual "wcevwo17" (f3e [| 1.0; 2.0 |])

    hf3 (vec [1.0; 2.0])
    |> fm.eval
    |> fm.toArray2D
    |> shouldEqual "wcevwo18" (expected (1.0, 2.0))

    node { use _ = fm.with_scope("foo")
           return vec [1.0; 2.0] + v 4.0 }
   // |> fm.Diff
    |> fm.eval

    node { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] }
    |> fm.eval

    node { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] + v 4.0 }
    |> fm.eval

    node { return fm.variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> fm.eval

    node { return fm.variable (1.0, name="hey") + 4.0 }
    |> fm.eval

    let var v nm = fm.variable (v, name=nm)
    
    // Specifying values for variables in the graph
    node { return var (vec [ 1.0 ]) "x" + v 4.0 }
    |> fun dt -> fm.eval(dt, ["x", (vec [2.0] :> DT)] )

    node { return var (vec [ 1.0 ]) "x" * var (vec [ 1.0 ]) "x" + v 4.0 }
    |> fm.eval

    node { return fm.variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> fun dt -> fm.eval(dt, ["hey", (vec [2.0] :> DT)])
       // Gives 6.0





module GradientAscentWithRecordOfParameters =

    // The mathematics of the core model is always authored with respet to DT<_> values for scalars, vectors, matrices
    // and tensors.  Parameters/weights will be vectors and scalars
    //
    // Live trajectory checking of your code will enrich your tooling with interactively
    // checked shape information.
    //
    // It is important that you feel comfortable with evaluating tensor values.
    type DParams = 
        { xs: DT<double>
          y: DT<double> } 

    // Inline helpers can be used to simplify some mathematical presentation
    let inline sqr x = x * x 

    // This is the function we are going to maximize using gradient ascent.
    // This defines a function taking DT values in and returning a scalar DT value.
    let f (ps: DParams) = 
        sin (v 0.5 * sqr ps.xs.[0] - v 0.25 * sqr ps.y + v 3.0) * cos (v 2.0 * ps.xs.[0] + v 1.0 - exp ps.y) - sqr ps.xs.[1]

    // Get the partial derivatives of the function
    let df (ps: DParams) = 
        // TODO: ideally this would be
        //    fm.gradients (f ps) ps
        // returning D<DParams>, a type derived structurally from DParams.
        let dfs = fm.gradients (f ps) [| ps.xs; ps.y |] 
        dfs.[0], dfs.[1]

    let rate = 0.1

    // In presenting the model externally we start with regular parameter values.
    type Params = 
        { xs: double[]
          y: double } 

    let inject (ps: Params) : DParams = { xs = vec ps.xs; y = v ps.y }

    // Gradient ascent
    let step (ps: Params) =
        printfn "ps = %A" ps
        let ps = inject ps
        let df_dxs, df_dy = df ps

        let xsNew, yNew = (ps.xs + v rate * df_dxs, ps.y + v rate * df_dy) |> fm.eval2
        { xs = xsNew.ToArray(); y = yNew.ToScalar()}

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
        node { return sin (v 0.5 * x * x - v 0.25 * y * y + v 3.0) * cos (v 2.0 * x + v 1.0 - exp y) }

    // Get the partial derivatives of the scalar function
    // computes [ 2*x1*x3 + x3*x3; 3*x2*x2; 2*x3*x1 + x1*x1 ]
    let df (x, y) = 
        let dfs = fm.gradients (f (x,y))  [| x; y |] 
        (dfs.[0], dfs.[1])

    let rate = 0.1
    let step (x, y) = 
        // Repeatedly run the derivative 
        let dzx, dzy = df (v x, v y) 
        let dzx = dzx |> fm.eval 
        let dzy = dzy |> fm.eval 
        printfn "size = %f" (sqrt (dzx*dzx + dzy*dzy))
        (x + rate * dzx, y + rate * dzy)

    (-0.3, 0.3) |> Seq.unfold (fun pos -> Some (pos, step pos)) |> Seq.truncate 200 |> Seq.toArray
*)
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
    let trueFunction (xs: double[]) = 
        Array.sum [| for i in 0 .. modelSize - 1 -> trueCoeffs.[i] * xs.[i]  |] + noise 0.5

    let makeData size = 
        [| for i in 1 .. size -> 
            let xs = [| for i in 0 .. modelSize - 1 -> rnd.NextDouble() |]
            xs, trueFunction xs |]
    
    let makeInput data = 
        let (xs, y) = Array.unzip data
        batchOfVecs xs, batchOfScalars y

    /// Make the data used to symbolically check the model
    let checkData = makeData checkSize |> makeInput

    /// Make the training data
    let trainData = makeData trainSize |> makeInput

    /// Make the validation data
    let validationInputs, validationOutputs = makeData validationSize |> makeInput

    /// evaluate the model for input and coefficients
    let model (xs: DT<double>, coeffs: DT<double>) = 
        fm.sum (xs * coeffs,axis= [| 1 |]) 

    /// evaluate the loss function for the model w.r.t. a true output
    let meanSquareError (z: DT<double>) tgt = 
        let dz = z - tgt 
        fm.sum (dz * dz) / v (double modelSize) / v (double z.Shape.[0].Value) 
          
    // Gradient of the loss function w.r.t. the coefficients
    let loss (xs, y) coeffs = 
        let coeffsBatch = batchExtend coeffs
        meanSquareError (model (xs, coeffsBatch)) y

    // Gradient of the objective function w.r.t. the coefficients
    let dloss_dcoeffs inputs coeffs = 
        let z = loss inputs coeffs
        fm.gradient z coeffs
 
    let validation coeffs = 
        let z = loss (validationInputs, validationOutputs) coeffs 
        z |> fm.eval

    // Note, the rate in this example is constant. Many practical optimizers use variable
    // update (rate) - often reducing - which makes them more robust to poor convergence.
    let rate = 2.0

    let step inputs coeffs = 
        let dz = dloss_dcoeffs inputs coeffs 
        let coeffsNew = coeffs - v rate * dz

        // An explicit evaluation step is needed to reduce the computation and get concrete values
        let coeffsNew, dz = (coeffsNew, dz) |> fm.eval2
        printfn "coeffsNew = %A" coeffsNew 
        printfn "dz = %A" dz 
        printfn "validation = %A" (validation coeffsNew)
        coeffsNew

    let initialCoeffs = vec [ for i in 0 .. modelSize - 1 -> rnd.NextDouble()  * double modelSize ]

    // Train the inputs in one batch
    let train input nsteps =
        initialCoeffs |> Seq.unfold (fun coeffs -> Some (coeffs, step input coeffs)) |> Seq.truncate nsteps |> Seq.last

    //[<LiveCheck>]
    //let check1 = train checkData checkSize

    //[<LiveTest>]
    //let test1 = train trainData 10

    //let learnedCoeffs = train trainData 200
     // [|1.017181246; 2.039034327; 2.968580146; 3.99544071; 4.935430581;
     //   5.988228378; 7.030374908; 8.013975714; 9.020138699; 9.98575733|]

    //validation (vec trueCoeffs) 
    //validation learnedCoeffs
    //   [|0.007351991009; 1.004220712; 2.002591797; 3.018333918; 3.996983572; 4.981999364; 5.986054734; 7.005387338; 8.005461854; 8.991150034|]

module NeuralTransferFragments =
    let input = tensor4 [ for i in 0 .. 9 -> [ for j in 1 .. 10 -> [ for k in 1 .. 10 -> [ for m in 1 .. 10 -> double (i+j+k+m) ]]]]
    let name = "a"
    let instance_norm (input, name) =
        node { 
             use _ = fm.with_scope(name + "/instance_norm")
             let mu, sigma_sq = fm.moments (input, axes=[0;1])
             let shift = fm.variable (v 0.0, name + "/shift")
             let scale = fm.variable (v 1.0, name + "/scale")
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift 
         }

    let friendly4D (d : 'T[,,,]) =
        [| for i in 0..Array4D.length1 d - 1 -> [| for j in 0..Array4D.length2 d - 1 -> [| for k in 0..Array4D.length3 d - 1 -> [| for m in 0..Array4D.length4 d - 1 -> d.[i,j,k,m]  |]|]|]|]
        |> array2D |> Array2D.map array2D

    // instance_norm (input, name) |> fm.eval
   
    let conv_layer (out_channels, filter_size, stride, name) input = 
        let filters = variable (fm.truncated_normal() * v 0.1) (name + "/weights")
        let x = fm.conv2d (input, filters, out_channels, filter_size=filter_size, stride=stride)
        instance_norm (x, name)
    
    let out_channels = 128
    let filter_size = 7 
    // conv_layer (out_channels, filter_size, 1, "layer") input  |> fm.eval
    //(fun input -> conv_layer (input, out_channels, filter_size, 1, true, "layer")) |> fm.gradient |> apply input |> fm.eval

    let residual_block (filter_size, name) input = 
        let tmp = conv_layer (128, filter_size, 1, name + "_c1") input  |> relu
        let tmp2 = conv_layer (128, filter_size, 1, name + "_c2") tmp
        input + tmp2 

    let conv2D_transpose (input, filter, stride) = 
        node { return fm.conv2d_backprop(filter, input, stride, padding = "SAME") }
  
    let conv_transpose_layer (out_channels, filter_size, stride, name) input =
        let filters = variable (fm.truncated_normal() * v 0.1) (name + "/weights")
        let x = fm.conv2d_backprop(filters, input, out_channels, filter_size=filter_size, stride=stride, padding="SAME")
        instance_norm (x, name)

    let to_pixel_value (input: DT<double>) = 
        node { return tanh input * v 150.0 + (v 255.0 / v 2.0) }

    // The style-transfer tf

(*
    node { let x = conv_layer (32, 9, 1, "conv1") input
         return relu x }
    |> fm.eval

    node { let x = input |> conv_layer (32, 9, 1, "conv1") |> relu
         let x = conv_layer (64, 3, 2, "conv2") x |> relu
         return x }
    |> fm.eval

    node { let x = 
            conv_layer (32, 9, 1, "conv1") input 
            |> relu 
            |> conv_layer (64, 3, 2, "conv2")
            |> relu 
            |> conv_layer (128, 3, 2, "conv3") 
            |> relu
         return x }
    |> fm.eval

    node { return
            input 
            |> conv_layer (32, 9, 1, "conv1")
            |> relu 
            |> conv_layer (64, 3, 2, "conv2")
            |> relu 
            |> conv_layer (128, 3, 2, "conv3")
            |> relu 
            |> residual_block (3, "resid1")
          }
    |> fm.eval

    node { return 
            input 
            |> conv_layer (32, 9, 1, "conv1")
            |> relu 
            |> conv_layer (64, 3, 2, "conv2")
            |> relu 
            |> conv_layer (128, 3, 2, "conv3")
            |> relu 
            |> residual_block (3, "resid1")
            |> residual_block (3, "resid2")
            |> residual_block (3, "resid3")
            |> residual_block (3, "resid4")
            |> residual_block (3, "resid5")
         }
    |> fm.eval


    let t1 = 
        node { return 
                input 
                |> conv_layer (32, 9, 1, "conv1")
                |> relu 
                |> conv_layer (64, 3, 2, "conv2")
                |> relu 
                |> conv_layer (128, 3, 2, "conv3")
                |> relu 
                |> residual_block (3, "resid1")
                |> residual_block (3, "resid2")
                |> residual_block (3, "resid3")
                |> residual_block (3, "resid4")
                |> residual_block (3, "resid5")
         }

    let t2 = 
        node { return conv_transpose_layer (64, 3, 2, "conv_t1") t1 }
        |> fm.eval

    node { return 
            input 
            |> conv_layer (32, 9, 1, "conv1")
            |> relu 
            |> conv_layer (64, 3, 2, "conv2")
            |> relu 
            |> conv_layer (128, 3, 2, "conv3")
            |> relu 
            |> residual_block (3, "resid1")
            |> residual_block (3, "resid2")
            |> residual_block (3, "resid3")
            |> residual_block (3, "resid4")
            |> residual_block (3, "resid5")
            |> conv_transpose_layer (64, 3, 2, "conv_t1")// TODO: check fails
         }
    |> fm.eval
*)

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
    U  = tf.variable(u_init)
    Ut = tf.variable(ut_init)

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
        node { 
             let x = conv_layer (input, 32, 9, 1, true, "conv1")
             let x = conv_layer (x, 64, 3, 2, true, "conv2")
             let x = conv_layer (x, 128, 3, 2, true, "conv3")
             let x = residual_block (x, 3, "resid1")
             let x = residual_block (x, 3, "resid2")
             let x = residual_block (x, 3, "resid3")
             let x = residual_block (x, 3, "resid4")
             let x = residual_block (x, 3, "resid5")
             let x = conv_transpose_layer (x, 64, 3, 2, "conv_t1") // TODO: check fails
             return x })
    |> fm.Diff |> apply input |> fm.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

//2019-01-24 17:21:17.095544: F tensorflow/core/grappler/costs/op_level_cost_estimator.cc:577] Check failed: iz == filter_shape.dim(in_channel_index).size() (64 vs. 128)
    *)


(*
let x = conv_layer (x, 3, 9, 1, false, "conv_t3")
             let x = to_pixel_value x
             let x = fm.clip_by_value (x, v 0.0, v 255.0)
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
    //graph.variable( .MakeName("a", "b")

    //TFTensor(TFDataType.Double, [| 2;2|], )
    //(TFDataType.Double, [| 2;2|])
    //TFOutput
    //graph.Inpu
