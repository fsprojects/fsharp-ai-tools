// Build the Debug 'FSAI.Tools.Tests' before using this

#if INTERACTIVE
#I __SOURCE_DIRECTORY__
#I "../../tests/bin/Debug/netcoreapp3.1"
#r "FSAI.Tools.dll"
#r "NumSharp.Core.dll"
#r "Tensorflow.Net.dll"
#nowarn "49"
#endif

#if INTERACTIVE && EDITING
#r @"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\Common7\IDE\CommonExtensions\Microsoft\FSharp\FSharp.Compiler.Interactive.Settings.dll"
#endif

#if NOTEBOOK
#r "nuget: TODO"
#endif


open System
open FSAI.Tools
open FSAI.Tools.FM

if not System.Environment.Is64BitProcess then printfn "64-bit expected";  exit 100

#if INTERACTIVE
fsi.AddPrintTransformer(DT.PrintTransform)
#endif


module FirstLiveCheck = 
    let f x shift = DT.sum(x * x, [| 1 |]) + v shift

    [<LiveCheck>]
    let check1() = f (DT.dummy_input (Shape [ Dim.Named "Size1" 10;  Dim.Named "Size2" 100 ])) 3.0
     
module PlayWithFM = 
    
    let f x = 
       x * x + scalar 4.0 * x 

    // Get the derivative of the function. This computes "x*2 + 4.0"
    let df x = fm.diff f x  



    // Run the function
    f (v 3.0) 

    // Most operators need explicit evaluation even in interactive mode
    df (v 3.0) 

    // returns 6.0 + 4.0 = 10.0
    df (v 3.0) |> fm.eval

    v 2.0 + v 1.0 

    // You can wrap in "node { return ... }" if you like
    node { return v 2.0 + v 1.0 }
      







    let test1() = 
        vec [1.0; 2.0] + vec [1.0; 3.0; 4.0]
    
    [<LiveCheck>]
    let check1() = test1() 













    // Math-multiplying matrices of the wrong sizes gives an error
    let test2() = 
        let matrix1 = 
            matrix [ [1.0; 2.0]
                     [1.0; 2.0]
                     [1.0; 2.0] ]
        let matrix2 =  
            matrix [ [1.0; 2.0]
                     [1.0; 2.0]
                     [1.0; 2.0] ] 
        matrix1 *! matrix2 









    
    // Things are only checked when there is a [<LiveCheck>] exercising the code path
    
    [<LiveCheck>]
    let check2() = test2() 




module GradientDescent =

    // Note, the rate in this example is constant. Many practical optimizers use variable
    // update (rate) - often reducing - which makes them more robust to poor convergence.
    let rate = 0.005

    // Gradient descent
    let step f xs =   
        // Evaluate to output values 
        xs - v rate * fm.diff f xs
    
    let train f initial numSteps = 
        initial |> Seq.unfold (fun pos -> Some (pos, step f pos  |> fm.eval)) |> Seq.truncate numSteps 

(*
module GradientDescentExample =

    // A numeric function of two parameters, returning a scalar, see
    // https://en.wikipedia.org/wiki/Gradient_descent
    let f (xs: Vector) : Scalar = 
        sin (0.5 * sqr xs.[0] - 0.25 * sqr xs.[1] + 3) * -cos (2 * xs.[0] + 1 - exp xs.[1])

    // Pass this to gradient descent
    let train numSteps = GradientDescent.train f (vec [ -0.3; 0.3 ]) numSteps

    [<LiveCheck>] 
    let check1() = train 4 |> Seq.last 
    
    let results = train 200 |> Seq.last
*)

module ModelExample =

    let modelSize = 10

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
         
    let prepare data = 
        let xs, y = Array.unzip data
        let xs = batchOfVecs xs
        let y = batchOfScalars y
        (xs, y)

    /// Make the training data
    let trainData = makeData trainSize |> prepare

    /// Make the validation data
    let validationData = makeData validationSize |> prepare
 
    /// Evaluate the model for input and coefficients
    let model (xs: Vectors, coeffs: Scalars) = 
        DT.sum (xs * coeffs,axis= [| 1 |])
           
    let meanSquareError (z: DT<double>) tgt = 
        let dz = z - tgt 
        DT.sum (dz * dz) / v (double modelSize) / v (double z.length)

    /// The loss function for the model w.r.t. a true output
    let loss (xs, y) coeffs = 
        let y2 = model (xs, batchExtend coeffs)
        meanSquareError y y2
          
    let validation coeffs = 
        let z = loss validationData (vec coeffs)
        z |> fm.eval

    let train initialCoeffs (xs, y) steps =
        GradientDescent.train (loss (xs, y)) initialCoeffs steps
           
    [<LiveCheck>]
    let check1() = 
        let DataSz = Dim.Named "DataSz" 100
        let ModelSz = Dim.Named "ModelSz" 10
        let coeffs = DT.dummy_input (Shape [ ModelSz ])
        let xs = DT.dummy_input (Shape [ DataSz; ModelSz ])
        let y = DT.dummy_input (Shape [ DataSz ])
        train coeffs (xs, y) 1  |> Seq.last

    let initialCoeffs = vec [ for i in 0 .. modelSize - 1 -> rnd.NextDouble()  * double modelSize ]
    //let learnedCoeffs = train initialCoeffs trainData 200 |> Seq.last |> DT.toArray
         // [|1.017181246; 2.039034327; 2.968580146; 3.99544071; 4.935430581;
         //   5.988228378; 7.030374908; 8.013975714; 9.020138699; 9.98575733|]

    validation trueCoeffs

    //validation learnedCoeffs

module ODEs = 
    let lotka_volterra(du, u: DT<double>, p: DT<double>, t) = 
        let x, y = u.[0], u.[1]
        let α, β, δ, γ = p.[0], p.[1], p.[2], p.[3]
        let dx = α*x - β*x*y
        let dy = -δ*y + γ*x*y 
        DT.pack [dx; dy]

    let u0 = vec [1.0; 1.0]
    let tspan = (0.0, 10.0)
    let p = vec [ 1.5; 1.0; 3.0; 1.0]
    //let prob = ODEProblem(lotka_volterra, u0, tspan, p)
    //let sol = solve(prob, Tsit5())
    //sol |> Chart.Lines 

module GradientDescentPreprocessed =
    // mucking about with model preparation/compilation

    // Note, the rate in this example is constant. Many practical optimizers use variable
    // update (rate) - often reducing - which makes them more robust to poor convergence.
    let rate = 0.005

    // Gradient descent
    let step f xs =   
        // Evaluate to output values 
        xs - v rate * fm.diff f xs
    
    let stepC f inputShape = DT.precompile (step f, inputShape)

    let trainC f inputShape = 
        let stepC = stepC f inputShape
        fun initial numSteps -> initial |> Seq.unfold (fun pos -> Some (pos, stepC pos)) |> Seq.truncate numSteps 

(*
module GradientDescentExamplePreprocessed =

    // A numeric function of two parameters, returning a scalar, see
    // https://en.wikipedia.org/wiki/Gradient_descent
    let f (xs: Vector) : Scalar = 
        sin (0.5 * sqr xs.[0] - 0.25 * sqr xs.[1] + 3) * -cos (2 * xs.[0] + 1 - exp xs.[1])

    // Pass this Define a numeric function of two parameters, returning a scalar
    let trainC = GradientDescentPreprocessed.trainC f (shape [ 2 ])

    let train numSteps = trainC (vec [ -0.3; 0.3 ]) numSteps

    [<LiveCheck>] 
    let check1() = f (vec [ -0.3; 0.3 ])
    
    let results = train 200 |> Seq.last
*)

module NeuralTransferFragments =

    let instance_norm (input, name) =
        use __ = DT.with_scope(name + "/instance_norm")
        let mu, sigma_sq = DT.moments (input, axes=[0;1])
        let shift = DT.variable (v 0.0, name + "/shift")
        let scale = DT.variable (v 1.0, name + "/scale")
        let epsilon = v 0.001
        let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
        normalized * scale + shift 

    let conv_layer (out_channels, filter_size, stride, name) input = 
        let filters = variable (DT.truncated_normal() * v 0.1) (name + "/weights")
        let x = DT.conv2d (input, filters, out_channels, filter_size=filter_size, stride=stride)
        instance_norm (x, name)

    let conv_transpose_layer (out_channels, filter_size, stride, name) input =
        let filters = variable (DT.truncated_normal() * v 0.1) (name + "/weights")
        let x = DT.conv2d_backprop(filters, input, out_channels, filter_size=filter_size, stride=stride, padding="SAME")
        instance_norm (x, name)

    let to_pixel_value (input: DT<double>) = 
        tanh input * v 150.0 + v (255.0 / 2.0) 

    let residual_block (filter_size, name) input = 
        let tmp = conv_layer (128, filter_size, 1, name + "_c1") input  |> relu
        let tmp2 = conv_layer (128, filter_size, 1, name + "_c2") tmp
        input + tmp2 

    let clip min max x = 
       DT.clip_by_value (x, v min, v max)

    // The style-transfer neural network
    let style_transfer input = 
        input 
        |> conv_layer (32, 9, 1, "conv1") |> relu
        |> conv_layer (64, 3, 2, "conv2") |> relu
        |> conv_layer (128, 3, 2, "conv3") |> relu
        |> residual_block (3, "resid1")
        |> residual_block (3, "resid2")
        |> residual_block (3, "resid3")
        |> residual_block (3, "resid4")
        |> residual_block (3, "resid5")
        |> conv_transpose_layer (64, 3, 2, "conv_t1") |> relu
        |> conv_transpose_layer (32, 3, 2, "conv_t2") |> relu
        |> conv_layer (3, 9, 1, "conv_t3")
        |> to_pixel_value
        |> clip 0.0 255.0
        |> fm.eval 

    [<LiveCheck>]
    let check1() = 
        let dummyImages = DT.dummy_input (Shape [ Dim.Named "BatchSz" 10; Dim.Named "H" 474;  Dim.Named "W" 712; Dim.Named "Channels" 3 ])
        style_transfer dummyImages
    
    do
        printfn "checking that livecheck runs..."
        use _holder = LiveChecking.WithLiveCheck() 
        check1() |> ignore

#if COMPILED

let v = sprintf "running test in %s at %A" __SOURCE_FILE__ System.DateTime.Now
open NUnit.Framework
[<Test>]
let ``run test`` () = 
    v |> ignore
#endif


    (*
    
    def gram_matrix(input_tensor):
      # We make the image channels first 
      channels = int(input_tensor.shape[-1])
      a = tf.reshape(input_tensor, [-1, channels])
      n = tf.shape(a)[0]
      gram = tf.matmul(a, a, transpose_a=True)
      return gram / tf.cast(n, tf.float32)
     
    def get_style_loss(base_style, gram_target):
      """Expects two images of dimension h, w, c"""
      # height, width, num filters of each layer
      height, width, channels = base_style.get_shape().as_list()
      gram_style = gram_matrix(base_style)
      
      return tf.reduce_mean(tf.square(gram_style - gram_target))
    
    def get_feature_representations(model, content_path, style_path):
      """Helper function to compute our content and style feature representations.
     
      This function will simply load and preprocess both the content and style 
      images from their path. Then it will feed them through the network to obtain
      the outputs of the intermediate layers. 
      
      Arguments:
        model: The model that we are using.
        content_path: The path to the content image.
        style_path: The path to the style image
        
      Returns:
        returns the style features and the content features. 
      """
      # Load our images in 
      content_image = load_and_process_img(content_path)
      style_image = load_and_process_img(style_path)
      
      # batch compute content and style features
      stack_images = np.concatenate([style_image, content_image], axis=0)
      model_outputs = model(stack_images)
      
      # Get the style and content feature representations from our model  
      style_features = [style_layer[0] for style_layer in model_outputs[:num_style_layers]]
      content_features = [content_layer[1] for content_layer in model_outputs[num_style_layers:]]
      return style_features, content_features
      

      def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
      """This function will compute the loss total loss.
      
      Arguments:
        model: The model that will give us access to the intermediate layers
        loss_weights: The weights of each contribution of each loss function. 
          (style weight, content weight, and total variation weight)
        init_image: Our initial base image. This image is what we are updating with 
          our optimization process. We apply the gradients wrt the loss we are 
          calculating to this image.
        gram_style_features: Precomputed gram matrices corresponding to the 
          defined style layers of interest.
        content_features: Precomputed outputs from defined content layers of 
          interest.
          
      Returns:
        returns the total loss, style loss, content loss, and total variational loss
      """
      style_weight, content_weight, total_variation_weight = loss_weights
      
      # Feed our init image through our model. This will give us the content and 
      # style representations at our desired layers. Since we're using eager
      # our model is callable just like any other function!
      model_outputs = model(init_image)
      
      style_output_features = model_outputs[:num_style_layers]
      content_output_features = model_outputs[num_style_layers:]
      
      style_score = 0
      content_score = 0

      # Accumulate style losses from all layers
      # Here, we equally weight each contribution of each loss layer
      weight_per_style_layer = 1.0 / float(num_style_layers)
      for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
        
      # Accumulate content losses from all layers 
      weight_per_content_layer = 1.0 / float(num_content_layers)
      for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
      
      style_score *= style_weight
      content_score *= content_weight
      total_variation_score = total_variation_weight * total_variation_loss(init_image)

      # Get total loss
      loss = style_score + content_score + total_variation_score 
      return loss, style_score, content_score, total_variation_score


      
      def compute_grads(cfg):
        with tf.GradientTape() as tape: 
          all_loss = compute_loss // *cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss
      *)
