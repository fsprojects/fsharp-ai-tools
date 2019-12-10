(**markdown
# Heading 1
This example shows the NeuralStyles transfer model using the FM F#-for-AI-models DSL.

See [the original python example](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398).

Build the Debug 'FSAI.Tools.Tests' before using this
*) 


#if INTERACTIVE
#I __SOURCE_DIRECTORY__
#I "../../tests/bin/Debug/netcoreapp2.0/"
#r "NumSharp.Core.dll"
#r "TensorFlow.Net.dll"
#r "FSAI.Tools.dll"
#r "nunit.framework.dll"
#endif
#if NOTEBOOK
#r "nuget: TODO"
#endif

(**markdown
Preliminaries for F# scripting
*)

open System
open System.IO
open FSAI.Tools
open FSAI.Tools.FM
open NPYReaderWriter

(**markdown
Check the process is 64-bit.  
*)
if not System.Environment.Is64BitProcess then  exit 100

(**markdown
Add the print helper for tensors in the REPL
*)
#if INTERACTIVE
fsi.AddPrintTransformer(fm.PrintTransform)
#endif

(**markdown
This is the NeuralStyles transfer model proper
*)
[<Model>]
module NeuralStyles = 

    let instance_norm name input  =
        use __ = fm.with_scope(name + "/instance_norm")
        let mu, sigma_sq = fm.moments (input, axes= (if input.rank = 3 then [0;1] else [1;2])) 
        let shift = fm.variable (v 0.0, name + "/shift")
        let scale = fm.variable (v 1.0, name + "/scale")
        let epsilon = v 0.001
        let normalized =  (input - mu) / sqrt (sigma_sq + epsilon)
        scale * normalized + shift 

    // Set up a convolution layer in the DNN
    let conv_layer (out_channels, filter_size, stride, name) input = 
        let filters = variable (fm.truncated_normal() * v 0.1) (name + "/weights") 
        fm.conv2d (input, filters, out_channels, stride=stride, filter_size=filter_size)
        |> instance_norm name

    let conv_transpose_layer (out_channels, filter_size, stride, name) input =
        let filters = variable (fm.truncated_normal() * v 0.1) (name + "/weights") 
        fm.conv2d_backprop(filters, input, out_channels, stride, padding = "SAME", filter_size=filter_size) 
        |> instance_norm name 

    let residual_block (filter_size, name) input = 
        let tmp = input |> conv_layer (128, filter_size, 1, name + "_c1") |> relu
        input + conv_layer (128, filter_size, 1, name + "_c2") tmp

    let to_pixel_value (input: DT<_>) = 
        tanh input * v 150.0 + v (255.0 / 2.0)

    let clip min max input = 
        fm.clip_by_value (input, v min, v max)

    // The style-transfer DNN
    let PretrainedFFStyleVGGCore (image: DT<double>) = 
        image
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

        // Use these to add shape assertions
        //let N, H, W, C = image.Shape.AsRank3OrMore()
        //|> fm.assert_shape (Shape [| H; W; C |])
        //|> (fun output -> printfn "image.Shape = %A, output.Shape = %A" image.Shape output.Shape; output)
        //|> fm.assert_shape (Shape [| H/2/2*2*2; W/2/2*2*2; Dim.Known 3 |])
        //|> fm.assert_shape image.Shape

    /// PretrainedFFStyleVGGCore is the complete model, taking either 
    ///    - an image to an image
    ///    - a batch of images to a batch of images
    ///
    let inputShapeForSingle = Shape [| Dim.Var "H";  Dim.Var "W"; Dim.Named "C" 3 |]
    let inputShapeForBatch  = Shape [| Dim.Var "N";  Dim.Var "H";  Dim.Var "W"; Dim.Named "C" 3 |]

    /// The LiveCheck is placed at the point where the model is finished.
    /// 
    /// We use dummy data for the LiveCheck.  
    ///
    /// We check the model on both single images and batches.
    [<LiveCheck>]
    let check_core_model() = 
        // Check the core model can be used on single decoded images
        let dummyImage = fm.dummy_input inputShapeForSingle
        PretrainedFFStyleVGGCore dummyImage  |> ignore

        // Check the core model can be used on batches of decoded images of the same size
        let dummyImages = fm.dummy_input inputShapeForBatch
        PretrainedFFStyleVGGCore dummyImages |> ignore

(**markdown
Read the weights map
*)

let readWeights weightsFile = 
    let npz = readFromNPZ (File.ReadAllBytes weightsFile)
    [| for KeyValue(k,(metadata, arr)) in npz do
            let shape = Shape.Known metadata.shape 
            let name = k.[0..k.Length-5]
            let value = fm.constant_array (arr, shape=shape) |> fm.cast<double> :> DT
            yield (name, value) |]

(**markdown
For convenience we add an entry point for one image
OK, now use the model on some sample data and pre-trained weights
*)

let imageFile imgName = Path.Combine(__SOURCE_DIRECTORY__,"../../images", imgName)

let weightsFile style = Path.Combine(__SOURCE_DIRECTORY__,"../../pretrained", sprintf "fast_style_weights_%s.npz" style)

let prepareCoreModelForStyle style = 
    //printfn "preparing model for style %s" style
    let modelForStyle = fm.precompile(NeuralStyles.PretrainedFFStyleVGGCore, NeuralStyles.inputShapeForSingle, weights = readWeights (weightsFile style))
    modelForStyle, style

let readImage imageName = 
    // read the image
    let fileName = imageFile imageName
    let jpg = fm.decode_jpeg(fileName) |> fm.cast<double>
    // subtract the mean
    let mean_pixel  = pixel [| 123.68; 116.778; 103.939 |] 
    jpg - mean_pixel |> fm.eval

let runModel (modelForStyle, styleName) imageName = 
    //printfn "processing image %s in style %s" imageName styleName
    let inputBytes = readImage imageName
    let image = modelForStyle inputBytes |> fm.cast<uint8> |> fm.toArray3D
    let png = image  |> ImageWriter.arrayToPNG_HWC 
    let outfile = Path.Combine(__SOURCE_DIRECTORY__, sprintf "%s_in_%s_style2.png" imageName styleName)
    File.WriteAllBytes(outfile, png)

#if NOTEBOOK
(**markdown
Now run
*)
let rain = prepareCoreModelForStyle "rain"
let wave = prepareCoreModelForStyle "wave" 
let starry_night = prepareCoreModelForStyle "starry_night"
for i in 1 .. 10 do 
    time (fun () -> runModel rain "chicago.jpg" )
     
runModel starry_night "chicago.jpg" 
//runModel wave "chicago.jpg" 
//runModel wave "chicago.jpg" 
//runModel wave "example_1.jpeg" 
//runModel wave "example_0.jpeg" 
#endif

#if COMPILED
open NUnit.Framework
[<Test>]
let ``read image `` () = 
    readImage "chicago.jpg" |> ignore 

[<Test>]
let ``prepare core model`` () = 
    prepareCoreModelForStyle "rain" |> ignore

[<Test>]
let ``run full model`` () = 
    let starry_night = prepareCoreModelForStyle "rain"
    runModel starry_night "chicago.jpg"

[<Test>]
let ``livecheck core model`` () = 
    use _holder = LiveChecking.WithLiveCheck()
    NeuralStyles.check_core_model()

#endif