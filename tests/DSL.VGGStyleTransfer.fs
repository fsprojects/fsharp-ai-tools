module FSAI.Tools.VGGStyleTransferDSL

// NOTE: Most operations were converted from double (float) to single as that is how this model was designed
// NOTE: I'm unsure why the shape resolver is not working 

// This example shows the NeuralStyles transfer model using the FM F#-for-AI-models DSL.
//
// See https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398
//
// Build the Debug 'FSAI.Tools.Tests' before using this

//#I __SOURCE_DIRECTORY__
//#r "netstandard"
//#r "../tests/bin/Debug/net472/FSAI.Tools.dll"
//#r "../tests/bin/Debug/net472/Tensorflow.Net.dll"
//#r "../tests/bin/Debug/net472/NumSharp.Core.dll"
//#load "shared/NPYReaderWriter.fsx"
//#load "shared/ScriptLib.fsx"


//------------------------------------------------------------------------------
// Preliminaries for F# scripting

open System
open System.IO
open FSAI.Tools
open FSAI.Tools.DSL
open NPYReaderWriter


// Check the process is 64-bit.  
if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

// Add the print helper for tensors in the REPL
//fsi.AddPrintTransformer(DT.PrintTransform)

//------------------------------------------------------------------------------
// This is the NeuralStyles transfer model proper

[<Model>]
module NeuralStyles = 
    let instance_norm name input  =
        use __ = DT.WithScope(name + "/instance_norm")
        let mu, sigma_sq = DT.Moments (input, axes= (if input.Shape.Rank = 3 then [0;1] else [1;2])) // TODO: the axes computation is needed to be usabe with 3D or 4D. Would be nice if this could be simpler
        let shift = variable (v 0.0f) (name + "/shift")
        let scale = variable (v 1.0f) (name + "/scale")
        let epsilon = v 0.001f
        let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
        scale * normalized + shift 

    // Set up a convolution layer in the DNN
    let conv_layer (out_channels, filter_size, stride, name) input = 
        let filters = variable (DT.TruncatedNormal() * v 0.1f) (name + "/weights") 
        DT.Conv2D (input, filters, out_channels, stride=stride, filter_size=filter_size)
        |> instance_norm name

    let conv_transpose_layer (out_channels, filter_size, stride, name) input =
        let filters = variable (DT.TruncatedNormal() * v 0.1f) (name + "/weights") 
        DT.Conv2DBackpropInput(filters, input, out_channels, stride, padding = "SAME", filter_size=filter_size) 
        |> instance_norm name 

    let residual_block (filter_size, name) input = 
        let tmp = input |> conv_layer (128, filter_size, 1, name + "_c1") |> relu
        input + conv_layer (128, filter_size, 1, name + "_c2") tmp

    let to_pixel_value input = 
        tanh input * v 150.0f + (v 255.0f / v 2.0f)

    let clip min max input = 
       DT.ClipByValue (input, v min, v max)

    // The style-transfer DNN
    let PretrainedFFStyleVGG (image:DT<single>) = 
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
        |> clip 0.0f 255.0f
        |> DT.Cast<single> 

    /// PretrainedFFStyleVGG is the complete model, taking either 
    ///    - an image to an image
    ///    - a batch of images to a batch of images
    ///
    let inputShapeForSingle = Shape [| Dim.Var "H";  Dim.Var "W"; Dim.Var "C" |]
    let inputShapeForBatch  = Shape [| Dim.Var "N";  Dim.Var "H";  Dim.Var "W"; Dim.Var "C" |]

    /// As usual, the LiveCheck is placed at the point where the model is finished.
    /// 
    /// We use dummy data for the LiveCheck.  
    ///
    /// We check the model on both single images and batches.
    //[<LiveCheck>]
    let test() = 
        // Check the code can be used on batches
        let dummyImages = DT.Dummy inputShapeForBatch
        PretrainedFFStyleVGG dummyImages |> ignore

       // Check the code can be used on single images
        let dummyImage = DT.Dummy inputShapeForSingle
        PretrainedFFStyleVGG dummyImage  |> ignore

//------------------------------------------------------------

//for i in 1 .. 10 do 
//    time (fun () -> processImage rain "chicago.jpg" )

let run() =
    // Read the weights map
    let readWeights weightsFile = 
        let npz = readFromNPZ (File.ReadAllBytes weightsFile)
        [| for KeyValue(k,(metadata, arr)) in npz do
                let shape = Shape.Known metadata.shape 
                let name = k.[0..k.Length-5]
                let value = DT.ConstArray (arr, shape=shape) |> DT.Cast<single> :> DT
                yield (name, value) |]

    /// Read an image from the path as a tensor value
    let readImage imgPath = 
        let imgName = File.ReadAllBytes(imgPath) |> DT.CreateString
        // read the image
        let jpg = DT.DecodeJpeg(imgName) |> DT.Cast<single> |> DT.Eval
        // subtract the mean
        let mean_pixel  = pixel [| 123.68f; 116.778f; 103.939f |] 
        jpg - mean_pixel

    /// For convenience we add an entry point for one image
    // OK, now use the model on some sample data and pre-trained weights

    let imageFile imgName = Path.Combine(__SOURCE_DIRECTORY__,"../tests/examples", imgName)

    let weightsFile style = Path.Combine(__SOURCE_DIRECTORY__,"../tests/pretrained", sprintf "fast_style_weights_%s.npz" style)

    let prepareModelForStyle style = 
        printfn "preparing model for style %s" style
        let modelForStyle = DT.Preprocess(NeuralStyles.PretrainedFFStyleVGG, NeuralStyles.inputShapeForSingle, weights = readWeights (weightsFile style))
        modelForStyle, style

    //let rain = prepareModelForStyle "rain"
    //let wave = prepareModelForStyle "wave" 
    let starry_night = prepareModelForStyle "starry_night"

    let processImage (modelForStyle, style) imageName = 
        printfn "processing image %s in style %s" imageName style
        // TODO the DSL for read image isn't working that well, I belive it's something to do with Tensorflow.Net needing additional String support that this DSL hits
        // Specifically NDArray.ToByteArray() is throwing a not implemented exception
        //let inputImage = readImage (imageFile imageName) 
        // This line may cause an issue with the dtype being a runtime type, although the shape is correct, could just be a debugging issue
        //let inputImage = DT.ConstArray3D(Array3D.zeroCreate<single> 712 474 3,false) // NOTE: Consts probably can't be used as inputs in this context
        //let inputImage : DT<single> = DT.FromNDArray(NumSharp.NDArray([|for _ in 1..(712*474*3) -> 0.f|],NumSharp.Shape(712,474,3)))// DT.ConstArray3D(Array3D.zeroCreate<single> 712 474 3,false)
        let inputImage : DT<single> = DT.Cast<single>(DT.ConstArray([|for _ in 1..(712*474*3) -> 0.f|] :> System.Array,DSL.Shape([|712;474;3|])))// DT.ConstArray3D(Array3D.zeroCreate<single> 712 474 3,false)
        let image = (modelForStyle inputImage) |> DT.toArray3D
        let png = image  |> ImageWriter.arrayToPNG_HWC 
        let outfile = Path.Combine(__SOURCE_DIRECTORY__, sprintf "%s_in_%s_style2.png" imageName style)
        File.WriteAllBytes(outfile, png)

    ()
    ///processImage starry_night "chicago.jpg" 
//    processImage wave "chicago.jpg" 
//    processImage wave "chicago.jpg" 
//    processImage wave "example_1.jpeg" 
//    processImage wave "example_0.jpeg" 



