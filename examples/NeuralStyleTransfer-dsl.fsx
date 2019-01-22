#r "netstandard"
#r "lib/Argu.dll"
#r "lib/TensorFlowSharp.dll"
#load "shared/NNImpl.fsx"
#load "shared/NNOps.fsx"
#load "shared/NPYReaderWriter.fsx"
#load "shared/ImageWriter.fsx"
#load "shared/dsl.fsx"
#nowarn "49"

open Argu
open NPYReaderWriter
open System
open System.IO
open TensorFlow
open TensorFlow.FSharp

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

type Argument =
    | [<AltCommandLine([|"-s"|])>] Style of string
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            |  Style _ -> "Specify a style of painting to use."

let style = ArgumentParser<Argument>().Parse(fsi.CommandLineArgs.[1..]).GetResult(<@ Argument.Style @>, defaultValue = "rain")

fsi.AddPrinter(fun (x:TFGraph) -> sprintf "TFGraph %i" (int64 x.Handle))

let pretrained_dir = Path.Combine(__SOURCE_DIRECTORY__,"pretrained")

let example_dir = Path.Combine(__SOURCE_DIRECTORY__,"examples")

[<TensorFlow>]
module NeuralStyles = 

    let conv_init_vars (out_channels:int, filter_size:int) =
        tf { let truncatedNormal = TF.TruncatedNormal(shape [filter_size; filter_size; Shape.DimInferred(); out_channels])
             return TF.Variable (truncatedNormal * v 0.1) }

    let instance_norm (input: V<double>) =
        tf { let mu, sigma_sq = TF.Moments (shape [1;2], input)
             let shift = TF.Variable (v 0.0)
             let scale = TF.Variable (v 1.0) 
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift }

    let conv_layer (input: V<double>, COut, F, stride, is_relu) = 
        tf { let filters = conv_init_vars (COut, F)
             let x = TF.Conv2D (input, filters, stride=stride)
             let x = instance_norm x
             if is_relu then 
                 return TF.Relu x 
             else 
                 return x }

    let residual_block (input, F) = 
        tf { let tmp = conv_layer(input, 128, F, 1, true)
             return input + conv_layer(tmp, 128, F, 1, false) }

    let conv2D_transpose (input, filter, stride) = 
        tf { return TF.Conv2DBackpropInput(filter, input, stride, padding = "SAME") }
  
    let conv_transpose_layer (input: V<double>, COut, F, Stride) =
        tf { let filters = conv_init_vars (COut, F)
             return TF.Relu (instance_norm (conv2D_transpose (input, filters, Stride)))
           }

    let to_pixel_value (input: V<double>) = 
        tf { return tanh input * v 150.0 + (v 255.0 / v 2.0) }

    // The style-transfer tf
    let PretrainedFFStyleVGG input_img = 
        tf { let x = conv_layer (input_img, 32, 9, 1, true)
             let x = conv_layer (x, 64, 3, 2, true)
             let x = conv_layer (x, 128, 3, 2, true)
             let x = residual_block (x, 3)
             let x = residual_block (x, 3)
             let x = residual_block (x, 3)
             let x = residual_block (x, 3)
             let x = conv_transpose_layer (x, 64, 3, 2)
             let x = conv_transpose_layer (x, 32, 3, 2)
             let x = conv_layer (x, 3, 9, 1, false)
             let x = to_pixel_value x
             let x = TF.ClipByValue (x, v 0.0, v 255.0)
             return x }

    // Compute the weights path
    // let weights_path = Path.Combine(pretrained_dir, sprintf "fast_style_weights_%s.npz" style)

    // Read the weights map
    //let weights_map = 
    //    readFromNPZ((File.ReadAllBytes(weights_path)))
    //    |> Map.toArray 
    //    |> Array.map (fun (k,(metadata, arr)) -> 
    //        k.Substring(0, k.Length-4), graph.Reshape(graph.Const(new TFTensor(arr)), graph.Const(Shape(metadata.shape |> Array.map int64).AsTensor()))) 
    //    |> Map.ofArray

    // The average pixel in the decoding
    let mean_pixel shape = 
        tf { return TF.ConstArray (shape, [| 123.68; 116.778; 103.939 |]) }

    // The decoding tf
    let img input_string = 
        tf { 
            let jpg = TF.DecodeJpeg(input_string)
            let decoded = TF.Cast<_, double>(jpg, TFDataType.Double)
            let preprocessed = decoded - mean_pixel decoded.Shape
            let expanded = TF.ExpandDims(preprocessed, 0)
            return expanded
        }

    // Tensor to read the input
    let img_tf = 
        tf { let bytes = File.ReadAllBytes(Path.Combine(example_dir,"chicago.jpg"))
             return TF.CreateString (bytes) } 

    // Run the decoding
    let img_tensor = img img_tf

    // Run the style transfer
    let img_styled = TF.Run (PretrainedFFStyleVGG img_tensor)

    // NOTE: Assumed NHWC dataformat
    let tensorToPNG (batchIndex:int) (imgs:TFTensor) =
        if imgs.TensorType <> TFDataType.Float then failwith "type unsupported"
        let shape = imgs.Shape |> Array.map int 
        let _N, H, W, _C = shape.[0], shape.[1], shape.[2], shape.[3]
        let pixels = 
            [|
                let result = imgs.GetValue() :?> Array
                for h in 0..H-1 do
                    for w in 0..W-1 do
                        let getV(c) = Math.Min(255.f, Math.Max(0.f, (result.GetValue(int64 batchIndex, int64 h, int64 w, int64 c) :?> float32))) |> byte
                        yield BitConverter.ToInt32([|getV 0; getV 1; getV 2; 255uy|], 0) // NOTE: Channels are commonly in RGB format
            |]
        ImageWriter.RGBAToPNG(H,W,pixels)

    // Write the result
    File.WriteAllBytes(Path.Combine(__SOURCE_DIRECTORY__, sprintf "chicago_in_%s_style.png" style), tensorToPNG 0 img_styled)
