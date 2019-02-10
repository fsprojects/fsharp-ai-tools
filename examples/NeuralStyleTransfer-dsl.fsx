// Build the Debug 'TensorFlow.FSharp.Tests' before using this

#I __SOURCE_DIRECTORY__
#r "netstandard"

#r "../tests/bin/Debug/net461/TensorFlow.FSharp.dll"
#r "../tests/bin/Debug/net461/TensorFlow.FSharp.Proto.dll"
#r "../tests/bin/Debug/net461/LinuxNativeWorkaround.dll"
#load "shared/NPYReaderWriter.fsx"

open NPYReaderWriter
open System
open System.IO
open TensorFlow.FSharp
open TensorFlow.FSharp.DSL

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

let style = "rain"

fsi.AddPrintTransformer(DT.PrintTransform)

[<TensorFlow>]
module NeuralStyles = 

    let conv_init_vars (out_channels:int, filter_size:int, is_transpose: bool, name) =
        let weights_shape = 
            if is_transpose then
                shape [ filter_size; filter_size; out_channels; -1 ]
            else
                shape [ filter_size; filter_size; -1; out_channels ]

        let truncatedNormal = DT.TruncatedNormal(weights_shape)
        variable (truncatedNormal * v 0.1) (name + "/weights") 

    let instance_norm (input: DT<double>, name) =
        tf { use _ = DT.WithScope(name + "/instance_norm")
             let mu, sigma_sq = DT.Moments (input, axes=[1;2])
             let shift = variable (v 0.0) (name + "/shift")
             let scale = variable (v 1.0) (name + "/scale")
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift }

    let conv_layer (input: DT<double>, out_channels, filter_size, stride, is_relu, name) = 
        tf { let filters = conv_init_vars (out_channels, filter_size, false, name)
             let x = DT.Conv2D (input, filters, stride=stride)
             let x = instance_norm (x, name)
             if is_relu then 
                 return relu x 
             else 
                 return x }

    let residual_block (input, filter_size, name) = 
        tf { let tmp = conv_layer(input, 128, filter_size, 1, true, name + "_c1")
             return input + conv_layer(tmp, 128, filter_size, 1, false, name + "_c2") }

    let conv2D_transpose (input, filter, stride) = 
        tf { return DT.Conv2DBackpropInput(filter, input, stride, padding = "SAME") }
  
    let conv_transpose_layer (input: DT<double>, out_channels, filter_size, stride, name) =
        tf { let filters = conv_init_vars (out_channels, filter_size, true, name)
             return relu (instance_norm (conv2D_transpose (input, filters, stride), name))
           }

    let to_pixel_value (input: DT<double>) = 
        tf { return tanh input * v 150.0 + (v 255.0 / v 2.0) }

    // The style-transfer tf
    let PretrainedFFStyleVGG input = 
        tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
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

    let dummyImages() = DT.Stack [ for i in 1 .. 10 -> DT.Reshape(DT.Zero, shape [474;  712; 3]) ]

    [<LiveCheck>]
    let test = PretrainedFFStyleVGG (dummyImages())

    // Compute the weights path
    let pretrained_dir = Path.Combine(__SOURCE_DIRECTORY__,"../tests/pretrained")

    let example_dir = Path.Combine(__SOURCE_DIRECTORY__,"../tests/examples")

    let weights_path = Path.Combine(pretrained_dir, sprintf "fast_style_weights_%s.npz" style)
    new TFShape([| 0L |])
    typeof<TFShape>.GetConstructors()
    // Read the weights map
    let readWeights (graph: TFGraph) = 
        readFromNPZ(File.ReadAllBytes(weights_path))
        |> Array.map (fun (k,(metadata, arr)) -> 
            k.Substring(0, k.Length-4), graph.Reshape(graph.Const(new TFTensor(arr)), graph.Const((new TFShape(metadata.shape |> Array.map int64)).AsTensor()))) 
        |> Map.ofArray

    // The average pixel in the decoding
    let mean_pixel () = 
        tf { return DT.ConstArray [| 123.68; 116.778; 103.939 |] }

    // Tensor to read the input
    let input_string = 
        tf { let bytes = File.ReadAllBytes(Path.Combine(example_dir,"chicago.jpg"))
             return DT.CreateString (bytes) } 

    // The decoding tf
    let input = 
        tf { 
            let jpg = DT.DecodeJpeg(input_string)
            let decoded = DT.Cast<_, double>(jpg)
            let preprocessed = decoded - mean_pixel ()
            let expanded = DT.ExpandDims(preprocessed)
            return expanded
        }

    // Run the style transfer
    let img_styled = DT.RunTFTensor (PretrainedFFStyleVGG input)

    // NOTE: Assumed NHWC dataformat
    let tensorToPNG (batchIndex:int) (imgs:TFTensor) =
        if imgs.TFDataType <> TFDataType.Single then failwith "type unsupported"
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
