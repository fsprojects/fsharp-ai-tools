#I __SOURCE_DIRECTORY__
#r "netstandard"

#load "shared/NPYReaderWriter.fsx"
#load "shared/ImageWriter.fsx"
#load "shared/dsl.fsx"
#nowarn "49"

//open Argu
open NPYReaderWriter
open System
open System.IO
open TensorFlow.FSharp

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

module Play1 = 
    let t1 = new TFTensor( [| 6 |])
    let shape1 = TFShape [| 1L |] 
    let shape2 = TFShape [| 1L |] 

    // TODO: TFShape should support basic things like equality
    (shape1 = shape2)
    (shape1.Equals(shape2))

    //tf.add(1, 2)
    //tf.add([1, 2], [3, 4])
    //tf.square(5)
    //tf.reduce_sum([1, 2, 3])
    //tf.encode_base64("hello world")



module PlayWithAddingConstants = 
    let t1 = new TFTensor( [| 6 |])
    let graph = new TFGraph()
    graph.MakeName("a", "b")
    let out1 = graph.Const(t1)
    let out2 = graph.Add(out1, out1)

    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], [| out1; out2 |])
    res1.[0].GetValue()
    res1.[1].GetValue()
    
module PlayWithUsingVariables = 
    let t1 = new TFTensor( [| 6 |])
    let graph = new TFGraph()
    let out1 = graph.Const(new TFTensor( [| 6 |]))
    let out2 = graph.Const(new TFTensor( [| 1 |]))
    let var2 : TFVariable = graph.Variable(out2, trainable = true, name= "var2")
    let out = graph.Add(out1, var2.Read)

    let session = new TFSession(graph)
    let res1 = session.Run( [|  |], [| |], [| out; var2.Read |])
    res1.[0].GetValue()
    res1.[1].GetValue()

    //graph.Const("a", "b")
    //graph.Variable( .MakeName("a", "b")

    //TFTensor(TFDataType.Double, [| 2;2|], )
    new TFTensor(array2D [| [| 0.0; 1.0 |]; [| 0.0; 1.0 |] |])
    (new TFTensor(array2D [| [| 0.0; 1.0 |]; [| 0.0; 1.0 |] ; [| 0.0; 1.0 |] |])).GetTensorDimension(1)
    TFOutput
    //graph.Inpu


module PlayWithTF = 
    tf { return v 1.0 }
    |> DT.Run
   
    tf { return v 1.0 + v 4.0 }
    |> DT.Run

    tf { return vec [1.0; 2.0] + v 4.0 }
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return vec [1.0; 2.0] + v 4.0 }
    |> DT.Diff
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { use _holder = DT.WithScope("foo")
         return vec [1.0; 2.0] + v 4.0 }
    |> DT.Diff
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] }
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] + v 4.0 }
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return var "x" + v 4.0 }
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return var "x" + v 4.0 }
    |> DT.Diff (var "x")
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> fun dt -> DT.Run(dt, ["hey", tf { return upcast (vec [2.0])}])
    |> fun x -> x.GetValue()
       // Gives 6.0

    let input = matrix [ for i in 0 .. 9 -> [ for j in 0 .. 9 -> double (i+j) ]]
    let name = "/a"
    let instance_norm =
        tf { let mu, sigma_sq = DT.Moments ([0;1], input)
             let shift = DT.Variable (v 0.0, name + "/shift")
             let scale = DT.Variable (v 1.0, name + "/scale")
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift }

    instance_norm |> DT.Run |> fun x -> x.GetValue()



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


(*
type Argument =
    | [<AltCommandLine([|"-s"|])>] Style of string
    interface IArgParserTemplate with
        member this.Usage =
            match this with
            |  Style _ -> "Specify a style of painting to use."

let style = ArgumentParser<Argument>().Parse(fsi.CommandLineArgs.[1..]).GetResult(<@ Argument.Style @>, defaultValue = "rain")
*)
let style = "rain"

fsi.AddPrinter(fun (x:TFGraph) -> sprintf "TFGraph %i" (int64 x.Handle))

let pretrained_dir = Path.Combine(__SOURCE_DIRECTORY__,"../tests/pretrained")

let example_dir = Path.Combine(__SOURCE_DIRECTORY__,"../tests/examples")


[<TensorFlow>]
module NeuralStyles = 

    let conv_init_vars (out_channels:int, filter_size:int, name) =
        tf { let truncatedNormal = DT.TruncatedNormal(Shape [| Dim filter_size; Dim filter_size; Dim.Inferred; Dim out_channels |])
             return DT.Variable (truncatedNormal * v 0.1, name + "/weights") }

    let instance_norm (input: DT<double>, name) =
        tf { let mu, sigma_sq = DT.Moments ([1;2], input)
             let shift = DT.Variable (v 0.0, name + "/shift")
             let scale = DT.Variable (v 1.0, name + "/scale")
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift }

    let conv_layer (input: DT<double>, num_filters, filter_size, stride, is_relu, name) = 
        tf { let filters = conv_init_vars (num_filters, filter_size, name)
             let x = DT.Conv2D (input, filters, stride=stride)
             let x = instance_norm (x, name)
             if is_relu then 
                 return DT.Relu x 
             else 
                 return x }

    let residual_block (input, filter_size, name) = 
        tf { let tmp = conv_layer(input, 128, filter_size, 1, true, name)
             return input + conv_layer(tmp, 128, filter_size, 1, false, name) }

    let conv2D_transpose (input, filter, stride) = 
        tf { return DT.Conv2DBackpropInput(filter, input, stride, padding = "SAME") }
  
    let conv_transpose_layer (input: DT<double>, num_filters, filter_size, Stride, name) =
        tf { let filters = conv_init_vars (num_filters, filter_size, name)
             return DT.Relu (instance_norm (conv2D_transpose (input, filters, Stride), name))
           }

    let to_pixel_value (input: DT<double>) = 
        tf { return tanh input * v 150.0 + (v 255.0 / v 2.0) }

    // The style-transfer tf
    let PretrainedFFStyleVGG input_img = 
        tf { let x = conv_layer (input_img, 32, 9, 1, true, "conv1")
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

    // Compute the weights path
    let weights_path = Path.Combine(pretrained_dir, sprintf "fast_style_weights_%s.npz" style)

    // Read the weights map
    let readWeights (graph: TFGraph) = 
        readFromNPZ(File.ReadAllBytes(weights_path))
        |> Array.map (fun (k,(metadata, arr)) -> 
            k.Substring(0, k.Length-4), graph.Reshape(graph.Const(new TFTensor(arr)), graph.Const(TFShape(metadata.shape |> Array.map int64).AsTensor()))) 
        |> Map.ofArray


    // The average pixel in the decoding
    let mean_pixel shape = 
        tf { return DT.ConstArray [| 123.68; 116.778; 103.939 |] }

    // Tensor to read the input
    let input_string = 
        tf { let bytes = File.ReadAllBytes(Path.Combine(example_dir,"chicago.jpg"))
             return DT.CreateString (bytes) } 

    // The decoding tf
    let input_img = 
        tf { 
            let jpg = DT.DecodeJpeg(input_string)
            let decoded = DT.Cast<_, double>(jpg, TFDataType.Single)
            let preprocessed = decoded - mean_pixel decoded.Shape
            let expanded = DT.ExpandDims(preprocessed, 0)
            return expanded
        }

    // Run the style transfer
    let img_styled = DT.Run (PretrainedFFStyleVGG input_img)

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
