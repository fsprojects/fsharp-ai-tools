#I __SOURCE_DIRECTORY__
#r "netstandard"
#I "../tests/bin/Debug/net461/"
#r "TensorFlow.FSharp.dll"
#r "TensorFlow.FSharp.Proto.dll"
#r "Argu.dll"
#r "TensorFlow.FSharp.Tests.dll"


open System
open System.IO
open Argu
open TensorFlow.FSharp
open TensorFlow.FSharp.Operations
open TensorFlow.FSharp.NPYReaderWriter

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

type Argument =
    | [<AltCommandLine([|"-s"|])>] Style of string
    with
        interface IArgParserTemplate with
            member this.Usage =
                match this with
                |  Style _ -> "Specify a style of painting to use."

let style = ArgumentParser<Argument>().Parse(fsi.CommandLineArgs.[1..]).GetResult(<@ Argument.Style @>, defaultValue = "rain")

let test_dir = Path.Combine(__SOURCE_DIRECTORY__, "..", "tests")
let pretrained_dir = Path.Combine(test_dir,"pretrained")
let example_dir = Path.Combine(test_dir,"examples")

let sess = new TFSession()
let graph = sess.Graph

let weights_path = Path.Combine(pretrained_dir, sprintf "fast_style_weights_%s.npz" style)
let weights = 
            readFromNPZ((File.ReadAllBytes(weights_path)))
            |> Map.toArray 
            |> Array.map (fun (k,(metadata, arr)) -> 
                k.Substring(0, k.Length-4), graph.Reshape(graph.Const(new TFTensor(arr)), graph.Const(TFShape(metadata.shape |> Array.map int64).AsTensor()))) 
            |> Map.ofArray

//let input = graph.Placeholder(TFDataType.Float32, TFShape(1L,474L,712L,3L),"input")

let input_string = graph.Placeholder(TFDataType.String)

let output_img = 
    let mean_pixel = graph.Const(new TFTensor([|123.68f; 116.778f; 103.939f|]))

    let img = 
        let decoded = graph.Cast(graph.DecodeJpeg(contents=input_string, channels=3L), TFDataType.Float32)
        let preprocessed = graph.Sub(decoded,mean_pixel)
        let expanded = graph.ExpandDims(input=preprocessed, dim = graph.Const(new TFTensor(0)))
        //let resized = graph.ResizeBicubic(expanded,graph.Const(new TFTensor([|256;256|])),align_corners=Nullable(true))
        expanded

    let output = FFStyleVGG.model(graph, img, weights)
    let output = graph.Print(output,[||],"output")

    let (+) x (y:float32) = graph.Add(x,graph.Const(new TFTensor(y)))
    let (*) x (y:float32) = graph.Mul(x,graph.Const(new TFTensor(y)))
    let tanh x = graph.Tanh(x)
    let clip_by_value(low:float32,hight:float32) x = 
        graph.ClipByValue(x,graph.Const(new TFTensor(low)), graph.Const(new TFTensor(hight)))

    let to_pixel_value x = (tanh(x) * 150.f) + (255.f/2.f)

    output
    |> to_pixel_value
    |> clip_by_value(0.f,255.f)

let img_tf = TFTensor.CreateString(File.ReadAllBytes(Path.Combine(example_dir,"chicago.jpg"))) 

let img_styled = sess.Run([|input_string|],[|img_tf|],[|output_img|]).[0]

/// NOTE: Assumed NHWC dataformat
/// TODO: Generalize this, enable the ability work on batches
let tensorToPNG(batchIndex:int) (imgs:TFTensor) =
    if imgs.TFDataType <> TFDataType.Float32 then failwith "type unsupported"
    match imgs.Shape |> Array.map int with
    | [|_N;H;W;_C|] ->
        let pixels = 
            [|
                let res_arr = imgs.GetValue() :?> Array
                for h in 0..H-1 do
                    for w in 0..W-1 do
                        let getV(c) = byte <| Math.Min(255.f, Math.Max(0.f, (res_arr.GetValue(int64 batchIndex, int64 h, int64 w, int64 c) :?> float32)))
                        yield BitConverter.ToInt32([|getV(0); getV(1); getV(2); 255uy|], 0) // NOTE: Channels are commonly in RGB format
            |]
        TensorFlow.FSharp.ImageWriter.RGBAToBitmap(H,W,pixels)
        //TensorFlow.FSharp.ImageWriter.RGBAToPNG(H,W,pixels)
    | _ -> failwithf "shape %A is unsupported" imgs.Shape


File.WriteAllBytes(Path.Combine(__SOURCE_DIRECTORY__, sprintf "chicago_in_%s_style.bmp" style), tensorToPNG 0 img_styled)
