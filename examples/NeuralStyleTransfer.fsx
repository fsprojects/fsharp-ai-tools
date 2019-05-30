#I __SOURCE_DIRECTORY__
#r "netstandard"
#I "../tests/bin/Debug/net461/"
#r "TensorFlow.FSharp.dll"
#r "TensorFlow.Net.dll"
#r "NumSharp.Core.dll"
#r "Argu.dll"
#r "TensorFlow.FSharp.Tests.dll"


open System
open System.IO
open Argu
open Tensorflow
open Tensorflow.Operations
open TensorFlow.FSharp
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

let sess = new Session()
let graph = sess.graph

let weights_path = Path.Combine(pretrained_dir, sprintf "fast_style_weights_%s.npz" style)
let weights = 
    readFromNPZ((File.ReadAllBytes(weights_path)))
    |> Map.toArray 
    |> Array.map (fun (k,(metadata, arr)) -> 
        k.Substring(0, k.Length-4), gen_ops.reshape(tf.constant(arr), tf.constant(TensorShape(metadata.shape)))) 
    |> Map.ofArray

//let input = graph.Placeholder(TFDataType.Float32, TFShape(1L,474L,712L,3L),"input")

let input_string = tf.placeholder(TF_DataType.TF_STRING)

let output_img = 
    let mean_pixel = tf.constant([|123.68f; 116.778f; 103.939f|])

    let img = 
        let decoded = tf.cast(gen_ops.decode_jpeg(contents=input_string, channels=Nullable(3)), TF_DataType.TF_FLOAT)
        let preprocessed = tf.sub(decoded,mean_pixel)
        let expanded = gen_ops.expand_dims(input=preprocessed, dim = tf.constant(0))
        //let resized = graph.ResizeBicubic(expanded,graph.Const(new TFTensor([|256;256|])),align_corners=Nullable(true))
        expanded

    let output = FFStyleVGG.model(img, weights)
    let output = gen_ops.print(output,[||],"output")

    let (+) x (y:float32) = tf.add(x,tf.constant(y))
    let (*) x (y:float32) = tf.multiply(x,tf.constant(y))
    let tanh x = tf.tanh(x)
    let clip_by_value(low:float32,hight:float32) x = 
        gen_ops.clip_by_value(x,tf.constant(low), tf.constant(hight))

    let to_pixel_value x = (tanh(x) * 150.f) + (255.f/2.f)

    output
    |> to_pixel_value
    |> clip_by_value(0.f,255.f)

let img_tf = new Tensor(File.ReadAllBytes(Path.Combine(example_dir,"chicago.jpg")))

// TODO update the run format here
//let img_styled = sess.Run([|input_string|],[|img_tf|],[|output_img|]).[0]


//File.WriteAllBytes(Path.Combine(__SOURCE_DIRECTORY__, sprintf "chicago_in_%s_style.bmp" style), tensorToPNG 0 img_styled)
