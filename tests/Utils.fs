module FSharp.AI.Tests.Utils

open Tensorflow
open FSharp.AI
open System.IO
open Tensorflow.Operations
open FSharp.AI.NPYReaderWriter

#if FS47
open Tensorflow.Binding
#else
let tf = Tensorflow.Binding.tf
#endif

let basePath = Path.Combine(__SOURCE_DIRECTORY__,"..")

let downloadFile(url:string,path:string) =
    Directory.CreateDirectory(Path.GetDirectoryName(path)) |> ignore
    if not(File.Exists(path)) then
        use wc = new System.Net.WebClient()
        printfn "Downloading %s -> %s" url path
        wc.DownloadFile(url,path)
        printfn "Completed %s -> %s" url path
    else 
        printfn "File %s already exists"  path

let loadWeight(path:string) =
    path 
    |> File.ReadAllBytes 
    |> readFromNPZ
    |> Map.map (fun k (metadata,arr) -> tf.reshape(tf.constant(arr ), tf.constant(metadata.shape)))

/// "rain" "wave" "starry_night"
let fetchStyleWeights(style:string) = 
    let file = sprintf "fast_style_weights_%s.npz" style
    let output = Path.Combine(basePath,"pretrained",file)
    if not(File.Exists(output)) then
        downloadFile(sprintf "https://s3-us-west-1.amazonaws.com/public.data13/TF_examples/%s" file,output)
    output
    |> File.ReadAllBytes |> readFromNPZ
    |> Map.toArray 
    |> Array.map (fun (k,(metadata, arr)) -> 
        // TODO: make initialization of tensor to be the right shape w/o doing a graph.Reshape here
        // This way the graph functions defined here will be defined as part of the model and not feeding into the model
        k.Substring(0, k.Length-4), gen_ops.reshape(tf.constant(arr), tf.constant(metadata.shape))) 
    |> Map.ofArray

// TODO: clean up this code
let fetchClassifierWeights() = 
    let file = "resnet_classifier_1000.npz"
    let output = Path.Combine(basePath,"pretrained",file)
    if not(File.Exists(output)) then
        downloadFile(sprintf "https://s3-us-west-1.amazonaws.com/public.data13/TF_examples/%s" file,output)
    output
    |> File.ReadAllBytes
    |> readFromNPZ
    |> Map.map (fun k (metadata,arr) -> tf.reshape(tf.constant(arr), tf.constant(metadata.shape)))

let is64 = if System.Environment.Is64BitProcess then true else System.Environment.Exit(-1); false
