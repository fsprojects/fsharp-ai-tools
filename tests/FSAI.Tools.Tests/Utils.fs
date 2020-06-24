module FSAI.Tools.Tests.Utils

open System
open System.IO
open Tensorflow
open FSAI.Tools
open Tensorflow.Operations

#if FS47
open Tensorflow.Binding
#else
let tf = Tensorflow.Binding.tf
#endif

type np = NumSharp.np

let basePath = Path.Combine(__SOURCE_DIRECTORY__,"..","..")

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
    [| for KeyValue(k,v) in np.Load_Npz<Array>(File.ReadAllBytes(path)) -> (k,tf.constant(NumSharp.NDArray(v))) |] |> Map.ofArray

/// "rain" "wave" "starry_night"
let fetchStyleWeights(style:string) = 
    let file = sprintf "fast_style_weights_%s.npz" style
    let output = Path.Combine(basePath,"pretrained",file)
    if not(File.Exists(output)) then
        downloadFile(sprintf "https://s3-us-west-1.amazonaws.com/public.data13/TF_examples/%s" file,output)
    [| for KeyValue(k,v) in np.Load_Npz<Array>(File.ReadAllBytes(output)) -> (k.Substring(0,k.Length-4),tf.constant(NumSharp.NDArray(v))) |] |> Map.ofArray

// TODO: clean up this code
let fetchClassifierWeights() = 
    let file = "resnet_classifier_1000.npz"
    let output = Path.Combine(basePath,"pretrained",file)
    if not(File.Exists(output)) then
        downloadFile(sprintf "https://s3-us-west-1.amazonaws.com/public.data13/TF_examples/%s" file,output)
    [| for KeyValue(k,v) in np.Load_Npz<Array>(File.ReadAllBytes(output)) -> (k,tf.constant(NumSharp.NDArray(v))) |] |> Map.ofArray

let is64 = if System.Environment.Is64BitProcess then true else exit 100; false
