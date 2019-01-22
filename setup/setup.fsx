#load "NugetDownload.fsx"
open NugetDownload

// TODO extract all OS specific files for Tensorflow

let nugetFiles = 
    [| 
        "HDF.PInvoke.NETStandard/1.10.200",
        [|
            match os with
            | Linux -> yield! [|"libhdf5_hl.so";"libhdf5_hl.so.101";"libhdf5.so";"libhdf5.so.101"|] |> Array.map (sprintf @"runtimes/linux-x64/native/%s")
            | Windows -> yield! [|"hdf5_hl.dll";"hdf5.dll";"zlib1.dll"|] |> Array.map (sprintf @"runtimes/win-x64/native/%s")
            | OSX -> yield! [|"libhdf5_hl.dylib"; "libhdf5.dylib"|] |> Array.map (sprintf @"runtimes/osx-x64/native/%s")
        |]
        "TensorFlowSharp/1.11.0",
        [|
            //yield @"lib/netstandard2.0/TensorFlowSharp.dll"
//            match os with
//            | Linux -> yield! [|"libtensorflow_framework.so"; "libtensorflow.so"|] |> Array.map (sprintf @"runtimes/linux/native/%s")
//            | Windows -> yield @"runtimes/win7-x64/native/libtensorflow.dll"
//            | OSX -> yield! [|"libtensorflow_framework.dylib"; "libtensorflow.dylib"|] |> Array.map (sprintf @"runtimes/osx/native/%s")
              yield! [|"libtensorflow_framework.so"; "libtensorflow.so"|] |> Array.map (sprintf @"runtimes/linux/native/%s")
              yield @"runtimes/win7-x64/native/libtensorflow.dll"
              yield! [|"libtensorflow_framework.dylib"; "libtensorflow.dylib"|] |> Array.map (sprintf @"runtimes/osx/native/%s")
        |]
        // https://eiriktsarpalis.wordpress.com/2013/03/27/a-declarative-argument-parser-for-f/ 
        "Argu/5.1.0",[|"lib/netstandard2.0/Argu.dll"|]
        "Google.Protobuf/3.6.1", [|"dll";"xml"|] |> Array.map (sprintf "lib/net45/Google.Protobuf.%s")
        "protobuf-net/2.4.0",[|"dll";"xml"|] |> Array.map (sprintf "lib/net40/protobuf-net.%s")
        "protobuf-net.protogen/2.3.17", [| for x in [|"protobuf-net";"protobuf-net.Reflection"|] do for y in [|"dll";"xml"|] -> sprintf "tools/netcoreapp2.1/any/%s.%s" x y |]
    |]

downloadAndExtractNugetFiles nugetFiles

open System.IO

let dir = __SOURCE_DIRECTORY__
let lib = Path.Combine(dir, "..","lib")

printfn "Building Tensorflow Proto"
runFSI (Path.Combine(dir,"BuildTensorflowProto.fsx"))
printfn "Finished building Tensorflow Proto"

printfn "Code genearting operations"
runFSC (sprintf "%s -o %s" (Path.Combine(dir,"LinuxNativeWorkaround.fs")) (Path.Combine(lib,"LinuxNativeWorkaround.dll")))
runFSI (Path.Combine(dir,"OperationCodeGenerationFSharp.fsx"))
printfn "Finished code genearting operations"

printfn "Fetching pre-trained weights for testing"
[| 
  yield! ["rain"; "starry_night"; "wave"] |> Seq.map (sprintf "fast_style_weights_%s.npz")
  yield "imagenet1000.txt"
  yield "resnet_classifier_1000.npz"
|]
|> Seq.iter (fun file -> 
    downloadFile(sprintf "https://s3-us-west-1.amazonaws.com/public.data13/TF_examples/%s" file, 
                 System.IO.Path.Combine(dir,"..","tests","pretrained",file)))
printfn "Finished fetching pre-trained weights for testing"

printfn "Setup has finished."