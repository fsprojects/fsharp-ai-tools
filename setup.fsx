#load "shared/NugetDownload.fsx"
open NugetDownload

let nugetFiles = 
    [| 
        "ionic.zlib/1.9.1.5", [|"lib/Ionic.Zlib.dll"|]
        "HDF.PInvoke.NETStandard/1.10.200",
        [|
            yield @"lib/netstandard2.0/HDF.PInvoke.dll"
            match os with
            | Linux -> yield! [|"libhdf5_hl.so";"libhdf5_hl.so.101";"libhdf5.so";"libhdf5.so.101"|] |> Array.map (sprintf @"runtimes/linux-x64/native/%s")
            | Windows -> yield! [|"hdf5_hl.dll";"hdf5.dll";"zlib1.dll"|] |> Array.map (sprintf @"runtimes/win-x64/native/%s")
            | OSX -> yield! [|"libhdf5_hl.dylib"; "libhdf5.dylib"|] |> Array.map (sprintf @"runtimes/osx-x64/native/%s")
        |]
        "TensorFlowSharp/1.11.0",
        [|
            yield @"lib/netstandard2.0/TensorFlowSharp.dll"
            match os with
            | Linux -> yield! [|"libtensorflow_framework.so"; "libtensorflow.so"|] |> Array.map (sprintf @"runtimes/linux/native/%s")
            | Windows -> yield @"runtimes/win7-x64/native/libtensorflow.dll"
            | OSX -> yield! [|"libtensorflow_framework.dylib"; "libtensorflow.dylib"|] |> Array.map (sprintf @"runtimes/osx/native/%s")
        |]
        // https://eiriktsarpalis.wordpress.com/2013/03/27/a-declarative-argument-parser-for-f/ 
        "Argu/5.1.0",[|"lib/netstandard2.0/Argu.dll"|]
        "Google.Protobuf/3.6.1", [|"dll";"xml"|] |> Array.map (sprintf "lib/net45/Google.Protobuf.%s")
        "protobuf-net/2.4.0",[|"dll";"xml"|] |> Array.map (sprintf "lib/net40/protobuf-net.%s")
        //"protobuf-net/2.4.0",[|"dll";"xml"|] |> Array.map (sprintf "lib/netcoreapp2.1/protobuf-net.%s")
    |]

downloadAndExtractNugetFiles nugetFiles

failwith "todo - fetch TensorFlowSharpProto from a yet to be created Nuget package"

printfn "Setup has finished."