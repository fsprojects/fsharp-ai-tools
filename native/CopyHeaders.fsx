// This copies headers from the tensorflow repository into this one
// This also inclues some cpp files that are (maybe?) needed
// This will need to be re-run upon updates of versions. It's probably best if this 
// stays a manually invoked process.

open System
open System.IO

/// NOTE: Actual tensorflow source directory will be specific to your computer
let tensorflowSource = Path.Combine(__SOURCE_DIRECTORY__, "..", "..","tensorflow")

let headerFiles = Directory.GetFiles(tensorflowSource,"*.h",SearchOption.AllDirectories)

let specificCPPFiles = 
    [|
        Path.Combine("tensorflow", "c", "tf_status_helper.cc")
        //Path.Combine("tensorflow", "c","record_reader.cc")
        Path.Combine("tensorflow", "core", "lib", "io", "record_reader.cc") // NOTE: This appears to have been moved from c -> core/lib/io
        Path.Combine("tensorflow", "c", "python_api.cc") // TODO: Check if this is needed
        Path.Combine("tensorflow", "c", "checkpoint_reader.cc")
        // NOTE: Location of the server code, the best way to incorporate this is TBD
        //Path.Combine("tensorflow", "core","distributed_runtime","server_lib.cc") 
        //Path.Combine("tensorflow", "core","distributed_runtime","grpc_server_lib.cc") 
    |] |> Array.map (fun x -> Path.Combine(tensorflowSource, x))

let targetFolder =  Path.Combine(__SOURCE_DIRECTORY__, "include")

[| yield! headerFiles ; yield! specificCPPFiles |]
|> Array.iter (fun source -> 
    let dest = Path.Combine(targetFolder, source.Substring(tensorflowSource.Length + 1))
    Directory.CreateDirectory(Path.GetDirectoryName(dest)) |> ignore
    File.Copy(source,dest,true)
    )
