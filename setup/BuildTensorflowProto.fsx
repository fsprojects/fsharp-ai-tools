
// This file is being used because the protogen.dll (exe) approach is fragile.

// NOTE: Don't include the common proto as these aready exist in Google.Protobuf and will conflict
// NOTE: But do include descriptor.proto

#r "netstandard.dll"
#I "../lib"
#r "protobuf-net.dll"
#r "protobuf-net.Reflection.dll"
#r "Google.Protobuf.dll"
#load "NugetDownload.fsx"

open ProtoBuf.Reflection
open System.IO
open NugetDownload

let directory = Path.Combine(__SOURCE_DIRECTORY__ , "..", "tensorflow","proto")
let protoFiles = Directory.GetFiles(directory,"*.proto",SearchOption.AllDirectories)
let googleDirectory = Path.Combine(__SOURCE_DIRECTORY__, "..", "tensorflow", "google")

/// fix for namespace errors
let namespaceErrors =
    [|
        ("error.Code","Code")
        ("tfprof.CodeDef","CodeDef")
        ("tfprof.OpLogProto", "OpLogProto")
        ("tpu.op_profile.Profile", "Profile")
        ("decision_trees.InequalityTest.Type", "InequalityTest.Type")
        ("decision_trees.TreeNode", "TreeNode")
        ("decision_trees.BinaryNode", "BinaryNode")
        ("decision_trees.Vector", "Vector")
        ("decision_trees.SparseVector","SparseVector")
    |]

let tryFixNamespace(text : string) = namespaceErrors |> Array.fold (fun (x:string) (s:string,t:string) -> x.Replace(s,t)) text

let toLinuxPath(x:string) = x |> String.map (function | '\\' -> '/' | x -> x)

let codeFiles = 
    [|
        yield! protoFiles  
        |> Array.filter (fun x -> not(x.Contains(Path.Combine("lite","toco"))))
        |> Array.map (fun x -> CodeFile(toLinuxPath(Path.Combine("tensorflow",x.Substring(directory.Length + 1))),tryFixNamespace(File.ReadAllText(x))))
        yield CodeFile("google/protobuf/descriptor.proto", File.ReadAllText(Path.Combine(googleDirectory , "protobuf", "descriptor.proto")))
    |]

codeFiles |> Array.rev |> Array.map (fun x -> x.Name)

let compilerResult = CSharpCodeGenerator.Default.Compile(codeFiles)

compilerResult.Errors

if compilerResult.Errors |> Array.filter (fun x -> not(x.IsWarning)) |> Array.length > 0 then failwith "Too many proto compiler errors"

let outDirectory = Path.Combine(__SOURCE_DIRECTORY__,  "protogen")

if Directory.Exists(outDirectory) then Directory.Delete(outDirectory,true) |> ignore
Directory.CreateDirectory(outDirectory) |> ignore


let rebaseNamespace(str:string) = 
 str.Replace("namespace Tensorflow","namespace TensorFlow.FSharp.Proto")
    .Replace("namespace tensorflow.","namespace TensorFlow.FSharp.Proto.")
    .Replace("global::tensorflow.","global::TensorFlow.FSharp.Proto.")
    .Replace("global::Tensorflow.","global::TensorFlow.FSharp.Proto.")

for file in compilerResult.Files do 
    let fn = Path.Combine(outDirectory, Path.GetFileName(file.Name))
    File.WriteAllText(fn, rebaseNamespace(file.Text))


(*
# previous bash commands for reference
csc -langversion:latest -target:library -reference:./lib/protobuf-net.dll -reference:./lib/Google.Protobuf.dll -out:TensorFlow.FSharp.Proto.dll -doc:TensorFlow.FSharp.Proto.xml ./protogen/*.cs
*)

//let requireRunProcessWithEnvs (name:string) (envs:(string*string)[]) (args:string)  = if runProcess None name envs args   <> 0 then failwith (sprintf "%s failed to exit correctly" name)

//let requireRunProcess (name:string) (args:string) = requireRunProcessWithEnvs name [||] args 

let libDir = Path.Combine(__SOURCE_DIRECTORY__, "..", "lib")

let csc  = 
    match os with
    | Windows -> 
        let tryDir dir f = if File.Exists dir then dir else f()
        tryDir @"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\MSBuild\15.0\Bin\Roslyn\csc.exe" (fun () -> 
            tryDir @"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\MSBuild\15.0\Bin\Roslyn\csc.exe" (fun () -> 
               failwith "replace csc.exe path with one which works on your computer"))

    | Linux -> "csc"
    | OSX -> failwith "todo - support Mac OSX"

if not(File.Exists(csc)) then failwith "replace csc.exe path with one which works on your computer"

let references = ["protobuf-net.dll"; "Google.Protobuf.dll"] |> List.map (sprintf @"-reference:%s%c%s" libDir Path.DirectorySeparatorChar) |> String.concat  " " 

// NOTE: documentation xml is empty
Directory.GetFiles(outDirectory, "*.cs") |> String.concat " "
|> sprintf @"-langversion:latest -target:library %s -out:%s%cTensorFlow.FSharp.Proto.dll %s" references libDir Path.DirectorySeparatorChar
|> runProcess csc 

// Clean up generated C# code
Directory.Delete(outDirectory,true) |> ignore
