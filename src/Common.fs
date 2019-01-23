namespace TensorFlow.FSharp

// This is common TensorFlow specific types / utils

open System.Runtime.InteropServices

// We use this TF_Xxx as the native "TF_Xxx *" as those are opaque
type TF_Status = System.IntPtr

type TF_SessionOptions = System.IntPtr

type TF_Graph = System.IntPtr

type TF_OperationDescription = System.IntPtr

type TF_Operation = System.IntPtr

type TF_Session = System.IntPtr

type TF_DeprecatedSession = System.IntPtr

type TF_Tensor = System.IntPtr

type TF_ImportGraphDefOptions = System.IntPtr

type TF_Library = System.IntPtr

type TF_BufferPtr = System.IntPtr

type TF_Function = System.IntPtr

type TF_DeviceList = System.IntPtr

// [<AutoOpen>]
// module TensorFlow.common

// open System
// open Utils
// open TensorFlow
// open TensorFlow
// open System.Collections.Generic

// module OS =
//     let tryGetEnv(name:string) = 
//         System.Environment.GetEnvironmentVariable(name,System.EnvironmentVariableTarget.Process)
//         |> Option.ofObj
        
        

// module context = 
//     /// For now eager execution is not supported
//     let executing_eagerly() = false

// //open tensorflow
// type LookupError(msg:string) =
//     inherit Exception(msg)

// type ValueError(msg:string) =
//     inherit Exception(msg)

// type TypeError(msg:string) =
//     inherit Exception(msg)

// type Operation with
//     member this.try_get_output(i:int) = if i >= 0 && i < this.NumOutputs then Some(this.[i]) else None
//     member this.``type`` = this.OpType
//     member this.graph = this.Graph


// module op_def_registery = 
//     let get_registered_ops() = failwith "todo"


// /// NOTE This is from python threading namespace and may not make sense
// module threading =
//     let local() : bool = failwith "todo"

// type Args = obj[]
// type KWArgs = Map<string,obj>


// let isSameObject = LanguagePrimitives.PhysicalEquality

// type Output with
//     member this.name : string = failwith "todo"
//     member this.device : string = failwith "todo"

// /// Breadth first search through a graph, returns first found node if there is one
// /// Skipping previously visited nodes
// /// This will eventually terminate as there is a finite number of nodes and expand is not done on visited nodes
// let breadthFirstSearch<'a when 'a : comparison>(x:'a, expand:'a-> 'a[], found:'a-> bool) =
//     ([x],Set.empty)
//     |> Seq.unfold (
//         function 
//         | [],_ -> None
//         | x::xs,visited ->
//             if visited |> Set.contains x then Some((None,(xs,visited)))
//             else 
//                 if x |> found then Some((Some(x),([],Set.empty)))
//                 else Some((None,(xs@(expand(x) |> Array.toList),visited.Add(x)))))
//     |> Seq.choose id
//     |> Seq.tryHead

// let isinstance<'a>(x:obj) = 
//     let t = typeof<'a>
//     x.GetType().IsSubclassOf(t) || x.GetType() = t

// type PendingCount = DictionaryCount<Operation>
 
// /// This is analogous to replace the 
// /// type GradientMapping = Map<int,Output>
// type OperationGradients = Output option [] //Map<int,Output>

// // NOTE: Behaviour here has been modified to hopefully be clearer
// type GradientDictionary = Dictionary<Operation,OperationGradients >
// // TODO, check return type here. I think it's a Output
// // Also, it seems that GradientMapping is optional
// type GradientFunction = (Operation*OperationGradients-> Output)

// type ICallable =
//     abstract member __call__ : unit -> unit