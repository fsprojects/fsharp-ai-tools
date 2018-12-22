// [<AutoOpen>]
// module Tensorflow.common

// open System
// open Utils
// open TensorFlow
// open Tensorflow
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

// type TFOperation with
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

// type TFOutput with
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



// /// TODO, this could be generalized with the ability
// /// to instantiate a default value. Though it's possible anything 
// /// beyond this is is not needed
// type DictionaryCount<'a when 'a : equality>() =
//     inherit Dictionary<'a,int>()
//     member this.Item 
//         with get(x:'a) = if this.ContainsKey(x) then this.[x] else 0
//         and set(x:'a) (v:int) = this.Add(x,v)
//     member this.Increment(x:'a) = this.[x] <- this.[x] + 1
//     member this.Decrement(x:'a) = this.[x] <- this.[x] - 1




// type PendingCount = DictionaryCount<TFOperation>
 
// /// This is analogous to replace the 
// /// type GradientMapping = Map<int,TFOutput>
// type TFOperationGradients = TFOutput option [] //Map<int,TFOutput>

// // NOTE: Behaviour here has been modified to hopefully be clearer
// type GradientDictionary = Dictionary<TFOperation,TFOperationGradients >
// // TODO, check return type here. I think it's a TFOutput
// // Also, it seems that GradientMapping is optional
// type GradientFunction = (TFOperation*TFOperationGradients-> TFOutput)

// type ICallable =
//     abstract member __call__ : unit -> unit