// System.TypeLoadException: Could not load type 'System.Runtime.InteropServices.GCHandle' from assembly 'System.Runtime, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a'.
(*

#I @"..\TensorFlow.FSharp\bin\Debug\netstandard2.0"
#r "Google.Protobuf.dll"
#r "HDF.PInvoke.dll"
#r "protobuf-net.dll"
#r "TensorFlow.Proto.dll"
#r "TensorFlow.FSharp.dll"
#r "netstandard"

open Tensorflow
open System

let c1 = TF.Const(new Tensor([|1;4|]))
let c2 = TF.Const(new Tensor(3))
let c3 = TF.Mul(c1,c2)

let Run(output:Output[]) =
    TF.DefaultSession.Run([||],[||],output)

Run([|c3|])


let x = TF.Placeholder(DType.Int32,name="p2")

x.Operation.ControlInputs
x.Operation.ControlOutputs
x.Operation.OpType

/// Breadth first search through a graph, returns first found node if there is one
/// Skipping previously visited nodes
/// This will eventually terminate as there is a finite number of nodes and expand is not done on visited nodes
let breadthFirstSearch<'a when 'a : comparison>(x:'a, expand:'a-> 'a[], found:'a-> bool) =
    ([x],Set.empty)
    |> Seq.unfold (
        function 
        | [],_ -> None
        | x::xs,visited ->
            if visited |> Set.contains x then Some((None,(xs,visited)))
            else 
                if x |> found then Some((Some(x),([],Set.empty)))
                else Some((None,(xs@(expand(x) |> Array.toList),visited.Add(x)))))
    |> Seq.choose id
    |> Seq.tryHead

(*
/// Raises an error if we backprop through a loop var.
let private RaiseNoGradWrtInitialLoopValError(op:TFOperation, from_ops:Set<TFOperation>, xs:TFOutput[]) = 
    match breadthFirstSearch(op, (fun x -> NonEagerInputs(x,xs) |> Array.map (fun x -> x.op)), from_ops.Contains) with
    | None -> failwith "Unable to find target op"
    | Some(target_op) ->
         [sprintf "Cannot compute gradient inside while loop with respect to op '%s'. " target_op.name
          "We do not support taking the gradient wrt or through the initial value "
          "of a loop variable. Gradients can be computed through loop invariants "
          "or wrt the input parameters to the loop body."] 
         |> String.concat "\n"
         |> ValueError |> raise
*)

//TF_OperationGetControlInputs
// TODO recursivly traverse upwards

//type Operation with
//    member this.Inputs = [|for x in 0 .. this.NumInputs - 1 -> this..[x]|]

// Const overloads and implicits
// Const implicits? can we pass 
// a Const function that uses implicits

//Const(0) * 1.0 / 2.0

//let [<Literal>] tflib = "C:\EE\Git\TensorFlow.FSharp\lib\libtensorflow.dll"

//module Native = 
//    // extern const char * TF_OperationOpType (TF_Operation *oper)
//    [<DllImport(tflib)>]
//    extern IntPtr TF_OperationOpType(TF_Operation oper)
//
//let res = Native.TF_OperationOpType(x.Operation.Handle)
//
//Marshal.PtrToStringAnsi(res)
*)
