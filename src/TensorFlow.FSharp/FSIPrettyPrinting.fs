module TensorFlow.FSharp.FSIPrettyPrinting

(*

// TODO figure out how to cleanly register Microsoft.FSharp.Compiler.Interface

open TensorFlow.FSharp
open Compiler.Interactive
do
    fsi.AddPrintTransform(fun (x:obj) ->
        match x with
        | :? IFSIPrint as x -> x.ToFSIString()
        | _ -> null)
*)
