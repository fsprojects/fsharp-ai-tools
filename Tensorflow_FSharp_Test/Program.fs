// Learn more about F# at http://fsharp.org

open System
open Tensorflow

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"
    let t = new Tensor([|0uy;0uy|])
    printfn "%s" (t.ToString())
    0 // return an integer exit code
