[<AutoOpen>]
module FSharp.ML.Onnx.Utils.Common
open System

let swap (x,y) = (y,x)

let (|TryFunc|_|) (f: 'a -> 'b option ) (x:'a) = f x
let (|Found|_|) map key = map |> Map.tryFind key

module Option =
    let all (xs: Option<'a>[]) = if xs |> Seq.exists ((=) None) then None else Some(xs |> Array.map Option.get) 

module List =
    let chop n xs =
        if n < 0 then invalidArg "n" "n must not be negative"
        let rec split l =
            match l with
            | 0, xs -> [], xs
            | n, x :: xs ->
                let front, back = split (n-1, xs)
                x :: front, back
            | _, [] -> failwith "List.chop: not enough elts list"
        split (n, xs)

    /// TODO, do a better name and more efficent implementation
    let splitWhen (f : 'a -> bool) (xs: 'a list) = (xs |> List.takeWhile f, xs |> List.skipWhile f)

    let reshape  (shape: int list) (xs: 'a list) : 'a list list =
        (([],xs),shape) ||> List.fold (fun (acc,xs) count -> chop (max count xs.Length) xs |> fun (head,xs) -> (head::acc,xs)) |> fst |> List.rev

type DisposableValue<'a>(value : 'a, dispose : unit -> unit) =
    let disposed = lazy dispose()
    member this.Disposed = disposed.IsValueCreated
    member this.Value = 
        if disposed.IsValueCreated then
            failwithf "Attempted access on disposed object"
        else value
    member this.Dispose() = disposed.Force()
    override this.Finalize() = disposed.Force()
    interface IDisposable with
        member this.Dispose() = disposed.Force()
