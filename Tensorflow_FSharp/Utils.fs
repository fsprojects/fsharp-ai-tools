[<AutoOpen>]
module Tensorflow.Utils
open System.Collections.Generic

module Option =
    let orNull (x:'a option) = match x with | None -> box null :?> 'a | Some(x) -> x
    let orDefault (default_:'a) (x:'a option)  = match x with | None -> default_ | Some(x) -> x
    let orDefaultLazy (default_:Lazy<'a>) (x:'a option)  = 
        match x with | None -> default_.Force() | Some(x) -> x
    let orDefaultDelay (f:(unit -> 'a)) (x:'a option)  = 
        match x with | None -> f() | Some(x) -> x
    let tryNull(x) = if box x <> null then Some(x) else None
    let ofRef (result:bool,byref:'a) = if result then Some(byref) else None
    let collect (x:'a[] option) = match x with | None -> [||] | Some(xs) -> xs; 
        

type Dictionary<'a,'b> with 
    member this.TryGet(x) = match this.TryGetValue(x) with | (true,v) -> Some(v) | _ -> None
    member this.TryFind(x) = this.TryGet(x) 
    member this.GetOrAdd(key:'a,value:unit -> 'b) : 'b =
        this.TryGet(key) 
        |> Option.orDefaultDelay(fun _ -> value() |> fun v -> this.Add(key,v);v)

module Result =
    let tryOkWithError(x:Result<'a,string>) = match x with | Ok x -> x | Error x -> failwith x

module Map =
    let values (x:Map<_,_>) = [|for KeyValue(k,v) in x -> v|]
    let keys (x:Map<_,_>) = [|for KeyValue(k,v) in x -> k|]

module Enum =
    let castMap (f:'a->'b) (x:'c): 'd = LanguagePrimitives.EnumOfValue(f(LanguagePrimitives.EnumToValue x))
    let cast (x:'a) : 'b = castMap id x

module Array =
    let all (f:'a -> bool ) (xs:'a[]) : bool = xs |> Array.exists (f >> not) |> not
    let enumerate (xs:'a[]) = xs |> Array.mapi (fun i x -> (i,x))
    let update (i:int) (x:'a) (xs:'a[]) = xs |> Array.mapi (fun j y -> if i = j then x else y)


/// to instantiate a default value. Though it's possible anything 
/// beyond this is is not needed
type DictionaryCount<'a when 'a : equality>() =
    inherit Dictionary<'a,int>()
    member this.Item 
        with get(x:'a) = if this.ContainsKey(x) then this.[x] else 0
        and set(x:'a) (v:int) = this.Add(x,v)
    member this.Increment(x:'a) = this.[x] <- this.[x] + 1
    member this.Decrement(x:'a) = this.[x] <- this.[x] - 1
    member this.GetThenIncrement(x:'a) = 
        let value = this.[x]
        this.Increment(x)
        value