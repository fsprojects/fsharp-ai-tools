[<AutoOpen>]
module Tensorflow.Utils
open System
open System.Collections.Generic
open System.Collections
open Microsoft.FSharp.NativeInterop
open System.Runtime.InteropServices

#nowarn "9"

module Option =
    let orNull (x:'a option) = match x with | None -> box null :?> 'a | Some(x) -> x
    let mapOrNull (f:'a->'b) (x:'a option)  = match x with | None -> box null :?> 'b | Some(x) -> f(x)
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
    // Not sure what to do with the warning here...
    let cast<'b> (xs:IEnumerable) = [|for x in xs -> x :?> 'b |]

module NativePtr =
    let nativeIntRead<'a when 'a : unmanaged> (ptr:IntPtr) = ptr |> NativePtr.ofNativeInt |> NativePtr.read<'a>
    let nativeIntWrite<'a when 'a : unmanaged> (ptr:IntPtr) (x:'a) = NativePtr.write (ptr |> NativePtr.ofNativeInt) x
    let nativeIntGet<'a when 'a : unmanaged> (ptr:IntPtr) (i:int) =  NativePtr.get<'a> (ptr |> NativePtr.ofNativeInt) i
    let nativeIntSet<'a when 'a : unmanaged> (ptr:IntPtr)(i:int) (x:'a) =  NativePtr.set (ptr |> NativePtr.ofNativeInt) i x 

let (|Integer|_|) (str: string) =
   let mutable intvalue = 0
   if System.Int32.TryParse(str, &intvalue) then Some(intvalue)
   else None

/// to instantiate a default value. Though it's possible anything 
/// beyond this is is not needed
type DictionaryCount<'a when 'a : equality>() =
    let dict = new Dictionary<'a,int>()
    /// NOTE: it would be nice to be able to override Item
    member this.Item 
        with get(x:'a) = if dict.ContainsKey(x) then dict.[x] else 0
        and set (x:'a) (v:int)  = dict.Add(x,v)
    member this.Increment(x:'a) = this.[x] <- this.[x] + 1
    member this.Decrement(x:'a) = this.[x] <- this.[x] - 1
    member this.GetThenIncrement(x:'a) = 
        let value = this.[x]
        this.Increment(x)
        value

let isAssignableTo<'a> (x:_) = typeof<'a>.IsAssignableFrom(x.GetType())

let inline (!>) (x:^a) : ^b = ((^a or ^b) : (static member op_Implicit : ^a -> ^b) x)

/// This is for value types
let valueToIntPtr (v:'a) = 
    let intPtr = Marshal.AllocHGlobal (sizeof<'a>)
    NativePtr.nativeIntWrite intPtr v
    intPtr