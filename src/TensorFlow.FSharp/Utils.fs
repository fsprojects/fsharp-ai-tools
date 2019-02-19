[<AutoOpen>]
module TensorFlow.FSharp.Utils
open System
open System.Collections
open System.Collections.Generic
open System.Collections.Concurrent
open FSharp.NativeInterop
open System.Runtime.InteropServices

#nowarn "9"

module Option =
    let orNull (x:'T option) = match x with | None -> box null :?> 'T | Some(x) -> x

    let mapOrNull (f:'T->'b) (x:'T option)  = match x with | None -> box null :?> 'b | Some(x) when box x = null -> box null :?> 'b | Some(x) -> f(x)

    let tryNull(x) = if box x <> null then Some(x) else None

    let ofRef (result:bool,byref:'T) = if result then Some(byref) else None

    let collect (x:'T[] option) = match x with | None -> [||] | Some(xs) -> xs; 
    /// nullable
    let ofNullable (x : 'T) = if box x <> null then Some x else None
    /// System.Nullable
    let ofSystemNullable (x : System.Nullable<'T>) = if x.HasValue then Some x.Value else None
    let tryGetAll (xs:'T option []) =
        if xs |> Array.exists (function | None -> true | _ -> false)
        then None
        else xs |> Array.map (Option.get) |> Some
    let ofType<'Out> : obj -> 'Out option = function | :? 'Out as x -> Some x | _ -> None
    let tryNullOfType<'Out>(x:obj) : 'Out option = x |> tryNull |> ofType<'Out>
    let ofUnrelatedType<'In, 'Out> (x : 'In) = box x |> ofType<'Out>
    let isOfType<'T> (x:obj) = match x with | :? 'T -> true | _ -> false
    let orElse (second:'T option) (first:'T option)  = match first with | Some(x) -> Some(x) | None -> match second with | Some(y) -> Some(y) | None -> None
    let tryFind (predicate:'T -> bool) (x:'T option) = match x with | Some(x) when predicate x -> Some(x) | _ -> None
    let dispose<'T when 'T :> IDisposable> (x:'T option) = x |> Option.iter (fun x -> x.Dispose())

type Dictionary<'T,'b> with 

    member this.TryGet(x) = match this.TryGetValue(x) with | (true,v) -> Some(v) | _ -> None

    member this.TryFind(x) = this.TryGet(x) 

    member this.GetOrAdd(key:'T,value:unit -> 'b) : 'b =
        this.TryGet(key) 
        |> Option.defaultWith(fun () -> value() |> fun v -> this.Add(key,v);v)

module Result =

    let tryOkWithError(x:Result<'T,string>) = match x with | Ok x -> x | Error x -> failwith x

module Map =

    let values (x:Map<_,_>) = [|for KeyValue(k,v) in x -> v|]

    let keys (x:Map<_,_>) = [|for KeyValue(k,v) in x -> k|]

module Enum =

    let castMap (f:'T->'b) (x:'c): 'd = LanguagePrimitives.EnumOfValue(f(LanguagePrimitives.EnumToValue x))

    let cast (x:'T) : 'b = castMap id x

module Array =

    let update (i:int) (x:'T) (xs:'T[]) = xs |> Array.mapi (fun j y -> if i = j then x else y)

    // Not sure what to do with the warning here...
    let cast<'b> (xs:IEnumerable) = [|for x in xs -> x :?> 'b |]

    /// Segments a sequence according to a provided sequence of segment lengths
    let segment (ns:int[]) (xs:'a[]) =  
        failwith "untested"
        [|for (start,length) in (ns.[..ns.Length-2] |> Array.scan (+) 0,ns) ||> Array.zip -> xs.[start..start+length]|]

    let removeByIndex (i:int) (xs:'T[]) = [|for (j,x) in xs |> Array.indexed do if j <> i then yield x |]

    // NOTE: could probably do this by mutation, uncomment if needed
    //let updateByIndex (i:int) (f : 'T -> 'T) (xs:'T[]) = [|for (j,x) in xs |> Array.indexed -> if j <> i then x else f(x)|]


module NativePtr =

    let nativeIntRead<'T when 'T : unmanaged> (ptr:IntPtr) = ptr |> NativePtr.ofNativeInt |> NativePtr.read<'T>

    let nativeIntWrite<'T when 'T : unmanaged> (ptr:IntPtr) (x:'T) = NativePtr.write (ptr |> NativePtr.ofNativeInt) x

    let nativeIntGet<'T when 'T : unmanaged> (ptr:IntPtr) (i:int) =  NativePtr.get<'T> (ptr |> NativePtr.ofNativeInt) i

    let nativeIntSet<'T when 'T : unmanaged> (ptr:IntPtr)(i:int) (x:'T) =  NativePtr.set (ptr |> NativePtr.ofNativeInt) i x 

    let intPtrToVoidPtr (ptr:IntPtr) = ptr |> NativePtr.ofNativeInt<int64> |> NativePtr.toVoidPtr

    /// This returns 
    let ofOption (x:Option<nativeptr<'T>>) = match x with | Some(x) -> x | None -> IntPtr.Zero |> NativePtr.ofNativeInt

let (|Integer|_|) (str: string) =
   let mutable intvalue = 0
   if System.Int32.TryParse(str, &intvalue) then Some(intvalue)
   else None

/// to instantiate a default value. Though it's possible anything 
/// beyond this is is not needed
type DictionaryCount<'T when 'T : equality>() =

    let dict = new Dictionary<'T,int>()

    /// NOTE: it would be nice to be able to override Item
    member this.Item 
        with get(x:'T) = if dict.ContainsKey(x) then dict.[x] else 0
        and set (x:'T) (v:int)  = dict.[x] <- v

    member this.Increment(x:'T) = this.[x] <- this.[x] + 1

    member this.Decrement(x:'T) = this.[x] <- this.[x] - 1

    member this.GetThenIncrement(x:'T) = 
        let value = this.[x]
        this.Increment(x)
        value

    member this.ContainsKey(x) = dict.ContainsKey(x)

let isAssignableTo<'T> (x:_) = typeof<'T>.IsAssignableFrom(x.GetType())

/// This is for value types
let valueToIntPtr (v:'T) = 
    let intPtr = Marshal.AllocHGlobal (sizeof<'T>)
    NativePtr.nativeIntWrite intPtr v
    intPtr

//// Not sure how to do this in a more type safe way, i.e. require contains parameter to be of same type as array
//type System.Collections.Generic.IEnumerable<'b> with
//
//    member this.Contains(x:'T) = seq { for x in this -> x} |> Seq.exists (fun y -> x.Equals(y))

// Not sure how to do this in a more type safe way, i.e. require contains parameter to be of same type as array
type System.Collections.IEnumerable with

    member this.Contains(x:'T) = seq { for x in this -> x} |> Seq.exists (fun y -> x.Equals(y))

type IntPtr with 

    member this.Add(x:int) = IntPtr(int64 this + int64 x)

    member this.Add(x:int64) = IntPtr(int64 this + x)

    member this.Sub(x:int) = IntPtr(int64 this - int64 x)

    member this.Sub(x:int64) = IntPtr(int64 this - x)

type UIntPtr with 

    member this.Add(x:int) = UIntPtr(uint64 this + uint64 x)

    member this.Add(x:int64) = UIntPtr(uint64 this + uint64 x)

    member this.Sub(x:int) = UIntPtr(uint64 this - uint64 x)

    member this.Sub(x:int64) = UIntPtr(uint64 this - uint64 x)


/// This interface enables FSI specific printing
type IFSIPrint =
    abstract member ToFSIString : unit -> string

type ConcurrentDictionary<'a,'b> with
    member this.TryGet(x) =
        match this.TryGetValue(x) with
        | (true,value) -> Some(value)
        | (false,_) -> None

    member this.Update(index:'a, update:'b->'b) = 
        let b' = update this.[index]
        this.[index] <- b'
        b'


