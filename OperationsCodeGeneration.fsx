// Depricated: See OperationCodeGeneartionNet instead
// There does not seem to be a clean way to extract lists of messages; the encoding of message boundaries is non trivial
// Instead we're using protobuf-net which has this functionality built in


/// TODO get this working to generate C# code then update to generate F# code
/// 
#r @"lib/Google.Protobuf.dll"
#r @"lib/TensorFlowSharpProto.dll"
#r @"lib/nativeWorkaround.dll"
#r @"lib/protobuf-net.dll"

open Google.Protobuf
open Google.Protobuf
open Google.Protobuf.Collections

open System
open ProtoBuf
open System.IO
open NativeWorkaround
open NativeWorkaround.Native
open Microsoft.FSharp.NativeInterop
open System.Collections.Generic
open System.Runtime.InteropServices
open Google.Protobuf.Collections
open System.Collections.Generic

open Tensorflow

#nowarn "9"

type Status() = 
    let mutable handle = TF_NewStatus()
    interface IDisposable with
        member this.Dispose() = TF_DeleteStatus(handle)
    member this.OK = TF_GetCode(handle) = 0
    member this.Error = TF_GetCode(handle) <> 0
    member this.Handle = handle


// let extractProtobuf (parser : Google.Protobuf.MessageParser<'a>) (f: Status -> nativeptr<LLBuffer>)  =
//     use status = new Status()
//     let ptr = f(status)
//     if status.Error then None
//     else
//         let llbuffer = ptr |> NativePtr.read
//         let length = (int(llbuffer.length))
//         let buffer = Array.zeroCreate<byte> length
//         Marshal.Copy (llbuffer.data, buffer, 0, length)
//         Some(parser.ParseFrom(buffer))

// type ApiDefMap(buffer : LLBuffer) = 
//     let mutable handle = 
//         let mutable buffer = buffer
//         use status = new Status()
//         let x = TF_NewApiDefMap ( &&buffer |> NativePtr.toNativeInt, status.Handle)
//         if status.Error then raise (ArgumentException("Failrue to call TF_NewApiDefMap"))
//         x
    
//     member this.Dispose (disposing : bool) =
//         if disposing then
//             if handle <> IntPtr.Zero then
//                 TF_DeleteApiDefMap (handle)
//                 handle <- IntPtr.Zero

//     override this.Finalize() = this.Dispose(false)

//     interface IDisposable with 
//         member this.Dispose() = 
//             this.Dispose(true)
//             GC.SuppressFinalize(this)

//     member this.TryGet (name : string) = extractProtobuf ApiDef.Parser (fun s -> TF_ApiDefMapGet(handle,name,IntPtr(name.Length),s.Handle)) 

//     member this.Put (text : string) =
//         use status = new Status()
//         TF_ApiDefMapPut (handle, text, IntPtr(text.Length),status.Handle)
//         status.OK


// ///
// /// Maps a TensorFlow type to a F# type
// /// 
// let fsharptype (tfType : string) =
//     let list, tfType = if tfType.StartsWith "list(" then true, tfType.Substring(5, tfType.Length - 6) else false, tfType
//     match tfType with
//     | "int"     -> "int64"     |> Some
//     | "float"   -> "float"     |> Some
//     | "bool"    -> "bool"      |> Some
//     | "type"    -> "DType"     |> Some
//     | "shape"   -> "TFShape"   |> Some
//     | "tensor"  -> "TFTensor"  |> Some
//     | "string"  -> "string"    |> Some
//     | _ -> printfn "Unknown data TensorFlow type %s" tfType; None
//     |> Option.map (fun fstype -> if list then fstype + "[]" else fstype)

// let isReferenceType (tfType : string) = tfType.StartsWith ("list(") || tfType = "tensor" || tfType = "string" || tfType = "shape"

// // Maps a parameter name to a F# acceptable name, to avoid classhes with langauge keywords
// let paramMap (paramName : string) = 
//     match paramName with
//     | "out" -> "output"
//     | "params" -> "parameters"
//     | "ref" -> "referecne"
//     | "event" -> "evnt"
//     | _ -> paramName


// // Determines if the specified ArgDef represents a TensorFlow list
// let isListArg (arg : OpDef.Types.ArgDef) = arg.TypeListAttr <> "" || arg.NumberAttr <> ""


// let getOpsList() =
let ptr = TF_GetAllOpList() 
let status = new Status()
let apiDefMapHandle = TF_NewApiDefMap(ptr |> NativePtr.toNativeInt, status.Handle)

TF_ApiDefMapPut(apiDefMapHandle,"_If",IntPtr("_If".Length), status.Handle)

1+1;;

//let buff = TF_ApiDefMapGet(apiDefMapHandle,"_If",IntPtr("_If".Length), status.Handle)

1 + 1;;

status.OK



//     let apimap = new ApiDefMap(ptr)

//     let length = (int ptr.length)
//     let ret = Array.zeroCreate<byte> length
//     Marshal.Copy (ptr.data, ret, 0, length)
//     OpDef.Parser.ParseDelimitedFrom()
//     (apimap,ops)
