namespace TensorFlow.FSharp

open System
open System.Numerics

/// <summary>
/// The data type for a specific tensor.
/// </summary>
/// <remarks>
/// Tensors have uniform data types, all the elements of the tensor are of this
/// type and they dictate how TensorFlow will treat the data stored.   
/// </remarks>
[<RequireQualifiedAccess>]
type TFDataType = 
    | Unknown       = 0u
    | Float32       = 1u
    | Float64       = 2u
    | Int32         = 3u
    | UInt8         = 4u
    | Int16         = 5u
    | Int8          = 6u
    | String        = 7u
    | Complex64     = 8u
    | Complex       = 8u
    | Int64         = 9u
    | Bool          = 10u
    | QInt8         = 11u
    | QUInt8        = 12u
    | QInt32        = 13u
    | BFloat16      = 14u
    | QInt16        = 15u
    | QUInt16       = 16u
    | UInt16        = 17u
    | Complex128    = 18u
    | Float16       = 19u
    | Resource      = 20u
    | Variant       = 21u
    | UInt32        = 22u
    | UInt64        = 23u

[<AutoOpen>]
module DTypeExtensions =
    let private toEnum = LanguagePrimitives.EnumOfValue
    let private ofEnum = LanguagePrimitives.EnumToValue

    type TFDataType with    
        member this.IsComplex =
            match this with
            | TFDataType.Complex | TFDataType.Complex64 | TFDataType.Complex128 -> true
            | _ -> false

        member this.IsFloat =
            match this with
            | TFDataType.Float16 | TFDataType.Float32 | TFDataType.Float64 | TFDataType.BFloat16 -> true
            | _ -> false

        member this.IsInteger =
            match this with
            | TFDataType.Int8 | TFDataType.Int16 | TFDataType.Int32 | TFDataType.Int64 -> true
            | _ -> false
        
        member this.Name =
            match this with
            | TFDataType.Float16 -> "float16"
            | TFDataType.Float32 -> "float32"
            | TFDataType.Float64 -> "float64"
            | TFDataType.Int32 -> "int32"
            | TFDataType.UInt8 -> "uint8"
            | TFDataType.UInt16 -> "uint16"
            | TFDataType.UInt32 -> "uint32"
            | TFDataType.UInt64 -> "uint64"
            | TFDataType.Int16 -> "int16"
            | TFDataType.Int8 -> "int8"
            | TFDataType.String -> "string"
            | TFDataType.Complex64 -> "complex64"
            | TFDataType.Complex128 -> "complex128"
            | TFDataType.Int64 -> "int64"
            | TFDataType.Bool -> "bool"
            // | TFDataType.QInt8 -> "qint8"
            // | TFDataType.QUInt8 -> "quint8"
            // | TFDataType.QInt16 -> "qint16"
            // | TFDataType.QUInt16 -> "quint16"
            // | TFDataType.QInt32 -> "qint32"
            | TFDataType.BFloat16 -> "bfloat16"
            | TFDataType.Resource -> "resource"
            | TFDataType.Variant -> "variant"
            | _ -> String.Empty
        
        /// Returns `True` if this `TFDataType` represents a reference type.
        member this.IsRefDtype = 
            ofEnum(this) > 100u

        /// Returns a reference `TFDataType` based on this `TFDataType`.
        member this.AsRef = 
            if this.IsRefDtype then this else  (toEnum(ofEnum(this) + 100u))

        /// Returns a non reference `TFDataType` based on this `TFDataType`.
        member this.BaseType = 
            if this.IsRefDtype then (toEnum(ofEnum(this) - 100u)) else this

        member this.ByteSize =
            match this with
            | TFDataType.Unknown       -> -1 
            | TFDataType.Float32       -> 4
            | TFDataType.Float64       -> 8
            | TFDataType.Int32         -> 4
            | TFDataType.UInt8         -> 1
            | TFDataType.Int16         -> 2
            | TFDataType.Int8          -> 1
            | TFDataType.String        -> -1
            | TFDataType.Complex64     -> 16
            //| TFDataType.Complex       -> 16, this will never be used because it's equal to Complex64
            | TFDataType.Int64         -> 8
            | TFDataType.Bool          -> 1
            | TFDataType.QInt8         -> 1
            | TFDataType.QUInt8        -> 1
            | TFDataType.QInt32        -> 4
            | TFDataType.BFloat16      -> 2
            | TFDataType.QInt16        -> 2
            | TFDataType.QUInt16       -> 2
            | TFDataType.UInt16        -> 2
            | TFDataType.Complex128    -> 32
            | TFDataType.Float16       -> 2
            | TFDataType.Resource      -> -1
            | TFDataType.Variant       -> -1
            | TFDataType.UInt32        -> 4
            | TFDataType.UInt64        -> 8
            | _                   -> -1

        /// <summary>
        /// Converts a <see cref="TFDataType"/> to a system type.
        /// </summary>
        /// <param name="type">The <see cref="TFDataType"/> to be converted.</param>
        /// <returns>The system type corresponding to the given <paramref name="type"/>.</returns>
        member this.ToType =
            match this with
            | TFDataType.Float32 -> typeof<single>
            | TFDataType.Float64 -> typeof<double>
            | TFDataType.Int32 -> typeof<int32>
            | TFDataType.UInt8 -> typeof<uint8>
            | TFDataType.UInt16 -> typeof<uint16>
            | TFDataType.UInt32 -> typeof<uint32>
            | TFDataType.UInt64 -> typeof<uint64>
            | TFDataType.Int16 -> typeof<int16>
            | TFDataType.Int8 -> typeof<int8> // sbyte?
            | TFDataType.String -> typeof<string> // TFString?
            | TFDataType.Complex128 -> typeof<Complex>
            | TFDataType.Int64 -> typeof<int64>
            | TFDataType.Bool -> typeof<bool>
            | _ -> box null :?> Type

        /// <summary>
        /// Converts a system type to a <see cref="TFDataType"/>.
        /// </summary>
        /// <param name="t">The system type to be converted.</param>
        /// <returns>The <see cref="TFDataType"/> corresponding to the given type.</returns>
        static member FromType (t:Type) : TFDataType = 
            //if true then TFDataType.Float32 else TFDataType.Unknown
            if   t = typeof<float32>     then TFDataType.Float32
            elif t = typeof<double>    then TFDataType.Float64
            elif t = typeof<int>       then TFDataType.Int32 
            elif t = typeof<byte>      then TFDataType.UInt8
            elif t = typeof<int16>     then TFDataType.Int16
            elif t = typeof<sbyte>     then TFDataType.Int8
            elif t = typeof<string>    then TFDataType.String
            elif t = typeof<int64>     then TFDataType.Int64
            elif t = typeof<bool>      then TFDataType.Bool
            elif t = typeof<uint16>    then TFDataType.UInt16
            elif t = typeof<Complex>   then TFDataType.Complex128
            else raise(ArgumentOutOfRangeException ("t", sprintf "The given type could not be mapped to an existing TFDataType."))
        
        static member Double = TFDataType.Float64
        static member Single = TFDataType.Float32
