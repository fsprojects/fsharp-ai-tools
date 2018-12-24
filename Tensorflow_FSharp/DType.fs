namespace Tensorflow
open System


/// <summary>
/// The data type for a specific tensor.
/// </summary>
/// <remarks>
/// Tensors have uniform data types, all the elements of the tensor are of this
/// type and they dictate how TensorFlow will treat the data stored.   
/// </remarks>
[<RequireQualifiedAccess>]
type DType = 
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

    type DType with    
        member this.IsComplex 
            with get() = 
                match this with
                | DType.Complex | DType.Complex64 | DType.Complex128 -> true
                | _ -> false

        member this.IsFloat
            with get() = 
                match this with
                | DType.Float16 | DType.Float32 | DType.Float64 | DType.BFloat16 -> true
                | _ -> false

        member this.IsInteger 
            with get() = 
                match this with
                | DType.Int8 | DType.Int16 | DType.Int32 | DType.Int64 -> true
                | _ -> false
        
        member this.Name 
            with get() = 
                match this with
                | DType.Float16 -> "float16"
                | DType.Float32 -> "float32"
                | DType.Float64 -> "float64"
                | DType.Int32 -> "int32"
                | DType.UInt8 -> "uint8"
                | DType.UInt16 -> "uint16"
                | DType.UInt32 -> "uint32"
                | DType.UInt64 -> "uint64"
                | DType.Int16 -> "int16"
                | DType.Int8 -> "int8"
                | DType.String -> "string"
                | DType.Complex64 -> "complex64"
                | DType.Complex128 -> "complex128"
                | DType.Int64 -> "int64"
                | DType.Bool -> "bool"
                // | DType.QInt8 -> "qint8"
                // | DType.QUInt8 -> "quint8"
                // | DType.QInt16 -> "qint16"
                // | DType.QUInt16 -> "quint16"
                // | DType.QInt32 -> "qint32"
                | DType.BFloat16 -> "bfloat16"
                | DType.Resource -> "resource"
                | DType.Variant -> "variant"
                | _ -> String.Empty
        
        /// Returns `True` if this `DType` represents a reference type.
        member this.IsRefDtype with get() = ofEnum(this) > 100u
        /// Returns a reference `DType` based on this `DType`.
        member this.AsRef with get() = if this.IsRefDtype then this else  (toEnum(ofEnum(this) + 100u))
        /// Returns a non reference `DType` based on this `DType`.
        member this.BaseType with get() = if this.IsRefDtype then (toEnum(ofEnum(this) - 100u)) else this






