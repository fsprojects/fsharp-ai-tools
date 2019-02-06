module TensorFlow.FShapr.API.Utilities

/// Files https://github.com/eaplatanios/tensorflow_scala/tree/a1a704fe79c1415df6eda9a8f1651c3b1984223a/modules/api/src/main/scala/org/platanios/tensorflow/api/utilities

///  Coding.scala

///  CRC32.scala
///  Collections.scala
///  Diposer.scala
///  NativeHandleWrapper.scala
///  Proto.scala
///  Reservoir.scala
///  package.scala

open System
open System.IO

module CRC32 = 
    /// TODO there are more functions that are likely to be needed here
    let crcTable : Lazy<uint32[]> =
        lazy 
            [|
                for n in 0..255 do
                    let mutable c = uint32 n
                    for k in 0..7 do
                        if ((c &&& 1u) <> 0u) then c <- uint32 (0xedb88320L ^^^ ((int64 c) >>> 1)) else c  <- c >>> 1
                    yield c
           |]         

    let getCrc(buf:byte[],start:int,finish:int) =
        let crcTable = crcTable.Force()
        let mutable c = 0xffffffffu 
        for n in start..finish-1 do 
            c <- crcTable.[int(c ^^^ uint32 buf.[n] &&& 0xffu)] ^^^ (c >>> 8)
        c ^^^ 0xffffffffu 
    

/// NOTE: depending on how often we have to do bigEndian we may skip out of this type all together
type Coding() =
    static member EncodeFixedInt16(value : int16, ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.GetBytes(value)
        else BitConverter.GetBytes(value) |> Array.rev
    
    static member EncodeFixedInt32(value : int32, ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.GetBytes(value)
        else BitConverter.GetBytes(value) |> Array.rev
        
    static member EncodeFixedInt64(value : int64, ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.GetBytes(value)
        else BitConverter.GetBytes(value) |> Array.rev

    static member DecodeFixedInt16(value : byte[], ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.ToInt16(value,0)
        else BitConverter.ToInt16(value |> Array.rev,0)
    
    static member DecodeFixedInt32(value : byte[], ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.ToInt32(value,0)
        else BitConverter.ToInt32(value |> Array.rev,0)
        
    static member DecodeFixedInt64(value : byte[], ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.ToInt64(value,0)
        else BitConverter.ToInt64(value |> Array.rev,0)

    static member EncodeStrings(values : String[]) : byte[] = 
        // NOTE: It must be assumed that the number of strings is encoded elsewhere
        let buffers = values |> Array.map System.Text.UTF8Encoding.UTF8.GetBytes
        [|
            for buffer in buffers do yield! BitConverter.GetBytes(buffer.Length)
            for buffer in buffers do yield! buffer
        |]
        
    static member VarIntLength(value  : int) : int = 
        // We are treating `value` as an unsigned integer.
        let B = 128L
        let mutable unsigned = (int64 value) &&& 0xffffffffL
        let mutable length = 1
        while unsigned >= B do
            unsigned <- unsigned >>> 7
            length <- length + 1
        length
    
    static member EncodeVarInt32(value : int) : byte [] =
        // We are treating `value` as an unsigned integer.
        let B = 128L
        let mutable unsigned = (int64 value) &&& 0xffffffffL
        let mutable position = 0
        [|
            while unsigned >= B do
                yield byte ((unsigned &&& (B - 1L)) ||| B)
                unsigned <- unsigned >>> 7
                position <- position + 1
        |]

    static member DecodeVarInt32(bytes : byte[]) : (int*int) = 
        // TODO double check this there may be errors introduced by casting
        let B = 128L
        let mutable b = bytes.[0]
        let mutable value = int b
        let mutable position = 1
        let mutable shift = 7
        while (b &&& byte B) <> 0uy && shift <= 28 && position < bytes.Length do
            // More bytes are present.
            b <- bytes.[position]
            value <- value ||| ((int (b &&& (byte B - 1uy))) <<< shift)
            position <- position + 1
            shift <- shift + 7
        if (b &&& 128uy) <> 0uy then (-1,-1)
        else (value,position)




            
