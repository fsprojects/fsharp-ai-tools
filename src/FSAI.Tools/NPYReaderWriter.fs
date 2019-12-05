module FSAI.Tools.NPYReaderWriter

open System
open System.IO
open System.IO.Compression

// WARN: This does not support nested record arrays and object arrays. For now this is by design.
// TODO: Support Strings

module Result =
    let requireOk (result:Result<'a,string>) = 
        match result with
        | Error(msg) -> failwith msg
        | Ok(x) -> x

[<RequireQualifiedAccess>]
type NPYDType = 
    | Bool
    | Byte
    | UInt8
    | UInt16
    | UInt32
    | UInt64
    | Int8
    | Int16
    | Int32
    | Int64
    | Float16
    | Float32
    | Float64
    with 
        member this.ToNPYString() =
            match this with
            | Bool      -> "b1"
            | Byte      -> "i1"
            | UInt8     -> "u1"
            | UInt16    -> "u2"
            | UInt32    -> "u4"
            | UInt64    -> "u8"
            | Int8      -> "i1"
            | Int16     -> "i2"
            | Int32     -> "i4"
            | Int64     -> "i8"
            | Float16   -> "f2"
            | Float32   -> "f4"
            | Float64   -> "f8"

        member this.ByteWidth =
            match this with
            | Bool      
            | Byte      
            | UInt8     
            | Int8      -> 1
            | UInt16    
            | Int16     
            | Float16   -> 2
            | UInt32    
            | Int32     
            | Float32   -> 4
            | UInt64    
            | Int64     
            | Float64   -> 8

        static member FromNPYString(npyString:string) =
            match npyString with
            | "b1" -> Bool      
            | "i1" -> Byte      
            | "u1" -> UInt8
            | "u2" -> UInt16    
            | "u4" -> UInt32    
            | "u8" -> UInt64    
            | "i2" -> Int16
            | "i4" -> Int32
            | "i8" -> Int64
            | "f2" -> Float16
            | "f4" -> Float32
            | "f8" -> Float64
            | _ -> failwith "unsupported"
        
        static member FromType(t:Type) =
            let t = if t.IsArray then t.GetElementType() else t
            if   t = typeof<bool>     then NPYDType.Bool
            elif t = typeof<byte>     then NPYDType.Byte
            elif t = typeof<uint8>    then NPYDType.UInt8
            elif t = typeof<uint16>   then NPYDType.UInt16
            elif t = typeof<uint32>   then NPYDType.UInt32
            elif t = typeof<uint64>   then NPYDType.UInt64
            elif t = typeof<int16>    then NPYDType.Int16
            elif t = typeof<int32>    then NPYDType.Int32
            elif t = typeof<int64>    then NPYDType.Int64
            elif t = typeof<float32>  then NPYDType.Float32
            elif t = typeof<double>   then NPYDType.Float64
            else failwith "unsupported"

let private ASCII = System.Text.ASCIIEncoding.ASCII

/// "x93NUMPY10"
let private magic_string = [|147uy; 78uy; 85uy; 77uy; 80uy; 89uy; 1uy; 0uy;|]

type NPYDescription = 
    {
        npyDType:NPYDType
        isLittleEnding:bool option
        /// Fortran is column-major and C is row-major
        fortran_order:bool 
        shape:int[]
    }
    with
        static member Default = 
            {
                    npyDType=NPYDType.Float32; 
                    isLittleEnding=Some(true); 
                    fortran_order=false;shape=[||]
            }

let readNumpy(bytes:byte[]) : Result<(NPYDescription*Array),string> =
    let parseHeader(header:string) =
        let headerRecord = 
            header.Split('{','}') 
            |> Array.filter (String.IsNullOrWhiteSpace >> not)
            |> Array.head

        let getPropertyValue(name) =
            match headerRecord.IndexOf(sprintf "'%s':" name) with
            | -1 -> Error (sprintf "not found %s" name)
            | i -> 
                // Scan through to the next ',' that is not inside a parens stack
                let charArray = 
                    Array.unfold (fun (index:int,parenStackCount:int) -> 
                        if index >= headerRecord.Length then None
                        else
                            let c = headerRecord.[index] 
                            match c with
                            | '(' -> Some(c,(index+1,parenStackCount+1))
                            | ')' -> Some(c,(index+1,parenStackCount-1))
                            | ',' when parenStackCount = 0 -> None
                            | _ -> Some(c,(index+1,parenStackCount))
                        ) (i + name.Length + 3,0)
                Ok (String(charArray).Trim())
        
        let (isLittle,npyDType) = 
            match (getPropertyValue("descr") |> Result.requireOk).ToCharArray() with
            | [|'\'';little;x;y;'\''|] -> 
                let isLittle = match little with | '|' -> None | '<' -> Some(true) | '>' -> Some(false) | _ -> failwith "unsupported"
                (isLittle, NPYDType.FromNPYString(String([|x;y|])))
            | _ -> failwith "error parsing descr"

        let fortran_order = 
            match getPropertyValue("fortran_order") |> Result.requireOk with
            | "False" -> false
            | "True" -> true 
            | x -> failwithf "fortan_order value %s is unsupported" x

        let shape = 
            (getPropertyValue("shape") 
            |> Result.requireOk
            |> String.filter (function | '(' | ')' | ' ' -> false | _ -> true)).Split(',')
            |> Array.filter (String.IsNullOrWhiteSpace >> not)
            |> Array.map (Int32.Parse)

        {npyDType=npyDType; isLittleEnding=isLittle; fortran_order=fortran_order; shape=shape;}

    let convertToArray(bytes:byte[],offset:int,shape:int[],dtype:NPYDType) : Array =
        let length = shape |> Seq.reduce (*)
        match dtype with
        | NPYDType.Bool ->
            let returnArray = Array.zeroCreate<float> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.Byte ->
            bytes.[offset..] :> Array
        | NPYDType.UInt8 -> 
            let returnArray = Array.zeroCreate<uint8> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.UInt16 -> 
            let returnArray = Array.zeroCreate<uint16> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.UInt32 ->
            let returnArray = Array.zeroCreate<uint32> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.UInt64 ->
            let returnArray = Array.zeroCreate<uint64> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.Int8 -> 
            let returnArray = Array.zeroCreate<int8> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.Int16 -> 
            let returnArray = Array.zeroCreate<int16> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.Int32 ->
            let returnArray = Array.zeroCreate<int32> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.Int64 ->
            let returnArray = Array.zeroCreate<int64> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.Float16 -> 
            let returnArray = Array.zeroCreate<uint16> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray |> Array.map System.Half.ToHalf :> Array
        | NPYDType.Float32 ->
            let returnArray = Array.zeroCreate<float32> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array
        | NPYDType.Float64 ->
            let returnArray = Array.zeroCreate<double> length
            Buffer.BlockCopy(bytes,offset,returnArray,0,length * dtype.ByteWidth)
            returnArray :> Array

    if magic_string  <> bytes.[..7] then
        Error "Magic String missmatch"
    else 
        let len = BitConverter.ToUInt16(bytes,8)
        let header = parseHeader <| ASCII.GetString(bytes,10,int len) 
        Ok (header, convertToArray(bytes,10 + int len, header.shape, header.npyDType))

/// Note: It is assumed that arr already has the propper C format data layout
let writeArrayToNumpy(arr:Array,shape:int[]) : byte[] = 
    let size = shape |> Array.reduce (*)
    let dtype = NPYDType.FromType(arr.GetType())
    let typeDesc = sprintf "%c%s" (if dtype.ByteWidth = 1 then '|' else '<') (dtype.ToNPYString())
    let header = sprintf "{'descr': '%s', 'fortran_order': False, 'shape': (%s), }" 
                    typeDesc (shape |> Array.map string |> String.concat ", ")
    let preambleLen = 10
    let len = header.Length + 1
    let headerSize = len + preambleLen
    let paddedHeader = header.PadRight(header.Length + (16 - (headerSize % 16))) + "\n"
    if (10 + paddedHeader.Length) % 16 <> 0 then
        failwith "header pad error"
    let bufferSize : int = preambleLen + paddedHeader.Length + size * dtype.ByteWidth
    let buffer = Array.zeroCreate<byte> bufferSize
    Buffer.BlockCopy(magic_string, 0, buffer, 0, magic_string.Length)
    let lenBits = BitConverter.GetBytes(uint16 (paddedHeader.Length))
    buffer.[8] <- lenBits.[0]
    buffer.[9] <- lenBits.[1]
    let headerBytes = ASCII.GetBytes(paddedHeader)
    Buffer.BlockCopy(headerBytes ,0, buffer, preambleLen, headerBytes.Length)
    /// NOTE: This only works for simple single dimention arrays
    Buffer.BlockCopy(arr,0,buffer,preambleLen + headerBytes.Length, size * dtype.ByteWidth)
    buffer
    
(*
// WARN: Untested
let saveToNPZ(npys:Map<string,NPYDescription*Array>) =
    use ms = new MemoryStream()
    use zip = new ZipArchive(ms,ZipArchiveMode.Create)
    for KeyValue(key,(desc,arr)) in npys do
        let entry = zip.CreateEntry(key)
        let s = entry.Open()
        let buffer = writeArrayToNumpy(arr,desc.shape)
        s.Write(buffer,0,buffer.Length)
        s.Flush()
        s.Close() // double check if any of these are redundant
        s.Dispose()
    ms.ToArray()
*)

let readFromNPZ(data:byte[]) : Map<string,(NPYDescription*Array)> =
    use ms = new MemoryStream(data)
    use zip = new ZipArchive(ms, ZipArchiveMode.Read)
    [|
        for entry in zip.Entries ->
            use targetMs = new MemoryStream()
            use entryStream = entry.Open()
            entryStream.CopyTo(targetMs)
            let bytes = targetMs.ToArray()
            let desc,arr = readNumpy(bytes) |> Result.requireOk
            entry.FullName, (desc,arr)
    |] |> Map.ofArray