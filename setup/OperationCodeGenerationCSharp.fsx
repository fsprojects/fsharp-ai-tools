// This is to test the protobuf-net implementation to see if it can read the tensorflow protbuf buffer

// TODO: figure out how to incorporate func into the type structure
// SkipType: FilterDataset due to attribute (func predicate lacking a mapping to C#
// SkipType: FlatMapDataset due to attribute (func f lacking a mapping to C#
// SkipType: For due to attribute (func body lacking a mapping to C#
// SkipType: GeneratorDataset due to attribute (func init_func lacking a mapping to C#
// SkipType: GroupByReducerDataset due to attribute (func key_func lacking a mapping to C#
// SkipType: GroupByWindowDataset due to attribute (func key_func lacking a mapping to C#
// SkipType: If due to attribute (func then_branch lacking a mapping to C#
// SkipType: InterleaveDataset due to attribute (func f lacking a mapping to C#
// SkipType: MapAndBatchDataset due to attribute (func f lacking a mapping to C#
// SkipType: MapAndBatchDatasetV2 due to attribute (func f lacking a mapping to C#
// SkipType: MapDataset due to attribute (func f lacking a mapping to C#
// SkipType: MapDefun due to attribute (func f lacking a mapping to C#
// SkipType: OneShotIterator due to attribute (func dataset_factory lacking a mapping to C#
// SkipType: ParallelInterleaveDataset due to attribute (func f lacking a mapping to C#
// SkipType: ParallelMapDataset due to attribute (func f lacking a mapping to C#
// SkipType: PartitionedCall due to attribute (func f lacking a mapping to C#
// SkipType: RemoteCall due to attribute (func f lacking a mapping to C#
// SkipType: ScanDataset due to attribute (func f lacking a mapping to C#
// SkipType: StatefulPartitionedCall due to attribute (func f lacking a mapping to C#
// SkipType: StatelessIf due to attribute (func then_branch lacking a mapping to C#
// SkipType: StatelessWhile due to attribute (func cond lacking a mapping to C#
// SkipType: SymbolicGradient due to attribute (func f lacking a mapping to C#
// SkipType: TPUReplicate due to attribute (func computation lacking a mapping to C#
// SkipType: While due to attribute (func cond lacking a mapping to C#)

#I "../lib"
#r @"Google.Protobuf.dll"
#r @"TensorFlow.Proto.dll"
#r @"nativeWorkaround.dll"
#r @"protobuf-net.dll"

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
open TensorFlow.Proto

let deserialize<'a> = Serializer.Deserialize<'a>

#nowarn "9"

type Status() = 
    let mutable handle = TF_NewStatus()
    interface IDisposable with
        member this.Dispose() = TF_DeleteStatus(handle)
    member this.OK = TF_GetCode(handle) = 0
    member this.Error = TF_GetCode(handle) <> 0
    member this.Handle = handle


let extractProtobuf (parser : Stream -> 'a) (f: Status -> nativeptr<LLBuffer>)  =
    use status = new Status()
    let ptr = f(status)
    if status.Error then None
    else
        let llbuffer = ptr |> NativePtr.read
        let length = (int(llbuffer.length))
        let buffer = Array.zeroCreate<byte> length
        Marshal.Copy (llbuffer.data, buffer, 0, length)
        Some(parser(new MemoryStream(buffer)))

type ApiDefMap(buffer : LLBuffer) = 
    let mutable handle = 
        let mutable buffer = buffer
        use status = new Status()
        let x = TF_NewApiDefMap ( &&buffer |> NativePtr.toNativeInt, status.Handle)
        if status.Error then raise (ArgumentException("Failrue to call TF_NewApiDefMap"))
        x
    
    member this.Dispose (disposing : bool) =
        if disposing then
            if handle <> IntPtr.Zero then
                TF_DeleteApiDefMap (handle)
                handle <- IntPtr.Zero

    override this.Finalize() = this.Dispose(false)

    interface IDisposable with 
        member this.Dispose() = 
            this.Dispose(true)
            GC.SuppressFinalize(this)

    member this.TryGet (name : string) = extractProtobuf deserialize<ApiDef> (fun s -> TF_ApiDefMapGet(handle,name,IntPtr(name.Length),s.Handle)) 

    member this.Put (text : string) =
        use status = new Status()
        TF_ApiDefMapPut (handle, text, IntPtr(text.Length),status.Handle)
        status.OK


///
/// Maps a TensorFlow type to a F# type
/// 
let fsharptype (tfType : string) =
    let list, tfType = if tfType.StartsWith "list(" then true, tfType.Substring(5, tfType.Length - 6) else false, tfType
    match tfType with
    | "int"     -> "int64"     |> Some
    | "float"   -> "float"     |> Some
    | "bool"    -> "bool"      |> Some
    | "type"    -> "TFDataType"     |> Some
    | "shape"   -> "TFShape"   |> Some
    | "tensor"  -> "TFTensor"  |> Some
    | "string"  -> "string"    |> Some
    | _ -> printfn "Unknown data TensorFlow type %s" tfType; None
    |> Option.map (fun fstype -> if list then fstype + "[]" else fstype)

let isReferenceType (tfType : string) = tfType.StartsWith ("list(") || tfType = "tensor" || tfType = "string" || tfType = "shape"

// Maps a parameter name to a F# acceptable name, to avoid classhes with langauge keywords
let paramMap (paramName : string) = 
    match paramName with
    | "out" -> "output"
    | "params" -> "parameters"
    | "ref" -> "referecne"
    | "event" -> "evnt"
    | _ -> paramName


// Determines if the specified ArgDef represents a TensorFlow list
let isListArg (arg : OpDef.ArgDef) = arg.TypeListAttr <> "" || arg.NumberAttr <> ""


let getOpsList() =
    let ptr = TF_GetAllOpList() |> NativePtr.read
    let apimap = new ApiDefMap(ptr)
    let length = (int ptr.length)
    let ret = Array.zeroCreate<byte> length
    Marshal.Copy (ptr.data, ret, 0, length)
    let ms = new MemoryStream(ret)
    let ops = deserialize<List<OpDef>>(ms)
    (ops |> Seq.toArray,apimap)


//sprintf 
let run(dirs : string []) =
    let mutable indent = 0
    let mutable text : string list = [] // NOTE: text is built up in reverse
    // let writef<'a> : Printf.StringFormat<'a, unit> -> 'a=
    //     Printf.kprintf (fun message -> text <- (sprintf "%s%s" (String.replicate indent "   ") message) :: text) 

    let p str = text <- sprintf "%s%s" (String.replicate indent "    ") str :: text
    let pi str = p str; indent <- indent + 1
    let pd str = p str; indent <- indent - 1
    // Writer.text <- []
    let operations, apimap = getOpsList()
    // Incorporates out-of-band data into the API definitions that we pulled out of the GelAllOplist
    dirs |> Array.collect (fun dir -> Directory.GetFiles(dir)) |> Array.iter (fun f -> apimap.Put(File.ReadAllText(f)) |>ignore)


    let setupArguments (oper : OpDef) = 
        // Attributes related ot the InputArg's type are inferred automatically
        // and are not exposed to the client
        let inferred = 
            oper.InputArgs |> Seq.toArray |> Array.choose (fun argdef ->
                if argdef.TypeAttr <> "" then Some(argdef.TypeAttr)
                elif argdef.TypeListAttr <> "" then Some(argdef.TypeListAttr)
                elif argdef.NumberAttr <> "" then Some(argdef.NumberAttr)
                else None) |> Set.ofArray
        let requiredAttrs, optionalAttrs = 
            oper.Attrs |> Seq.toArray
            |> Array.filter (fun attr -> not(inferred.Contains(attr.Name)))
            |> Array.partition (fun attr -> attr.DefaultValue = null)

        (requiredAttrs,optionalAttrs, oper.OutputArgs.Count > 0) 

    let comment (text) =
        if String.IsNullOrWhiteSpace(text) then ()
        else
            let lines = text.Split('\n')
            let mutable isOpen = true
            let quote (input : string) =
                match input.IndexOf ('`') with
                | -1 -> input
                | _-> input |> String.collect (function | '`' -> (if isOpen then "<c>" else "</c>" |> fun x -> isOpen <- not isOpen; x) | c -> string c)
            let mutable blockOpen = true
            for line in lines do
                // TODO probably should remove this
                if  line.IndexOf ("in image height coordinates.") <> -1 then printfn "Hello"
                let line2 = line.Trim().Replace("<","&lt;").Replace(">","&gt;").Replace("&","&amp;")
                if line2.StartsWith "```" then
                    p(sprintf "///   %s"  (if blockOpen then "<code>" else "</code>"))
                    blockOpen <- not blockOpen
                    match line2 with 
                    | "```python" | "```c++" | "```" -> ()
                    | _ ->
                        let line2 = line2.Substring 3
                        if line2.EndsWith "```" then
                            let line3 = line2.Substring (0, line2.Length - 3)
                            p (sprintf "///    %s" (quote line3))
                            p (sprintf "///    %s" (if blockOpen then "<code>" else "</code>"))
                            blockOpen <- not blockOpen
                        else
                            p (sprintf "///    %s" (quote line2))
                else
                    p (sprintf "///    %s" (quote line2))


    // Produces the C# inline documentation
    let genDocs (oper : OpDef, optionalAttrs : OpDef.AttrDef [], requiredAttrs : OpDef.AttrDef [], hasReturnValue : bool) =
        let api = apimap.TryGet oper.Name |> Option.get
        p "/// <summary>"
        comment api.Summary
        p "/// </summary>"
        for input in api.InArgs do
            p (sprintf "/// <param name=\"%s\">" (paramMap input.Name))
            comment input.Description
            p "/// </param>"
#if DOCS        
        // TODO Maybe?
#endif       
        p "/// <param name=\"name\">"
        p (sprintf "/// If specified, the created operation in the graph will be this one, otherwise it will be named '%s'." oper.Name)
        p "/// <param>"
        let pAttr (attr : OpDef.AttrDef) isOptional =
            p (sprintf "/// <param name=\"%s\">" (paramMap attr.Name))
            if isOptional then comment "Optional argument"
            api.Attrs 
            |> Seq.filter (fun x -> x.Name = attr.Name) 
            |> Seq.tryHead |> Option.iter (fun x -> comment x.Description)
            p "/// </param>"

        optionalAttrs |> Array.iter (fun attr -> pAttr attr true)
        requiredAttrs |> Array.iter (fun attr -> pAttr attr false)
        p "/// <returns>" 
        if hasReturnValue then
            if oper.OutputArgs.Count = 1 then
                api.OutArgs |> Seq.tryHead |> Option.iter (fun x -> comment x.Description)
                comment "The TFOperation can be fetched from the resulting TFOutput, by fetching the Operation property from the result."
            else
                comment "Returns a tuple with multiple values, as follows:"
                oper.OutputArgs |> Seq.iter (fun arg ->
                    api.OutArgs |> Seq.filter (fun x -> x.Name = arg.Name) 
                    |> Seq.tryHead |> Option.iter (fun oapi -> comment (sprintf "%s : %s" (paramMap arg.Name) oapi.Description)))
                comment "The TFOperation can be fetched from any of the TFOutputs returned in the tuple values, by fetching the Operation property."
        else
            comment "Returns the description of the operation"
        p "/// </returns>"
        if not(String.IsNullOrEmpty (api.Description)) then
            p "/// <remarks>"
            comment (api.Description)
            p "/// </remarks>"


    let setAttribute (_type : string, attrName : string, csAttrName : string) =
        if _type = "shape" then p (sprintf "desc.SetAttrShape (\"%s\", %s)" attrName csAttrName)
        elif _type.StartsWith ("list(shape") then p (sprintf "desc.SetAttrShape (\"%s\", %s)" attrName csAttrName)
        else
            match fsharptype _type |> Option.get with
            | "int64"
            | "int64[]"
            | "string"
            | "string[]"
            | "float"
            | "float[]"
            | "bool"
            | "bool[]" -> p (sprintf "desc.SetAttr (\"%s\", %s);" attrName csAttrName)
            | "TFDataType"
            | "TFDataType[]" -> p (sprintf "desc.SetAttrType (\"%s\", %s);" attrName csAttrName)
            // this should pass the cstatus, but requries the
            // function to take a TFStatus as well, so need to weave that
            // in the parameters
            | "TFTensor"
            | "TFTensor[]" -> p (sprintf "desc.SetAttr (\"%s\", %s (* cstatus *));" attrName csAttrName)
            | fstype -> failwithf "Unexpected type: %s" fstype


    let fillArguments (oper : OpDef, requiredAttrs : OpDef.AttrDef[], optionalAttrs : OpDef.AttrDef[]) =
        [|
            yield! oper.InputArgs |> Seq.toArray |> Array.map (fun inarg -> (if isListArg inarg then "TFOutput[] " else "TFOutput ") + inarg.Name) 
            yield! requiredAttrs |> Array.map (fun attr -> sprintf "%s %s" (fsharptype attr.Type |> Option.get) attr.Name) 
            yield! optionalAttrs |> Array.map (fun attr -> 
                let reftype = isReferenceType attr.Type
                let fstype = fsharptype attr.Type |> Option.get
                let fstypesuffix = if reftype then "" else "?"
                sprintf "%s%s %s= null" fstype fstypesuffix attr.Name ) 
        |] |> String.concat ", "

    let generate (oper : OpDef) =
        let requiredAttrs, optionalAttrs, hasReturnValue = setupArguments (oper)
        genDocs (oper, requiredAttrs, optionalAttrs, hasReturnValue)
        let name = oper.Name
        let retType = 
            if hasReturnValue then
                match oper.OutputArgs |> Seq.toArray with
                | [||] -> failwith "should have at least one return value"
                | [|x|] -> if isListArg x then "TFOutput[]" else "TFOutput"
                | xs -> xs |> Array.map (fun arg ->  (if isListArg arg then "TFOutput[]" else "TFOutput") + " " + (paramMap arg.Name)) |> String.concat ", " |> sprintf "(%s)"
            else "TFOperation"
        p (sprintf "public %s %s (%s, string name = null)" retType name (fillArguments(oper, requiredAttrs, optionalAttrs)))
        pi "{"
        let needStatus = [|yield! requiredAttrs; yield! optionalAttrs|] |> Array.exists (fun x -> x.Type.Contains("TFTensor"))
        p (sprintf "var desc = new TFOperationDesc (this, \"%s\", MakeName (\"%s\", name));" oper.Name oper.Name)
        oper.InputArgs |> Seq.iter (fun arg -> p (sprintf "desc.AddInput%s (%s);" ( if isListArg arg then "s" else "")  arg.Name))
        pi "foreach ( TFOperation control in CurrentDependencies )"
        p "desc.AddControlInput (control);"
        pd ""

        // If we have attributes
        if requiredAttrs.Length > 0 || optionalAttrs.Length > 0 then
            for attr in requiredAttrs do
                setAttribute (attr.Type, attr.Name, paramMap attr.Name)
            for attr in optionalAttrs do
                let reftype = isReferenceType attr.Type
                let fsattr = paramMap attr.Name
                if reftype then pi (sprintf"if (%s) != null" fsattr) else pi (sprintf "if (%s.HasValue)" fsattr)

                setAttribute(attr.Type, attr.Name, fsattr + (if reftype then "" else ".Value"))
                pd ""
        
        p "var op = desc.FinishOperation ();"
        if oper.OutputArgs.Count > 0 then
            p "int _idx=0;"
        if (oper.OutputArgs |> Seq.exists (fun x -> isListArg x)) then
            p "int _n=0;"
        for arg in oper.OutputArgs do
            if isListArg arg then
                p (sprintf "_n = op.OutputListLength (\"%s\");" (paramMap arg.Name))
                p (sprintf "var %s = new TFOutput [_n];" (paramMap arg.Name))
                pi "for (int i = 0; i < _n; i++)"
                p (sprintf "%s [i] = new TFOutput (op, _idx++);" (paramMap arg.Name))
                pd ""
            else
                p (sprintf "var %s = new TFOuptut (op, _idx++);" (paramMap arg.Name))
        
        if hasReturnValue then
            if oper.OutputArgs.Count = 1 then
                p (sprintf "return %s;" (paramMap (oper.OutputArgs |> Seq.head).Name))
            else
                p (sprintf "return (%s);" (oper.OutputArgs |> Seq.map (fun x -> paramMap x.Name) |> String.concat ", "))
        else p "return op;"
        pd "}\n"
            
    p "using System;\n"
    pi "namespace Tensorflow {"
    pi "public partial class TFGraph {"

    for oper in operations |> Array.sortBy (fun x -> x.Name) do
        // Skip internal operations
        if oper.Name.StartsWith ("_") then ()
        else
            // Ignore functions where we lack a C# type mapping
            match oper.Attrs |> Seq.tryFind (fun attr -> attr.Type |> fsharptype |> Option.isNone) with
            | Some(attr) -> printfn "SkipType: %s due to attribute (%s %s lacking a mapping to C#" oper.Name attr.Type attr.Name
            | None -> 
                let def = apimap.TryGet oper.Name |> Option.get
                // Undocumented operation, perhaps we should not surface
                if not (String.IsNullOrWhiteSpace(def.Summary)) 
                then
                    // Generate
                    generate(oper)
    pd "}"
    pd "}"
    text |> List.rev |> String.concat "\n"


let res = run([|"/home/moloneymb/EE/Git/tensorflow/tensorflow/core/api_def/base_api"|])
                

File.WriteAllText(__SOURCE_DIRECTORY__ + "/gen.cs", res)

