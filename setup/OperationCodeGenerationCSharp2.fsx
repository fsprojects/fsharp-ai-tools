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
#r @"TensorFlow.FSharp.Proto.dll"
#r @"nativeWorkaround.dll"
#r @"protobuf-net.dll"

open System
open ProtoBuf
open System.IO
open NativeWorkaround
open NativeWorkaround.Native
open FSharp.NativeInterop
open System.Collections.Generic
open System.Runtime.InteropServices
open Google.Protobuf.Collections
open System.Collections.Generic
open TensorFlow.FSharp.Proto

let deserialize<'a> = Serializer.Deserialize<'a>

/// If character is capital then we lower case it
/// If the capital letter is not the first character nor the has a digit preceding character then insert '_' before the lower case character
let CPPtoPythonName(name:string) =
    name.ToCharArray() |> Array.fold (fun ((supress,acc) : (bool*string list)) (c : char) -> 
        (Char.IsDigit(c),(if Char.IsUpper(c) then (if supress then "" else "_") + (string <| Char.ToLower(c)) else string c)::acc)) (true,[])
   |> snd |> List.toArray |> Array.rev |> String.concat ""

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
    | "int"     -> "int"     |> Some
    | "float"   -> "float"     |> Some
    | "bool"    -> "bool"      |> Some
    | "type"    -> "TF_DataType"     |> Some
    | "shape"   -> "TensorShape"   |> Some
    | "tensor"  -> "Tensor"  |> Some
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
            oper.InputArgs |> Seq.toArray |> Array.collect (fun argdef ->
                [|
                    if argdef.TypeAttr <> "" then yield argdef.TypeAttr
                    elif argdef.TypeListAttr <> "" then yield argdef.TypeListAttr
                    if argdef.NumberAttr <> "" then yield argdef.NumberAttr
                |]) |> Set.ofArray
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
                | _-> input |> String.collect (function | '`' -> ((if isOpen then "<c>" else "</c>") |> fun x -> isOpen <- not isOpen; x) | c -> string c)
            let mutable blockOpen = true
            for line in lines do
                // TODO probably should remove this
                if  line.IndexOf ("in image height coordinates.") <> -1 then printfn "Hello"
                let line2 = line.Trim().Replace("<","&lt;").Replace(">","&gt;").Replace("&","&amp;")
                    //.Replace(",","").Replace("")
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
        p "/// </param>"
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
                comment "The Operation can be fetched from the resulting Tensor, by fetching the Operation property from the result."
            else
                comment "Returns a tuple with multiple values, as follows:"
                oper.OutputArgs |> Seq.iter (fun arg -> api.OutArgs |> Seq.filter (fun x -> x.Name = arg.Name) |> Seq.tryHead |> Option.iter (fun oapi -> comment (sprintf "%s : %s" (paramMap arg.Name) oapi.Description)))
                comment "The Operation can be fetched from any of the Tensorreturned in the tuple values, by fetching the Operation property."
        else
            comment "Returns the description of the operation"
        p "/// </returns>"
        if not(String.IsNullOrEmpty (api.Description)) then
            p "/// <remarks>"
            comment (api.Description)
            p "/// </remarks>"


//    let setAttribute (_type : string, attrName : string, csAttrName : string) =
//        if _type = "shape" then p (sprintf "desc.SetAttrShape (\"%s\", %s)" attrName csAttrName)
//        elif _type.StartsWith ("list(shape") then p (sprintf "desc.SetAttrShape (\"%s\", %s)" attrName csAttrName)
//        else
//            match fsharptype _type |> Option.get with
//            | "int"
//            | "int[]"
//            | "string"
//            | "string[]"
//            | "float"
//            | "float[]"
//            | "bool"
//            | "bool[]" -> p (sprintf "desc.SetAttr (\"%s\", %s);" attrName csAttrName)
//            | "TF_DataType"
//            | "TF_DataType[]" -> p (sprintf "desc.SetAttrType (\"%s\", %s);" attrName csAttrName)
//            // this should pass the cstatus, but requries the
//            // function to take a TFStatus as well, so need to weave that
//            // in the parameters
//            | "Tensor"
//            | "Tensor[]" -> p (sprintf "desc.SetAttr (\"%s\", %s (* cstatus *));" attrName csAttrName)
//            | fstype -> failwithf "Unexpected type: %s" fstype


    let fillArguments (oper : OpDef, requiredAttrs : OpDef.AttrDef[], optionalAttrs : OpDef.AttrDef[]) =
        [|
            yield! oper.InputArgs |> Seq.toArray |> Array.map (fun inarg -> (if isListArg inarg then "Tensor[] " else "Tensor ") + paramMap(inarg.Name))
            yield! requiredAttrs |> Array.map (fun attr -> sprintf "%s %s" (fsharptype attr.Type |> Option.get) (paramMap(attr.Name)))
            yield! optionalAttrs |> Array.map (fun attr -> 
                let reftype = isReferenceType attr.Type
                let fstype = fsharptype attr.Type |> Option.get
                let fstypesuffix = if reftype then "" else "?"
                sprintf "%s%s %s%s" fstype fstypesuffix attr.Name  (if false then "" else " = null")) 
        |] |> String.concat ", "

    let generate (oper : OpDef) =
        let requiredAttrs, optionalAttrs, hasReturnValue = setupArguments (oper)
        genDocs (oper, requiredAttrs, optionalAttrs, hasReturnValue)
        let name = oper.Name
        let retType = 
            if hasReturnValue then
                match oper.OutputArgs |> Seq.toArray with
                | [||] -> failwith "should have at least one return value"
                | [|x|] -> if isListArg x then "Tensor[]" else "Tensor"
                | xs -> xs |> Array.map (fun arg ->  (if isListArg arg then "Tensor[]" else "Tensor") + " " + (paramMap arg.Name)) |> String.concat ", " |> sprintf "(%s)"
            else "Operation"
        let keywordSubstitution(x:string) =
            match x with 
            | "const" -> "constant"
            | "switch" -> "switch_"
            | x -> x
        p (sprintf "public static %s %s (%s%sstring name = \"%s\")" retType (name |> CPPtoPythonName |> keywordSubstitution) (fillArguments(oper, requiredAttrs, optionalAttrs)) (if oper.InputArgs.Count > 0 || requiredAttrs.Length > 0 || optionalAttrs.Length > 0 then ", " else "") name)
        pi "{"
        //let needStatus = [|yield! requiredAttrs; yield! optionalAttrs|] |> Array.exists (fun x -> x.Type.Contains("Tensor"))
        p "var dict = new Dictionary<string, object>();"
        oper.InputArgs |> Seq.iter (fun arg -> p (sprintf "dict[\"%s\"] = %s;" arg.Name (paramMap arg.Name)))
        // If we have attributes
        if requiredAttrs.Length > 0 || optionalAttrs.Length > 0 then
            for attr in requiredAttrs do
                //setAttribute (attr.Type, attr.Name, paramMap attr.Name)
                p (sprintf "dict[\"%s\"] = %s;" attr.Name (paramMap attr.Name))
            for attr in optionalAttrs do
                let reftype = isReferenceType attr.Type
                let fsattr = paramMap attr.Name
                if reftype then pi (sprintf"if (%s != null)" fsattr) else pi (sprintf "if (%s.HasValue)" fsattr)
                pd (sprintf "dict[\"%s\"] = %s%s;" attr.Name (paramMap attr.Name) (if reftype then "" else ".Value"))
                //setAttribute(attr.Type, attr.Name, fsattr + (if reftype then "" else ".Value"))
        
        p (sprintf "var op = _op_def_lib._apply_op_helper(\"%s\", name: name, keywords: dict);" name)
        if oper.OutputArgs.Count  = 1 && (not <| isListArg oper.OutputArgs.[0]) then
            pd "return op.output;"
        else
            if oper.OutputArgs.Count > 0 then
                p "int _idx = 0;"
            //if (oper.OutputArgs |> Seq.exists (fun x -> isListArg x)) then
            //    p "int _n=0;"
            for arg in oper.OutputArgs do
                if isListArg arg then
                    //p (sprintf "_n = op.OutputListLength (\"%s\");" (paramMap arg.Name))
                    //p (sprintf "var %s = new Tensor[_n];" (paramMap arg.Name))
                    //pi "for (int i = 0; i < _n; i++)"
                    //p (sprintf "%s [i] = new Tensor(op, _idx++);" (paramMap arg.Name))
                    //pd ""
                    //()
                    p (sprintf "var %s = Enumerable.Range(0, op.OutputListLength(\"%s\")).Select(_ => op.outputs[_idx++]).ToArray();" (paramMap arg.Name) arg.Name)
                else
                    p (sprintf "var %s = op.outputs[_idx++];" (paramMap arg.Name))
            
            if hasReturnValue then
//                if oper.OutputArgs.Count = 1 then
//                    pd (sprintf "return %s;" (paramMap (oper.OutputArgs |> Seq.head).Name))
//                else
                    pd (sprintf "return (%s);" (oper.OutputArgs |> Seq.map (fun x -> paramMap x.Name) |> String.concat ", "))
            else pd "return op;"

        p "}\n"
            
    p "using System;"
    p "using System.Collections.Generic;"
    p "using System.Linq;\n"
    pi "namespace Tensorflow {"
    p "// This class name should be changed"
    pi "public class GenOpsCSharp {"
    p "static readonly OpDefLibrary _op_def_lib;"
    p "static GenOpsCSharp() { _op_def_lib = new OpDefLibrary(); }"

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


//let res = run([|"/home/moloneymb/EE/Git/tensorflow/tensorflow/core/api_def/base_api"|])
//File.WriteAllText(__SOURCE_DIRECTORY__ + "/gen.cs", res)

let root = Path.Combine(__SOURCE_DIRECTORY__, "..")

run([|Path.Combine(root, "tensorflow", "api_def")|]) 
|> fun res -> File.WriteAllText(Path.Combine(root,"TensorFlowCSharpWrapper", "TFNetWrapper.g.cs"), res)

//C:\EE\Git\Tensorflow_FSharp\TensorFlowCSharpWrapper
