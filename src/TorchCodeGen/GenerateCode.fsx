#if INTERACTIVE
#I @".\bin\Debug\netcoreapp3.1"
#r @"YamlDotNet.dll"
#r @"Argu.dll"
#endif

open System
open System.IO

let INDENT = "    "
let EXPORT_API = "EXPORT_API"
let THSTensor = "THSTensor__"
let TargetLineWidth = 120

module CodeGenCommon = 

    module String = 
        let indent (n: int) (x: string) = x.Insert(0,String(Array.create n ' '))

    let capitalizeFirst (x: string) = x |> String.mapi (function | 0 -> Char.ToUpperInvariant | _ -> id)

    let camelToUnderscore = 
        let r = System.Text.RegularExpressions.Regex("(?!^)([A-Z])([a-z])")
        fun (name:string) -> 
            r.Replace(name,"_${1}${2}").ToLowerInvariant()

    let underscoreToCamel(name:string) = 
        let c1 = 
            name.ToCharArray() 
            |> Array.fold (fun (lastUnderscore: bool ,xs:char list) (c:char) -> 
                match c with
                | '_' -> (true,xs)
                | _ -> (false, (if lastUnderscore then Char.ToUpperInvariant(c) else c) :: xs)
                ) (false,[])
                |> snd |> List.toArray |> Array.rev |> String
        if name.EndsWith("_") then c1 + "_" else c1

    let asTuple(xs:string[]) = xs |> String.concat ", " |> sprintf "(%s)"

    module Array = 
        let prepend xs ys = Array.append ys xs

        let interleaveWith (xs:seq<'a[]>) (ys:'a[]) = 
            let rec f (xs:List<'a[]>) : 'a[] = 
                [|
                    match xs with
                    | [] -> ()
                    | [xs] -> yield! xs
                    | (x::xs) -> yield! x; yield! ys; yield! f xs
                |]
            f (xs |> Seq.toList)

    let interleaveWithNewLine (xs:string[][]) : string[] = Array.interleaveWith xs [|""|]

    let indent (xs: string[]) = xs |> Array.map (fun x -> INDENT + x)

    let multiLineParams(xs:string[]) = 
        match xs with 
        | [||] -> [|")"|]
        | _ -> xs |> Array.mapi (fun i x -> x + if xs.Length - 1 = i then ")" else ",")

    let addFinalSemiColon(xs: string[]) = 
        match xs with
        | [||] -> [|";"|]
        | _ -> xs |> Array.mapi (fun i x -> if xs.Length - 1 = i then x+";" else x)

    let closeParen(xs: string[]) = 
        xs |> Array.mapi (fun i x -> if xs.Length - 1 = i then x+")" else x)

    module Cpp = 
        // adds a semicolon to the last line
        let semicolon (xs:string[]) = match xs with | [||] -> [|";"|] | _ -> [|yield! xs.[0..xs.Length - 2]; yield xs.[xs.Length-1] + ";"|]
        // Require new line
        let macro(name: string, break_ : bool) (lines: string[]) = 
            match lines, break_ with
            | [|x|], false -> [|sprintf "%s(%s)" name x|]
            | _,true -> [|yield sprintf "%s(" name; yield! lines |> indent; yield ")"|]
            | _, false -> failwith "err" // multi-line must break

        let ifThenElse(conditional : string, then_: string[], else_: string[]) = 
            [|
                yield sprintf "if (%s)" conditional
                yield "{"
                yield! then_ |> indent
                match else_ with
                | [||] -> () 
                | _ -> 
                    yield "} else {"
                    yield! else_ |> indent
                yield "}"
            |]

        let ifThen(conditional : string) (then_: string[]) = ifThenElse(conditional, then_, [||])

        let func(firstLine) (body: string[]) = 
            [| yield firstLine; yield "{"; yield! body |> indent; yield "}" |]

        let funcMany (functionLines: string[]) (body: string[]) = 
            match functionLines with
            | [||] -> failwith "err"
            | [|x|] -> func x body
            | xs -> [|yield xs.[0]; yield! xs.[1..] |> indent; yield "{"; yield! body |> indent; yield "}"|]
            
        let ternaryIfThenElse(conditional : string, then_: string, else_: string) = 
            sprintf "%s ? %s : %s" conditional then_ else_

    module CSharp = 
        open Cpp

        let DLLImport = "[DllImport(\"LibTorchSharp\")]"

        let keywords = 
            set [|
                "abstract"; "as"; "base"; "bool"; "break"; "byte"; "case"; "catch"; "char";
                "checked"; "class"; "const"; "continue"; "decimal"; "default"; "delegate";
                "do"; "double"; "else"; "enum"; "event"; "explicit"; "extern"; "false";
                "finally"; "fixed"; "float"; "for"; "foreach"; "goto"; "if"; "implicit";
                "in"; "int"; "interface"; "internal"; "is"; "lock"; "long"; "namespace";
                "new"; "null"; "object"; "operator"; "out"; "override"; "params";
                "private"; "protected"; "public"; "readonly"; "ref"; "return"; "sbyte";
                "sealed"; "short"; "sizeof"; "stackalloc"; "static"; "string"; "struct";
                "switch"; "this"; "throw"; "true"; "try"; "typeof"; "uint"; "ulong";
                "unchecked"; "unsafe"; "ushort"; "using"; "virtual"; "void"; "volatile";
                "while"|]

        let sainitize (x:string) = if keywords.Contains(x) then x+"_" else x

        let extern_(body: string) = [| DLLImport; sprintf "extern static %s" body|]

        let namespace_(namespace_: string) (body: string[]) =
            [| 
                yield sprintf "namespace %s" namespace_
                yield "{"
                yield! body |> indent
                yield "}"
            |]

        let getSetMember(fistLine: string,get: string[],set: string[]) =
            [| 
                yield fistLine + " {"
                yield! [| yield "get {"; yield! get |> indent; yield "}" |] |> indent
                yield! [| yield "set {"; yield! set |> indent; yield "}" |] |> indent
                yield "}"
            |]

        let using(firstLine) (body:string[]) = func(sprintf "using (%s)" firstLine) body
        let unsafe (body: string[]) = func "unsafe" body 
        let fixed_(args: string) (body: string[]) = func (sprintf "fixed(%s)" args) body
        let nestedFixed (args: string[]) (body: string[]) =
            match args with
            | [||] -> body
            | [|x|] -> fixed_ x body
            | _ -> (args,body) ||> Array.foldBack (fun x body -> fixed_ x body) 

    let compressLines (maxLine:int) (xs: string[]) = 
        if xs.Length = 0 || xs.Length = 0 then xs
        else
            let singleLine = [|yield xs.[0]; yield! xs.[1..] |> Array.map (fun x -> x.Trim())|] |> String.concat " "
            if singleLine.Length <= maxLine then [|singleLine|]
            else xs

    let prefixFirstLine (prefix:string) (xs:string[]) = 
        if xs.Length = 0 then [||]
        else [|yield prefix + xs.[0]; yield! xs.[1..]|]


module Parser = 
    open YamlDotNet.RepresentationModel
    open CodeGenCommon

    // method_of -> namespace

    [<RequireQualifiedAccess>]
    type BT = 
        | Bool | Bool2 | Bool3 | Bool4
        | BoolOptional
        | ConstQuantizerPtr
        | ConstTensor
        | Device
        | Dimname
        | DimnameList
        | DimnameListOptional
        | Double
        | DoubleOptional
        | Generator
        | Int
        | IntList
        | IntOptional
        | MemoryFormat
        | MemoryFormatOptional
        | Scalar
        | ScalarOptional
        | ScalarType
        | ScalarTypeOptional
        | Storage
        | String
        | Tensor
        | TensorAnd
        | TensorList
        | TensorVector
        | TensorOptionsAnd
        | TensorOptions
        | QScheme
        override this.ToString() = 
            match this with
            | BT.Bool -> "bool"
            | BT.Bool2 -> "std::array<bool,2>"
            | BT.Bool3 -> "std::array<bool,3>"
            | BT.Bool4 -> "std::array<bool,4>"
            | BT.BoolOptional -> "c10::optional<bool>"
            | BT.ConstQuantizerPtr -> "ConstQuantizerPtr"
            | BT.ConstTensor -> "const Tensor &"
            | BT.Device -> "Device"
            | BT.Dimname -> "Dimname"
            | BT.DimnameList -> "DimnameList"
            | BT.DimnameListOptional -> "c10::optional<DimnameList>"
            | BT.Double -> "double"
            | BT.DoubleOptional -> "c10::optional<double>"
            | BT.Generator -> "Generator *"
            | BT.Int -> "int64_t"
            | BT.IntList -> "IntArrayRef"
            | BT.IntOptional -> "c10::optional<int64_t>"
            | BT.MemoryFormat -> "MemoryFormat"
            | BT.MemoryFormatOptional -> "c10::optional<MemoryFormat>"
            | BT.Scalar -> "Scalar"
            | BT.ScalarOptional -> "c10::optional<Scalar>"
            | BT.ScalarType -> "ScalarType"
            | BT.ScalarTypeOptional -> "c10::optional<ScalarType>"
            | BT.Storage -> "Storage"
            | BT.String ->  "std::string"
            | BT.Tensor -> "Tensor"
            | BT.TensorAnd -> "Tensor &"
            | BT.TensorList -> "TensorList"
            | BT.TensorVector -> "std::vector<Tensor>"
            | BT.TensorOptionsAnd -> "const TensorOptions &"
            | BT.TensorOptions -> "TensorOptions"
            | BT.QScheme -> "QScheme"
        static member Parse(str: string) = 
            match str with
            | "bool" -> BT.Bool
            | "std::array<bool,2>" -> BT.Bool2
            | "std::array<bool,3>" -> BT.Bool3
            | "std::array<bool,4>" -> BT.Bool4
            | "c10::optional<bool>" -> BT.BoolOptional
            | "ConstQuantizerPtr" -> BT.ConstQuantizerPtr
            | "const Tensor &" -> BT.ConstTensor
            | "Device" -> BT.Device
            | "Dimname" -> BT.Dimname
            | "DimnameList" -> BT.DimnameList
            | "c10::optional<DimnameList>" -> BT.DimnameListOptional
            | "double" -> BT.Double
            | "c10::optional<double>" -> BT.DoubleOptional
            | "Generator *" -> BT.Generator
            | "int64_t" -> BT.Int
            | "IntArrayRef" -> BT.IntList
            | "c10::optional<int64_t>" -> BT.IntOptional
            | "MemoryFormat" -> BT.MemoryFormat
            | "c10::optional<MemoryFormat>" -> BT.MemoryFormatOptional
            | "Scalar" -> BT.Scalar
            | "c10::optional<Scalar>" -> BT.ScalarOptional
            | "ScalarType" -> BT.ScalarType
            | "c10::optional<ScalarType>" -> BT.ScalarTypeOptional
            | "Storage" -> BT.Storage
            | "std::string" -> BT.String 
            | "Tensor" -> BT.Tensor
            | "Tensor &" -> BT.TensorAnd
            | "TensorList" -> BT.TensorList
            | "std::vector<Tensor>" -> BT.TensorVector
            | "const TensorOptions &" -> BT.TensorOptionsAnd
            | "TensorOptions" -> BT.TensorOptions
            | "QScheme" -> BT.QScheme
            | _ -> failwithf "BT Parse error %s" str
        member this.Metadata = // isArray, isOptional
            match this with
            | BT.Bool -> false,false
            | BT.Bool2 
            | BT.Bool3 
            | BT.Bool4 -> true,false// is it??
            | BT.BoolOptional -> false,true
            | BT.ConstQuantizerPtr -> false,false
            | BT.ConstTensor 
            | BT.Device 
            | BT.Dimname -> false,false
            | BT.DimnameList -> true,false
            | BT.DimnameListOptional -> true,true
            | BT.Double -> false,false
            | BT.DoubleOptional -> false,true
            | BT.Generator -> false,false
            | BT.Int -> false,false
            | BT.IntList -> true,true
            | BT.IntOptional -> false,true
            | BT.MemoryFormat -> false,false
            | BT.MemoryFormatOptional -> false,true
            | BT.Scalar -> false,false
            | BT.ScalarOptional -> false,true
            | BT.ScalarType -> false,false
            | BT.ScalarTypeOptional -> false,true
            | BT.Storage -> false,false
            | BT.String ->  false,false
            | BT.Tensor -> false,false
            | BT.TensorAnd -> false,false
            | BT.TensorList -> true,false
            | BT.TensorVector -> true,false
            | BT.TensorOptionsAnd -> false,false
            | BT.TensorOptions -> false,false
            | BT.QScheme -> false,false
        member this.IsArray = fst this.Metadata
        member this.IsOptional = snd this.Metadata

    let (|Tensor|_|) (x:BT) = 
        match x with
        | BT.Tensor
        | BT.TensorAnd
        | BT.TensorList
        | BT.TensorVector
        | BT.ConstTensor -> Some(x)
        | _ -> None

    let (|TensorOrScalar|_|) (x:BT) = 
        match x with
        | Tensor _ 
        | BT.Scalar
        | BT.ScalarOptional -> Some(x)
        | _ -> None

    type Arg = {
        name            : string
        annotation      : (char*bool) option
        isNullable     : bool
        defaultValue    : string option
        type_           : BT
        dynamicType     : BT
    } 

    type Return = {
        dynamicType : BT
        type_ : BT
        fieldName : string option
        name : string
    } 

    type Schema = {
        name : string
        operatorName : string
        overloadName : string
        args : Arg[]
        returns : Return[]
        depricated : bool
        methodOfTensor : bool option
        methodOfNamespace : bool option
    } with member this.ntensors =  
            if this.returns 
                |> Array.exists (fun x -> 
                    match x.dynamicType with 
                    | BT.Tensor | BT.TensorAnd | BT.ConstTensor -> false 
                    | _ -> true) 
                |> not then
                Some(Some(this.returns.Length))
            else 
                match this.returns with
                | [|x|] -> 
                    match x.dynamicType with 
                    | BT.TensorVector | BT.TensorList -> Some(None)
                    | _ -> None
                | _ -> None

    let schemas(path: string) : Schema[] = 
        let ys = YamlStream()
        use sr = new StringReader(File.ReadAllText(path))
        ys.Load(sr)
        let doc = ys.Documents.[0]
        let rootNode = (doc.RootNode :?> YamlSequenceNode)
        let returns = YamlScalarNode("returns")
        let arguments = YamlScalarNode("arguments")
        let name = YamlScalarNode("name")
        let operatorName = YamlScalarNode("operator_name")
        let overloadName = YamlScalarNode("overload_name")
        let deprecated = YamlScalarNode("deprecated")
        let method_of = YamlScalarNode("method_of")
        let tensor = YamlScalarNode("Tensor")
        let namespace_ = YamlScalarNode("namespace")

        [|
             for op in rootNode.Children |> Seq.cast<YamlMappingNode> do 
                let args = 
                    if op.Children.Keys.Contains(arguments) then
                        op.Children.Item(arguments) :?> YamlSequenceNode
                        |> Seq.cast<YamlMappingNode>
                        |> Seq.map (fun (arg: YamlMappingNode) -> 
                            let getV(name:string) = (arg.Item(YamlScalarNode(name)) :?> YamlScalarNode).Value 
                            let annotation = 
                                match getV "annotation" with
                                | "null" -> None
                                | "a" -> Some('a', false)
                                | "a!" -> Some('a', true)
                                | "b!" -> Some('b', true)
                                | "c!" -> Some('c', true)
                                | _ -> failwith "err"
                            {
                                name            = getV "name"
                                annotation      = annotation
                                isNullable      = match getV "is_nullable" with | "true" -> true | "false" -> false | _ -> failwith "err"
                                defaultValue    = if arg.Children.Keys.Contains(YamlScalarNode("default")) then Some(getV "default") else None
                                type_           = BT.Parse(getV "type")
                                dynamicType     = BT.Parse(getV "dynamic_type")
                            })
                        |> Seq.toArray
                    else [||]
                let returns = 
                    if op.Children.Keys.Contains(returns) then
                        op.Children.Item(returns) :?> YamlSequenceNode
                        |> Seq.cast<YamlMappingNode>
                        |> Seq.map (fun ret -> 
                            let getV(name:string) = (ret.Item(YamlScalarNode(name)) :?> YamlScalarNode).Value 
                            {
                                name         = getV "name"
                                dynamicType  = BT.Parse(getV "dynamic_type")
                                type_        = BT.Parse(getV "type")
                                fieldName    = if ret.Children.Keys.Contains(YamlScalarNode("field_name")) then Some(getV "field_name") else None
                            })
                        |> Seq.toArray
                    else [||]

                let mo = 
                    if op.Children.Keys.Contains(method_of) then
                        let x = (op.Children.Item(method_of) :?> YamlSequenceNode)
                        Some(x.Children.Contains(tensor), x.Children.Contains(namespace_))
                    else None
                {
                    name = (op.Children.Item(name) :?> YamlScalarNode).Value
                    operatorName = (op.Children.Item(operatorName) :?> YamlScalarNode).Value
                    overloadName = (op.Children.Item(overloadName) :?> YamlScalarNode).Value
                    args = args
                    returns = returns
                    depricated = (match (op.Children.Item(deprecated) :?> YamlScalarNode).Value with | "true" -> true | "false" -> false | _ -> failwith "err")
                    methodOfTensor = mo |> Option.map fst
                    methodOfNamespace = mo |> Option.map snd
                }
        |]


module CodeGenLL = 
    open CodeGenCommon
    open Parser
    open CSharp
    open Cpp

    let TensorAllocator = "Tensor* (*allocator)(size_t length)"
    let TensorAllocatorCSharp = "AllocatePinnedArray allocator"

    let toCSharpName(name: string) = name |> underscoreToCamel |> CSharp.sainitize
    let compressLines = compressLines TargetLineWidth

    type Arg with
        member x.Gated = 
            let canBeNull(t: BT) = 
                if t.IsArray then true
                else 
                    match t with
                    | TensorOrScalar _ -> true
                    | _ -> false
            if canBeNull x.type_ then false
            else
                (x.isNullable ||  x.defaultValue.IsSome) &&
                not(x.name = "options" && (match x.type_ with | BT.TensorOptions | BT.TensorOptionsAnd -> true | _ -> false))

        member x.IsOptional = x.isNullable || x.defaultValue.IsSome


        member arg.CSignature = 
            match arg.type_ with
            | BT.TensorOptions
            | BT.TensorOptionsAnd when arg.name = "options" -> "const int8_t scalar_type, const char* device, const bool requires_grad"
            | BT.TensorOptions
            | BT.TensorOptionsAnd -> sprintf "int %s_kind, int %s_device" arg.name arg.name
            | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" arg
            | BT.IntList -> sprintf "const int64_t* %s, const int %s_length" arg.name arg.name
            | BT.TensorList  -> sprintf "const Tensor* %s, const int %s_length" arg.name arg.name
            | BT.DimnameList -> failwithf "todo %A" arg
            | BT.String -> Printf.sprintf "const char* %s" arg.name
            | _ when arg.type_.IsArray -> failwithf "err %A" arg
            | _ ->
                match arg.type_ with
                | BT.Bool -> "const bool"
                | BT.Int
                | BT.IntOptional-> "const int64_t"
                | BT.Double -> "double"
                | BT.DoubleOptional -> "double"
                | BT.Tensor -> "Tensor"
                | BT.TensorAnd -> "const Tensor"
                | BT.ConstTensor -> "const Tensor"
                | BT.ScalarType 
                | BT.ScalarTypeOptional -> "const int8_t"
                | BT.Device -> "const int8_t" 
                | BT.Scalar -> "const Scalar"
                | BT.ScalarOptional -> "const Scalar" 
                | BT.MemoryFormat 
                | BT.MemoryFormatOptional -> "const MemoryFormat"
                | _ -> (string arg.type_) 
                |> fun x -> sprintf "%s %s" x arg.name 
            |> fun x -> 
                if arg.Gated 
                then sprintf "const bool with_%s, %s" arg.name x
                else x

        member arg.CSharpInteropSignature = 
            let argName = arg.name |> toCSharpName
            match arg.type_ with
            | _ when arg.IsTensorOption ->
                "sbyte scalarType, [MarshalAs(UnmanagedType.LPStr)] string device, bool requiresGrad"
            | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" arg
            | BT.IntList -> sprintf "IntPtr %s, int %s_length" argName argName
            | BT.TensorList -> sprintf "IntPtr %s, int %s_length" argName argName
            | BT.DimnameList -> failwithf "todo %A" arg
            | _ when arg.type_.IsArray -> failwithf "err %A" arg
            | _ ->
                match arg.type_ with
                | BT.String -> "[MarshalAs(UnmanagedType.LPStr)] string"
                | BT.Bool -> "bool" 
                | BT.Int
                | BT.IntOptional-> "long"
                | BT.Double -> "double"
                | BT.DoubleOptional -> "double"
                | BT.Tensor -> "IntPtr"
                | BT.TensorAnd -> "IntPtr"
                | BT.ConstTensor -> "IntPtr"
                | BT.ScalarType 
                | BT.ScalarTypeOptional -> "sbyte"
                | BT.Device -> "sbyte" 
                | BT.Scalar -> "IntPtr"
                | BT.ScalarOptional -> "IntPtr" 
                | BT.MemoryFormat 
                | BT.MemoryFormatOptional -> "MemoryFormat" 
                | _ -> (string arg.type_) 
                |> fun x -> sprintf "%s %s" x argName 
            |> fun x -> 
                if arg.Gated 
                then sprintf "bool with_%s, %s" argName x
                else x

        member arg.CppToC =
            if arg.type_.IsArray then
                match arg.type_ with
                | BT.IntList -> "at::ArrayRef<int64_t>" + (sprintf "(%s, %s_length)" arg.name arg.name)
                | BT.TensorList -> "toTensors<at::Tensor>((torch::Tensor**)"  + (sprintf "%s, %s_length)" arg.name arg.name)
                | _ -> failwithf "todo cppToC %A " arg
                
            else
                match arg.type_ with
                | TensorOrScalar _ -> "*" + arg.name
                | _ -> arg.name
            |> fun y -> 
                let defaultValue = 
                    arg.defaultValue 
                    |> Option.map (fun x ->
                        match arg.type_,x with
                        | Tensor _, "{}" -> "at::Tensor()"
                        | BT.Scalar, "{}"
                        | BT.ScalarOptional,"{}" -> "at::Scalar()"
                        | _ ,_ -> x)
                let y = 
                    match arg.type_ with
                    | BT.ScalarOptional -> 
                         sprintf "c10::optional<c10::Scalar>(%s)" y // Note: it's converted from at::Scalar to c10::Scalar
                    | BT.IntOptional -> 
                         sprintf "c10::optional<int64_t>(%s)" y
                    | BT.DoubleOptional -> 
                         sprintf "c10::optional<double>(%s)" y
                    | BT.ScalarType -> sprintf "c10::ScalarType(%s)" y
                    | BT.ScalarTypeOptional -> sprintf "c10::optional<c10::ScalarType>(c10::ScalarType(%s))" y
                    | BT.String -> sprintf "std::string(%s)" y
                    | _ -> y
                if arg.Gated then
                    match arg.type_ with
                    | BT.ScalarTypeOptional -> 
                        ternaryIfThenElse(sprintf "with_%s" arg.name, y, "c10::nullopt")
                    | _ -> 
                        ternaryIfThenElse(sprintf "with_%s" arg.name, y, defaultValue.Value)
                else
                    if (match arg.type_ with | BT.TensorOptions | BT.TensorOptionsAnd -> true | _ -> false) && arg.name = "options" then
                        y
                    else
                        if arg.isNullable && arg.defaultValue.IsSome then
                            ternaryIfThenElse(arg.name, y, defaultValue.Value)
                        else y
        member this.IsTensorOrScalar = match this.type_ with | TensorOrScalar _ -> true | _ -> false
        member this.ParameterNullable = this.isNullable || this.defaultValue.IsSome 
        /// Not wrappable in Nullable<T>
        member this.Nullableable = 
            match this.type_ with
            | BT.IntList 
            | TensorOrScalar _ -> false
            | _ -> true

        member this.IsTensorOption = 
            match this.type_ with
            | BT.TensorOptions
            | BT.TensorOptionsAnd when this.name = "options" -> true
            | _ -> false

    type BT with
        member x.ReqPinning = x.IsArray || match x with | _ -> false 

    [<RequireQualifiedAccess>]
    type RT = 
     | Empty
     | Single of BT
     | SingleTensor
     | SingleScalar
     | TensorTuple of int
     | ManyTensor
     with 
        member this.HasAllocator = 
            match this with
            | RT.TensorTuple _ 
            | RT.ManyTensor -> true
            | _ -> false

        member this.Return = 
            match this with
            | RT.SingleTensor
            | RT.Single BT.Tensor -> "Tensor"
            | RT.SingleScalar
            | RT.Single BT.Scalar -> "Scalar"
            | RT.Single BT.Bool -> "bool"
            | RT.Single BT.Int -> "int" 
            | RT.Single BT.Double -> "double" 
            | RT.Empty
            | RT.ManyTensor
            | RT.TensorTuple _ -> "void"
            | RT.Single BT.ScalarType -> "int8_t"
            | _ -> failwithf "TODO %A" this

    type Schema with
        member this.Return =
            match this.returns with
            | [||] -> RT.Empty
            | [|x|] -> 
                match x.dynamicType with
                | BT.Tensor -> RT.SingleTensor
                | BT.Scalar -> RT.SingleScalar
                | BT.Bool -> RT.Single BT.Bool
                | BT.Int -> RT.Single BT.Int
                | BT.Double -> RT.Single BT.Double
                | BT.ScalarType -> RT.Single BT.ScalarType
                | BT.TensorVector
                | BT.TensorList -> RT.ManyTensor
                | _ -> failwith "todo"
            | xs when (xs |> Array.exists (fun x -> x.dynamicType <> BT.Tensor) |> not) -> 
                 RT.TensorTuple xs.Length
            | _ -> failwith "todo"

        member this.FunctionName = 
            this.name + this.overloadName + (if this.name.EndsWith("_") then "_" else "")

        member this.HasOption = this.args |> Array.exists (fun x -> x.IsTensorOption)

        member schema.IsInstanceAndSelfName = 
            let isInstanceMember = schema.methodOfTensor = Some(true)
            let name = 
                if not isInstanceMember then None
                else 
                    // it appears that instance memebers may not be the first argument (e.g. polygamma)
                    // special casing for the second argument
                    // TODO maybe generalize this
                    match schema.args |> Array.toList with
                    | [] -> None
                    | (x::_) when (match x.type_ with | Tensor _ -> true | _ -> false) -> Some(schema.args.[0].name)
                    | (_::x::_) when (match x.type_ with | Tensor _ -> true | _ -> false) ->  Some(schema.args.[1].name)
                    | _ ->  None
            (isInstanceMember,name)

    let genImport(singleLine: bool) (schema: Schema): string  = 
        let ret = 
            match schema.Return with
            | RT.Empty -> "void"
            | RT.Single BT.Bool -> "bool"
            | RT.Single BT.Int -> "int"
            | RT.Single BT.Double -> "double"
            | RT.Single BT.ScalarType -> "sbyte"
            | RT.SingleScalar
            | RT.SingleTensor -> "IntPtr"
            | RT.ManyTensor -> "void"
            | RT.TensorTuple _ -> "void"
            | _ -> failwithf "todo return name %A" schema.Return

        let args = 
            schema.args |> Array.map (fun x -> x.CSharpInteropSignature)
            |> fun xs -> if schema.Return.HasAllocator then Array.append [|TensorAllocatorCSharp;|] xs else xs 
        let firstLine = 
            sprintf "private static extern %s %s%s(" ret THSTensor schema.FunctionName 
        if singleLine 
        then sprintf "%s%s);" firstLine (args |> String.concat ", ")
        else sprintf "%s%s    %s);" firstLine Environment.NewLine (args |> String.concat (sprintf ",%s    " Environment.NewLine))

    let genHeader (singleLine: bool, forExport: bool) (schema: Schema) : string[] = 
        let r = schema.Return
        let overloadedName = schema.FunctionName
        let args = 
            schema.args |> Array.map (fun x -> x.CSignature)
            |> fun xs -> if r.HasAllocator then Array.append [|TensorAllocator|] xs  else xs
        let returnWithExport = 
            if forExport then sprintf "%s(%s)" EXPORT_API r.Return
            else r.Return
        if singleLine then
            [|sprintf "%s %s%s(%s)" returnWithExport THSTensor overloadedName (args |> String.concat ", ")|]
        else 
            [|
                yield sprintf "%s %s%s(" returnWithExport THSTensor  overloadedName 
                yield! 
                    match args |> multiLineParams with
                    | [||] -> [|")"|]
                    | xs -> xs
            |] 

    let genCpp(schema: Schema) : string[] = 
        let r = schema.Return
        let first,tailArgs = 
            match schema.methodOfNamespace, schema.methodOfTensor with
            | Some(true),_ -> (sprintf "torch::%s(" schema.name, schema.args)
            | _, Some(true) -> ((sprintf "%s->%s(") schema.args.[0].name schema.name, schema.args.[1..])
            | _, _ -> failwith "err - come back to this if needed"

        let options = [|
            "auto options = at::TensorOptions()"
            "    .dtype(at::ScalarType(scalar_type))"
            "    .device(device)"
            "    .requires_grad(requires_grad);" |]
       
        let simpleFunc(firstLine,args,m: string option) = 
            [|yield firstLine; yield! args |> multiLineParams |> indent|]
            |> compressLines |> addFinalSemiColon
            |> fun xs -> 
                match m with
                | None -> xs
                | Some(m) -> xs |> macro(m,true) |> compressLines
        [|
            let xs = tailArgs |> Array.map (fun x -> x.CppToC) 
            let manyLines = tailArgs |> Array.map (fun x -> x.CppToC)
            if schema.HasOption then yield! options
            match r with
            | RT.SingleScalar -> yield!  simpleFunc(first, manyLines,Some("CATCH_SCALAR"))  
            | RT.SingleTensor -> yield! simpleFunc(first, manyLines,Some("CATCH_TENSOR"))
            | RT.TensorTuple c-> 
                yield!
                    [|
                        yield! simpleFunc(sprintf "auto res = %s" first, manyLines, None) 
                        yield sprintf "Tensor * result = allocator(%i);" c
                        for i in 0..c-1 do
                            yield sprintf "result[%i] = new torch::Tensor(std::get<%i>(res));" i i
                    |] //|> macro("CATCH",true)
            | RT.ManyTensor ->
                yield!
                    [|
                        yield! simpleFunc(sprintf "auto res = %s" first, manyLines,None) 
                        yield "const size_t sz = res.size();"
                        yield "Tensor * result = allocator(sz);"
                        yield "for (size_t i = 0; i < sz; i++)"
                        yield "    result[i] = new torch::Tensor(res[i]);"
                    |] //|> macro("CATCH",true)
            | RT.Empty -> 
                yield! simpleFunc(sprintf "    %s" first, manyLines,Some("CATCH"))
            | RT.Single BT.Int
            | RT.Single BT.Double
            | RT.Single BT.Bool -> 
                yield! simpleFunc(sprintf "return %s" first, manyLines, None)
            | RT.Single BT.ScalarType -> 
                yield! simpleFunc(sprintf "return (int8_t) %s" first, manyLines, None)
            | _ -> failwithf "todo %A" r
        |]
        |> fun body -> 
            let singleLine = (genHeader (true,false) schema).[0] 
            if singleLine.Length < TargetLineWidth then func singleLine body
            else funcMany (genHeader (false,false) schema) body

    let genCSharp(schema: Schema) : string[] = 
        let rt = 
            match schema.Return with
            | RT.Empty -> "void"
            | RT.SingleTensor -> "TorchTensor"
            | RT.SingleScalar -> "Scalar"
            | RT.Single BT.Bool -> "bool"
            | RT.Single BT.Int -> "int"
            | RT.Single BT.Double -> "double"
            | RT.Single BT.ScalarType -> "ScalarType"
            | RT.Single x -> failwithf "return type not yet supported %A" x
            | RT.TensorTuple n -> [|for x in schema.returns -> sprintf "TorchTensor %s" (x.name |> toCSharpName)|] |> asTuple
            | RT.ManyTensor -> "TorchTensor[]"
        
        let isInstanceMember,selfName = schema.IsInstanceAndSelfName
        let args = schema.args

        let parameters = 
            args 
            |> Array.partition (fun x -> x.IsOptional) 
            |> fun (xs,ys) -> [| yield! ys; yield! xs |]
            |> Array.filter (fun x -> x.IsTensorOption |> not)
            |> Array.map (fun arg -> 
                let argType = 
                    match arg.type_ with
                    | BT.TensorOptions
                    | BT.TensorOptionsAnd when arg.name = "options" -> 
                        failwith "options argument should not reach this stage"
                    | BT.Bool2 | BT.Bool3 | BT.Bool4 -> failwithf "todo %A" arg
                    | BT.IntList -> "long[]"
                    | BT.TensorList -> "TorchTensor[]"
                    | BT.DimnameList -> failwithf "todo %A" arg 
                    | BT.String -> "string"
                    | _ when arg.type_.IsArray -> failwithf "err %A" arg
                    | _ ->
                        match arg.type_ with
                        | BT.Bool -> "bool" 
                        | BT.Int -> "long"
                        | BT.IntOptional-> "long"
                        | BT.Double -> "double"
                        | BT.DoubleOptional -> "double"
                        | BT.Tensor -> "TorchTensor"
                        | BT.TensorAnd -> "TorchTensor"
                        | BT.ConstTensor -> "TorchTensor"
                        | BT.ScalarType 
                        | BT.ScalarTypeOptional -> "ScalarType"
                        | BT.Device -> failwith "todo"
                        | BT.Scalar -> "Scalar"
                        | BT.ScalarOptional -> "Scalar" 
                        | BT.MemoryFormat 
                        | BT.MemoryFormatOptional -> "MemoryFormat" // unsure about this
                        | _ -> failwithf "todo %A" arg
                let name = (arg.name |> toCSharpName)
                match arg.ParameterNullable, arg.Nullableable with 
                | true,true-> sprintf "%s? %s = null" argType  name 
                | true,false -> sprintf "%s %s = null" argType name 
                | false,_ -> sprintf "%s %s" argType name
            )
            |> fun xs -> 
                if not schema.HasOption then xs
                else 
                    xs |> Array.prepend [|
                        "ScalarType dtype = ScalarType.Float"
                        "string device = \"cpu\""
                        "bool requiresGrad = true"
                    |]
            |> String.concat ", "
        let name = (schema.name|> toCSharpName) // NOTE: Not using overload
        let firstLine = 
            sprintf "public %s%s %s(%s)" 
                (if not isInstanceMember then "static " else "") rt name parameters
        let anyPinning = args.Length > 0 && args |> Array.exists (fun x -> x.type_.ReqPinning)
        let isUnsafe = anyPinning // || ... todo other reasons
        let hasAllocator = match schema.Return with | RT.TensorTuple _ | RT.ManyTensor -> true | _ -> false
        
        let cArgs = 
            [| 
                // This complication is around the allocator function, instead we will add it to the first parameter
                if hasAllocator then yield "pa.CreateArray"
                for arg in args do
                    match arg.type_ with
                    | BT.TensorOptions
                    | BT.TensorOptionsAnd when arg.name = "options" -> ()
                    | _ -> 
                        if isInstanceMember && Some(arg.name) = selfName then 
                            yield "Handle" 
                        else
                            let argName = arg.name |> toCSharpName 
                            let conversion = 
                                match arg.type_ with
                                | BT.ScalarType
                                | BT.ScalarTypeOptional-> "(sbyte) "
                                | _ -> ""
                            if arg.IsTensorOrScalar then 
                                match arg.ParameterNullable, arg.Nullableable, arg.type_.IsArray with
                                | _,_,true-> yield sprintf "(IntPtr) p%s" argName
                                | true,false,_ -> yield sprintf "%sPtr" argName
                                | _,_,false -> yield argName + ".Handle"
                            elif arg.Gated then 
                                // This is based on type and is completely ignored
                                let cSharpDefault = 
                                    match arg.type_ with
                                    | BT.BoolOptional
                                    | BT.Bool -> "false"
                                    | BT.DoubleOptional
                                    | BT.Double -> "0.0"
                                    | BT.Int 
                                    | BT.IntOptional -> "-1"
                                    | TensorOrScalar _ -> "IntPtr.Zero"
                                    | BT.ScalarType 
                                    | BT.ScalarTypeOptional -> "ScalarType.Float"
                                    | BT.DoubleOptional
                                    | _ -> "todo"
                                yield sprintf "%s.HasValue" argName
                                yield sprintf "%s%s.GetValueOrDefault(%s)" conversion argName cSharpDefault
                            elif arg.isNullable then 
                                yield conversion + argName
                            else 
                                if arg.type_.ReqPinning 
                                then yield "(IntPtr) p" + argName
                                else yield conversion + argName 
                            if arg.type_.IsArray then
                                yield sprintf "%s.Length" argName
                if schema.HasOption then yield! [|"(sbyte) dtype"; "device"; "requiresGrad"|]
            |] 

        let checkForErrors = "Torch.CheckForErrors();"

        let emptyDefaults = 
            [| 
                for arg in args do
                    let name = arg.name |> toCSharpName
                    match arg.ParameterNullable, arg.Nullableable with
                    | false,true-> ()
                    | false,false -> 
                        yield sprintf "_ = %s ?? throw new ArgumentNullException(nameof(%s));" name name
                    | true,true -> () // Nullable<T> are handled elsewhere
                    | true,false -> 
                        yield
                            match arg.type_ with
                            | BT.TensorVector
                            | BT.TensorList -> 
                                sprintf "%s ??= new TorchTensor[0];" name
                            | TensorOrScalar _ -> 
                                sprintf "IntPtr %sPtr = %s is null ? IntPtr.Zero : %s.Handle;" name name name
                            | BT.IntList -> sprintf "%s ??= new long[0];" name 
                            | _ -> sprintf "// todo %A" arg.type_
            |]

        let nativeCall = 
            [| yield sprintf "%s%s(" THSTensor schema.FunctionName; yield! cArgs |> multiLineParams |> indent|]
            |> compressLines |> addFinalSemiColon

        [|
            match schema.Return with 
            | RT.TensorTuple _ | RT.ManyTensor ->
                yield "IntPtr[] ptrArray;"
                yield! 
                    [|yield! nativeCall; yield checkForErrors ; yield "ptrArray = pa.Array;"|]
                    |> func "using (var pa = new PinnedArray<IntPtr>())"
                match schema.Return with
                | RT.TensorTuple n -> 
                    yield!
                        [|
                            yield "return ("
                            yield! 
                                [|for i in 0 .. n - 1 -> sprintf "new TorchTensor(ptrArray[%i])" i|]
                                |> multiLineParams |> addFinalSemiColon
                        |] |> compressLines
                | RT.ManyTensor -> 
                    yield "return ptrArray.Select(x => new TorchTensor(x)).ToArray();"
                | _ -> failwith "err"
            | RT.SingleTensor -> 
                yield! prefixFirstLine  "var res = " nativeCall
                yield checkForErrors
                yield "return new TorchTensor(res);"
            | RT.SingleScalar -> failwith "todo"
            | RT.Single BT.Bool 
            | RT.Single BT.Int
            | RT.Single BT.Double
                -> yield! prefixFirstLine "return "  nativeCall
            | RT.Single BT.ScalarType 
                -> yield! prefixFirstLine "return (ScalarType) " nativeCall
            | _ -> failwith "todo"
        |] 
        |> fun xs -> 
            if anyPinning then 
                args 
                |> Array.filter (fun x -> x.type_.ReqPinning)
                |> Array.groupBy (fun x -> x.type_)
                |> Array.map (fun (_,ys) -> 
                    let fixedType = 
                        match ys.[0].type_ with
                        | BT.IntList -> "long"
                        | BT.TensorVector
                        | BT.TensorList -> "IntPtr"
                        | _ -> "todo"
                    sprintf "%s* %s" fixedType
                        (ys |> Array.map (fun y -> 
                            let yName = y.name |> toCSharpName
                            let yName' = 
                                match y.type_ with 
                                | BT.TensorVector 
                                | BT.TensorList -> yName + ".Select(x => x.Handle).ToArray()" 
                                | _ -> yName
                            sprintf "p%s = %s" yName yName') |> String.concat ", "))
                |> fun ys -> nestedFixed ys xs
            else xs
        |> Array.append emptyDefaults
        |> fun xs -> if isUnsafe then unsafe xs else xs
        |> func firstLine 

module Filter = 
    let aditionalFiltered = 
        set [
            // These have mixed return tuples
            "_batch_norm_impl_index","" // (Tensor * Tensor * Tensor * Tensor * int)
            "fbgemm_linear_quantize_weight","" // (Tensor * Tensor * double * int)
            "convolution_backward_overrideable","" // Bool3
            "from_file","" // BoolOptional
            "to","device" // MemoryFormatOptional
            "item","" //Scalar return type
            "_local_scalar_dense","" // Scalar return type
            "set_quantizer_", "" // ConstQuantizerPtr
            "qscheme", "" // QScheme
        ]

    let excludedFunctions  = set [
          "multi_margin_loss";
          "multi_margin_loss_out";
          "log_softmax_backward_data";
          "softmax_backward_data";
          "clone";
          "copy_";
          "conv_transpose2d_backward_out";
          "conv_transpose3d_backward_out";
          "slow_conv_transpose2d_backward_out";
          "slow_conv_transpose3d_backward_out";
          "slow_conv3d_backward_out";
          "normal";
          "_cufft_set_plan_cache_max_size";
          "_cufft_clear_plan_cache";
          "backward";
          "set_data";
          "_amp_non_finite_check_and_unscale_";
          "_cummin_helper";
          "_cummax_helper";
          "retain_grad"; ]

    let excludedPrefixes = set [ "_thnn_"; "_th_"; "thnn_"; "th_" ]
    let excludedSuffixes = set [ "_forward"; "_forward_out" ]

module TorchCodeGen = 
    open Parser
    open Filter
    open CodeGenCommon

    // This is for adding functions that are not included
    let methods() = 
        let baseFunc : Schema = 
            {name = ""; operatorName = ""; overloadName = ""; args = [||]; returns = [||]; depricated = false; methodOfTensor = Some(true); methodOfNamespace = None} 
        let f (name: string)  (baseType: BT) : Arg = 
            { type_ = baseType;  name = name; defaultValue = None; isNullable = false; annotation = None; dynamicType = baseType }
        [|
            "grad", [||]
            "set_requires_grad", [|f "r" BT.Bool|]
            "toType", [|f "scalar_type" BT.ScalarType|]
            "to", [|f "device" BT.Device|]
        |] |> Array.map (fun (name,inputs) -> 
            {baseFunc with name = name; args = [|yield f "self" BT.Tensor; yield! inputs|]})

    let filterFunctionsSummary(fs : (string * ('a -> bool))[], key : 'a -> string) (xs: 'a[]) = 
        let filterSets = 
            fs 
            |> Array.map (fun (fName,f) -> (fName, xs |> Array.filter (f >> not) |> Array.map key |> Set)) 
            |> Array.sortByDescending (fun (_,xs) -> xs.Count)
        printfn "Total Unfiltered %i" xs.Length
        let unfilteredSet = (xs,fs) ||> Array.fold (fun xs (_,f) -> xs |> Array.filter f) |> Array.map key |> Set
        printfn "Total Filtered %i" unfilteredSet.Count
        printfn "Total filtered"
        for (name,set) in filterSets do
            printfn "%s %i" name set.Count
        printfn "Incremental filtered"
        filterSets |> Array.map (fun (name,_) -> 
           (xs,fs) 
           ||> Array.fold (fun xs (n,f) -> if n = name then xs else xs |> Array.filter f) 
           |> Array.map key |> Set 
           |> fun otherSet -> name, (otherSet.Count - unfilteredSet.Count))
        |> Array.sortByDescending snd
        |> Array.iter (fun (name,c) -> printfn "%s %i" name c)

    let filterFuncs = 
        [|
            "Depricated",          (fun x -> not (x.overloadName = "deprecated" || x.depricated)) 
            "Exclude Prefixes",    (fun x -> excludedPrefixes |> Seq.exists (fun y -> x.name.StartsWith(y)) |> not)
            "Exclude Suffixes",    (fun x -> excludedSuffixes|> Seq.exists (fun y -> x.name.EndsWith(y)) |> not)
            "Generators",          (fun x -> not( x.overloadName.EndsWith("generator") || x.overloadName.EndsWith("generator_out")))
            "source",              (fun x -> x.overloadName.StartsWith("source_") |> not)
            "Excluded Functions",  (fun x -> excludedFunctions.Contains(x.name) |> not) // 20
            "Arg Types Dimname",   (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.Dimname | BT.DimnameList | BT.DimnameListOptional -> true | _ -> false) |> not)
            "Backward",            (fun x -> not(x.name.EndsWith("backward") || x.name.EndsWith("backward_out")))
            "Arg Types Generator", (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.Generator -> true | _ -> false) |> not)
            "Excluded Functions 2",(fun x -> aditionalFiltered.Contains(x.name,x.overloadName) |> not)
            "Arg Types Memory",    (fun x -> x.args |> Array.exists (fun x -> match x.type_ with | BT.MemoryFormat | BT.MemoryFormatOptional -> true | _ -> false) |> not)
        |]

    let codeGen(metadataPath: string) =
        let schemas = (Parser.schemas(metadataPath),filterFuncs) ||> Array.fold (fun xs (_,f) -> xs |> Array.filter f) 

        let HeaderHeader = 
            [|  "#pragma once"
                "#include \"../Stdafx.h\""
                "#include \"TH/THTensor.h\""
                "#include \"torch/torch.h\""
                "#include \"Utils.h\"" |]

        let CPPHeader = 
            [|  "#include \"THSTensorG.h\""
                "#include <iostream>"
                "#include <fstream>" |]

        let CSharpHeader = 
            [|  "using System;"
                "using System.Linq;"
                "using System.Runtime.CompilerServices;"
                "using System.Runtime.InteropServices;"
                "using System.Text;" |]

        let CSharpBody = 
            [| 
                for schema in schemas do
                    yield CSharp.DLLImport
                    yield schema |> CodeGenLL.genImport(true)
                    yield ""
                    yield! schema |> CodeGenLL.genCSharp
            |] 
            |> Cpp.func ("public partial class TorchTensor : IDisposable")
            |> Cpp.func ("namespace TorchSharp.Tensor")

        ([|yield! CSharpHeader; yield! CSharpBody|],
         [|yield! CPPHeader; yield! schemas |> Array.map CodeGenLL.genCpp |> interleaveWithNewLine|],
         [|yield! HeaderHeader; yield! schemas |> Array.map (CodeGenLL.genHeader (true,true) >> addFinalSemiColon) |> Array.concat|])

open Argu

type Arguments = 
    | [<AltCommandLine("-wd")>] Output_Directory of path:string
    | [<AltCommandLine("-d")>] [<Mandatory>] Declarations of path:string
    | [<AltCommandLine("-cs")>] CSharp of path:string
    | [<AltCommandLine("-hpp")>] CPP_Header of path:string
    | [<AltCommandLine("-cpp")>]CPP_Body of path:string

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Output_Directory _ -> "Output directory for output paths without root"
            | Declarations _ -> "PyTorch declarations file"
            | CSharp _ -> "Code generated C# file"
            | CPP_Header _ -> "Code generated Cpp Header file"
            | CPP_Body _ -> "Code generated Cpp Body file"


#if INTERACTIVE
#else
[<EntryPoint>]
let main argv =
    let parser = ArgumentParser.Create<Arguments>(programName = "TorchSharpCodeGen.exe")
    try
         let results = parser.ParseCommandLine(inputs = argv, raiseOnUsage = true)
         let wd = defaultArg (results.TryGetResult(<@ Output_Directory @>)) System.Environment.CurrentDirectory
         let rootWithWD (x:string) = if Path.IsPathRooted x then x else Path.Combine(wd,x)
         let csharp, cpp_header, cpp_body = 
            [| "TorchTensorG.cs", <@ CSharp @>; "THSTensorG.h", <@ CPP_Header @>; "THSTensorG.cpp", <@ CPP_Body @> |]
            |> Array.map (fun (defaultName, arg) -> defaultArg (results.TryGetResult arg) defaultName  |> rootWithWD)
            |> fun xs -> xs.[0], xs.[1], xs.[2]
         let declarations = results.GetResult <@ Declarations @> |> rootWithWD
         let csharpCode,cppBodyCode,cppHeaderCode = TorchCodeGen.codeGen(declarations)
         File.WriteAllLines(csharp, csharpCode)
         File.WriteAllLines(cpp_body, cppBodyCode)
         File.WriteAllLines(cpp_header, cppHeaderCode)
         let nl = Environment.NewLine
         printfn "Source files %s%s%s have been written" nl ([|csharp; cpp_header; cpp_body|] |> Array.map (sprintf "  * %s") |> String.concat nl) nl
         0
     with e -> printfn "%s" e.Message; 0
#endif
