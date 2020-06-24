// TODO - don't use this, use code gen protobuf instead
(*
#I "C:/Users/moloneymb/.nuget/packages"

#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r "microsoft.ml.onnxruntime/1.1.2/lib/netstandard1.1/Microsoft.ML.OnnxRuntime.dll"
#r @"google.protobuf/3.11.2/lib/netstandard2.0/Google.Protobuf.dll"
#r "../protobuf/onnx/csharp/OnnxMLProto.dll"

// TODO code-gen basic datatype

open Onnx

type SupportType = 
| COMMON
| EXPERIMENTAL

type UseType = 
| DEFAULT
| CONSUME_ALLOWED
| CONSUME_ENFORCED

type AttrType = 
| FLOAT
| INT
| STRING
| TENSOR
| GRAPH
| SPARSE_TENSOR
| FLOATS
| INTS
| STRINGS
| TENSORS
| GRAPHS
| SPARSE_TENSORS

type FormalParameterOption =
| Single
| Optional
| Variadic

type Attribute = {
    name : string
    description : string
    ``type`` : AttrType
    default_value : AttributeProto
    required : bool
}

type FormalParameter = {
    name : string
    types : string[]
    typeStr : string
    description : string
    option : FormalParameterOption
    isHomogeneous : bool
}

type TypeConstraintParam = {
    type_param_str : string
    description : string
    allowed_type_strs : string[]
}

type OpSchema = {
    file : string
    line : int
    support_level : SupportType
    doc : string option
    since_version : int
    deprecated : bool
    domain : string
    name : string
    min_input : int
    max_input : int
    min_output : int
    max_output : int
    attributes : (string*Attribute)[]
    inputs : FormalParameter[]
    outputs : FormalParameter[]
    type_constraints : TypeConstraintParam[]
    has_type_and_shape_inference_function : bool
}

//[|
//    {
//        file = ""
//        line = 0
//        support_level = SupportType.COMMON
//        doc = Some("")
//        since_version = 0
//        deprecated = false
//        domain = ""
//        name = ""
//        min_input = 0
//        max_input = 0
//        min_output = 0
//        max_output = 0
//        attributes = [|
//            ("auto_pad",{
//                name = ""
//                description = ""
//                ``type`` = AttrType.FLOAT
//                default_value = AttributeProto()
//                required = false
//            })
//        |]
//        inputs = [|
//            {
//                name = ""
//                types = [|""|]
//                typeStr = ""
//                description = ""
//                option = FormalParameterOption.Single
//                isHomogeneous = false
//            } 
//        |]
//        outputs = [|
//            {
//                name = ""
//                types = [|""|]
//                typeStr = ""
//                description = ""
//                option = FormalParameterOption.Single
//                isHomogeneous = false
//            } 
//        |]
//        type_constraints = [|
//            {
//                type_param_str = ""
//                description = ""
//                allowed_type_strs = [|""|]
//            }
//        |]
//        has_type_and_shape_inference_function  = true
//    } 
//|]
//
*)
