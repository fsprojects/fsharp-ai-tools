# This is 'throw away' code 
# This should be re-written in F# with some kind of F# bindings


# NOTE: Curried API requires overloading on the curried parameter which we cannot do at this stage
#       A case could be made for using non-generic Tensors which would not need overloaded functions
# NOTE: This was run in Linux, getting onnx wheel to work in windows requires python 3.7
# NOTE: Single inputs allways appear before Optional
# NOTE: Variadic inputs are at most one and are always last
# NOTE: Only 'Loop' has Variadic and Optional
# NOTE: 'NonMaxSuppression' #has no type constraint
# TODO handle AttrType.TENSOR and AttrType.SPARSE_TENSOR for 'Constant' nodes
# TODO documentation
# TODO consider returning anonomous types instead of tuples
# TODO ['DictVectorizer', 'ConcatFromSequence', 'SplitToSequence', 'SequenceErase', 'SequenceAt', 'SequenceInsert', 'SequenceConstruct', 'ConstantOfShape', 'If', 'ZipMap', 'SequenceLength', 'Scan', 'Loop', 'Split', 'Constant', 'CastMap']
# TODO QuantizeLinear, input y_zero_point should not be optional as it's the only parameter with output of T2
# TODO NonMaxSuppression, output type is a single list that happens to match an input typeStr, should jsut be DataType.INT64

from onnx import defs
from onnx import AttributeProto
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
from collections import defaultdict
import re
import os

# TODO relative path
dirname = os.path.dirname(__file__)
snake_path = os.path.join(dirname, 'Onnx.API.SnakeCase.g.fs')
snake_module = "FSharp.ML.Onnx.API.SnakeCase"
pascal_path = os.path.join(dirname, 'Onnx.API.PascalCase.g.fs')
pascal_module = "FSharp.ML.Onnx.API.PascalCase"
curried_path = os.path.join(dirname, 'Onnx.API.Curried.g.fs')
curried_module = "FSharp.ML.Onnx.API.Curried"
graph_path = os.path.join(dirname, 'Onnx.API.Graph.g.fs')

fsharp_keywords = set ([
    "abstract", "as", "base", "begin", "class", "default", "delegate", "do", "done", "downcast", "downto", "elif",
    "else", "end", "exception", "extern", "false", "finally", "fixed", "for", "fun", "function", "global", "if", "in",
    "inherit", "inline", "interface", "internal", "lazy", "let", "let!", "match", "match!", "member", "module", "mutable",
    "namespace", "new", "not", "null", "of", "open", "or", "override", "private", "public", "rec", "return", "return!",
    "select", "static", "struct", "then", "to", "true", "try", "type", "upcast", "use", "use!", "val", "void", "when",
    "while", "with", "yield", "yield!", "mod", "and"])

def pascal_to_underscore(name):
    underscore_name = re.sub(r'(?!^)([A-Z])([a-z])', r'_\1\2', name).lower()
    return underscore_name + "_" if underscore_name in fsharp_keywords else underscore_name

def countby(f, seq):
    d = defaultdict(int)
    for i in seq: d[f(i)] += 1
    return dict(d)

def groupby(f, seq):
    d = defaultdict(list)
    for i in seq: d[f(i)].append(i)
    return dict(d)

AttrType = OpSchema.AttrType
FormalParameterOption = OpSchema.FormalParameterOption

def getSchemas():
    schemas = [x for x in defs.get_all_schemas_with_history() if not x.deprecated]
    max_version = {}
    for x in schemas:
        if x.name in max_version:
            max_version[x.name] = max(max_version[x.name],x.since_version)
        else:
            max_version[x.name] = x.since_version
    return [x for x in schemas if x.since_version == max_version[x.name]]

schemas = getSchemas()

def hasType(z,schema):
    return len([tc for tc in schema.type_constraints if len([y for y in tc.allowed_type_strs if z in y]) > 0]) > 0

def mapNames(xs):
    return [x.name for x in xs]

todo_schemas = (['ConstantOfShape','Constant'] + # Attribute TENSOR or SPARSE_TENSOR #[x.name for x in schemas if len([attr for (_,attr) in x.attributes.items() if (attr.type == AttrType.TENSOR or attr.type == AttrType.SPARSE_TENSOR)]) > 0]
                ['ConcatFromSequence', 'SplitToSequence', 'SequenceErase', 'SequenceAt', 'SequenceInsert', 'SequenceConstruct', 'SequenceEmpty', 'SequenceLength'] + #sequences [x.name for x in schemas if hasType("seq",x)]
                ['DictVectorizer', 'ZipMap', 'CastMap'] + #maps [x.name for x in schemas if hasType("map",x)]
                ['If', 'Scan', 'Loop', ] + # GRAPH Attribute # Loop has Combined Optional and Variadic inputs
                ['Split'] + # variadic output
                ['OneHot'] + # remove OneHot for now, will spcial case it later
                ['Cast'] + #Done 'to' attribute
                ['SequenceEmpty', 'EyeLike', 'Multinomial', 'RandomUniformLike', 'RandomNormalLike', 'RandomNormal', 'RandomUniform'] + # Done 'dtype' in the attributes
                ['TreeEnsembleClassifier', 'LSTM', 'LinearClassifier', 'SVMClassifier', 'MaxPool', 'GRU', 'TopK', 'Dropout', 'Unique', 'DynamicQuantizeLinear', 'RNN', 'BatchNormalization'] + # Done multi-outputs [x for x in schemas if (x.max_output != 1 or x.min_output != 1)]
                ['LabelEncoder', 'CategoryMapper']) #Done # special case, these appear to have a single input and a single output with the same type...

filtered = set(todo_schemas)

schemas1 = list(filter(lambda x: x.name not in filtered, schemas))

def getSchema(name):
    return [x for x in schemas if x.name == name][0]

def filterSchemas(xs,yss):
    filterx = set(mapNames([y for ys in yss for y in ys]))
    return [x for x in xs if x.name not in filterx]

so_zero_type = ([x for x in schemas1 if len(x.type_constraints) == 0]) # ['NonMaxSuppression', 'StringNormalizer'] # 2
so_single_type = ([x for x in schemas1 if len(x.type_constraints) == 1]) # 105
so_single_output_type = ([x for x in schemas1 if len(x.type_constraints) == 2 and len(list(x.outputs[0].types)) == 1]) # 11
so_multi_type = [x for x in filterSchemas(schemas1,[so_single_type, so_single_output_type ]) if len(x.type_constraints) > 1] 

def getTypeMappings(schema):
    def combinations(xss):
        def recCombInner(acc,xss):
            if len(xss) == 0:
                yield acc
            else:
                head, *tail = xss
                for x in head:
                    yield from recCombInner(acc + [x], tail)
        return recCombInner([],xss)
    return [{y:x for (x,y) in zip(xs,[z.type_param_str for z in schema.type_constraints])}  for xs in list(combinations([x.allowed_type_strs for x in schema.type_constraints]))]

def wrap(x,y):
    def f(z):
        return x + z + y
    return f
quote = wrap("\"","\"")
record = wrap("{","}")

def getBool(x):
    return "true" if x else "false"

def getArray(xs):
    return wrap("[|","|]")(";".join(xs))

def getByteStrings(xs):
    return getArray(map(lambda x: quote(x.decode()), xs))

def getStrings(xs):
    return getArray(map(lambda x: quote(str(x)), xs))

def getInt32s(xs):
    return getArray(xs)

def getInt64s(xs):
    return getArray(map(lambda x: str(x) + 'L', xs))

def getFloat32s(xs):
    return getArray(map(lambda x: ("%.15f" % x).rstrip('0') + 'f', xs))

def getFloat64s(xs):
    return getArray(map(lambda x: ("%.15f" % x).rstrip('0'), xs))

# This is for diagnostics
def print_schema(x):
    print(x.name)
    print("(%i,%i) -> (%i,%i)" % (x.min_input, x.max_input, x.min_output, x.max_output))
    print("version %i" % x.since_version)
    print("Attrs")
    for (key,attr) in x.attributes.items():
        print("%s %s %s" % (attr.name, attr.type, ("r" if attr.required else "")))
    def print_formal_parameter(fp):
        print("%s %s %s %s %s" % (fp.name, getStrings(fp.types), fp.typeStr, fp.option, ("t" if fp.isHomogeneous else "f")))
    print("Inputs")
    for y in x.inputs:
        print_formal_parameter(y)
    print("Outputs")
    for y in x.outputs:
        print_formal_parameter(y)
    print("TypeConstraints")
    for y in x.type_constraints:
        print("%s %s" % (y.type_param_str,getStrings(y.allowed_type_strs)))

def mapONNXToFSharp(name):
    mapping = {
        "tensor(uint8)" :"uint8", 
        "tensor(uint16)" :"uint16",
        "tensor(uint32)" : "uint32",
        "tensor(uint64)" : "uint64",
        "tensor(int8)" : "int8",
        "tensor(int16)" : "int16",
        "tensor(int32)" : "int",
        "tensor(int64)" : "int64",
        "tensor(float16)" : None,
        "tensor(float)" : "float32",
        "tensor(double)" : "double", 
        "tensor(string)" : "string",
        "tensor(bool)" : "bool",
        "tensor(complex64)" : None,
        "tensor(complex128)" : "Complex",
    }
    return mapping.get(name)

def mapONNXToDataType(name):
    mapping = {
        "tensor(uint8)" :"UINT8", 
        "tensor(uint16)" :"UINT16",
        "tensor(uint32)" : "UINT32",
        "tensor(uint64)" : "UINT64",
        "tensor(int8)" : "INT8",
        "tensor(int16)" : "INT16",
        "tensor(int32)" : "INT32",
        "tensor(int64)" : "INT64",
        "tensor(float16)" : "FLOAT16",
        "tensor(float)" : "FLOAT32",
        "tensor(double)" : "DOUBLE", 
        "tensor(string)" : "STRING",
        "tensor(bool)" : "BOOL",
        "tensor(complex64)" : "COMPLEX64",
        "tensor(complex128)" : "COMPLEX128",
    }
    if mapping.get(name):
        return "DataType." + mapping.get(name)
    else:
        return None

def mapONNXToDtype(name):
    mapping = {
        "tensor(uint8)"     : 2,
        "tensor(uint16)"    : 4,
        "tensor(uint32)"    : 12,
        "tensor(uint64)"    : 13, 
        "tensor(int8)"      : 3, 
        "tensor(int16)"     : 5, 
        "tensor(int32)"     : 6,
        "tensor(int64)"     : 7,
        "tensor(float16)"   : None, #10
        "tensor(float)"     : 1,
        "tensor(double)"    : 11, 
        "tensor(string)"    : 8,
        "tensor(bool)"      : 9,
        "tensor(complex64)" : None, #14
        "tensor(complex128)": 15,
    }
    return mapping.get(name)

def choseFSharpTypes(type_constraint):
    return [mapONNXToFSharp(x) for x in type_constraint.allowed_type_strs if mapONNXToFSharp(x)]

def mapAttrType(attr):
    if attr.type == AttrType.FLOATS:
        return "float32[]"
    elif attr.type == AttrType.FLOAT:
        return "float32"
    elif attr.type == AttrType.INTS:
        return "int64[]"
    elif attr.type == AttrType.INT:
        return "int64"
    elif attr.type == AttrType.STRINGS:
        return "string[]"
    elif attr.type == AttrType.STRING:
        return "string"
    else:
        raise Exception(f'unsupported attribute type {attr.type}' )

def mapAttrFunction(attr):
    if attr.type == AttrType.FLOATS:
        return "floats"
    elif attr.type == AttrType.FLOAT:
        return "float"
    elif attr.type == AttrType.INTS:
        return "ints"
    elif attr.type == AttrType.INT:
        return "int"
    elif attr.type == AttrType.STRINGS:
        return "strings"
    elif attr.type == AttrType.STRING:
        return "string"
    else:
        raise Exception(f'unsupported attribute type {attr.type}' )

#This returns the inline F# code
def mapDefaultValue(default_value):
    if default_value.type == 0:   return '' #undefined
    elif default_value.type == 1: return f', {default_value.f}f' #float
    elif default_value.type == 2: return  f', {default_value.i}L' #int
    elif default_value.type == 3: return f', "{default_value.s.decode()}"' #string
    elif default_value.type == 7: return f', {getInt64s(default_value.ints)}' #ints
    elif default_value.type == 8: return f', {getByteStrings(default_value.strings)}' #strings
    else:
        raise Exception(f'unsupported default value of type {default_value.type}')

def partition(f,xs):
    return ([x for x in xs if f(x)],[x for x in xs if not f(x)])

def filterAttrType(t):
    def f(x):
        return t in set([y.type for (_,y) in x.attributes.items()])
    return f

def optionalChoose(x):
    return f'([|{x}|] |> Array.choose id)' if x else '[||]'

def inputParamString(x,typeMap):
    t = f'Tensor<{typeMap(x.typeStr) if typeMap(x.typeStr) is not None else mapONNXToFSharp(x.typeStr)}>' 
    if x.option == FormalParameterOption.Single: return f'{x.name}: {t}'
    elif x.option == FormalParameterOption.Optional: return f'?{x.name}: {t}'
    elif  x.option == FormalParameterOption.Variadic: return f'[<ParamArray>]{x.name}: {t}[]'
    else: raise Exception("shouldn't happen")

def partitionInputs(inputs):
    xs,ys,zs = [],[],[]
    for x in inputs:
        if x.option == FormalParameterOption.Single:
            xs.append(x)
        elif x.option == FormalParameterOption.Optional:
            ys.append(x)
        elif  x.option == FormalParameterOption.Variadic:
            zs.append(x)
        else: raise Exception("shouldn't happen")
    return (xs,ys,zs)

def get_part_inputs_and_attrs(schema):
    (req_inputs,opt_inputs,var_inputs) = partitionInputs(schema.inputs)
    (req_attrs, opt_attrs) = partition(lambda x: x.required, [x for (_,x) in schema.attributes.items()])
    return (req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs)

def get_params(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap):
    params = ", ".join(
        [inputParamString(x, typeMap) for x in req_inputs] + 
        [f'{x.name}: {mapAttrType(x)}' for x in req_attrs] + 
        [inputParamString(x, typeMap) for x in var_inputs] +
        [inputParamString(x, typeMap) for x in opt_inputs] + 
        [f'?{x.name}: {mapAttrType(x)}' for x in opt_attrs])
    return params

def get_params_curried(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap):
    if ((len(req_inputs) > 0) and ((len(req_inputs) + len(opt_inputs) + len(var_inputs) + len(req_attrs) + len(opt_attrs)) > 1)):
        params = ", ".join(
            [inputParamString(x, typeMap) for x in req_inputs[1:]] + 
            [f'{x.name}: {mapAttrType(x)}' for x in req_attrs] + 
            [inputParamString(x, typeMap) for x in var_inputs] +
            [inputParamString(x, typeMap) for x in opt_inputs] + 
            [f'?{x.name}: {mapAttrType(x)}' for x in opt_attrs])
        return params + ") (" + inputParamString(req_inputs[0], typeMap) 
    else:
        return get_params(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap)

def params_passthrough(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap):
    params = ", ".join(
        [f'{x.name} = {x.name}' for x in (req_inputs + req_attrs + var_inputs)] + 
        [f'?{x.name} = {x.name}' for x in (opt_inputs + opt_attrs)])
    return params

####################################################################################################
#                     Code Gen
####################################################################################################
#list(range(0,10))[1:]

def write_api(path, module, snake_case, curried):
    fo = open(path,"w")
    fo.write("module " + module + "\n")
    fo.write("\n")
    fo.write("open System\n")
    fo.write("open System.Numerics\n")
    fo.write("open Microsoft.ML.OnnxRuntime.Tensors\n")
    if snake_case:
        fo.write("type on = FSharp.ML.Onnx.API.PascalCase.Onnx\n")
        fo.write("\n")
        fo.write(f'[<ReflectedDefinition>]\n')
        fo.write("type Onnx() =\n")
    else:
        fo.write("open FSharp.ML.Onnx.Protobuf")
        fo.write("\n")
        fo.write("type Onnx() =\n")

    def change_cap(x):
         return pascal_to_underscore(x) if snake_case else x

    # NOTE: Req inputs, Req attr, Opt inputs, Opt attr, Var inputs

    schemas_out = []
    def get_params2(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap):
        if curried:
            return get_params_curried(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap)
        else:
            return get_params(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap)

    def code_gen_single_output(fo,schema,typeMap,outputTypes):
        (req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs) = get_part_inputs_and_attrs(schema)
        opt_attrs = [x for x in opt_attrs if not ("Pool" in schema.name and x.name == "ceil_mode")] # Should figure out how
        params = get_params2(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap)
        if snake_case:
            fo.write(f'    static member {change_cap(schema.name)}({params}) =\n')
            fo.write(f'        on.{schema.name}({params_passthrough(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap)})\n')
        else:
            fo.write(f'    static member {change_cap(schema.name)}({params}) =')
            attrProto = "; ".join([f'Attr.{mapAttrFunction(attr)}("{attr.name}", {attr.name}{mapDefaultValue(attr.default_value)})' for attr in (req_attrs + opt_attrs)])
            inputs = ""
            def wrapTensor(x):
                return f'mv.c({x})'
            # NOTE We're assuming inputs are matched structually by order
            if len(opt_inputs) == 0 and len(var_inputs) == 0: #only req_inputs
                inputs = '[|' + '; '.join([wrapTensor(x.name) for x in req_inputs]) + '|]'
            elif len(opt_inputs) != 0 and len(var_inputs) == 0:
                inputs = '([|' + '; '.join([f'Some({wrapTensor(x.name)})' for x in req_inputs] + [wrapTensor(x.name) for x in opt_inputs]) + '|] |> Array.choose id)'
            elif len(opt_inputs) == 0 and len(var_inputs) != 0:
                if len(req_inputs) == 0: inputs = f'({wrapTensor(var_inputs[0].name)})'
                else: inputs = '([|' + '; '.join([f'yield {wrapTensor(x.name)}' for x in req_inputs] + [f'yield! {wrapTensor(x.name)}' for x in var_inputs]) + '|])'
            elif len(opt_inputs) != 0 and len(var_inputs) != 0:
                raise Exception("ops with both optional and variadic inputs are not yet supported")
            else:
                raise Exception("shouldn't happen")
            if isinstance(outputTypes,list):
                outputs = ", ".join(outputTypes)
                print(outputTypes)
                fo.write(f'\n        MV() |> fun mv -> execNodeTuple{len(outputTypes)}<{outputs}> "{schema.name}" {inputs} {optionalChoose(attrProto)}\n')
            else:
                fo.write(f'\n        MV() |> fun mv -> execNode<{outputTypes}> "{schema.name}" {inputs} {optionalChoose(attrProto)}\n')

    # single type constraints
    for schema in so_single_type:
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        for t in choseFSharpTypes(schema.type_constraints[0]):
            code_gen_single_output(fo,schema,(lambda x: t if mapONNXToFSharp(x) is None else mapONNXToFSharp(x)),t)

    # two type constraints one for input one for output, output only has one type
    for schema in so_single_output_type:
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        output_type = [tc.allowed_type_strs for tc in schema.type_constraints if tc.type_param_str == schema.outputs[0].typeStr][0][0]
        #TODO check that inputs all have the same typeStr...
        input_types = [tc.allowed_type_strs for tc in schema.type_constraints if tc.type_param_str == schema.inputs[0].typeStr][0]
        for t in [mapONNXToFSharp(x) for x in input_types if mapONNXToFSharp(x)]:
            code_gen_single_output(fo,schema,(lambda x: t if mapONNXToFSharp(x) is None else mapONNXToFSharp(x)),mapONNXToFSharp(output_type))

    for schema in so_multi_type:
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        for type_mappings in getTypeMappings(schema):
            # skip if we contain unsupported types
            if len([v for (_,v) in type_mappings.items() if mapONNXToFSharp(v) is None ]) == 0:
                output_type = mapONNXToFSharp(type_mappings.get(schema.outputs[0].typeStr))
                code_gen_single_output(fo,schema,(lambda x: mapONNXToFSharp(type_mappings.get(x))),output_type)

    for schema in so_zero_type:
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        code_gen_single_output(fo,schema,(lambda x: None),mapONNXToFSharp(schema.outputs[0].typeStr))

    for schemaName in ['LabelEncoder', 'CategoryMapper']:
        schema = getSchema(schemaName)
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        # TODO WARN NOTE The output type appears unconstrained, we're going to assume that it is constrained to the only input
        for t in schema.type_constraints[0].allowed_type_strs:
            code_gen_single_output(fo,schema,(lambda x: mapONNXToFSharp(t)),mapONNXToFSharp(t))

    for schemaName in ['SequenceEmpty', 'EyeLike', 'Multinomial', 'RandomUniformLike', 'RandomNormalLike', 'RandomNormal', 'RandomUniform', 'Cast']:
        schema = getSchema(schemaName)
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        typeMaps = [t for t in schema.inputs[0].types if mapONNXToFSharp(t) is not None] if len(schema.inputs) == 1 else [""]
        for t in typeMaps:
            typeMap = lambda x: mapONNXToFSharp(t)
            (req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs) = get_part_inputs_and_attrs(schema)
            for (gen1,gen2) in ([("<'a>","'a")] if (len(schema.inputs) == 0 or schema.name == "Cast") else [("<'a>","'a"),("",mapONNXToFSharp(t))]):
                req_attrs = [x for x in req_attrs if x.name != 'dtype' and x.name != 'to']
                opt_attrs = [x for x in opt_attrs if x.name != 'dtype' and x.name != 'to']
                params = get_params2(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap)
                if snake_case:
                    fo.write(f'    static member {change_cap(schema.name)}{gen1}({params}) =\n')
                    fo.write(f'        on.{schema.name}{gen1}({params_passthrough(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap)})\n')
                else:
                    fo.write(f'    static member {change_cap(schema.name)}{gen1}({params}) =')
                    attrProto = "; ".join([f'Attr.{mapAttrFunction(attr)}("{attr.name}", {attr.name}{mapDefaultValue(attr.default_value)})' for attr in (req_attrs + opt_attrs)])
                    # NOTE We're assuming inputs are matched structually by order
                    inputs = '[|' + '; '.join([f'MV.mv(1,{x.name})' for x in req_inputs]) + '|]'
                    dtypes = "; ".join([str(mapONNXToDtype(x)) + "L" for x in  schema.outputs[0].types if mapONNXToDtype(x) is not None])
                    fo.write(f'\n        execNodeCheck<{gen2}> "{schema.name}" {inputs} [|{dtypes}|] {optionalChoose(attrProto)}\n')

    for schemaName in ['TreeEnsembleClassifier', 'LSTM', 'LinearClassifier', 'SVMClassifier', 'MaxPool', 'GRU', 'TopK', 'Dropout', 'Unique', 'DynamicQuantizeLinear', 'RNN', 'BatchNormalization']:
        schema = getSchema(schemaName)
        schemas_out.append(schema.name)
        print(f'{schema.name}')
        for type_mappings in getTypeMappings(schema):
            if not('Classifier' in schemaName and mapONNXToFSharp(type_mappings.get(schema.outputs[0].typeStr)) == "string"):
                # skip if we contain unsupported types
                if len([v for (_,v) in type_mappings.items() if mapONNXToFSharp(v) is None ]) == 0:
                    output_types = [mapONNXToFSharp(type_mappings.get(x.typeStr) if type_mappings.get(x.typeStr) is not None else x.typeStr) for x in schema.outputs]
                    code_gen_single_output(fo,schema,(lambda x: mapONNXToFSharp(type_mappings.get(x))),output_types)

    fo.flush()
    fo.close()


def write_graph_api(path):
    fo = open(path,"w")
    fo.write("module FSharp.ML.Onnx.API.Graph\n")
    fo.write("\n")
    fo.write("open System\n")
    fo.write("open FSharp.ML.Onnx.Protobuf\n")
    fo.write("\n")
    fo.write("type OnnxGraph() =\n")

    schemas_out = []

    def inputParamStringGraph(x,typeMap):
        t = 'ValueInfo'
        if x.option == FormalParameterOption.Single: return f'{x.name}: {t}'
        elif x.option == FormalParameterOption.Optional: return f'?{x.name}: {t}'
        elif  x.option == FormalParameterOption.Variadic: return f'[<ParamArray>]{x.name}: {t}[]'
        else: raise Exception("shouldn't happen")

    def get_params_graph(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap):
        params = ", ".join(
            [inputParamStringGraph(x, typeMap) for x in req_inputs] + 
            [f'{x.name}: {mapAttrType(x)}' for x in req_attrs] + 
            [inputParamStringGraph(x, typeMap) for x in var_inputs] +
            [inputParamStringGraph(x, typeMap) for x in opt_inputs] + 
            [f'?{x.name}: {mapAttrType(x)}' for x in opt_attrs])
        return params

    def code_gen_single_output_graph(fo,schema,typeMap,outputStr,outputCount):
        (req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs) = get_part_inputs_and_attrs(schema)
        opt_attrs = [x for x in opt_attrs if not ("Pool" in schema.name and x.name == "ceil_mode")] # Should figure out how
        params = get_params_graph(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMap)
        fo.write(f'    static member {schema.name}(graph: Graph, {params}) =')
        attrProto = "; ".join([f'Attr.{mapAttrFunction(attr)}("{attr.name}", {attr.name}{mapDefaultValue(attr.default_value)})' for attr in (req_attrs + opt_attrs)])
        inputs = ""
        # NOTE We're assuming inputs are matched structually by order
        if len(opt_inputs) == 0 and len(var_inputs) == 0: #only req_inputs
            inputs = '[|' + '; '.join([x.name for x in req_inputs]) + '|]'
        elif len(opt_inputs) != 0 and len(var_inputs) == 0:
            inputs = '([|' + '; '.join([f'Some({x.name})' for x in req_inputs] + [x.name for x in opt_inputs]) + '|] |> Array.choose id)'
        elif len(opt_inputs) == 0 and len(var_inputs) != 0:
            if len(req_inputs) == 0: inputs = f'({var_inputs[0].name})'
            else: inputs = '([|' + '; '.join([f'yield {x.name}' for x in req_inputs] + [f'yield! {x.name}' for x in var_inputs]) + '|])'
        elif len(opt_inputs) != 0 and len(var_inputs) != 0:
            raise Exception("ops with both optional and variadic inputs are not yet supported")
        else:
            raise Exception("shouldn't happen")
        fo.write(f'\n        graph.AddNode("{schema.name}", {inputs}, [|{outputStr}|], [|{attrProto}|]) |> toTuple{outputCount}\n')

    def mapOutputTypeNames(schema):
        tcs = {z.type_param_str:z.allowed_type_strs for z in schema.type_constraints}
        # This reversal makes sure it matches the first input with the typeStr
        # which is less likely to be an optional attribute
        ins = {x.typeStr : x.name for x in list(schema.inputs)[::-1]}
        isVariadic = {x.typeStr : (x.option == FormalParameterOption.Variadic) for x in list(schema.inputs)[::-1]}
        # TODO check if variadic, if so use the first first item
        def mapOutputTypeName(typeStr):
            if typeStr in ins:
                 return ins[typeStr] + (".[0].dt" if isVariadic[typeStr] else ".dt")
            else:
                 if typeStr in tcs:
                     ats = tcs[typeStr]
                     if len(ats) != 1:
                        print(ins)
                        print(ats)
                        print(tcs)
                        raise Exception("err")
                     return mapONNXToDataType(ats[0])
                 else:
                     # TODO perhaps add a check here
                     return mapONNXToDataType(typeStr)
        return [mapOutputTypeName(output.typeStr)  for output in schema.outputs]

    for schema in (so_zero_type + so_single_type + so_single_output_type + so_multi_type):
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        outputs = mapOutputTypeNames(schema)
        if schema.name == "NonMaxSuppression":
            code_gen_single_output_graph(fo,schema,(lambda x: "never"),"DataType.INT64", 1)
        elif schema.name == "QuantizeLinear":
            # NODE: the output type depends on this variable
            # On the off chance this is used then without a y_zero_point parameter then we'll fix it
            code_gen_single_output_graph(fo,schema,(lambda x: "never"),"y_zero_point.Value.dt", 1)
        else:
            code_gen_single_output_graph(fo,schema,(lambda x: "never"),"; ".join(outputs), len(outputs))

    for schemaName in [ 'LSTM',   'MaxPool', 'GRU', 'TopK', 'Dropout', 'Unique', 'DynamicQuantizeLinear', 'RNN', 'BatchNormalization']:
        schema = getSchema(schemaName)
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        outputs = mapOutputTypeNames(schema)
        code_gen_single_output_graph(fo,schema,(lambda x: "never"),"; ".join(outputs), len(outputs))

    for schemaName in ['SequenceEmpty', 'EyeLike', 'Multinomial', 'RandomUniformLike', 'RandomNormalLike', 'RandomNormal', 'RandomUniform', 'Cast']:
        schema = getSchema(schemaName)
        print(f'{schema.name}')
        schemas_out.append(schema.name)
        typeMaps = [t for t in schema.inputs[0].types if mapONNXToFSharp(t) is not None] if len(schema.inputs) == 1 else [""]
        (req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs) = get_part_inputs_and_attrs(schema)
        for (gen1,gen2) in ([("<'a>","getDataType(typeof<'a>)")] if (len(schema.inputs) == 0 or schema.name == "Cast") else [("<'a>","getDataType(typeof<'a>)"),("","input.dt")]):
            req_attrs = [x for x in req_attrs if x.name != 'dtype' and x.name != 'to']
            opt_attrs = [x for x in opt_attrs if x.name != 'dtype' and x.name != 'to']
            params = get_params_graph(req_inputs,opt_inputs,var_inputs, req_attrs, opt_attrs, typeMaps)
            params = ", " + params if params else ""
            fo.write(f'    static member {schema.name}{gen1}(graph: Graph{params}) =')
            attrProto = "; ".join([f'Attr.{mapAttrFunction(attr)}("{attr.name}", {attr.name}{mapDefaultValue(attr.default_value)})' for attr in (req_attrs + opt_attrs)])
            # NOTE We're assuming inputs are matched structually by order
            inputs = '[|' + '; '.join([f'{x.name}' for x in req_inputs]) + '|]'
            fo.write(f'\n        graph.AddNode("{schema.name}", {inputs}, [|{gen2}|], [|{attrProto}|])\n')

    fo.flush()
    fo.close()


write_graph_api(graph_path)
write_api(snake_path, snake_module, True, False)
write_api(pascal_path, pascal_module, False, False)
#write_api(curried_path, curried_module, True,True)


