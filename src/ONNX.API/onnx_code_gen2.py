# NOTE: This was run in Linux, it may be difficult to get onnx to work in Windows
# NOTE: This is only API code generation, quotations transforms will need some other object model

# TODO maybe do different type constraints

# TODO nary inputs (in progress)
#     unifying unary and binary
#     variadic
#     NOTE: The inputs appear to be structual, as in only the order matters
#     multiple type constraints

# TODO optional number of outputs?
# TODO code gen documentation

#len(conv.type_constraints)

from onnx import defs
from onnx import AttributeProto
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
from collections import defaultdict

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

# TODO filter out earlier version ops

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
unary_schemas = [schema for schema in schemas if (schema.min_input == 1) and (schema.max_input == 1) and (schema.min_output == 1) and (schema.max_output == 1)]
binary_schemas = [schema for schema in schemas if (schema.min_input == 2) and (schema.max_input == 2) and (schema.min_output == 1) and (schema.max_output == 1)]

# TODO code gen for multiple type constraints
unary_schemas1 = [x for x in unary_schemas if len(x.type_constraints) == 1] # 75
binary_schemas1 = [x for x in binary_schemas if len(x.type_constraints) == 1] # 16

#binary_schemas1[0].inputs[0].name
#binary_schemas1[0].inputs[1].name

schema = [x for x in schemas if x.name == "Softmax"][0]

conv = [x for x in schemas if x.name == "Conv"][0]

conv.inputs[0].name
conv.inputs[1].name

conv.inputs[1].option == FormalParameterOption.Optional
FormalParameterOption.Optional
FormalParameterOption.Single
FormalParameterOption.Single
FormalParameterOption.Variadic

# NOTE: Single inputs allways appear before Optional
# NOTE: Variadic inputs are at most one and are always last
# TODO: Only 'Loop' has Variadic and Optional

def anyOptionalBeforeSingle(schema):
    hasOptional = False
    for x in schema.inputs:
        if x.option == FormalParameterOption.Optional or x.option == FormalParameterOption.Variadic:
            hasOptional = True
        else:
            if x.option == FormalParameterOption.Single:
                if hasOptional:
                    return True
    return False

#set([anyOptionalBeforeSingle(schema) for x in schemas])

def countVariadic(schema):
    return len([x for x in schema.inputs if x.option == FormalParameterOption.Variadic])

def countOptional(schema):
    return len([x for x in schema.inputs if x.option == FormalParameterOption.Optional])

#def         FormalParameterOption.Variadic]

#[schema.inputs for schema in schemas]

#countby(lambda x: x,[countVariadic(schema) for schema in schemas])

countby(lambda x: x,[countVariadic(schema) for schema in schemas])

#[schema.inputs[len(schema.inputs) - 1].option == FormalParameterOption.Variadic for schema in schemas if countVariadic(schema) == 1]

variadic_schemas = [schema for schema in schemas if countVariadic(schema) == 1]

variadic_schemas[7].inputs[0].name
variadic_schemas[7].inputs[1].name
variadic_schemas[7].inputs[2].name

single_outputs = ([x for x in schemas if x.min_input > 0 and x.max_output == 1 and x.min_output == 1 and x.name != 'Loop' and len(x.type_constraints) == 1])


'NonMaxSuppression' #has no type constraint
'StringNormalizer'


#150/175 have single output

# NOTE 110/174 ops have a single output and a single type constraint
# NOTE 38/174 ops have a single output and a single type constraint

len(unary_schemas)
len(binary_schemas)


#93+28
#110+39

# NOTE Loop has multiple outputs

#single_output_schemas = 
#len(schemas)

for (x,count) in countby(lambda x: x, ["(%i,%i) -> (%i,%i)" % (x.min_input, x.max_input, x.min_output, x.max_output) for x in schemas]).items():
#variadic_schemas[7].inputs[2].option
#variadic_schemas[7].name
#variadic_schemas[7].type_constraints

[countOptional(schema) for schema in schemas if countVariadic(schema) == 1]

variadic = [x for schema in schemas for x in schema.inputs if x.option == FormalParameterOption.Variadic]

#len(variadic)

#NOTE: Presuming Variadic inputs as assumed to be last


#conv.inputs[1].option


dir(conv.inputs[2])

list(schema.attributes.items())[0]

def wrap(x,y):
    def f(z):
        return x + z + y
    return f

quote = wrap("\"","\"")
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
        "tensor(uint16)" :None,
        "tensor(uint32)" : None,
        "tensor(uint64)" : None,
        "tensor(int8)" : "int8",
        "tensor(int16)" : None,
        "tensor(int32)" : "int",
        "tensor(int64)" : "int64",
        "tensor(float16)" : None,
        "tensor(float)" : "float32",
        "tensor(double)" : None, #"float", #limiting it for now
        "tensor(string)" : "string",
        "tensor(bool)" : "bool",
        "tensor(complex64)" : None,
        "tensor(complex128)" : None,
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


#f_test = [attr.default_value for schema in schemas for (name,attr) in schema.attributes.items() if attr.default_value.type == 8][0]

mapDefaultValue(f_test)


#f_test.ints
#f', {getByteStrings(f_test.strings)}'
#b'a'.decode()

#type(b'a')

#f_test.strings[0]

#TODO get one of the string attributes with default values


#0,3,2,1,7,8

def partition(f,xs):
    return ([x for x in xs if f(x)],[x for x in xs if not f(x)])


#TODO split attributes by optional
#[f'{attr.name}: {mapAttrType(attr)}' for attr in req_attrs]

#schema.attributes
#len(unary_schemas)
#print_schema(unary_schemas[0])
#mapONNXToFSharp("tensor(uint8)")
#mapONNXToFSharp("tensor(float)")
#unary
#[len(x.type_constraints) for x in unary_schemas] 
#[len(x.type_constraints) for x in unary_schemas]
#[len(x.type_constraints) for x in binary_schemas][9]

#TODO handle seq, map, tensor
#print_schema(unary_schemas[2])
#print_schema(binary_schemas[9])

#special case SequenceAt

#len([x  for x in schemas if len(x.type_constraints) == 3])
#[x.name  for x in schemas if len(x.type_constraints) == 3]

#def getSchema(name):
#    return [x for x in schemas if x.name == name][0]

#print_schema(getSchema("QLinearConv"))
#print_schema(getSchema("MatMulInteger"))

for (x,count) in countby(lambda x: x, ["(%i,%i) -> (%i,%i)" % (x.min_input, x.max_input, x.min_output, x.max_output) for x in schemas]).items():
    print(f'{count}: {x}')

# NOTE Figuring out how to do attributes
x.attribute for x in schemas

countby(lambda x: x, [attr.default_value.type for schema in schemas for (name,attr) in schema.attributes.items()])

defaults = [attr.default_value for schema in schemas for (name,attr) in schema.attributes.items() if attr.default_value.type != 0]


#NOTE Nothing with defaults is required
set([attr.required for schema in schemas for (name,attr) in schema.attributes.items() if attr.default_value.type != 0])

[attr.name for schema in schemas for (name,attr) in schema.attributes.items() if attr.default_value.type == 0 and attr.required]

attr = [attr for schema in schemas for (name,attr) in schema.attributes.items() if attr.default_value.type == 0 and attr.required][0]

#for now filter it down to uint8, int32, int64, float32, bool, string
# [x for x in schemas if x.min_output == 2 and x.max_output == 2]

#set([attr.type for x in schemas for (_,attr) in x.attributes.items()])

# TODO examine AttrType.TENSOR, AttrType.GRAPH, and AttrType.SPARSE_TENSOR

[x.name for x in schemas if x.min_output == 1 and x.max_output == 2]
[x.name for x in schemas if x.min_output == 3 and x.max_output == 3]
[x.name for x in schemas if x.max_output > 1000]
[x.name for x in schemas if x.max_input == 1 and x.max_output > 1000]


def filterAttrType(t):
    def f(x):
        return t in set([y.type for (_,y) in x.attributes.items()])
    return f


#[x.name for x in schemas if filterAttrType(AttrType.GRAPH)(x)]
#GRAPH is If, Scan, Loop
#[x.name for x in schemas if filterAttrType(AttrType.TENSOR)(x)]
#TENSOR is ConstantOfShape, Constant
#[x.name for x in schemas if filterAttrType(AttrType.SPARSE_TENSOR)(x)]
#SPARSE_TENSOR is Constant

#[x.attributes.items() for x in  schemas]
#['If', 'Scan', 'Loop', 'Split']

# -> (0,2) "GRU","RNN"
# -> (0,3) "LSTM"

#print_schema(getSchema('LSTM'))
#print_schema(getSchema('GRU'))
# (5,5) -> (1,5)
#print_schema(getSchema('BatchNormalization'))

# TODO figure out how optional inputs result in optional outputs
# AFAIK Optional outputs are always available but don't have to be used

# TODO, how do we deal with Optional outputs such as LSTM, it seems that it depends on the optional inputs. May need to special case it.


# NOTE: These are uncommon
#[x.name for x in schemas if x.min_output == 2 and x.max_output == 2]
#['TreeEnsembleClassifier', 'LinearClassifier', 'SVMClassifier', 'TopK']

####################################################################################################
#                     Code Gen
####################################################################################################

fo = open("/mnt/c/EE/Git/ONNXBackend/ONNXBackend/ONNXAPI.g.fs","w")
fo.write("module ONNXAPI\n")
fo.write("\n")
fo.write("open System\n")
fo.write("open System.IO\n")
fo.write("open System.Text\n")
fo.write("open Onnx\n")
fo.write("open Google.Protobuf.Collections\n")
fo.write("open Microsoft.ML.OnnxRuntime.Tensors\n")
fo.write("open Microsoft.ML.OnnxRuntime\n")
fo.write("open ProtoBuf\n")
fo.write("\n")
fo.write("type ONNX() =\n")

def optionalParam(x):
    return ', ' + x if x else ''

def optionalChoose(x):
    return f'([|{x}|] |> Array.choose id)' if x else '[||]'

def inputParamString(x,typeMap):
    t = f'Tensor<{typeMap(x.typeStr)}>' 
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

# NOTE: All required appear before optional
# NOTE: Variadic appear last and without optional (unless it's a 'Loop' op)


def typeMap(x): 
    return "float"

#        params = ", ".join([f'{"" if req else "?"}{attr.name}: {mapAttrType(attr)}' for (req,attr) in attrs])

", ".join(
    [inputParamString(x, typeMap) for x in req_inputs] + 
    [f'{x.name}: {mapAttrType(x)}' for x in req_attrs] + 
    [inputParamString(x, typeMap) for x in opt_inputs] + 
    [f'?{x.name}: {mapAttrType(x)}' for x in opt_attrs] + 
    [inputParamString(x, typeMap) for x in var_inputs]
    )

schema = conv

(req_inputs,opt_inputs,var_inputs) = partitionInputs(schema.inputs)
(req_attrs, opt_attrs) = partition(lambda x: x.required, [x for (_,x) in schema.attributes.items()])
# NOTE: Req inputs, Req attr, Opt inputs, Opt attr, Var inputs
param_string = ""

#params = ", ".join([f'{"" if req else "?"}{attr.name}: {mapAttrType(attr)}' for (req,attr) in attrs])



#parameterString(schemas[10],(lambda x: "x"))

#countVariadic(schema)
#countOptional(schema)


def code_gen_single_output(fo,schema):
    for t in choseFSharpTypes(schema.type_constraints[0]):
        req_attrs, opt_attrs = partition(lambda x: x.required, [x for (_,x) in schema.attributes.items()])
        #attrs = [(True,x) for x in req_attrs] + [(False,x) for x in opt_attrs]
        fo.write(f'    static member {schema.name}({input_name}: Tensor<{t}>{optionalParam(params)}) =')
        attrProto = "; ".join([f'Attr.{mapAttrFunction(attr)}("{attr.name}", {attr.name}{mapDefaultValue(attr.default_value)})' for (req,attr) in attrs])
        fo.write(f'        execNode "{schema.name}" {input_name} {optionalChoose(attrProto)}')
        fo.write('\n')

#def code_gen_unary(fo,schema):
#    for t in choseFSharpTypes(schema.type_constraints[0]):
#        input_name = schema.inputs[0].name
#        req_attrs, opt_attrs = partition(lambda x: x.required, [x for (_,x) in schema.attributes.items()])
#        attrs = [(True,x) for x in req_attrs] + [(False,x) for x in opt_attrs]
#        params = ", ".join([f'{"" if req else "?"}{attr.name}: {mapAttrType(attr)}' for (req,attr) in attrs])
#        fo.write(f'    static member {schema.name}({input_name}: Tensor<{t}>{optionalParam(params)}) =')
#        attrProto = "; ".join([f'Attr.{mapAttrFunction(attr)}("{attr.name}", {attr.name}{mapDefaultValue(attr.default_value)})' for (req,attr) in attrs])
#        fo.write(f'        buildAndRunUnary "{schema.name}" {input_name} {optionalChoose(attrProto)}')
#        fo.write('\n')
#
#def code_gen_binary(fo,schema):
#    for t in choseFSharpTypes(schema.type_constraints[0]):
#        input_nameA = schema.inputs[0].name
#        input_nameB = schema.inputs[1].name
#        req_attrs, opt_attrs = partition(lambda x: x.required, [x for (_,x) in schema.attributes.items()])
#        attrs = [(True,x) for x in req_attrs] + [(False,x) for x in opt_attrs]
#        params = ", ".join([f'{"" if req else "?"}{attr.name}: {mapAttrType(attr)}' for (req,attr) in attrs])
#        fo.write(f'    static member {schema.name}({input_nameA}: Tensor<{t}>, {input_nameB}: Tensor<{t}>{optionalParam(params)}) =')
#        fo.write('\n')
#        attrProto = "; ".join([f'Attr.{mapAttrFunction(attr)}("{attr.name}", {attr.name}{mapDefaultValue(attr.default_value)})' for (req,attr) in attrs])
#        fo.write(f'        buildAndRunBinary "{schema.name}" {input_nameA} {input_nameB} ([|{attrProto}|] |> Array.choose id)')
#        fo.write('\n')

for schema in unary_schemas1:
    code_gen_unary(fo,schema)

for schema in binary_schemas1:
    code_gen_binary(fo,schema)

fo.flush()
fo.close()
#import time
#import numpy as np
#input1 = np.ones((10000000,40),np.float32) * -2.0
#input2 = np.ones((40,10000000),np.float32) * -2.0
#
#start = time.time()
#t = np.matmul(input2,input1) 
#end = time.time()
#end-start
#
#end
#start


####################################################################################################
#                     Default Experiments
####################################################################################################

#defaults = [attr.default_value for x in schemas for (_,attr) in x.attributes.items() if attr.default_value.type != 0]
#len(defaults)
