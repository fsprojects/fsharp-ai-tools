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
# TODO handle seq, map, tensor
# TODO special case SequenceAt
# NOTE All required appear before optional
# NOTE Variadic appear last and without optional (unless it's a 'Loop' op)
# TODO handle AttrType.TENSOR and AttrType.SPARSE_TENSOR for 'Constant' node


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

def hasType(z,schema):
    return len([tc for tc in schema.type_constraints if len([y for y in tc.allowed_type_strs if z in y]) > 0]) > 0


todo_schemas = (['ConstantOfShape','Constant'] + # Attribute TENSOR or SPARSE_TENSOR #[x.name for x in schemas if len([attr for (_,attr) in x.attributes.items() if (attr.type == AttrType.TENSOR or attr.type == AttrType.SPARSE_TENSOR)]) > 0]
                ['Loop'] +  # Combined Optional and Variadic inputs
                ['SequenceEmpty', 'EyeLike', 'Multinomial', 'RandomUniformLike', 'RandomNormalLike', 'RandomNormal', 'RandomUniform'] + # 'dtype' in the attributes
                ['Cast'] + # 'to' attribute
                ['ConcatFromSequence', 'SplitToSequence', 'SequenceErase', 'SequenceAt', 'SequenceInsert', 'SequenceConstruct', 'SequenceEmpty', 'SequenceLength'] + #sequences [x.name for x in schemas if hasType("seq",x)]
                ['DictVectorizer', 'ZipMap', 'CastMap']) #maps [x.name for x in schemas if hasType("map",x)]

filtered = set(todo_schemas)

schemas1 = list(filter(lambda x: x.name not in filtered, schemas))

def getSchema(name):
    return [x for x in schemas if x.name == name][0]

# Foldable type constraints; all types are singular and equal
#[x.name for x in schemas1 if len(set([t for tc in x.type_constraints for t in tc.allowed_type_strs])) == 1]
# NOTE it turns out only bool are foldable

single_output_schemas = [x for x in schemas1 if x.max_output == 1 and x.min_output == 1]
single_outputs_single_type = ([x for x in single_output_schemas if len(x.type_constraints) == 1])
single_outputs_single_output_type = ([x for x in single_output_schemas if len(x.type_constraints) == 2 and len([tc.allowed_type_strs for tc in x.type_constraints if tc.type_param_str == x.outputs[0].typeStr][0]) == 1])

def mapNames(xs):
    return [x.name for x in xs]

filtered2 = set(mapNames(single_outputs_single_type) + mapNames(single_outputs_single_output_type))

schemas2 = [x for x in schemas1 if x.name not in filtered2]

len(schemas2)

len([x for x in schemas if (x.max_output != 1 or x.min_output != 1)])

#multi-output is 16

#len(single_outputs_single_type) + len(single_outputs_single_output_type)

len(set(mapNames(schemas)))

single_outputs_multi_type = [x for x in single_output_schemas if x.max_output == 1 and x.min_output == 1 and len(x.type_constraints) > 1]

#check to make sure output shares a typeStr with at least one input

len(([x.name for x in single_outputs_multi_type if x.max_output == 1 and x.min_output == 1 and len(x.type_constraints) > 1 and x.outputs[0].typeStr in [y.typeStr for y in x.inputs]]))
len(([x.name for x in single_outputs_multi_type if x.max_output == 1 and x.min_output == 1 and len(x.type_constraints) > 1 and x.outputs[0].typeStr not in [y.typeStr for y in x.inputs]]))


done_set = set(todo_schemas + mapNames(single_outputs_single_type) + mapNames(single_outputs_single_output_type))

len([x for x in mapNames(schemas) if x not in done_set])
#36... different outputs...

print_schema(getSchema('Slice'))

#Mixed input type constraints

len(getSchema('Slice').type_constraints)

#[x.name for x in schemas1 if len(x.type_constraints) == 2 and x.type_constraints[0].allowed_type_strs == x.type_constraints[1].allowed_type_strs][0]
#[x.name for x in schemas1 if len(x.type_constraints) == 2 and x.type_constraints[0].allowed_type_strs == x.type_constraints[1].allowed_type_strs][0]


#[print_schema(x) for x in single_outputs2]

and_schema = [x for x in schemas if x.name == "And"][0]

print_schema(and_schema)

and_schema.inputs

and_schema.inputs[0].typeStr
#and_schema.inputs[1].typeStr
#and_schema.outputs[0].typeStr

[(tc.type_param_str,tc.allowed_type_strs) for tc in and_schema.type_constraints]

#[x for x in schemas if x.name == "And"][0].outputs

dir(conv.type_constraints[0].allowed_type_strs)
dir(conv.type_constraints[0].type_param_str)

tcs[11].type_constraints[0].allowed_type_strs == tcs[11].type_constraints[1].allowed_type_strs

conv.type_constraints

#[x.name for x in schemas if len(x.type_constraints) == 2 and ("seq(tensor(float))" in x.type_constraints[0].allowed_type_strs or "seq(tensor(float))" in x.type_constraints[1].allowed_type_strs ) ]
#[x.name for x in schemas if len(x.type_constraints) == 2 and ("seq(tensor(float))" in x.type_constraints[0].allowed_type_strs or "seq(tensor(float))" in x.type_constraints[1].allowed_type_strs ) ]

#TODO support seq(tensor(..)) this accounts for 4 ops

#TODO single output

[(set([y.typeStr for y in x.inputs]),set([y.typeStr for y in x.outputs])) for x in single_outputs2]

# Filter Type Constraints which have a different

# Filter Type Constraints which have an input

(schema1,tcs) = [(x,x.type_constraints)  for x in single_outputs2 if len(set([y.typeStr for y in x.inputs])) == 1][4]
(schema1,tcs) = [(x,x.type_constraints)  for x in single_outputs2 if len(set([y.typeStr for y in x.inputs])) == 1][1]

single_outputs4 = ([x for x in schemas if (x.max_output != 1 or x.min_output != 1) ])

len(single_outputs4 )

single_outputs4[1].min_output
single_outputs4[1].max_output
[x.name for x in single_outputs4[1].outputs]

#155/174

# NOTE 110/174 ops have a single output and a single type constraint
# NOTE 38/174 ops have a single output and a single type constraint
# NOTE Loop has multiple outputs


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

def partition(f,xs):
    return ([x for x in xs if f(x)],[x for x in xs if not f(x)])

for (x,count) in countby(lambda x: x, ["(%i,%i) -> (%i,%i)" % (x.min_input, x.max_input, x.min_output, x.max_output) for x in schemas]).items():
    print(f'{count}: {x}')

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

# NOTE: Req inputs, Req attr, Opt inputs, Opt attr, Var inputs

def code_gen_single_output(fo,schema,inputType,outputType):
    (req_inputs,opt_inputs,var_inputs) = partitionInputs(schema.inputs)
    (req_attrs, opt_attrs) = partition(lambda x: x.required, [x for (_,x) in schema.attributes.items()])
    def typeMap(x): 
        return inputType
    params = ", ".join(
        [inputParamString(x, typeMap) for x in req_inputs] + 
        [f'{x.name}: {mapAttrType(x)}' for x in req_attrs] + 
        [inputParamString(x, typeMap) for x in var_inputs] +
        [inputParamString(x, typeMap) for x in opt_inputs] + 
        [f'?{x.name}: {mapAttrType(x)}' for x in opt_attrs])
    fo.write(f'    static member {schema.name}({params}) =')
    attrProto = "; ".join([f'Attr.{mapAttrFunction(attr)}("{attr.name}", {attr.name}{mapDefaultValue(attr.default_value)})' for attr in (req_attrs + opt_attrs)])
    inputs = ""
    # NOTE We're assuming inputs are matched structually by order
    if len(opt_inputs) == 0 and len(var_inputs) == 0: #only req_inputs
        inputs = '[|' + '; '.join([x.name for x in req_inputs]) + '|]'
    elif len(opt_inputs) != 0 and len(var_inputs) == 0:
        inputs = '([|' + '; '.join([f'Some({x.name})' for x in req_inputs] + [x.name for x in opt_inputs]) + '|] |> Array.choose id)'
    elif len(opt_inputs) == 0 and len(var_inputs) != 0:
        if len(req_inputs) == 0: inputs = f'{var_inputs[0].name}'
        else: inputs = '([|' + '; '.join([f'yield {x.name}' for x in req_inputs] + [f'yield! {x.name}' for x in var_inputs]) + '|])'
    elif len(opt_inputs) != 0 and len(var_inputs) != 0:
        raise Exception("ops with both optional and variadic inputs are not yet supported")
    else:
        raise Exception("shouldn't happen")
    fo.write('\n')
    fo.write(f'        execNode<{inputType},{outputType}> "{schema.name}" {inputs} {optionalChoose(attrProto)}')
    fo.write('\n')

# single type constraints
for schema in single_outputs_single_type:
    print(f'{schema.name}')
    for t in choseFSharpTypes(schema.type_constraints[0]):
        code_gen_single_output(fo,schema,t,t)

# two type constraints one for input one for output, output only has one type
for schema in single_outputs_single_output_type:
    print(f'{schema.name}')
    output_type = [tc.allowed_type_strs for tc in schema.type_constraints if tc.type_param_str == schema.outputs[0].typeStr][0][0]
    #TODO check that inputs all have the same typeStr...
    input_types = [tc.allowed_type_strs for tc in schema.type_constraints if tc.type_param_str == schema.inputs[0].typeStr][0]
    for t in [mapONNXToFSharp(x) for x in input_types if mapONNXToFSharp(x)]:
        code_gen_single_output(fo,schema,t,mapONNXToFSharp(output_type))


fo.flush()
fo.close()

#print_schema(getSchema('Or'))

####################################################################################################
#                     Default Experiments
####################################################################################################

#defaults = [attr.default_value for x in schemas for (_,attr) in x.attributes.items() if attr.default_value.type != 0]
#len(defaults)

####################################################################################################
#                     NOTES
####################################################################################################

#conv.inputs[0].name
#conv.inputs[1].name
#conv.inputs[1].option == FormalParameterOption.Optional
#FormalParameterOption.Single
#FormalParameterOption.Optional
#FormalParameterOption.Variadic

# NOTE: Single inputs allways appear before Optional
# NOTE: Variadic inputs are at most one and are always last
# TODO: Only 'Loop' has Variadic and Optional

#def anyOptionalBeforeSingle(schema):
#    hasOptional = False
#    for x in schema.inputs:
#        if x.option == FormalParameterOption.Optional or x.option == FormalParameterOption.Variadic:
#            hasOptional = True
#        else:
#            if x.option == FormalParameterOption.Single:
#                if hasOptional:
#                    return True
#    return False
#
##set([anyOptionalBeforeSingle(schema) for x in schemas])
#
#def         FormalParameterOption.Variadic]

#[schema.inputs for schema in schemas]

#countby(lambda x: x,[countVariadic(schema) for schema in schemas])

#countby(lambda x: x,[countVariadic(schema) for schema in schemas])

#[schema.inputs[len(schema.inputs) - 1].option == FormalParameterOption.Variadic for schema in schemas if countVariadic(schema) == 1]

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

#'NonMaxSuppression' #has no type constraint
#'StringNormalizer'




#def countVariadic(schema):
#    return len([x for x in schema.inputs if x.option == FormalParameterOption.Variadic])
#
#def countOptional(schema):
#    return len([x for x in schema.inputs if x.option == FormalParameterOption.Optional])
#
#variadic_schemas = [schema for schema in schemas if countVariadic(schema) == 1]

