module FSharp.ML.Onnx.API.Graph

open System
open FSharp.ML.Onnx.Protobuf

type OnnxGraph() =
    static member NonMaxSuppression(graph: Graph, boxes: ValueInfo, scores: ValueInfo, ?max_output_boxes_per_class: ValueInfo, ?iou_threshold: ValueInfo, ?score_threshold: ValueInfo, ?center_point_box: int64) =
        graph.AddNode("NonMaxSuppression", ([|Some(boxes); Some(scores); max_output_boxes_per_class; iou_threshold; score_threshold|] |> Array.choose id), [|DataType.INT64|], [|Attr.int("center_point_box", center_point_box, 0L)|]) |> toTuple1
    static member StringNormalizer(graph: Graph, X: ValueInfo, ?case_change_action: string, ?is_case_sensitive: int64, ?locale: string, ?stopwords: string[]) =
        graph.AddNode("StringNormalizer", [|X|], [|X.dt|], [|Attr.string("case_change_action", case_change_action, "NONE"); Attr.int("is_case_sensitive", is_case_sensitive, 0L); Attr.string("locale", locale); Attr.strings("stopwords", stopwords)|]) |> toTuple1
    static member LinearRegressor(graph: Graph, X: ValueInfo, ?coefficients: float32[], ?intercepts: float32[], ?post_transform: string, ?targets: int64) =
        graph.AddNode("LinearRegressor", [|X|], [|DataType.FLOAT32|], [|Attr.floats("coefficients", coefficients); Attr.floats("intercepts", intercepts); Attr.string("post_transform", post_transform, "NONE"); Attr.int("targets", targets, 1L)|]) |> toTuple1
    static member Imputer(graph: Graph, X: ValueInfo, ?imputed_value_floats: float32[], ?imputed_value_int64s: int64[], ?replaced_value_float: float32, ?replaced_value_int64: int64) =
        graph.AddNode("Imputer", [|X|], [|X.dt|], [|Attr.floats("imputed_value_floats", imputed_value_floats); Attr.ints("imputed_value_int64s", imputed_value_int64s); Attr.float("replaced_value_float", replaced_value_float, 0.0f); Attr.int("replaced_value_int64", replaced_value_int64, 0L)|]) |> toTuple1
    static member FeatureVectorizer(graph: Graph, [<ParamArray>]X: ValueInfo[], ?inputdimensions: int64[]) =
        graph.AddNode("FeatureVectorizer", (X), [|DataType.FLOAT32|], [|Attr.ints("inputdimensions", inputdimensions)|]) |> toTuple1
    static member Binarizer(graph: Graph, X: ValueInfo, ?threshold: float32) =
        graph.AddNode("Binarizer", [|X|], [|X.dt|], [|Attr.float("threshold", threshold, 0.0f)|]) |> toTuple1
    static member ArrayFeatureExtractor(graph: Graph, X: ValueInfo, Y: ValueInfo) =
        graph.AddNode("ArrayFeatureExtractor", [|X; Y|], [|X.dt|], [||]) |> toTuple1
    static member SVMRegressor(graph: Graph, X: ValueInfo, ?coefficients: float32[], ?kernel_params: float32[], ?kernel_type: string, ?n_supports: int64, ?one_class: int64, ?post_transform: string, ?rho: float32[], ?support_vectors: float32[]) =
        graph.AddNode("SVMRegressor", [|X|], [|DataType.FLOAT32|], [|Attr.floats("coefficients", coefficients); Attr.floats("kernel_params", kernel_params); Attr.string("kernel_type", kernel_type, "LINEAR"); Attr.int("n_supports", n_supports, 0L); Attr.int("one_class", one_class, 0L); Attr.string("post_transform", post_transform, "NONE"); Attr.floats("rho", rho); Attr.floats("support_vectors", support_vectors)|]) |> toTuple1
    static member Det(graph: Graph, X: ValueInfo) =
        graph.AddNode("Det", [|X|], [|X.dt|], [||]) |> toTuple1
    static member TreeEnsembleRegressor(graph: Graph, X: ValueInfo, ?aggregate_function: string, ?base_values: float32[], ?n_targets: int64, ?nodes_falsenodeids: int64[], ?nodes_featureids: int64[], ?nodes_hitrates: float32[], ?nodes_missing_value_tracks_true: int64[], ?nodes_modes: string[], ?nodes_nodeids: int64[], ?nodes_treeids: int64[], ?nodes_truenodeids: int64[], ?nodes_values: float32[], ?post_transform: string, ?target_ids: int64[], ?target_nodeids: int64[], ?target_treeids: int64[], ?target_weights: float32[]) =
        graph.AddNode("TreeEnsembleRegressor", [|X|], [|DataType.FLOAT32|], [|Attr.string("aggregate_function", aggregate_function, "SUM"); Attr.floats("base_values", base_values); Attr.int("n_targets", n_targets); Attr.ints("nodes_falsenodeids", nodes_falsenodeids); Attr.ints("nodes_featureids", nodes_featureids); Attr.floats("nodes_hitrates", nodes_hitrates); Attr.ints("nodes_missing_value_tracks_true", nodes_missing_value_tracks_true); Attr.strings("nodes_modes", nodes_modes); Attr.ints("nodes_nodeids", nodes_nodeids); Attr.ints("nodes_treeids", nodes_treeids); Attr.ints("nodes_truenodeids", nodes_truenodeids); Attr.floats("nodes_values", nodes_values); Attr.string("post_transform", post_transform, "NONE"); Attr.ints("target_ids", target_ids); Attr.ints("target_nodeids", target_nodeids); Attr.ints("target_treeids", target_treeids); Attr.floats("target_weights", target_weights)|]) |> toTuple1
    static member Round(graph: Graph, X: ValueInfo) =
        graph.AddNode("Round", [|X|], [|X.dt|], [||]) |> toTuple1
    static member Range(graph: Graph, start: ValueInfo, limit: ValueInfo, delta: ValueInfo) =
        graph.AddNode("Range", [|start; limit; delta|], [|start.dt|], [||]) |> toTuple1
    static member ThresholdedRelu(graph: Graph, X: ValueInfo, ?alpha: float32) =
        graph.AddNode("ThresholdedRelu", [|X|], [|X.dt|], [|Attr.float("alpha", alpha, 1.0f)|]) |> toTuple1
    static member MeanVarianceNormalization(graph: Graph, X: ValueInfo, ?axes: int64[]) =
        graph.AddNode("MeanVarianceNormalization", [|X|], [|X.dt|], [|Attr.ints("axes", axes, [|0L;2L;3L|])|]) |> toTuple1
    static member NonZero(graph: Graph, X: ValueInfo) =
        graph.AddNode("NonZero", [|X|], [|DataType.INT64|], [||]) |> toTuple1
    static member Shrink(graph: Graph, input: ValueInfo, ?bias: float32, ?lambd: float32) =
        graph.AddNode("Shrink", [|input|], [|input.dt|], [|Attr.float("bias", bias, 0.0f); Attr.float("lambd", lambd, 0.5f)|]) |> toTuple1
    static member Erf(graph: Graph, input: ValueInfo) =
        graph.AddNode("Erf", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Atanh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Atanh", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Acosh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Acosh", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Expand(graph: Graph, input: ValueInfo, shape: ValueInfo) =
        graph.AddNode("Expand", [|input; shape|], [|input.dt|], [||]) |> toTuple1
    static member Atan(graph: Graph, input: ValueInfo) =
        graph.AddNode("Atan", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Asin(graph: Graph, input: ValueInfo) =
        graph.AddNode("Asin", [|input|], [|input.dt|], [||]) |> toTuple1
    static member LpNormalization(graph: Graph, input: ValueInfo, ?axis: int64, ?p: int64) =
        graph.AddNode("LpNormalization", [|input|], [|input.dt|], [|Attr.int("axis", axis, -1L); Attr.int("p", p, 2L)|]) |> toTuple1
    static member Ceil(graph: Graph, X: ValueInfo) =
        graph.AddNode("Ceil", [|X|], [|X.dt|], [||]) |> toTuple1
    static member LogSoftmax(graph: Graph, input: ValueInfo, ?axis: int64) =
        graph.AddNode("LogSoftmax", [|input|], [|input.dt|], [|Attr.int("axis", axis, 1L)|]) |> toTuple1
    static member MatMul(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("MatMul", [|A; B|], [|A.dt|], [||]) |> toTuple1
    static member BitShift(graph: Graph, X: ValueInfo, Y: ValueInfo, direction: string) =
        graph.AddNode("BitShift", [|X; Y|], [|X.dt|], [|Attr.string("direction", direction)|]) |> toTuple1
    static member Sinh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Sinh", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Acos(graph: Graph, input: ValueInfo) =
        graph.AddNode("Acos", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Identity(graph: Graph, input: ValueInfo) =
        graph.AddNode("Identity", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Pow(graph: Graph, X: ValueInfo, Y: ValueInfo) =
        graph.AddNode("Pow", [|X; Y|], [|X.dt|], [||]) |> toTuple1
    static member Mod(graph: Graph, A: ValueInfo, B: ValueInfo, ?fmod: int64) =
        graph.AddNode("Mod", [|A; B|], [|A.dt|], [|Attr.int("fmod", fmod, 0L)|]) |> toTuple1
    static member Softplus(graph: Graph, X: ValueInfo) =
        graph.AddNode("Softplus", [|X|], [|X.dt|], [||]) |> toTuple1
    static member Normalizer(graph: Graph, X: ValueInfo, ?norm: string) =
        graph.AddNode("Normalizer", [|X|], [|DataType.FLOAT32|], [|Attr.string("norm", norm, "MAX")|]) |> toTuple1
    static member Hardmax(graph: Graph, input: ValueInfo, ?axis: int64) =
        graph.AddNode("Hardmax", [|input|], [|input.dt|], [|Attr.int("axis", axis, 1L)|]) |> toTuple1
    static member HardSigmoid(graph: Graph, X: ValueInfo, ?alpha: float32, ?beta: float32) =
        graph.AddNode("HardSigmoid", [|X|], [|X.dt|], [|Attr.float("alpha", alpha, 0.20000000298023224f); Attr.float("beta", beta, 0.5f)|]) |> toTuple1
    static member LpPool(graph: Graph, X: ValueInfo, kernel_shape: int64[], ?auto_pad: string, ?p: int64, ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("LpPool", [|X|], [|X.dt|], [|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("p", p, 2L); Attr.ints("pads", pads); Attr.ints("strides", strides)|]) |> toTuple1
    static member Min(graph: Graph, [<ParamArray>]data_0: ValueInfo[]) =
        graph.AddNode("Min", (data_0), [|data_0.[0].dt|], [||]) |> toTuple1
    static member Sum(graph: Graph, [<ParamArray>]data_0: ValueInfo[]) =
        graph.AddNode("Sum", (data_0), [|data_0.[0].dt|], [||]) |> toTuple1
    static member Transpose(graph: Graph, data: ValueInfo, ?perm: int64[]) =
        graph.AddNode("Transpose", [|data|], [|data.dt|], [|Attr.ints("perm", perm)|]) |> toTuple1
    static member ScatterND(graph: Graph, data: ValueInfo, indices: ValueInfo, updates: ValueInfo) =
        graph.AddNode("ScatterND", [|data; indices; updates|], [|data.dt|], [||]) |> toTuple1
    static member GlobalLpPool(graph: Graph, X: ValueInfo, ?p: int64) =
        graph.AddNode("GlobalLpPool", [|X|], [|X.dt|], [|Attr.int("p", p, 2L)|]) |> toTuple1
    static member Gemm(graph: Graph, A: ValueInfo, B: ValueInfo, ?C: ValueInfo, ?alpha: float32, ?beta: float32, ?transA: int64, ?transB: int64) =
        graph.AddNode("Gemm", ([|Some(A); Some(B); C|] |> Array.choose id), [|A.dt|], [|Attr.float("alpha", alpha, 1.0f); Attr.float("beta", beta, 1.0f); Attr.int("transA", transA, 0L); Attr.int("transB", transB, 0L)|]) |> toTuple1
    static member InstanceNormalization(graph: Graph, input: ValueInfo, scale: ValueInfo, B: ValueInfo, ?epsilon: float32) =
        graph.AddNode("InstanceNormalization", [|input; scale; B|], [|input.dt|], [|Attr.float("epsilon", epsilon, 9.999999747378752e-06f)|]) |> toTuple1
    static member AveragePool(graph: Graph, X: ValueInfo, kernel_shape: int64[], ?auto_pad: string, ?count_include_pad: int64, ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("AveragePool", [|X|], [|X.dt|], [|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.int("count_include_pad", count_include_pad, 0L); Attr.ints("pads", pads); Attr.ints("strides", strides)|]) |> toTuple1
    static member Sign(graph: Graph, input: ValueInfo) =
        graph.AddNode("Sign", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Clip(graph: Graph, input: ValueInfo, ?min: ValueInfo, ?max: ValueInfo) =
        graph.AddNode("Clip", ([|Some(input); min; max|] |> Array.choose id), [|input.dt|], [||]) |> toTuple1
    static member DequantizeLinear(graph: Graph, x: ValueInfo, x_scale: ValueInfo, ?x_zero_point: ValueInfo) =
        graph.AddNode("DequantizeLinear", ([|Some(x); Some(x_scale); x_zero_point|] |> Array.choose id), [|x_scale.dt|], [||]) |> toTuple1
    static member LRN(graph: Graph, X: ValueInfo, size: int64, ?alpha: float32, ?beta: float32, ?bias: float32) =
        graph.AddNode("LRN", [|X|], [|X.dt|], [|Attr.int("size", size); Attr.float("alpha", alpha, 9.999999747378752e-05f); Attr.float("beta", beta, 0.75f); Attr.float("bias", bias, 1.0f)|]) |> toTuple1
    static member Elu(graph: Graph, X: ValueInfo, ?alpha: float32) =
        graph.AddNode("Elu", [|X|], [|X.dt|], [|Attr.float("alpha", alpha, 1.0f)|]) |> toTuple1
    static member Sin(graph: Graph, input: ValueInfo) =
        graph.AddNode("Sin", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Pad(graph: Graph, data: ValueInfo, pads: ValueInfo, ?constant_value: ValueInfo, ?mode: string) =
        graph.AddNode("Pad", ([|Some(data); Some(pads); constant_value|] |> Array.choose id), [|data.dt|], [|Attr.string("mode", mode, "constant")|]) |> toTuple1
    static member GatherND(graph: Graph, data: ValueInfo, indices: ValueInfo) =
        graph.AddNode("GatherND", [|data; indices|], [|data.dt|], [||]) |> toTuple1
    static member Relu(graph: Graph, X: ValueInfo) =
        graph.AddNode("Relu", [|X|], [|X.dt|], [||]) |> toTuple1
    static member Conv(graph: Graph, X: ValueInfo, W: ValueInfo, ?B: ValueInfo, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("Conv", ([|Some(X); Some(W); B|] |> Array.choose id), [|X.dt|], [|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|]) |> toTuple1
    static member ArgMax(graph: Graph, data: ValueInfo, ?axis: int64, ?keepdims: int64) =
        graph.AddNode("ArgMax", [|data|], [|DataType.INT64|], [|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member Div(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Div", [|A; B|], [|A.dt|], [||]) |> toTuple1
    static member MaxRoiPool(graph: Graph, X: ValueInfo, rois: ValueInfo, pooled_shape: int64[], ?spatial_scale: float32) =
        graph.AddNode("MaxRoiPool", [|X; rois|], [|X.dt|], [|Attr.ints("pooled_shape", pooled_shape); Attr.float("spatial_scale", spatial_scale, 1.0f)|]) |> toTuple1
    static member Add(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Add", [|A; B|], [|A.dt|], [||]) |> toTuple1
    static member LeakyRelu(graph: Graph, X: ValueInfo, ?alpha: float32) =
        graph.AddNode("LeakyRelu", [|X|], [|X.dt|], [|Attr.float("alpha", alpha, 0.009999999776482582f)|]) |> toTuple1
    static member ReduceLogSum(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceLogSum", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member Floor(graph: Graph, X: ValueInfo) =
        graph.AddNode("Floor", [|X|], [|X.dt|], [||]) |> toTuple1
    static member ArgMin(graph: Graph, data: ValueInfo, ?axis: int64, ?keepdims: int64) =
        graph.AddNode("ArgMin", [|data|], [|DataType.INT64|], [|Attr.int("axis", axis, 0L); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member DepthToSpace(graph: Graph, input: ValueInfo, blocksize: int64, ?mode: string) =
        graph.AddNode("DepthToSpace", [|input|], [|input.dt|], [|Attr.int("blocksize", blocksize); Attr.string("mode", mode, "DCR")|]) |> toTuple1
    static member Tan(graph: Graph, input: ValueInfo) =
        graph.AddNode("Tan", [|input|], [|input.dt|], [||]) |> toTuple1
    static member ReduceSum(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceSum", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member Concat(graph: Graph, axis: int64, [<ParamArray>]inputs: ValueInfo[]) =
        graph.AddNode("Concat", (inputs), [|inputs.[0].dt|], [|Attr.int("axis", axis)|]) |> toTuple1
    static member OneHotEncoder(graph: Graph, X: ValueInfo, ?cats_int64s: int64[], ?cats_strings: string[], ?zeros: int64) =
        graph.AddNode("OneHotEncoder", [|X|], [|DataType.FLOAT32|], [|Attr.ints("cats_int64s", cats_int64s); Attr.strings("cats_strings", cats_strings); Attr.int("zeros", zeros, 1L)|]) |> toTuple1
    static member ConvTranspose(graph: Graph, X: ValueInfo, W: ValueInfo, ?B: ValueInfo, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?output_padding: int64[], ?output_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("ConvTranspose", ([|Some(X); Some(W); B|] |> Array.choose id), [|X.dt|], [|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("output_padding", output_padding); Attr.ints("output_shape", output_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|]) |> toTuple1
    static member ReverseSequence(graph: Graph, input: ValueInfo, sequence_lens: ValueInfo, ?batch_axis: int64, ?time_axis: int64) =
        graph.AddNode("ReverseSequence", [|input; sequence_lens|], [|input.dt|], [|Attr.int("batch_axis", batch_axis, 1L); Attr.int("time_axis", time_axis, 0L)|]) |> toTuple1
    static member Max(graph: Graph, [<ParamArray>]data_0: ValueInfo[]) =
        graph.AddNode("Max", (data_0), [|data_0.[0].dt|], [||]) |> toTuple1
    static member GlobalMaxPool(graph: Graph, X: ValueInfo) =
        graph.AddNode("GlobalMaxPool", [|X|], [|X.dt|], [||]) |> toTuple1
    static member Exp(graph: Graph, input: ValueInfo) =
        graph.AddNode("Exp", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Reshape(graph: Graph, data: ValueInfo, shape: ValueInfo) =
        graph.AddNode("Reshape", [|data; shape|], [|data.dt|], [||]) |> toTuple1
    static member GlobalAveragePool(graph: Graph, X: ValueInfo) =
        graph.AddNode("GlobalAveragePool", [|X|], [|X.dt|], [||]) |> toTuple1
    static member Mean(graph: Graph, [<ParamArray>]data_0: ValueInfo[]) =
        graph.AddNode("Mean", (data_0), [|data_0.[0].dt|], [||]) |> toTuple1
    static member Mul(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Mul", [|A; B|], [|A.dt|], [||]) |> toTuple1
    static member Neg(graph: Graph, X: ValueInfo) =
        graph.AddNode("Neg", [|X|], [|X.dt|], [||]) |> toTuple1
    static member Not(graph: Graph, X: ValueInfo) =
        graph.AddNode("Not", [|X|], [|X.dt|], [||]) |> toTuple1
    static member ReduceL1(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceL1", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member Flatten(graph: Graph, input: ValueInfo, ?axis: int64) =
        graph.AddNode("Flatten", [|input|], [|input.dt|], [|Attr.int("axis", axis, 1L)|]) |> toTuple1
    static member PRelu(graph: Graph, X: ValueInfo, slope: ValueInfo) =
        graph.AddNode("PRelu", [|X; slope|], [|X.dt|], [||]) |> toTuple1
    static member Unsqueeze(graph: Graph, data: ValueInfo, axes: int64[]) =
        graph.AddNode("Unsqueeze", [|data|], [|data.dt|], [|Attr.ints("axes", axes)|]) |> toTuple1
    static member Tanh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Tanh", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Abs(graph: Graph, X: ValueInfo) =
        graph.AddNode("Abs", [|X|], [|X.dt|], [||]) |> toTuple1
    static member Reciprocal(graph: Graph, X: ValueInfo) =
        graph.AddNode("Reciprocal", [|X|], [|X.dt|], [||]) |> toTuple1
    static member ReduceLogSumExp(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceLogSumExp", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member ReduceMax(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceMax", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member ReduceMean(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceMean", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member Cosh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Cosh", [|input|], [|input.dt|], [||]) |> toTuple1
    static member ReduceMin(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceMin", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member ReduceProd(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceProd", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member Squeeze(graph: Graph, data: ValueInfo, ?axes: int64[]) =
        graph.AddNode("Squeeze", [|data|], [|data.dt|], [|Attr.ints("axes", axes)|]) |> toTuple1
    static member Selu(graph: Graph, X: ValueInfo, ?alpha: float32, ?gamma: float32) =
        graph.AddNode("Selu", [|X|], [|X.dt|], [|Attr.float("alpha", alpha, 1.6732631921768188f); Attr.float("gamma", gamma, 1.0507010221481323f)|]) |> toTuple1
    static member Sigmoid(graph: Graph, X: ValueInfo) =
        graph.AddNode("Sigmoid", [|X|], [|X.dt|], [||]) |> toTuple1
    static member ReduceSumSquare(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceSumSquare", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member Softmax(graph: Graph, input: ValueInfo, ?axis: int64) =
        graph.AddNode("Softmax", [|input|], [|input.dt|], [|Attr.int("axis", axis, 1L)|]) |> toTuple1
    static member Softsign(graph: Graph, input: ValueInfo) =
        graph.AddNode("Softsign", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Cos(graph: Graph, input: ValueInfo) =
        graph.AddNode("Cos", [|input|], [|input.dt|], [||]) |> toTuple1
    static member SpaceToDepth(graph: Graph, input: ValueInfo, blocksize: int64) =
        graph.AddNode("SpaceToDepth", [|input|], [|input.dt|], [|Attr.int("blocksize", blocksize)|]) |> toTuple1
    static member Asinh(graph: Graph, input: ValueInfo) =
        graph.AddNode("Asinh", [|input|], [|input.dt|], [||]) |> toTuple1
    static member ReduceL2(graph: Graph, data: ValueInfo, ?axes: int64[], ?keepdims: int64) =
        graph.AddNode("ReduceL2", [|data|], [|data.dt|], [|Attr.ints("axes", axes); Attr.int("keepdims", keepdims, 1L)|]) |> toTuple1
    static member Sqrt(graph: Graph, X: ValueInfo) =
        graph.AddNode("Sqrt", [|X|], [|X.dt|], [||]) |> toTuple1
    static member Log(graph: Graph, input: ValueInfo) =
        graph.AddNode("Log", [|input|], [|input.dt|], [||]) |> toTuple1
    static member Sub(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Sub", [|A; B|], [|A.dt|], [||]) |> toTuple1
    static member Scaler(graph: Graph, X: ValueInfo, ?offset: float32[], ?scale: float32[]) =
        graph.AddNode("Scaler", [|X|], [|DataType.FLOAT32|], [|Attr.floats("offset", offset); Attr.floats("scale", scale)|]) |> toTuple1
    static member Upsample(graph: Graph, X: ValueInfo, scales: ValueInfo, ?mode: string) =
        graph.AddNode("Upsample", [|X; scales|], [|X.dt|], [|Attr.string("mode", mode, "nearest")|]) |> toTuple1
    static member IsInf(graph: Graph, X: ValueInfo, ?detect_negative: int64, ?detect_positive: int64) =
        graph.AddNode("IsInf", [|X|], [|DataType.BOOL|], [|Attr.int("detect_negative", detect_negative, 1L); Attr.int("detect_positive", detect_positive, 1L)|]) |> toTuple1
    static member TfIdfVectorizer(graph: Graph, X: ValueInfo, max_gram_length: int64, max_skip_count: int64, min_gram_length: int64, mode: string, ngram_counts: int64[], ngram_indexes: int64[], ?pool_int64s: int64[], ?pool_strings: string[], ?weights: float32[]) =
        graph.AddNode("TfIdfVectorizer", [|X|], [|DataType.FLOAT32|], [|Attr.int("max_gram_length", max_gram_length); Attr.int("max_skip_count", max_skip_count); Attr.int("min_gram_length", min_gram_length); Attr.string("mode", mode); Attr.ints("ngram_counts", ngram_counts); Attr.ints("ngram_indexes", ngram_indexes); Attr.ints("pool_int64s", pool_int64s); Attr.strings("pool_strings", pool_strings); Attr.floats("weights", weights)|]) |> toTuple1
    static member Shape(graph: Graph, data: ValueInfo) =
        graph.AddNode("Shape", [|data|], [|DataType.INT64|], [||]) |> toTuple1
    static member Greater(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Greater", [|A; B|], [|DataType.BOOL|], [||]) |> toTuple1
    static member Equal(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Equal", [|A; B|], [|DataType.BOOL|], [||]) |> toTuple1
    static member And(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("And", [|A; B|], [|DataType.BOOL|], [||]) |> toTuple1
    static member Size(graph: Graph, data: ValueInfo) =
        graph.AddNode("Size", [|data|], [|DataType.INT64|], [||]) |> toTuple1
    static member IsNaN(graph: Graph, X: ValueInfo) =
        graph.AddNode("IsNaN", [|X|], [|DataType.BOOL|], [||]) |> toTuple1
    static member Less(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Less", [|A; B|], [|DataType.BOOL|], [||]) |> toTuple1
    static member Xor(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Xor", [|A; B|], [|DataType.BOOL|], [||]) |> toTuple1
    static member Or(graph: Graph, A: ValueInfo, B: ValueInfo) =
        graph.AddNode("Or", [|A; B|], [|DataType.BOOL|], [||]) |> toTuple1
    static member CumSum(graph: Graph, x: ValueInfo, axis: ValueInfo, ?exclusive: int64, ?reverse: int64) =
        graph.AddNode("CumSum", [|x; axis|], [|x.dt|], [|Attr.int("exclusive", exclusive, 0L); Attr.int("reverse", reverse, 0L)|]) |> toTuple1
    static member RoiAlign(graph: Graph, X: ValueInfo, rois: ValueInfo, batch_indices: ValueInfo, ?mode: string, ?output_height: int64, ?output_width: int64, ?sampling_ratio: int64, ?spatial_scale: float32) =
        graph.AddNode("RoiAlign", [|X; rois; batch_indices|], [|X.dt|], [|Attr.string("mode", mode, "avg"); Attr.int("output_height", output_height, 1L); Attr.int("output_width", output_width, 1L); Attr.int("sampling_ratio", sampling_ratio, 0L); Attr.float("spatial_scale", spatial_scale, 1.0f)|]) |> toTuple1
    static member QLinearConv(graph: Graph, x: ValueInfo, x_scale: ValueInfo, x_zero_point: ValueInfo, w: ValueInfo, w_scale: ValueInfo, w_zero_point: ValueInfo, y_scale: ValueInfo, y_zero_point: ValueInfo, ?B: ValueInfo, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("QLinearConv", ([|Some(x); Some(x_scale); Some(x_zero_point); Some(w); Some(w_scale); Some(w_zero_point); Some(y_scale); Some(y_zero_point); B|] |> Array.choose id), [|y_zero_point.dt|], [|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|]) |> toTuple1
    static member ConvInteger(graph: Graph, x: ValueInfo, w: ValueInfo, ?x_zero_point: ValueInfo, ?w_zero_point: ValueInfo, ?auto_pad: string, ?dilations: int64[], ?group: int64, ?kernel_shape: int64[], ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("ConvInteger", ([|Some(x); Some(w); x_zero_point; w_zero_point|] |> Array.choose id), [|DataType.INT32|], [|Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.int("group", group, 1L); Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|]) |> toTuple1
    static member QLinearMatMul(graph: Graph, a: ValueInfo, a_scale: ValueInfo, a_zero_point: ValueInfo, b: ValueInfo, b_scale: ValueInfo, b_zero_point: ValueInfo, y_scale: ValueInfo, y_zero_point: ValueInfo) =
        graph.AddNode("QLinearMatMul", [|a; a_scale; a_zero_point; b; b_scale; b_zero_point; y_scale; y_zero_point|], [|y_zero_point.dt|], [||]) |> toTuple1
    static member Where(graph: Graph, condition: ValueInfo, X: ValueInfo, Y: ValueInfo) =
        graph.AddNode("Where", [|condition; X; Y|], [|X.dt|], [||]) |> toTuple1
    static member MaxUnpool(graph: Graph, X: ValueInfo, I: ValueInfo, kernel_shape: int64[], ?output_shape: ValueInfo, ?pads: int64[], ?strides: int64[]) =
        graph.AddNode("MaxUnpool", ([|Some(X); Some(I); output_shape|] |> Array.choose id), [|X.dt|], [|Attr.ints("kernel_shape", kernel_shape); Attr.ints("pads", pads); Attr.ints("strides", strides)|]) |> toTuple1
    static member GatherElements(graph: Graph, data: ValueInfo, indices: ValueInfo, ?axis: int64) =
        graph.AddNode("GatherElements", [|data; indices|], [|data.dt|], [|Attr.int("axis", axis, 0L)|]) |> toTuple1
    static member QuantizeLinear(graph: Graph, x: ValueInfo, y_scale: ValueInfo, ?y_zero_point: ValueInfo) =
        graph.AddNode("QuantizeLinear", ([|Some(x); Some(y_scale); y_zero_point|] |> Array.choose id), [|y_zero_point.Value.dt|], [||]) |> toTuple1
    static member Resize(graph: Graph, X: ValueInfo, roi: ValueInfo, scales: ValueInfo, ?sizes: ValueInfo, ?coordinate_transformation_mode: string, ?cubic_coeff_a: float32, ?exclude_outside: int64, ?extrapolation_value: float32, ?mode: string, ?nearest_mode: string) =
        graph.AddNode("Resize", ([|Some(X); Some(roi); Some(scales); sizes|] |> Array.choose id), [|X.dt|], [|Attr.string("coordinate_transformation_mode", coordinate_transformation_mode, "half_pixel"); Attr.float("cubic_coeff_a", cubic_coeff_a, -0.75f); Attr.int("exclude_outside", exclude_outside, 0L); Attr.float("extrapolation_value", extrapolation_value, 0.0f); Attr.string("mode", mode, "nearest"); Attr.string("nearest_mode", nearest_mode, "round_prefer_floor")|]) |> toTuple1
    static member MatMulInteger(graph: Graph, A: ValueInfo, B: ValueInfo, ?a_zero_point: ValueInfo, ?b_zero_point: ValueInfo) =
        graph.AddNode("MatMulInteger", ([|Some(A); Some(B); a_zero_point; b_zero_point|] |> Array.choose id), [|DataType.INT32|], [||]) |> toTuple1
    static member Compress(graph: Graph, input: ValueInfo, condition: ValueInfo, ?axis: int64) =
        graph.AddNode("Compress", [|input; condition|], [|input.dt|], [|Attr.int("axis", axis)|]) |> toTuple1
    static member Gather(graph: Graph, data: ValueInfo, indices: ValueInfo, ?axis: int64) =
        graph.AddNode("Gather", [|data; indices|], [|data.dt|], [|Attr.int("axis", axis, 0L)|]) |> toTuple1
    static member ScatterElements(graph: Graph, data: ValueInfo, indices: ValueInfo, updates: ValueInfo, ?axis: int64) =
        graph.AddNode("ScatterElements", [|data; indices; updates|], [|data.dt|], [|Attr.int("axis", axis, 0L)|]) |> toTuple1
    static member Slice(graph: Graph, data: ValueInfo, starts: ValueInfo, ends: ValueInfo, ?axes: ValueInfo, ?steps: ValueInfo) =
        graph.AddNode("Slice", ([|Some(data); Some(starts); Some(ends); axes; steps|] |> Array.choose id), [|data.dt|], [||]) |> toTuple1
    static member Tile(graph: Graph, input: ValueInfo, repeats: ValueInfo) =
        graph.AddNode("Tile", [|input; repeats|], [|input.dt|], [||]) |> toTuple1
    static member Scatter(graph: Graph, data: ValueInfo, indices: ValueInfo, updates: ValueInfo, ?axis: int64) =
        graph.AddNode("Scatter", [|data; indices; updates|], [|data.dt|], [|Attr.int("axis", axis, 0L)|]) |> toTuple1
    static member LSTM(graph: Graph, X: ValueInfo, W: ValueInfo, R: ValueInfo, ?B: ValueInfo, ?sequence_lens: ValueInfo, ?initial_h: ValueInfo, ?initial_c: ValueInfo, ?P: ValueInfo, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?input_forget: int64) =
        graph.AddNode("LSTM", ([|Some(X); Some(W); Some(R); B; sequence_lens; initial_h; initial_c; P|] |> Array.choose id), [|X.dt; X.dt; X.dt|], [|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("input_forget", input_forget, 0L)|]) |> toTuple3
    static member MaxPool(graph: Graph, X: ValueInfo, kernel_shape: int64[], ?auto_pad: string, ?dilations: int64[], ?pads: int64[], ?storage_order: int64, ?strides: int64[]) =
        graph.AddNode("MaxPool", [|X|], [|X.dt; DataType.INT64|], [|Attr.ints("kernel_shape", kernel_shape); Attr.string("auto_pad", auto_pad, "NOTSET"); Attr.ints("dilations", dilations); Attr.ints("pads", pads); Attr.int("storage_order", storage_order, 0L); Attr.ints("strides", strides)|]) |> toTuple2
    static member GRU(graph: Graph, X: ValueInfo, W: ValueInfo, R: ValueInfo, ?B: ValueInfo, ?sequence_lens: ValueInfo, ?initial_h: ValueInfo, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64, ?linear_before_reset: int64) =
        graph.AddNode("GRU", ([|Some(X); Some(W); Some(R); B; sequence_lens; initial_h|] |> Array.choose id), [|X.dt; X.dt|], [|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size); Attr.int("linear_before_reset", linear_before_reset, 0L)|]) |> toTuple2
    static member TopK(graph: Graph, X: ValueInfo, K: ValueInfo, ?axis: int64, ?largest: int64, ?sorted: int64) =
        graph.AddNode("TopK", [|X; K|], [|X.dt; DataType.INT64|], [|Attr.int("axis", axis, -1L); Attr.int("largest", largest, 1L); Attr.int("sorted", sorted, 1L)|]) |> toTuple2
    static member Dropout(graph: Graph, data: ValueInfo, ?ratio: float32) =
        graph.AddNode("Dropout", [|data|], [|data.dt; DataType.BOOL|], [|Attr.float("ratio", ratio, 0.5f)|]) |> toTuple2
    static member Unique(graph: Graph, X: ValueInfo, ?axis: int64, ?sorted: int64) =
        graph.AddNode("Unique", [|X|], [|X.dt; DataType.INT64; DataType.INT64; DataType.INT64|], [|Attr.int("axis", axis); Attr.int("sorted", sorted, 1L)|]) |> toTuple4
    static member DynamicQuantizeLinear(graph: Graph, x: ValueInfo) =
        graph.AddNode("DynamicQuantizeLinear", [|x|], [|DataType.UINT8; DataType.FLOAT32; DataType.UINT8|], [||]) |> toTuple3
    static member RNN(graph: Graph, X: ValueInfo, W: ValueInfo, R: ValueInfo, ?B: ValueInfo, ?sequence_lens: ValueInfo, ?initial_h: ValueInfo, ?activation_alpha: float32[], ?activation_beta: float32[], ?activations: string[], ?clip: float32, ?direction: string, ?hidden_size: int64) =
        graph.AddNode("RNN", ([|Some(X); Some(W); Some(R); B; sequence_lens; initial_h|] |> Array.choose id), [|X.dt; X.dt|], [|Attr.floats("activation_alpha", activation_alpha); Attr.floats("activation_beta", activation_beta); Attr.strings("activations", activations, [|"Tanh";"Tanh"|]); Attr.float("clip", clip); Attr.string("direction", direction, "forward"); Attr.int("hidden_size", hidden_size)|]) |> toTuple2
    static member BatchNormalization(graph: Graph, X: ValueInfo, scale: ValueInfo, B: ValueInfo, mean: ValueInfo, var: ValueInfo, ?epsilon: float32, ?momentum: float32) =
        graph.AddNode("BatchNormalization", [|X; scale; B; mean; var|], [|X.dt; X.dt; X.dt; X.dt; X.dt|], [|Attr.float("epsilon", epsilon, 9.999999747378752e-06f); Attr.float("momentum", momentum, 0.8999999761581421f)|]) |> toTuple5
    static member SequenceEmpty<'a>(graph: Graph) =
        graph.AddNode("SequenceEmpty", [||], [|getDataType(typeof<'a>)|], [||])
    static member EyeLike<'a>(graph: Graph, input: ValueInfo, ?k: int64) =
        graph.AddNode("EyeLike", [|input|], [|getDataType(typeof<'a>)|], [|Attr.int("k", k, 0L)|])
    static member EyeLike(graph: Graph, input: ValueInfo, ?k: int64) =
        graph.AddNode("EyeLike", [|input|], [|input.dt|], [|Attr.int("k", k, 0L)|])
    static member Multinomial<'a>(graph: Graph, input: ValueInfo, ?sample_size: int64, ?seed: float32) =
        graph.AddNode("Multinomial", [|input|], [|getDataType(typeof<'a>)|], [|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|])
    static member Multinomial(graph: Graph, input: ValueInfo, ?sample_size: int64, ?seed: float32) =
        graph.AddNode("Multinomial", [|input|], [|input.dt|], [|Attr.int("sample_size", sample_size, 1L); Attr.float("seed", seed)|])
    static member RandomUniformLike<'a>(graph: Graph, input: ValueInfo, ?high: float32, ?low: float32, ?seed: float32) =
        graph.AddNode("RandomUniformLike", [|input|], [|getDataType(typeof<'a>)|], [|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|])
    static member RandomUniformLike(graph: Graph, input: ValueInfo, ?high: float32, ?low: float32, ?seed: float32) =
        graph.AddNode("RandomUniformLike", [|input|], [|input.dt|], [|Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|])
    static member RandomNormalLike<'a>(graph: Graph, input: ValueInfo, ?mean: float32, ?scale: float32, ?seed: float32) =
        graph.AddNode("RandomNormalLike", [|input|], [|getDataType(typeof<'a>)|], [|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|])
    static member RandomNormalLike(graph: Graph, input: ValueInfo, ?mean: float32, ?scale: float32, ?seed: float32) =
        graph.AddNode("RandomNormalLike", [|input|], [|input.dt|], [|Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|])
    static member RandomNormal<'a>(graph: Graph, shape: int64[], ?mean: float32, ?scale: float32, ?seed: float32) =
        graph.AddNode("RandomNormal", [||], [|getDataType(typeof<'a>)|], [|Attr.ints("shape", shape); Attr.float("mean", mean, 0.0f); Attr.float("scale", scale, 1.0f); Attr.float("seed", seed)|])
    static member RandomUniform<'a>(graph: Graph, shape: int64[], ?high: float32, ?low: float32, ?seed: float32) =
        graph.AddNode("RandomUniform", [||], [|getDataType(typeof<'a>)|], [|Attr.ints("shape", shape); Attr.float("high", high, 1.0f); Attr.float("low", low, 0.0f); Attr.float("seed", seed)|])
    static member Cast<'a>(graph: Graph, input: ValueInfo) =
        graph.AddNode("Cast", [|input|], [|getDataType(typeof<'a>)|], [||])
