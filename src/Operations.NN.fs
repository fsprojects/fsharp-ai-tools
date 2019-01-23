[<AutoOpen>]
module TensorFlow.NNImpl
//#r "netstandard"
//#r "../lib/TensorFlowSharp.dll"

open Utils
open System
type TF with
    /// https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/nn_impl.py
    static member Moments(x:Output, ?axes:Output, ?shift, ?name, ?keep_dims) =
        let keep_dimsV = defaultArg keep_dims false
        use name = TF.DefaultGraph.WithScope("moments") // NOTE: this needs control parameters
        let y = if x.DType = DType.Float16 then TF.Cast(x,DType.Float32) else x
        let mean = TF.ReduceMean(y, ?axis=axes, keep_dims=true, name ="mean")
        let variance = TF.ReduceMean(
                         TF.SquaredDifference(y, TF.StopGradient(mean)), 
                         ?axis=axes, 
                         keep_dims=true,
                         name="variance")

        let maybeSqueezeAndCast (y:Output) = 
            let y = if keep_dimsV then y else TF.Squeeze(y)
            if x.DType = DType.Float16 then TF.Cast(y,DType.Float16) else y
        (mean |> maybeSqueezeAndCast, variance |> maybeSqueezeAndCast)

    static member Conv2DTranspose(value,filter, output_shape:int64[], strides:int64[], ?padding:string, ?data_format:string,?name:string) = 
        let paddingV     = defaultArg padding "SAME"
        let data_formatV = defaultArg data_format "NHWC"
        use name_scope = TF.DefaultGraph.WithScope("conv2d_transpose") // NOTE: this needs control parameters
        if not (data_formatV = "NCHW" || data_formatV = "NHWC") then 
            failwith "dataformat has to be either NCHW or NHWC."
        let axis = if data_formatV = "NHWC" then 3 else 1
        let value_shape = TF.DefaultGraph.GetShape(value)
        let filter_shape = TF.DefaultGraph.GetShape(filter)
        if output_shape.Length <> 4 then
            failwithf "output_shape must have shape (4,) got %A" output_shape
        if value_shape.[axis] <> filter_shape.[3] then
            failwithf "input channels does not match filter's input channels, \n %i != %i" 
                value_shape.[axis]
                filter_shape.[3]
        if output_shape.[3] <> filter_shape.[2] then
            failwithf "output_shape does does not match filter's output channels, \n %i != %i" 
                value_shape.[axis]
                filter_shape.[3]
        if paddingV <> "VALID" && paddingV <> "SAME" then
            failwithf "padding must be either VALID or SAME: %s" paddingV

        TF.Conv2DBackpropInput(
            input_sizes = TF.Const(Shape(output_shape).AsTensor()),
            filter = filter,
            out_backprop = value,
            strides = strides,
            padding = paddingV,
            data_format = data_formatV,
            ?name = name 
        )