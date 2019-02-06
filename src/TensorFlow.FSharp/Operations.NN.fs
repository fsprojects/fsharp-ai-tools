[<AutoOpen>]
module TensorFlow.FSharp.NNImpl

open TensorFlow.FSharp.Utils
open System

type TFGraph with
    /// https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/nn_impl.py
    member graph.Moments(x:TFOutput, ?axes:TFOutput, ?shift, ?name, ?keep_dims) =
        let keep_dimsV = defaultArg keep_dims false
        use name = graph.NameScope("moments",[|yield x; yield! axes |> Option.toArray|]) // NOTE: this needs control parameters
        let y = if x.TFDataType = TFDataType.Float16 then graph.Cast(x,TFDataType.Float32) else x
        let mean = graph.ReduceMean(y, ?axis=axes, keep_dims=true, name ="mean")
        let variance = graph.ReduceMean(
                         graph.SquaredDifference(y, graph.StopGradient(mean)), 
                         ?axis=axes, 
                         keep_dims=true,
                         name="variance")

        let maybeSqueezeAndCast (y:TFOutput) = 
            let y = if keep_dimsV then y else graph.Squeeze(y)
            if x.TFDataType = TFDataType.Float16 then graph.Cast(y,TFDataType.Float16) else y
        (mean |> maybeSqueezeAndCast, variance |> maybeSqueezeAndCast)
        

    member graph.Conv2DTranspose(value:TFOutput, filter:TFOutput, outputShape:TFOutput, strides:int64[], ?padding:string, ?data_format:string,?name:string) = 
        let paddingV     = defaultArg padding "SAME"
        let data_formatV = defaultArg data_format "NHWC"
        use name_scope = graph.NameScope("conv2d_transpose",value,filter,outputShape)
        if not (data_formatV = "NCHW" || data_formatV = "NHWC") then 
            failwith "dataformat has to be either NCHW or NHWC."
        let axis = if data_formatV = "NHWC" then 3 else 1
        let value_shape = graph.GetShape(value)
        let filter_shape = graph.GetShape(filter)
        let output_shape = graph.GetShape(outputShape)
        if not (value_shape.[axis].IsCompatibleWith(filter_shape.[3])) then
            sprintf "input channels does not match filter's input channels, \n %i != %i" 
                (int64 value_shape.[axis]) (int64 filter_shape.[3])
            |> ValueError |> raise
        if not (output_shape.IsCompatibleWith(TFShape.Vector(4L))) then
            sprintf "output shape must have shape (4,), got %O" output_shape
            |> ValueError |> raise
        if not (filter_shape.[2].IsCompatibleWith(output_shape.[axis])) then
            sprintf "output shape does not match filter's output channels, %O != %O" 
                output_shape.[axis] filter_shape.[2]
            |> ValueError |> raise
            
        if paddingV <> "VALID" && paddingV <> "SAME" then
            failwithf "padding must be either VALID or SAME: %s" paddingV

        graph.Conv2DBackpropInput(
            input_sizes = outputShape,
            filter = filter,
            out_backprop = value,
            strides = strides,
            padding = paddingV,
            data_format = data_formatV,
            name = name_scope.Scope
        )