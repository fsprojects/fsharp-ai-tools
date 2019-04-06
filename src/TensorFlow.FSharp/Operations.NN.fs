[<AutoOpen>]
module TensorFlow.FSharp.NNImpl

open System
open TensorFlow.FSharp.Utils
open Tensorflow
open Tensorflow.Operations

type gen_ops with

    static member conv2d_transpose(value : Tensor, filter : Tensor, outputShape : Tensor, strides:int[], ?padding:string, ?data_format:string,?name:string) = 
        let paddingV     = defaultArg padding "SAME"
        let data_formatV = defaultArg data_format "NHWC"

        //use name_scope = graph.NameScope("conv2d_transpose",value,filter,outputShape)

        if not (data_formatV = "NCHW" || data_formatV = "NHWC") then 
            failwith "dataformat has to be either NCHW or NHWC."
        let axis = if data_formatV = "NHWC" then 3 else 1
        let value_shape =  gen_ops.GetShape(value)
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

        gen_ops.conv2d_backprop_input(
            input_sizes = outputShape,
            filter = filter,
            out_backprop = value,
            strides = strides,
            padding = paddingV,
            data_format = data_formatV,
            name = name
        )