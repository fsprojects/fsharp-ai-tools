[<AutoOpen>]
module FSAI.Tools.NNImpl

open Tensorflow
open Tensorflow.Operations

#if FS47
open Tensorflow.Binding
#else
let tf = Tensorflow.Binding.tf
#endif

type gen_ops with

    static member reduce_dims(input: Tensor, ?axis: Tensor) =
        
        match axis with
        | Some axis -> axis
        | None ->
            // Fast path: avoid creating Rank and Range ops if ndims is known.
            let shape = input.TensorShape
            if shape.is_fully_defined() then
                // NOTE: The python code distinguishes between tensor and sparsetensor
                tf.constant([|0 .. shape.size - 1|], TF_DataType.TF_INT32)
            else
                // Otherwise, we rely on Range and Rank to do the right thing at run-time.
                gen_ops.range(tf.constant(0), gen_ops.rank (input), tf.constant(1))


    static member conv2d_transpose(value: Tensor, filter: Tensor, outputShape: Tensor, strides:int[], ?padding:string, ?data_format:string, ?name:string) = 
        let paddingV     = defaultArg padding "SAME"
        let data_formatV = defaultArg data_format "NHWC"
        let name = defaultArg name "conv2d_transpose"
        // TODO re-do Dimension and Shape functions 
        // https://github.com/fsprojects/FSAI.Tools/blob/cdbd841bc86136f8ef24524cfc346e77bf21e6af/src/FSAI.Tools/Tensorflow.fs#L409
//        if not (data_formatV = "NCHW" || data_formatV = "NHWC") then 
//            failwith "dataformat has to be either NCHW or NHWC."
//        let axis = if data_formatV = "NHWC" then 3 else 1
//        let value_shape =  value.GetShape()
//        let filter_shape = filter.GetShape()
//        let output_shape = outputShape.GetShape()
//        if not (value_shape.[axis].IsCompatibleWith(filter_shape.[3])) then
//            sprintf "input channels does not match filter's input channels, \n %i != %i" 
//                (int64 value_shape.[axis]) (int64 filter_shape.[3])
//            |> ValueError |> raise
//        if not (output_shape.IsCompatibleWith(TFShape.Vector(4L))) then
//            sprintf "output shape must have shape (4,), got %O" output_shape
//            |> ValueError |> raise
//        if not (filter_shape.[2].IsCompatibleWith(output_shape.[axis])) then
//            sprintf "output shape does not match filter's output channels, %O != %O" 
//                output_shape.[axis] filter_shape.[2]
//            |> ValueError |> raise
//        if paddingV <> "VALID" && paddingV <> "SAME" then
//            failwithf "padding must be either VALID or SAME: %s" paddingV
        gen_ops.conv2d_backprop_input(
            input_sizes = outputShape,
            filter = filter,
            out_backprop = value,
            strides = strides,
            padding = paddingV,
            data_format = data_formatV,
            name = name
        )