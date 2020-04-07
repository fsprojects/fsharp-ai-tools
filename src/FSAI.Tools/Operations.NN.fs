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
        gen_ops.conv2d_backprop_input(
            input_sizes = outputShape,
            filter = filter,
            out_backprop = value,
            strides = strides,
            padding = paddingV,
            data_format = data_formatV,
            name = name
        )