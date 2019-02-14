[<AutoOpen>]
module TensorFlow.FSharp.NNImpl

open TensorFlow.FSharp.Utils
open System

type TFGraph with
    /// <summary>
    ///   Calculate the mean and variance of <c>x</c>
    /// </summary>
    /// <param name="inputs">
    ///   A Tensor.
    /// </param>
    /// <param name="axes">
    /// axes: Array of ints.  Axes along which to compute mean and
    ///  variance.
    /// <param>
    /// <param name="name">
    /// Name used to scope the operations that compute the moments.
    /// </param>
    /// <param name="keep_dims">
    ///  produce moments with the same dimensionality as the input.
    /// </param>
    /// <returns>
    /// </returns>
    /// Two <c>Tensor</c> objects: <c>mean</c> and <c>variance</c> .
    /// <remarks>
    /// The mean and variance are calculated by aggregating the contents of <c>x</c> 
    /// across <c>axes</c> .  If <c>x</c> is 1-D and <c>axes = [0]</c> this is just the mean
    /// and variance of a vector.
    /// Note: shift is currently not used; the true mean is computed and used.
    /// When using these moments for batch normalization (see
    /// <c>tf.nn.batch_normalization</c>):
    ///  * for so-called "global normalization", used with convolutional filters with
    ///    shape <c>[batch, height, width, depth]</c> , pass <c>axes=[0, 1, 2]</c> .
    ///  * for simple batch normalization pass <c>axes=[0]</c> (batch only).
    /// </remarks>
    ///
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