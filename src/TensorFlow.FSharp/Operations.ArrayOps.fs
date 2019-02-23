[<AutoOpen>]
module TensorFlow.FSharp.Operations.ArrayOps
// https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
// 2860 LOC

open System
open TensorFlow.FSharp.Utils
open TensorFlow.FSharp

type TFGraph with
    /// <summary>
    /// Outputs Zero values based on shape of tensor
    /// </summary>
    /// <param name="shape">Shape of the output tensor</param>
    /// <param name="dtype">Optional Type of the Zero value. Default: Double</param>
    /// <param name="operName">Operation name, optional.</param>
    /// <returns></returns>
    member graph.Zeros (shape : TFShape, ?dtype : TFDataType, ?name : string) = 
        graph.Constant (0, shape, ?dtype = dtype, ?name = name)

    /// <summary>
    /// Outputs One values based on shape of tensor
    /// </summary>
    /// <param name="shape">Shape of the output tensor</param>
    /// <param name="dtype">Optional Type of the Zero value. Default: Double</param>
    /// <param name="operName">Operation name, optional.</param>
    /// <returns></returns>
    member graph.Ones (shape : TFShape, ?dtype : TFDataType, ?name : string) = 
        graph.Constant (1, shape, ?dtype = dtype, ?name = name)

    /// <summary>
    /// Create a constant tensor based on a shape
    /// Used by Zeros and Ones
    /// </summary>
    /// <param name="value">Value for tensor</param>
    /// <param name="tfshape">Shape of the tensor</param>
    /// <param name="dtype">Optional Type of the Zero value. Default: Double</param>
    /// <param name="operName">Operation name, optional.</param>
    /// <returns></returns>
    /// see https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/python/framework/constant_op.py
    member graph.Constant (value : obj, tfshape : TFShape, ?dtype : TFDataType, ?name : string) =
        let dtype = defaultArg dtype TFDataType.Float32
        //convert the .net type to relevant tensorflow type
        let dtvalue = TFTensor.FetchSimpleObj(dtype, value)
        let shape = tfshape.ToLongArray ()
        let idx = Array.zeroCreate shape.Length
        for i = 0 to shape.Length - 1 do
            if int64 shape.[i] > int64 Int32.MaxValue then 
                raise(new ArgumentOutOfRangeException ("Shape can not be longer than 32 bits"))
        let data = 
            if tfshape.IsLongArray then Array.CreateInstance (dtvalue.GetType (), tfshape.ToLongArray ())
            else Array.CreateInstance (dtvalue.GetType (), tfshape.ToIntArray ())
        TFTensor.Set (data, dtype, shape, idx, 0, value)
        let tensor_value = new TFTensor (data)
        graph.Const (tensor_value, tensor_value.TFDataType, ?name = name)
