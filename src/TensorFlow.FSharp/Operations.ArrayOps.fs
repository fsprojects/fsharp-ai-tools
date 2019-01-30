module TensorFlow.FSharp.Operations.ArrayOps
// https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
// 2860 LOC

open System
open System.Runtime.InteropServices
open System.Collections.Generic
open System.Text
open FSharp.NativeInterop
open TensorFlow.FSharp.Utils
open TensorFlow.FSharp
open System

type TFGraph with
    ///Creates a tensor with all elements set to zero.
    ///This operation returns a tensor of type `dtype` with shape `shape` and
    ///all elements set to zero.
    ///For example:
    ///```python
    ///tf.zeros([3, 4], tf.int32)  # [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ///```
    ///Args:
    ///shape: A list of integers, a tuple of integers, or a 1-D `Tensor` of type
    ///  `int32`.
    ///dtype: The type of an element in the resulting `Tensor`.
    ///name: A name for the operation (optional).
    ///Returns:
    ///A `Tensor` with all elements set to zero.
    member graph.Zeros(shape:TFShape,?dtype:TFDataType, ?name:string) =
        let dtype = defaultArg dtype TFDataType.Float32
        //use name = graph.
        failwith "todo"

    member graph.Ones(shape:TFShape,?dtype:TFDataType, ?name:string) =
        failwith "todo"
