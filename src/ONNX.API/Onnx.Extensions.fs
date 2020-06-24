[<AutoOpen>]
module FSharp.ML.Onnx.Extensions
open Microsoft.ML.OnnxRuntime.Tensors
open FSharp.ML.Onnx.API

type 'a``[]`` with
    member x.ToTensor() = ArrayTensorExtensions.ToTensor(x)

type SnakeCase.Onnx with
    [<ReflectedDefinition>]
    static member reshape(x: Tensor<float32>,shape: int32[]) = 
        SnakeCase.Onnx.reshape(x,(shape |> Array.map int64).ToTensor())

type PascalCase.Onnx with
    [<ReflectedDefinition>]
    static member Reshape(x: Tensor<float32>,shape: int32[]) = 
        PascalCase.Onnx.Reshape(x,(shape |> Array.map int64).ToTensor())
