// TODO use FSI in Ionide for .net core
#I @"C:\EE\Git\ONNXBackend\FSharpMLOnnx\bin\Debug\netcoreapp3.1"
#I @"C:\Users\moloneymb\.nuget\packages\"
#r @"google.protobuf\3.11.4\lib\netstandard2.0\Google.Protobuf.dll"
#r @"microsoft.ml.onnxruntime\1.1.2\lib\netstandard1.1\Microsoft.ML.OnnxRuntime.dll"
#r @"fsharp.quotations.evaluator\2.1.0\lib\netstandard2.0\FSharp.Quotations.Evaluator.dll"
#r "OnnxMLProto.dll"
#r "FSharpMLOnnx.dll"

open FSharp.ML.Onnx.Protobuf
open Microsoft.ML.OnnxRuntime.Tensors
open System.IO
open Microsoft.ML.OnnxRuntime.Tensors
open Onnx
open FSharp.ML.Onnx.Protobuf
open System
open System.IO
open Microsoft.ML.OnnxRuntime
open Microsoft.FSharp.Quotations
open FSharp.ML.Onnx.Utils
open FSharp.ML.Onnx.Utils.Expr
open FSharp.ML.Onnx.Expr
open FSharp.ML.Onnx.Extensions


let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","data","mnist")
type on = FSharp.ML.Onnx.API.SnakeCase.Onnx
type DV<'a> = DisposableValue<'a>


let getTensorF(name,shape) =
    let dts = File.ReadAllBytes(Path.Combine(mnistDir, name)) |> bytesToFloats
    on.reshape(ArrayTensorExtensions.ToTensor(dts) ,ArrayTensorExtensions.ToTensor(shape))


type MNIST() = 
    let p193 = getTensorF("Parameter193", [|16L; 4L; 4L; 10L|])
    let p87  = getTensorF("Parameter87",  [|16L; 8L; 5L; 5L|])
    let p5   = getTensorF("Parameter5",  [|8L; 1L; 5L; 5L|])
    let p6   = getTensorF("Parameter6", [|8L; 1L; 1L|])
    let p88  = getTensorF("Parameter88", [|16L; 1L; 1L|])
    let p194 = getTensorF("Parameter194", [|1L; 10L|]) 

    [<ReflectedDefinition>]
    member this.Layer(x:Tensor<float32>,p1,p2,k) = 
       on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst

    [<ReflectedDefinition>]
    member this.Forward(x: Tensor<float32>) = 
        on.add(on.mat_mul(on.reshape((this.Layer(this.Layer(x,p5,p6,2L),p87,p88,3L)),[|1;256|]),on.reshape(p193,[|256;10|])),p194)

    [<ReflectedDefinition>]
    member this.Forward2(x: Tensor<float32>) = 
        let layer (p1,p2,k) (x:Tensor<float32>) = 
            on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
        on.add(on.mat_mul(on.reshape(x |> layer(p5,p6,2L) |> layer (p87, p88, 2L),[|1;256|]),on.reshape(p193,[|256;10|])),p194)

    [<ReflectedDefinition>]
    member this.Forward3(x: Tensor<float32>) = 
        let layer (p1,p2,k)  = 
            fun (x:Tensor<float32>) -> on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
        on.add(on.mat_mul(on.reshape(x |> layer(p5,p6,2L) |> layer (p87, p88, 2L),[|1;256|]),on.reshape(p193,[|256;10|])),p194)

    [<ReflectedDefinition>]
    member this.Forward4(x: Tensor<float32>) = 
        let layer (p1,p2) (x:Tensor<float32>) = 
            let k = 2L
            on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
        on.add(on.mat_mul(on.reshape(x |> layer(p5,p6) |> layer (p87, p88),[|1;256|]),on.reshape(p193,[|256;10|])),p194)

    [<ReflectedDefinition>]
    member this.Forward5(x: Tensor<float32>) = 
        let layer(p,k) (x:Tensor<float32>) = 
            on.max_pool(on.conv(x,p,auto_pad = "SAME_UPPER"),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
        x |> layer(p5,2L)

//let mnistG = MNIST()
// I know about the new Type mismatch bug, I will work on it
//let graphFunction : DV<Tensor<float32> -> DV<Tensor<float32>>> = toOnnxGraph(<@ mnistG.Forward @>)
//let graphFunction2 : DV<Tensor<float32> -> DV<Tensor<float32>>> = toOnnxGraph(<@ mnistG.Forward2 @>)






