# Onnx for FSharp
This is an experimental F# API for Onnx. 
The problem this library aims to solve is to combine the ease of use of an Eager Execution API with the speed of a Graph API.  The API is code generated from the Onnx operator definitions. 

# Example
```fsharp
type on = FSharp.ML.Onnx.API.Snake.Onnx

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
```

## DONE
*	Eager Execution API for snake and pascal
*	F# Quotation -> Onnx transform

## TODO
*	Nuget package
*	Encapsulate Tensor type to support operator overloads
*	Curried API for forward piping
*	Code Analyzer to check code is transformable at design time
*	Shape Analyzer to graph has valid shapes at design time
*	Onnx graph optimizer passes
*	Onnx training

## Limitations
*	Eager Execution needs to clone memory which penalizes performance
*	Normal quotation limitations apply for the F# Quotation -> Onnx Graph transform 
*	The generic arguments in the expression only support a limited subset of structual typing
	*	Tuples and Records of Tensors of types Int32, Int64, Float32, or Float64
	*	Arrays are not supported
*	Not all Onnx Ops are supported at this time

## Out of scope
*	Partial graph evaluation - this requires external knowledge of the generated graph which breaks the abstraction
*	C++ wrapper interface for Onnx
