namespace TensorFlow.FSharp 

#r "netstandard"
#r "../lib/TensorFlowSharp.dll"
#load "NNImpl.fsx"
#load "NNOps.fsx"
#load "NPYReaderWriter.fsx"
#load "ImageWriter.fsx"

open System
open TensorFlow

type Shape(shape: int[]) =

    member __.Item with get v = shape.[v]

    member __.Dimensions = shape

    override __.ToString() = sprintf "%A" shape

    member shape.AsTFShape() = TFShape(shape.Dimensions |> Array.map int64)

    member shape.AsTFTensor() = shape.AsTFShape().AsTensor()

    static member NewInferred() = failwith "tbd"
    static member DimInferred() = failwith "tbd"

    static member EquivShapes (shape1: Shape) (shape2: Shape) = 
        if shape1.Dimensions = shape2.Dimensions then 
            shape1 
        else 
            failwithf "mismatched shapes: %A and %A" shape1 shape2 

// tf { ... }  is a computational DSL (not using ReflectedDefinition) that does a layer of shape inference 
// in the first runtime phase, and then the actual graph construction in the second runtime phase.   
type V<'T>(shape: Shape, eval: (TFGraph -> TFOutput)) =

    member __.Apply(graph) = eval graph

    static member (+) (v1: V<'T>, v2: V<'T>) : V<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        V (outputShape, fun graph -> graph.Add(v1.Apply(graph), v2.Apply(graph)))

    static member (-) (v1: V<'T>, v2: V<'T>) : V<'T> = 
        V (Shape.EquivShapes v1.Shape v2.Shape, fun graph -> graph.Sub(v1.Apply(graph), v2.Apply(graph)))

    static member ( * ) (v1: V<'T>, v2: V<'T>) : V<'T> = 
        V (Shape.EquivShapes v1.Shape v2.Shape, fun graph -> graph.Mul(v1.Apply(graph), v2.Apply(graph)))

    static member (/) (v1: V<'T>, v2: V<'T>) : V<'T> = 
        V (Shape.EquivShapes v1.Shape v2.Shape, fun graph -> graph.Div(v1.Apply(graph), v2.Apply(graph)))

    static member Sqrt (v: V<'T>) : V<'T> = 
        V (v.Shape, fun graph -> graph.Sqrt(v.Apply(graph)))

    static member Tanh (v: V<'T>) : V<'T> = 
        V (v.Shape, fun graph -> graph.Tanh(v.Apply(graph)))

    static member Tan (v: V<'T>) : V<'T> =  
        V (v.Shape, fun graph -> graph.Tan(v.Apply(graph)))

    member __.Shape : Shape = shape

type TF() =

    static member Const (shape: Shape, value: 'T) : V<'T> = 
        V (shape, fun graph -> graph.Reshape(graph.Const(new TFTensor(value)), graph.Const(shape.AsTFTensor())))

    static member ConstArray (shape: Shape, value: 'T[]) : V<'T> = 
        V (shape, fun graph -> graph.Reshape(graph.Const(new TFTensor(value)), graph.Const(shape.AsTFTensor())))

    static member TruncatedNormal (shape: Shape) : V<double> = 
        V (shape, fun graph -> graph.TruncatedNormal(graph.Const(shape.AsTFTensor()), TFDataType.Double ))

    static member Variable (value: V<'T>) : V<'T> = 
        V (value.Shape, fun graph -> graph.Variable(value.Apply(graph)).Read)

    static member Conv2D (input: V<'T>, filters: V<'T>, ?stride: int, ?padding: string) : V<'T> = 
    //[N,H,W,C], filters: V[C;COut;F]) -> V[N,H,W,COut] 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let inputShape = input.Shape
        let filtersShape = filters.Shape
        let outputShape = Shape [| inputShape.[0]; inputShape.[1]; inputShape.[2]; filtersShape.[1] |]
        V (outputShape, fun graph -> graph.Conv2D(input.Apply(graph), filters.Apply(graph),strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    static member Conv2DBackpropInput(filters: V<'T>, out_backprop: V<'T>, ?stride: int, ?padding: string) : V<'T> = 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let shape = out_backprop.Shape
        let filtersShape = filters.Shape
        let outputShape = Shape [| shape.[0]; shape.[1]; shape.[2]; filtersShape.[1] |]
        V (outputShape, fun graph -> 
           let batch_size = shape.[0]
           let rows = shape.[1]
           let cols = shape.[2]
           let num_filters = filtersShape.[2]
           let output_shape = Shape [|batch_size; rows*stride; cols*stride; num_filters |]
           let input_sizes = graph.Const(output_shape.AsTFTensor())
           graph.Conv2DBackpropInput(input_sizes, filters.Apply(graph), out_backprop.Apply(graph), strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))
         // , output_shape=[input.Shape.[0],H*Stride,W*Stride,filters.Shape.[3]]

    static member ClipByValue (input: V<'T>, low: V<'T>, high: V<'T>) : V<'T> = 
        let outputShape = Shape.EquivShapes (Shape.EquivShapes input.Shape low.Shape) high.Shape
        V (outputShape, fun graph -> graph.ClipByValue(input.Apply(graph), low.Apply(graph), high.Apply(graph)))

    static member Moments(shape: Shape, input: V<'T>) : V<'T> * V<'T> = 
        let outputShape = shape
        // TODO: we make two Moments nodes here
        // TODO: keep_dims
        V (outputShape, fun graph -> fst (graph.Moments(input.Apply(graph), graph.Const(shape.AsTFTensor())))),
        V (outputShape, fun graph -> snd (graph.Moments(input.Apply(graph), graph.Const(shape.AsTFTensor()))))

    static member Relu(input: V<'T>) : V<'T> = 
        let outputShape = input.Shape
        V (outputShape, fun graph -> graph.Relu(input.Apply(graph)))

    static member DecodeJpeg(contents:V<string>, ?channels: int) : V<int> = // V[int,H,W,C]
        let channels = defaultArg channels 3 // CHECK ME
        let outputShape = Shape [| -1; -1; channels |]
        V (outputShape, fun graph -> graph.DecodeJpeg(contents=contents.Apply(graph), channels=Nullable(3L)))

    static member Cast<'T, 'T2>(input: V<'T>, dt) : V<'T2> = 
        let outputShape = input.Shape
        V (outputShape, fun graph -> graph.Cast(input.Apply(graph), dt))

    static member CreateString(value: byte[]) : V<string> = 
        let outputShape = Shape [| 1 |]
        V (outputShape, fun graph -> graph.Const(TFTensor.CreateString(value)))

    static member ExpandDims(value: V<'T>, [<ParamArray>] dims: int[]) : V<'T> = 
        let outputShape = Shape dims
        V (outputShape, fun graph -> graph.ExpandDims(value.Apply(graph), graph.Const(outputShape.AsTFTensor())))
    //TF.ExpandDims[Dim](value: V[shape]) : V[Expanded(Dim,shape)]

    static member Run(value: V<'T>) : TFTensor = 
        let sess = new TFSession()
        let graph = sess.Graph
        let node = value.Apply(graph)
        sess.Run([||],[||],[|node|]).[0]

type TensorFlow = ReflectedDefinitionAttribute

type TFBuilder() =
    member x.Return(v: 'T) = v
    //member x.Zero() = V()
    //member x.Run(v) = v

//type NumericLiteralG() =
//    member x.FromString(s: string) : V<double> = failwith "tbd"
    
[<AutoOpen>]
module DSLHelpers = 
    let tf = TFBuilder()

    let shape (ints: int list) = Shape(Array.ofSeq ints)

    let v (d:double) : V<double> = 
        let shape = Shape.NewInferred ()
        V (shape, (fun graph -> graph.Const(new TFTensor(d))))

