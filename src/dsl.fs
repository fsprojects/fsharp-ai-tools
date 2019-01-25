namespace TensorFlow.FSharp.DSL

open System
open System.Collections.Generic
open TensorFlow.FSharp

/// Represents an inferred dimension
type Dim =
    | DimMulInt of Dim * int
    | DimDivInt of Dim * int
    | DimVar of Dim option ref
    | Dim of int
    override dim.ToString() = 
        match dim.TryValue with 
        | Some v -> string v
        | None ->  
        match dim with 
        | DimMulInt (dim2, n) -> dim2.ToString() + "*" + string n 
        | DimDivInt (dim2, n) -> dim2.ToString() + "/" + string n 
        | Dim n -> string n 
        | DimVar v -> 
            match v.Value with 
            | None -> "?" 
            | Some v -> v.ToString()

    member dim.TryValue = 
        match dim with 
        | DimMulInt (dim2,n) -> match dim2.TryValue with None -> None | Some dimv -> Some (dimv*n) 
        | DimDivInt (dim2,n) -> match dim2.TryValue with None -> None | Some dimv -> Some (dimv/n) 
        | Dim n -> Some n 
        | DimVar v -> 
            match v.Value with 
            | None -> None 
            | Some v -> v.TryValue

    member dim.Value = 
        match dim.TryValue with 
        | Some v -> v
        | None -> -1

    static member ( * ) (dim: Dim, stride: int) = if stride = 1 then dim else DimMulInt (dim, stride)

    static member ( / ) (dim: Dim, stride: int) = if stride = 1 then dim else DimDivInt (dim, stride)

    static member Inferred = DimVar (ref None)

    static member Unify (dim1: Dim) (dim2: Dim) = 
        match dim1.TryValue, dim2.TryValue with 
        | Some v1, Some v2 -> 
            if v1 <> v2 then 
                failwithf "mismatched dimension %d and %d" v1 v2 
        | _ -> 
        match dim1, dim2 with 
        | DimVar ({ contents = None } as v1), DimVar ({ contents = None } as v2) when Object.ReferenceEquals(v1,v2) -> ()
        | DimVar ({ contents = None } as v1), _ -> v1 := Some dim2
        | _, DimVar ({ contents = None } as v2) -> v2 := Some dim1
        | DimVar { contents = Some soln1}, _ -> Dim.Unify soln1 dim2
        | _, DimVar { contents = Some soln2} -> Dim.Unify dim1 soln2
        | Dim d1, Dim d2 -> 
            if d1 <> d2 then 
                failwithf "mismatched dimensions, d1 = %d, d2 = %d" d1 d2
        | DimMulInt (d1, n1), Dim d2 -> 
            if d2 % n1 <> 0 then 
                failwithf "mismatched dimension %d and multiplier %d" d2 n1
            Dim.Unify d1 (Dim (d2 / n1))
        | Dim d1, DimMulInt (d2, n2) -> 
            if d1 % n2 <> 0 then 
                failwithf "mismatched dimension %d and multiplier %d" d1 n2
            Dim.Unify (Dim (d1 / n2)) d2
        | DimMulInt (d1, n1), DimMulInt (d2, n2) -> 
            if n1 <> n2 then 
                failwithf "mismatched dimension multipliers %d and %d" n1 n2
            Dim.Unify d1 d2
        | DimDivInt (d1, n1), DimDivInt (d2, n2) -> 
            if n1 <> n2 then 
                failwithf "mismatched dimension divisors %d and %d" n1 n2
            Dim.Unify d1 d2
        | _ -> 
            match dim1.TryValue, dim2.TryValue with 
            | None, _ -> failwithf "incomplete dimension %s" (dim1.ToString()) 
            | _, None -> failwithf "incomplete dimension %s" (dim2.ToString()) 
            | _ -> () // equal, see above

/// Represents an inferred shape
type Shape =
    | ShapeVar of Shape option ref
    | Shape of Dim[] 

    member shape.Item 
        with get idx = 
            match shape with 
            | Shape dims -> dims.[idx]
            | ShapeVar v -> 
                match v.Value with 
                | None -> Dim.Inferred
                | Some sln -> sln.[idx]

    member shape.AsRank1() = shape.[0]
    member shape.AsRank2() = shape.[0], shape.[1]
    member shape.AsRank3() = shape.[0], shape.[1], shape.[2]
    member shape.AsRank4() = shape.[0], shape.[1], shape.[2], shape.[3] 

    member shape.Dimensions = 
        match shape with 
        | Shape n -> n 
        | ShapeVar v -> 
            match v.Value with 
            | None -> failwith "unsolved shape variable" 
            | Some sln -> sln.Dimensions

    override shape.ToString() = 
        match shape with 
        | Shape n -> sprintf "shape %A" [ for i in n -> i.ToString() ] 
        | ShapeVar v -> 
            match v.Value with 
            | None -> "shape ?" 
            | Some sln -> sln.ToString()

    member shape.AsTFShape() = TFShape(shape.Dimensions |> Array.map (fun dim -> int64 dim.Value))

    member shape.AsTFTensor() = shape.AsTFShape().AsTensor()

    static member Inferred = ShapeVar (ref None)

    static member Unify (shape1: Shape) (shape2: Shape) = 
        match shape1, shape2 with 
        | ShapeVar ({ contents = None } as v1), ShapeVar ({ contents = None } as v2) when Object.ReferenceEquals(v1,v2) -> ()
        | ShapeVar ({ contents = None } as v1), _ -> v1 := Some shape2
        | _, ShapeVar ({ contents = None } as v2) -> v2 := Some shape1
        | ShapeVar { contents = Some soln1}, _ -> Shape.Unify soln1 shape2
        | _, ShapeVar { contents = Some soln2} -> Shape.Unify shape1 soln2
        | Shape dims1, Shape dims2 -> 
            if dims1.Length <> dims2.Length then 
                failwithf "mismatched shapes: %A and %A, dims1.Length = %d, dims2.Length = %d" shape1 shape2 dims1.Length dims2.Length
            (dims1, dims2) ||> Array.iter2 (fun d1 d2 -> Dim.Unify d1 d2)

    static member EquivShapes (shape1: Shape) (shape2: Shape) = 
        Shape.Unify shape1 shape2
        shape1

[<AutoOpen>]
module ShapeHelpers = 

    /// Create a non-inferred shape
    let shape (ints: seq<int>) = 
        ints |> Array.ofSeq |> Array.map (fun n -> if n = -1 then Dim.Inferred else Dim n) |> Shape

/// Represents a differentiable tensor value, which later corresponds to a node in a TensorFlow graph
type DT(shape: Shape) = 

    /// Get the inferred shape of the differentiable tensor 
    member __.Shape = shape

/// Represents a context for turning differentiable tensors into a TensorFlow graph
type internal Ctxt = 
    { Graph: TFGraph 
      Nodes: Dictionary<DT, TFOutput> // uses reference identity to ensure unique nodes from unique DT values
      MomentNodes: Dictionary<DT, TFOutput * TFOutput> // uses reference identity to ensure unique Moment nodes from unique DT values
      Values: Map<string,DT> }

type internal WithScopeDisposable(name:string) = 
    interface IDisposable with 
        member __.Dispose() = ()
    member __.Name = name        

/// Represents a differentiable tensor value
type DT<'T> internal (shape: Shape, eval: (Ctxt -> TFOutput)) =
    inherit DT(shape)

    member internal dt.Apply(ctxt: Ctxt) = 
        match ctxt.Nodes.TryGetValue(dt) with 
        | true, node -> node
        | _ -> 
           let res = eval ctxt
           ctxt.Nodes.[dt] <- res
           res

    static member (+) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Add(v1.Apply(ctxt), v2.Apply(ctxt)))

    static member (-) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Sub(v1.Apply(ctxt), v2.Apply(ctxt)))

    static member ( * ) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Mul(v1.Apply(ctxt), v2.Apply(ctxt)))

    static member (/) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Div(v1.Apply(ctxt), v2.Apply(ctxt)))

    static member Sqrt (v: DT<'T>) : DT<'T> = 
        DT<_> (v.Shape, fun ctxt -> ctxt.Graph.Sqrt(v.Apply(ctxt)))

    static member Tanh (v: DT<'T>) : DT<'T> = 
        DT<_> (v.Shape, fun ctxt -> ctxt.Graph.Tanh(v.Apply(ctxt)))

    static member Tan (v: DT<'T>) : DT<'T> =  
        DT<_> (v.Shape, fun ctxt -> ctxt.Graph.Tan(v.Apply(ctxt)))

    override dt.ToString() = dt.Shape.ToString()

    static member Diff (f: DT<'T> -> DT<'T>) (x: DT<'T>) : DT<'T> =  
        let y = f x
        DT<_> (y.Shape, fun ctxt -> 
            let xnode = x.Apply(ctxt)
            let ynode = y.Apply(ctxt)
            let dynodes = ctxt.Graph.AddGradients([| ynode |], [| xnode |])
            dynodes.[0])

    static member Const (value: double, ?shape: Shape) : DT<'T> = 
        let shape = shape |> Option.defaultWith (fun () -> Shape.Inferred)
        DT<_> (shape, fun ctxt -> ctxt.Graph.Reshape(ctxt.Graph.Const(new TFTensor(value)), ctxt.Graph.Const(shape.AsTFTensor())))

    static member ConstArray (value: 'T[], ?shape: Shape) : DT<'T> = 
        let shape = shape |> Option.defaultWith (fun () -> Shape.Inferred)
        DT<_> (shape, fun ctxt -> ctxt.Graph.Reshape(ctxt.Graph.Const(new TFTensor(value)), ctxt.Graph.Const(shape.AsTFTensor())))

    static member TruncatedNormal (shape: Shape) : DT<double> = 
        DT<_> (shape, fun ctxt -> ctxt.Graph.TruncatedNormal(ctxt.Graph.Const(shape.AsTFTensor()), TFDataType.Float64 ))

    static member Variable (value: DT<'T>, ?name: string) : DT<'T> = 
        DT<_> (value.Shape, fun ctxt -> 
                     let name2 = defaultArg name ""
                     match ctxt.Values.TryFind name2 with 
                     | None -> 
                         printfn "variable nodes not yet supported, and weight '%s' not found in Values, assuming constant" name2
                         //ctxt.Graph.Variable(value.Apply(ctxt),name=name2).Read
                         value.Apply(ctxt)
                     | Some t -> 
                         match t with 
                         | :? DT<'T> as vt -> vt.Apply(ctxt)
                         | _ -> 
                         printfn "incorrect type in values, got '%A' expected '%A', assuming variable node is constant" (t.GetType()) (typeof<DT<'T>>)
                         value.Apply(ctxt)
                         )

    static member Conv2D (input: DT<'T>, filters: DT<'T>, ?stride: int, ?padding: string) : DT<'T> = 
    //input: V[N,H,W,C], filters: V[F1;F2;C;COut]) -> output:V[N,H,W,COut] 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let filtersShape = filters.Shape
        let N, H, W, C = input.Shape.AsRank4()
        let F1, F2, C2, COut = filtersShape.AsRank4()
        Dim.Unify C C2
        let outputShape = Shape [| N; H/stride; W/stride; COut |]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Conv2D(input.Apply(ctxt), filters.Apply(ctxt),strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    // filter: 4-D with shape [filter_height, filter_width, in_channels, out_channels].
    // out_backprop: 4-D with shape [batch, out_height, out_width, out_channels]. Gradients w.r.t. the output of the convolution.
    // input_sizes: An integer vector representing the shape of input, where input is a 4-D [batch, in_height, in_width, in_channels] tensor.
    // Output: 4-D with shape [batch, in_height, in_width, in_channels]. Gradient w.r.t. the input of the convolution.

    static member Conv2DBackpropInput(filters: DT<'T>, out_backprop: DT<'T>, ?stride: int, ?padding: string) : DT<'T> = 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let N, out_height, out_width, out_channels = out_backprop.Shape.AsRank4()
        let _filter_height, _filter_width, in_channels, out_channels2 = filters.Shape.AsRank4()
        Dim.Unify out_channels out_channels2
        let input_shape = Shape [| N; out_height*stride; out_width*stride; in_channels |]
        DT<'T> (input_shape, fun ctxt -> 
           let input_sizes = ctxt.Graph.Const(input_shape.AsTFTensor())
           ctxt.Graph.Conv2DBackpropInput(input_sizes, filters.Apply(ctxt), out_backprop.Apply(ctxt), strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    /// Clips tensor values to a specified min and max.
    static member ClipByValue (input: DT<'T>, low: DT<'T>, high: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes (Shape.EquivShapes input.Shape low.Shape) high.Shape
        DT<'T> (outputShape, fun ctxt -> ctxt.Graph.ClipByValue(input.Apply(ctxt), low.Apply(ctxt), high.Apply(ctxt)))

    static member Moments(input: DT<'T>, ?axes: seq<int>) : DT<'T> * DT<'T> = 
        // Note: keep_dims = true
        let outputShape = input.Shape
        let compute (ctxt: Ctxt) = 
            match ctxt.MomentNodes.TryGetValue (input) with 
            | true, res -> res
            | _ -> 
                let axes = match axes with None -> None | Some v -> Some (ctxt.Graph.Const((shape v).AsTFTensor()))
                let res = ctxt.Graph.Moments(input.Apply(ctxt), ?axes=axes,keep_dims=true)
                ctxt.MomentNodes.[input] <- res
                res

        DT<'T> (outputShape, fun ctxt -> fst (compute ctxt)),
        DT<'T> (outputShape, fun ctxt -> snd (compute ctxt))

    static member Relu(input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        DT<'T> (outputShape, fun ctxt -> ctxt.Graph.Relu(input.Apply(ctxt)))

    static member DecodeJpeg(contents:DT<string>, ?channels: int) : DT<int> = // V[int,H,W,C]
        let channels = defaultArg channels 3 // CHECK ME
        let outputShape = Shape [| Dim.Inferred; Dim.Inferred; Dim channels |]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.DecodeJpeg(contents=contents.Apply(ctxt), channels=3L))

    static member Cast<'T, 'T2>(input: DT<'T>) : DT<'T2> = 
        let outputShape = input.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Cast(input.Apply(ctxt), TFDataType.FromType(typeof<'T2>)))

    static member WithScope(name: string) : IDisposable = 
        new WithScopeDisposable(name) :> _

    static member UsingWithScope (name: string) (f: unit -> DT<'T>) : DT<'T> = 
        let dt = f()
        DT<'T> (dt.Shape, fun ctxt -> use _scope = ctxt.Graph.WithScope(name) in dt.Apply(ctxt))

    static member CreateString(value: byte[]) : DT<string> = 
        let outputShape = shape [ 1 ]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Const(TFTensor.CreateString(value)))

    static member ExpandDims(value: DT<'T>, [<ParamArray>] dims: int[]) : DT<'T> = 
        let outputShape = shape dims
        DT<'T> (outputShape, fun ctxt -> ctxt.Graph.ExpandDims(value.Apply(ctxt), ctxt.Graph.Const(outputShape.AsTFTensor())))
    //TF.ExpandDims[Dim](value: V[shape]) : V[Expanded(Dim,shape)]

    static member Run(value: DT<'T>, ?weights: seq<string * DT>) : TFTensor = 
        let sess = new TFSession()
        let graph = sess.Graph
        let ctxt = 
            { Graph = graph
              MomentNodes = Dictionary(HashIdentity.Reference)
              Nodes = Dictionary(HashIdentity.Reference)
              Values = Map.ofSeq (defaultArg weights Seq.empty)}
        let node = value.Apply(ctxt)
        sess.Run([||],[||],[|node|]).[0]


type TensorFlow = ReflectedDefinitionAttribute

type TF = DT

/// tf { ... }  is a computational DSL (not yet using ReflectedDefinition) that does a layer of shape inference 
/// in the first runtime phase, and then the actual graph construction in the second runtime phase.   
type TFBuilder() =
    member x.Return(v: DT<'T>) = v
    
    /// Supports the use of `use _ = ...` in tf expressions
    member x.Using(v: IDisposable, f: (unit -> DT<'T>)) = 
        match v with 
        | :? WithScopeDisposable as w -> DT.UsingWithScope w.Name f
        | _ -> use x = v in f()

    //member x.Zero() = V()
    //member x.Run(v) = v

//type NumericLiteralG() =
//    member x.FromString(s: string) : V<double> = failwith "tbd"
    
[<AutoOpen>]
module TFHelpers = 
    let tf = TFBuilder()

    let shape (ints: int list) = Shape(Array.map Dim (Array.ofSeq ints))

    let v (d:double) : DT<double> = 
        let shape = Shape.Inferred 
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(d))))

    let vec (d:seq<double>) : DT<double> = 
        let d = Array.ofSeq d
        let shape = shape [ d.Length ]
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(d))))

    let matrix (d: seq< #seq<'T>>) : DT<'T> = 
        let d = array2D d 
        let shape = shape [ d.GetLength(0); d.GetLength(1)  ]
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(d))))

    let matrix3 (d: seq< #seq< #seq<'T>>>) : DT<'T> = 
        let d = d |> Array.ofSeq |> Array.map array2D
        let ds = Array3D.init d.Length (d.[0].GetLength(0)) (d.[0].GetLength(1)) (fun i j k -> d.[i].[j,k])
        let shape = shape [ d.Length; d.[0].GetLength(0); d.[0].GetLength(1)  ]
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(ds))))

    let matrix4 (d: seq< #seq< #seq< #seq<'T>>>>) : DT<'T> = 
        let d = d |> array2D |> Array2D.map array2D
        let r1,r2,r3,r4 = (d.GetLength(0), d.GetLength(1), d.[0,0].GetLength(0),d.[0,0].GetLength(1))
        let ds = Array4D.init r1 r2 r3 r4 (fun i j k m -> d.[i,j].[k,m])
        let sh = shape [ r1; r2; r3; r4 ]
        DT<_> (sh, (fun ctxt -> ctxt.Graph.Const(new TFTensor(ds))))

    let inline relu (x: ^T) : ^T = 
        (^T: (static member Relu : ^T -> ^T) (x))

    let variable value name = DT.Variable (value, name)
