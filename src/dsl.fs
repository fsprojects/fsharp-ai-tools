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
        match dim.TryValue() with 
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

    member dim.TryValue() = 
        match dim with 
        | DimMulInt (dim2,n) -> match dim2.TryValue() with None -> None | Some dimv -> Some (dimv*n) 
        | DimDivInt (dim2,n) -> match dim2.TryValue() with None -> None | Some dimv -> Some (dimv/n) 
        | Dim n -> Some n 
        | DimVar v -> 
            match v.Value with 
            | None -> None 
            | Some v -> v.TryValue()

    member dim.Value = 
        match dim.TryValue() with 
        | Some v -> v
        | None -> -1

    static member ( * ) (dim: Dim, stride: int) = if stride = 1 then dim else DimMulInt (dim, stride)

    static member ( / ) (dim: Dim, stride: int) = if stride = 1 then dim else DimDivInt (dim, stride)

    static member Inferred = DimVar (ref None)

    static member Unify (dim1: Dim) (dim2: Dim) = 
        match dim1.TryValue(), dim2.TryValue() with 
        | Some v1, Some v2 -> 
            if v1 <> v2 then 
                failwithf "mismatched dimension %d and %d" v1 v2 
        | _ -> 
        match dim1, dim2 with 
        // strip equations
        | DimVar { contents = Some soln1}, _ -> Dim.Unify soln1 dim2
        | _, DimVar { contents = Some soln2} -> Dim.Unify dim1 soln2
        // check for identical variables
        | DimVar ({ contents = None } as v1), DimVar ({ contents = None } as v2) when Object.ReferenceEquals(v1,v2) -> ()
        // solve
        | DimVar ({ contents = None } as v1), _ -> v1 := Some dim2
        | _, DimVar ({ contents = None } as v2) -> v2 := Some dim1
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
            match dim1.TryValue(), dim2.TryValue() with 
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
    static member D with get() = (Shape [| Dim 1 |])
    static member DV with get() = (Shape [| Dim.Inferred |])
    static member DM with get() = (Shape [| Dim.Inferred; Dim.Inferred |])

    static member Unify (shape1: Shape) (shape2: Shape) = 
        match shape1, shape2 with 
        | ShapeVar { contents = Some soln1}, _ -> Shape.Unify soln1 shape2
        | _, ShapeVar { contents = Some soln2} -> Shape.Unify shape1 soln2
        | ShapeVar ({ contents = None } as v1), ShapeVar ({ contents = None } as v2) when Object.ReferenceEquals(v1,v2) -> ()
        | ShapeVar ({ contents = None } as v1), _ -> v1 := Some shape2
        | _, ShapeVar ({ contents = None } as v2) -> v2 := Some shape1
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
      AddGradientNodes: Dictionary<DT * DT * DT option, TFOutput> // uses reference identity to ensure unique ValueAndDiff nodes from unique DT values
      Values: Map<string,DT> }

type internal WithScopeDisposable(name:string) = 
    interface IDisposable with 
        member __.Dispose() = ()
    member __.Name = name        

/// Represents a differentiable tensor value
type DT<'T> internal (shape: Shape, eval: (Ctxt -> TFOutput)) =
    inherit DT(shape)

    static let memoize (dict: Dictionary<_,_>) key f = 
        match dict.TryGetValue (key) with 
        | true, res -> res
        | _ -> 
            let res = f ()
            dict.[key] <- res
            res

    member internal dt.Apply(ctxt: Ctxt) = 
        memoize ctxt.Nodes (upcast dt) (fun () -> eval ctxt)

    static member AddN (vs: DT<'T>[]) : DT<'T> = 
        let outputShape = vs.[0].Shape 
        for v in vs do Shape.Unify outputShape v.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.AddN(vs |> Array.map (fun v -> v.Apply ctxt), int64 vs.Length))

    // TODO: accept axis argument for stacking, take this into account in inferred shape
    // TODO: this gives wrong shape when all inputs have size 1. Ugh
    static member Stack (vs: DT<'T>[]) : DT<'T> = 
        let inputShape = vs.[0].Shape
        for v in vs do Shape.Unify inputShape v.Shape
        let outputShape = Shape [| yield Dim vs.Length; yield! inputShape.Dimensions |]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Stack(vs |> Array.map (fun v -> v.Apply ctxt)))

    static member (+) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Add(v1.Apply ctxt, v2.Apply ctxt))

    static member (-) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Sub(v1.Apply ctxt, v2.Apply ctxt))

    static member ( * ) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Mul(v1.Apply ctxt, v2.Apply ctxt))

    static member (/) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Div(v1.Apply ctxt, v2.Apply ctxt))

    static member internal ReduceOp keep_dims (v: DT<'T>) f : DT<'T> = 
        let outputShape = if keep_dims = Some true then v.Shape else Shape.D
        DT<_> (outputShape, fun ctxt -> f ctxt (v.Apply ctxt))

    static member ReduceSum (v: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims v (fun ctxt vnode -> ctxt.Graph.ReduceSum(vnode, ?keep_dims=keep_dims))

    static member ReduceMean (v: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims v (fun ctxt vnode -> ctxt.Graph.ReduceMean(vnode, ?keep_dims=keep_dims))

    static member ReduceProd (v: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims v (fun ctxt vnode -> ctxt.Graph.ReduceProd(vnode, ?keep_dims=keep_dims))

    static member ReduceMin (v: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        let outputShape = if keep_dims = Some true then v.Shape else Shape.D
        DT<_> (outputShape, fun ctxt -> 
           let vnode = v.Apply ctxt
           ctxt.Graph.Min(vnode, ctxt.Graph.ReduceDims(vnode), ?keep_dims=keep_dims))

    static member ReduceMax (v: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        let outputShape = if keep_dims = Some true then v.Shape else Shape.D
        DT<_> (outputShape, fun ctxt -> 
           let vnode = v.Apply ctxt
           ctxt.Graph.Max(vnode, ctxt.Graph.ReduceDims(vnode), ?keep_dims=keep_dims))

    static member ReduceNorm (v: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        DT.Sqrt(DT.ReduceSum(v*v, ?keep_dims= keep_dims))

    // TODO : generalize beyond vectors
    member v.Item 
        with get (n: int) : DT<'T> = 
            Shape.Unify v.Shape Shape.DV
            let outputShape = Shape.D
            DT<'T>(outputShape, fun ctxt -> 
               let vnode = v.Apply ctxt
               let graph = ctxt.Graph
               graph.Squeeze(graph.Slice(vnode, graph.Const(new TFTensor( [| n |])),graph.Const(new TFTensor( [| 1 |]))), [| 0L |])) // y = xv1

    // TODO : generalize beyond vectors
    member v.Item 
        with get (n1: int, n2: int) : DT<'T> = 
            Shape.Unify v.Shape Shape.DV
            let outputShape = Shape.D
            DT<'T>(outputShape, fun ctxt -> 
               let vnode = v.Apply ctxt
               let graph = ctxt.Graph
               graph.Squeeze(graph.Slice(vnode, graph.Const(new TFTensor( [| n1; n2 |])),graph.Const(new TFTensor( [| 1; 1 |]))), [| 0L; 1L |])) // y = xv1

    // TODO : generalize beyond vectors
    member v.GetSlice(startIndex: int option, endIndex: int option) =
        Shape.Unify v.Shape Shape.DV
        // TODO attach a constraint to the dimension that the endIndex is in-bounds
        let startIndex = defaultArg startIndex 0
        if endIndex.IsNone then failwith "end index must be specified"
        let len = endIndex.Value - startIndex + 1
        let outputShape = Shape [| Dim len |]
        DT<'T>(outputShape, fun ctxt -> 
            let vnode = v.Apply ctxt
            let graph = ctxt.Graph
            graph.Slice(vnode, graph.Const(new TFTensor( [| startIndex |])),graph.Const(new TFTensor( [| len |])))) // y = xv1

    static member Sqrt (v: DT<'T>) : DT<'T> = 
        DT<'T>(v.Shape, fun ctxt -> ctxt.Graph.Sqrt(v.Apply ctxt))

    static member ReverseV2 (v: DT<'T>) : DT<'T> = 
        DT<'T>(v.Shape, fun ctxt -> ctxt.Graph.ReverseV2(v.Apply ctxt, ctxt.Graph.Const (new TFTensor( [| 0 |]))))

    static member DiagPart (v: DT<'T>) : DT<'T> = 
        let dims = v.Shape.Dimensions
        let n = dims.Length
        if n % 2 <> 0 then invalidArg "DiagPart: v" "expected a tensor with even rank"
        for i in 0 .. n - 1 do 
            Dim.Unify dims.[i] dims.[n/2 + i]
        let outputShape = Shape (dims.[0 .. n/2 - 1 ])
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.DiagPart(v.Apply ctxt))

    static member Tanh (v: DT<'T>) : DT<'T> = 
        DT<'T>(v.Shape, fun ctxt -> ctxt.Graph.Tanh(v.Apply ctxt))

    static member Tan (v: DT<'T>) : DT<'T> =  
        DT<'T>(v.Shape, fun ctxt -> ctxt.Graph.Tan(v.Apply ctxt))

    override dt.ToString() = dt.Shape.ToString()

    /// Add partial deriviatives of 1-dim value 
    static member AddGradients (y: (* D *) DT<'T>, (* D, DV, DM, ...  *) x: DT<'T>, (* D *) ?dy: DT<'T>) : DT<'T> =  
        Shape.Unify y.Shape Shape.D
        let outputShape = x.Shape
        DT<'T>(outputShape, fun ctxt -> 
            memoize ctxt.AddGradientNodes ((upcast y,upcast x, (match dy with None -> None | Some z -> Some (upcast z)))) (fun () -> 
                let xnode = x.Apply ctxt
                let ynode = y.Apply ctxt
                let dynodesIn = match dy with None -> None | Some z -> Some [| z.Apply ctxt |]
                let dynodes = ctxt.Graph.AddGradients([| ynode |], [| xnode |], ?dy=dynodesIn)
                let dynode = dynodes.[0]
                dynode))

(* 

        DT<'T>(outputShape, fun ctxt -> fst (compute ctxt)),
        DT<'T>(outputShape, fun ctxt -> snd (compute ctxt))

    let xv1 = 3.0
    let xv2 = 4.0
    let graph = new TFGraph()
    let x1 = graph.Const (new TFTensor( [| xv1 |]))
    let x2 = graph.Const (new TFTensor( [| xv2 |]))

    let y1 = graph.Add(graph.Mul(graph.Square x1, x2), x2) // y1 = x1^2 * x2 + x2
    let y2 = graph.Add(graph.Mul(graph.Square x2, x1), x1) // y2 = x2^2 * x1 + x1
    // first element differentiates by first variable, second element by second variable
    let g1 = graph.AddGradients ([| y1 |], [| x1; x2 |]) // [dy1/dx1; dy1/dx2] 
    let g2 = graph.AddGradients ([| y2 |], [| x1; x2 |]) // [dy2/dx1; dy2/dx2] 

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y1; yield y2; yield! g1; yield! g2 |])
    let primal1 = results.[0].GetValue() :?> double[] |> Array.item 0
    let primal2 = results.[1].GetValue() :?> double[] |> Array.item 0
    let dy1_dx1 = results.[2].GetValue() :?> double[] |> Array.item 0
    let dy1_dx2 = results.[3].GetValue() :?> double[] |> Array.item 0
    let dy2_dx1 = results.[4].GetValue() :?> double[] |> Array.item 0
    let dy2_dx2 = results.[5].GetValue() :?> double[] |> Array.item 0
    if primal1 <> xv1 ** 2.0 * xv2 + xv2 then failwith "fail"
    if primal2 <> xv2 ** 2.0 * xv1 + xv1 then failwith "fail"
    if dy1_dx1 <> 2.0 * xv1 * xv2 then failwith "fail"
    if dy1_dx2 <> xv1 ** 2.0 + 1.0 then failwith "fail"
    if dy2_dx1 <> xv2 ** 2.0 + 1.0 then failwith "fail"
    if dy2_dx2 <> 2.0 * xv2 * xv1 then failwith "fail"
*)

    static member Const (value: double, ?shape: Shape) : DT<'T> = 
        let shape = shape |> Option.defaultWith (fun () -> Shape.Inferred)
        DT<'T>(shape, fun ctxt -> ctxt.Graph.Reshape(ctxt.Graph.Const(new TFTensor(value)), ctxt.Graph.Const(shape.AsTFTensor())))

    static member ConstArray (value: 'T[], ?shape: Shape) : DT<'T> = 
        let shape = shape |> Option.defaultWith (fun () -> Shape.Inferred)
        DT<'T>(shape, fun ctxt -> ctxt.Graph.Reshape(ctxt.Graph.Const(new TFTensor(value)), ctxt.Graph.Const(shape.AsTFTensor())))

    static member TruncatedNormal (shape: Shape) : DT<double> = 
        DT<double> (shape, fun ctxt -> ctxt.Graph.TruncatedNormal(ctxt.Graph.Const(shape.AsTFTensor()), TFDataType.Float64 ))

    static member Variable (value: DT<'T>, ?name: string) : DT<'T> = 
        DT<'T>(value.Shape, fun ctxt -> 
                     let name2 = defaultArg name ""
                     match ctxt.Values.TryFind name2 with 
                     | None -> 
                         printfn "variable nodes not yet supported, and weight '%s' not found in Values, assuming constant" name2
                         //ctxt.Graph.Variable(value.Apply ctxt,name=name2).Read
                         value.Apply ctxt
                     | Some t -> 
                         match t with 
                         | :? DT<'T> as vt -> vt.Apply ctxt
                         | _ -> 
                         printfn "incorrect type in values, got '%A' expected '%A', assuming variable node is constant" (t.GetType()) (typeof<DT<'T>>)
                         value.Apply ctxt
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
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Conv2D(input.Apply ctxt, filters.Apply ctxt,strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

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
        DT<'T>(input_shape, fun ctxt -> 
           let input_sizes = ctxt.Graph.Const(input_shape.AsTFTensor())
           ctxt.Graph.Conv2DBackpropInput(input_sizes, filters.Apply ctxt, out_backprop.Apply ctxt, strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    /// Clips tensor values to a specified min and max.
    static member ClipByValue (input: DT<'T>, low: DT<'T>, high: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes (Shape.EquivShapes input.Shape low.Shape) high.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.ClipByValue(input.Apply ctxt, low.Apply ctxt, high.Apply ctxt))

    static member Moments(input: DT<'T>, ?axes: seq<int>) : DT<'T> * DT<'T> = 
        // Note: keep_dims = true
        let outputShape = input.Shape
        let compute (ctxt: Ctxt) = 
            memoize ctxt.MomentNodes (upcast input) (fun () -> 
                let axes = match axes with None -> None | Some v -> Some (ctxt.Graph.Const((shape v).AsTFTensor()))
                ctxt.Graph.Moments(input.Apply ctxt, ?axes=axes,keep_dims=true))

        DT<'T>(outputShape, fun ctxt -> fst (compute ctxt)),
        DT<'T>(outputShape, fun ctxt -> snd (compute ctxt))

    static member Relu(input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Relu(input.Apply ctxt))

    static member DecodeJpeg(contents:DT<string>, ?channels: int) : DT<int> = // V[int,H,W,C]
        let channels = defaultArg channels 3 // CHECK ME
        let outputShape = Shape [| Dim.Inferred; Dim.Inferred; Dim channels |]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.DecodeJpeg(contents=contents.Apply ctxt, channels=3L))

    static member Cast<'T, 'T2>(input: DT<'T>) : DT<'T2> = 
        let outputShape = input.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Cast(input.Apply ctxt, TFDataType.FromType(typeof<'T2>)))

    static member WithScope(name: string) : IDisposable = 
        new WithScopeDisposable(name) :> _

    static member UsingWithScope (name: string) (f: unit -> DT<'T>) : DT<'T> = 
        let dt = f()
        DT<'T>(dt.Shape, fun ctxt -> use _scope = ctxt.Graph.WithScope(name) in dt.Apply ctxt)

    static member CreateString(value: byte[]) : DT<string> = 
        let outputShape = shape [ 1 ]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Const(TFTensor.CreateString(value)))

    static member ExpandDims(value: DT<'T>, [<ParamArray>] dims: int[]) : DT<'T> = 
        let outputShape = shape dims
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.ExpandDims(value.Apply ctxt, ctxt.Graph.Const(outputShape.AsTFTensor())))
    //TF.ExpandDims[Dim](value: V[shape]) : V[Expanded(Dim,shape)]

    static member RunTFTensor(value: DT<'T>, ?weights: seq<string * DT>) : TFTensor = 
        let sess = new TFSession()
        let graph = sess.Graph
        let ctxt = 
            { Graph = graph
              MomentNodes = Dictionary(HashIdentity.Reference)
              AddGradientNodes = Dictionary(HashIdentity.Structural)
              Nodes = Dictionary(HashIdentity.Reference)
              Values = Map.ofSeq (defaultArg weights Seq.empty)}
        let node = value.Apply ctxt
        sess.Run([||],[||],[|node|]).[0]

    static member RunScalar(value: DT<'T>, ?weights: seq<string * DT>) : 'T = 
        DT.RunTFTensor(value, ?weights=weights).GetValue() :?> 'T

    static member RunArray(value: DT<'T>, ?weights: seq<string * DT>) : 'T[] = 
        DT.RunTFTensor(value, ?weights=weights).GetValue() :?> 'T[]

    static member RunArray2D(value: DT<'T>, ?weights: seq<string * DT>) : 'T[,] = 
        DT.RunTFTensor(value, ?weights=weights).GetValue() :?> 'T[,]

    static member RunArray3D(value: DT<'T>, ?weights: seq<string * DT>) : 'T[,,] = 
        DT.RunTFTensor(value, ?weights=weights).GetValue() :?> 'T[,,]

    static member RunArray4D(value: DT<'T>, ?weights: seq<string * DT>) : 'T[,,,] = 
        DT.RunTFTensor(value, ?weights=weights).GetValue() :?> 'T[,,,]

    static member Run(value: DT<'T>, ?weights: seq<string * DT>) : obj = 
        DT.RunTFTensor(value, ?weights=weights).GetValue() 


/// Forward and reverse differentiation operations module (automatically opened)
module DT =

    /// Alias for a differentiable scalar.
    type D<'T> = DT<'T>

    /// Alias for a differentiable vector.
    type DV<'T> = DT<'T>

    /// Alias for a differentiable matrix
    type DM<'T> = DT<'T>

    /// First derivative of a tensor-to-scalar function `f`, at point `x`. Forward AD.
    let gradient (y: D<'T>) (x: DT<'T>) = 
        DT.AddGradients (y, x)

    /// Original value and first derivative of a tensor-to-scalar function `f`, at point `x`. Forward AD.
    let evalAndDiff (f: DT<'T> -> D<'T>) (x: DT<'T>) = 
        let y = f x
        y, gradient y x

    /// First derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diff (f: D<'T> -> D<'T>) x = evalAndDiff f x |> snd

    /// Second derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diff2 (f: D<'T> -> D<'T>) x  : D<'T> =
        diff (diff f) x

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let evalAndDiffAndDiff2 (f: D<'T> -> D<'T>) x : D<'T> * D<'T> * D<'T> =
        let v, d = evalAndDiff f x
        let d2 = diff2 f x
        (v, d, d2)

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diffAndDiff2 (f: D<'T> -> D<'T>) x  : D<'T> * D<'T> =
        evalAndDiffAndDiff2 f x |> (fun (a,_,c) -> a,c)

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let diffN n (f: D<'T> -> D<'T>) x  : D<'T> =
        if n < 0 then invalidArg "n" "must be positive"
        elif n = 0 then f x
        else
            let rec d n f =
                match n with
                | 1 -> diff f
                | _ -> d (n - 1) (diff f)
            x |> d n f

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`, at point `x`. Forward AD.
    let evalAndDiffN n (f: D<'T> -> D<'T>) x  : D<'T> * D<'T> =
        (x |> f, diffN n f x)

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let evalAndGradient (f: DV<'T> -> D<'T>) (x: DV<'T>) : D<'T> * DV<'T> = 
        Shape.Unify x.Shape Shape.DV
        let y = f x
        let dy = gradient y x
        y, dy

    /// Gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let grad (f: DV<'T> -> D<'T>) x : DV<'T> =
        evalAndGradient f x |> snd

(*
    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`. Forward AD.
    let gradv' (f: DV<'T> -> D<'T>) x (v: DV<'T>) : D<'T> * D<'T> =
        let yv = f v
        let y = f x
        let dyv = DT.AddGradients (y, x, yv)
        y, dyv

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`. Forward AD.
    let gradv (f: DV<'T> -> D<'T>) x v : D<'T> =
        gradv' f x v |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Forward AD.
    let jacobianv' (f: DV<'T> -> DV<'T>) (x: DV<'T>) (v: DV<'T>) : DV<'T> * DV<'T> =
        Shape.Unify x.Shape Shape.DV
        Shape.Unify v.Shape Shape.DV
        let yv = f v
        let y = f x
        let ysize = 
            match y.Shape.Dimensions.[0].TryValue() with 
            | None -> failwith "unknown vector output size in jacobian"
            | Some d -> d
        let dyv = DT.Stack [| for i in 0 .. ysize-1 -> DT.AddGradients (y.[i], x, yv.[i]) |]
        y, dyv

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`. Forward AD.
    let jacobianv (f: DV<'T> -> DV<'T>) x v : DV<'T> =
        jacobianv' f x v |> snd
*)
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let evalAndJacobian (f: DV<'T> -> DV<'T>) (x:DV<'T>) : DV<'T> * DM<'T> =
        let y = f x
        let ysize = 
            match y.Shape.Dimensions.[0].TryValue() with 
            | None -> failwith "unknown vector output size in jacobian"
            | Some d -> d
        let jydx = DT.Stack [| for i in 0 .. ysize - 1 -> DT.AddGradients (y.[i], x) |]
        y, jydx

    /// Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let jacobian (f: DV<'T> -> DV<'T>) x : DM<'T> =
        evalAndJacobian f x |> snd

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let gradAndHessian (f: DV<'T> -> D<'T>) x : DV<'T> * DM<'T> =
        evalAndJacobian (grad f) x

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let evalAndGradAndHessian (f: DV<'T> -> D<'T>) x : D<'T> * DV<'T> * DM<'T> =
        let g, h = gradAndHessian f x
        (f x, g, h)

    /// Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let hessian (f: DV<'T> -> D<'T>) x : DM<'T> =
        jacobian (grad f) x

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let evalAndHessian (f: DV<'T> -> D<'T>) x : D<'T> * DM<'T> =
        (x |> f, hessian f x)

(*
    /// Original value, gradient-vector product (directional derivative), and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let gradAndHessianv' (f: DV<'T> -> D<'T>) x v =
        let gv, hv = evalAndGradient (fun xx -> gradv f xx v) x
        (x |> f, gv, hv)

    /// Gradient-vector product (directional derivative) and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let gradAndHessianv (f: DV<'T> -> D<'T>) x v : D<'T> * DV<'T> =
        gradAndHessianv' f x v |> (fun (_,b,c) -> b,c)

    /// Original value and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let hessianv' (f: DV<'T> -> D<'T>) x v =
        gradAndHessianv' f x v |> (fun (a,_,c) -> a,c)

    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let hessianv (f: DV<'T> -> D<'T>) x v : DV<'T> =
        hessianv' f x v |> snd
*)

    let trace v = DT.ReduceSum (DT.DiagPart v)

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let evalAndLaplacian (f: DV<'T> -> D<'T>) x : D<'T> * D<'T> = // TODO: reimplement faster
        let v, h = evalAndHessian f x
        (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let laplacian (f: DV<'T> -> D<'T>) x : D<'T> =
        evalAndLaplacian f x |> snd

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix. Forward AD.
    let evalAndCurl (f: DV<'T> -> DV<'T>) x =
        let v, j = evalAndJacobian f x
        //if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        v, DT.Stack [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix. Forward AD.
    let curl (f: DV<'T> -> DV<'T>) x : DV<'T> =
        evalAndCurl f x |> snd

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix. Forward AD.
    let evalAndDivergenceergence (f: DV<'T> -> DV<'T>) x =
        let v, j = evalAndJacobian f x
        //if j.Rows <> j.Cols then ErrorMessages.InvalidArgDiv()
        v, DT.ReduceSum (DT.DiagPart j)

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix. Forward AD.
    let divergence (f: DV<'T> -> DV<'T>) x : D<'T> =
        evalAndDivergenceergence f x |> snd

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix. Forward AD.
    let evalAndCurlAndDivergence (f: DV<'T> -> DV<'T>) x =
        let v, j = evalAndJacobian f x
        //if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        v, DT.Stack [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix. Forward AD.
    let curlAndDivergence (f: DV<'T> -> DV<'T>) x : DV<'T> * D<'T> =
        evalAndCurlAndDivergence f x |> (fun (_,b,c) -> b,c)

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

    let reduceSum (x: DT<'T>) : DT<'T> = DT.ReduceSum(x)
    let reduceProd (x: DT<'T>) : DT<'T> = DT.ReduceProd(x)
    let reduceMean (x: DT<'T>) : DT<'T> = DT.ReduceMean(x)
    let reduceMax (x: DT<'T>) : DT<'T> = DT.ReduceMax(x)
    let reduceMin (x: DT<'T>) : DT<'T> = DT.ReduceMin(x)
    let reduceNorm (x: DT<'T>) : DT<'T> = DT.ReduceNorm(x)

    let variable value name = DT.Variable (value, name)
