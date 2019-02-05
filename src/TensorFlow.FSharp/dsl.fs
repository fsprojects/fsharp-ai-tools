namespace TensorFlow.FSharp.DSL

open System
open System.Collections.Generic
open TensorFlow.FSharp

[<AutoOpen>]
module LiveChecking = 
    let livecheck = 
        try 
            match System.Environment.GetEnvironmentVariable("LIVECHECK") with null | "0" -> false | _ -> true 
        with _ -> false

type internal InferenceVarSoln<'T> =
    | Solved of 'T
    | Unsolved

type internal InferenceVar<'T>() = 
    let mutable solution : InferenceVarSoln<'T> = Unsolved
    
    member __.IsSolved = match solution with Solved _ -> true | Unsolved -> false
    member __.Solve sln = solution <- Solved sln
    member __.Solution = solution

/// Represents an inferred dimension
type Dim =
    internal
    /// One dimension is a multiple of another
    | DimMulInt of Dim * int

    /// One dimension is a divisor of another
    | DimDivInt of Dim * int

    /// The dimension is a variable, possibly solved
    | DimVar of InferenceVar<Dim>

    /// The dimension is known
    | DimKnown of int

    override dim.ToString() = 
        match dim.TryValue() with 
        | Some v -> string v
        | None ->  
        match dim with 
        | DimMulInt (expected, n) -> expected.ToString() + "*" + string n 
        | DimDivInt (expected, n) -> expected.ToString() + "/" + string n 
        | DimKnown n -> string n 
        | DimVar v -> 
            match v.Solution with 
            | Unsolved -> "?" 
            | Solved v -> v.ToString()

    member internal dim.StripFlex() = 
        match dim with 
        | DimVar v -> 
            match v.Solution with 
            | Unsolved -> dim
            | Solved v -> v.StripFlex()
        | _ -> dim

    member dim.TryValue() = 
        match dim with 
        | DimMulInt (expected,n) -> match expected.TryValue() with None -> None | Some dimv -> Some (dimv*n) 
        | DimDivInt (expected,n) -> match expected.TryValue() with None -> None | Some dimv -> Some (dimv/n) 
        | DimKnown n -> Some n 
        | DimVar v -> 
            match v.Solution with 
            | Unsolved -> None 
            | Solved v -> v.TryValue()

    member dim.Value = 
        match dim.TryValue() with 
        | Some v -> v
        | None -> -1

    static member ( * ) (dim: Dim, stride: int) = if stride = 1 then dim else DimMulInt (dim, stride)

    static member ( / ) (dim: Dim, stride: int) = if stride = 1 then dim else DimDivInt (dim, stride)

    static member Known n = DimKnown n

    static member Inferred = DimVar (InferenceVar())

    static member Unify op (actual: Dim) (expected: Dim) = 
        match Dim.UnifyInner op actual expected with
        | Ok () -> ()
        | Error msg -> failwithf "mismatched dimensions: expected '%s' but got '%s' for operator %s (%s)" (expected.ToString())  (actual.ToString()) op msg

    static member UnifyInner op (actual: Dim) (expected: Dim) = 
        match actual.TryValue(), expected.TryValue() with 
        | Some v1, Some v2 -> if v1 <> v2 then Error "unequal values" else Ok()
        | _ -> 
        match actual.StripFlex(), expected.StripFlex() with 
        // check for identical variables
        | DimVar v1, DimVar v2 when Object.ReferenceEquals(v1,v2) -> Ok ()
        // solve
        | DimVar v1, _ -> v1.Solve expected; Ok()
        | _, DimVar v2 -> v2.Solve actual; Ok()
        | DimKnown d1, DimKnown d2 -> failwith "unreachable - each dimension had value"
        | DimMulInt (d1, n1), DimKnown d2 -> 
            if d2 % n1 <> 0 then 
                Error "not divisible"
            else
                Dim.UnifyInner op d1 (DimKnown (d2 / n1))
        | DimKnown d1, DimMulInt (d2, n2) -> 
            if d1 % n2 <> 0 then 
                Error "not divisible"
            else
                Dim.UnifyInner op (DimKnown (d1 / n2)) d2
        | DimMulInt (d1, n1), DimMulInt (d2, n2) -> 
            if n1 <> n2 then 
                Error "different multipliers"
            else
                Dim.UnifyInner op d1 d2
        | DimDivInt (d1, n1), DimDivInt (d2, n2) -> 
            if n1 <> n2 then 
                Error "different multipliers"
            else
                Dim.UnifyInner op d1 d2
        | _ -> 
            match actual.TryValue(), expected.TryValue() with 
            | None, _ | _, None -> Error "incomplete dimension"
            | _ -> Ok () // equal, see above

/// Represents an inferred shape
type Shape =
    internal
    /// Represents a shape with possible flexibile variable + possible solution
    | ShapeImpl of Dim[] * InferenceVar<Shape> option

    member shape.Item 
        with get idx = 
            match shape with 
            | ShapeImpl (dims, flexopt) -> 
                if idx < dims.Length then 
                    dims.[idx]
                else
                    match flexopt with 
                    | None -> failwithf "index %d out of bounds" idx
                    | Some v ->
                    match v.Solution with 
                    | Unsolved -> Dim.Inferred
                    | Solved sln -> sln.[idx - dims.Length ]

    member shape.AsRank1() = shape.[0]

    member shape.AsRank2() = shape.[0], shape.[1]

    member shape.AsRank3() = shape.[0], shape.[1], shape.[2]

    member shape.AsRank4() = shape.[0], shape.[1], shape.[2], shape.[3] 

    member internal shape.DimensionsWithFlexVar = 
        match shape with 
        | ShapeImpl (dims, None) -> dims, None
        | ShapeImpl (dims, Some v) -> 
            match v.Solution with 
            | Unsolved -> dims, Some v
            | Solved sln -> let dims2, flex = sln.DimensionsWithFlexVar in Array.append dims dims2, flex

    member internal shape.DimensionsEliminatingFlex = 
        let dims, flexvar = shape.DimensionsWithFlexVar
        match flexvar with 
        | None -> ()
        | Some v -> v.Solve (ShapeImpl ([| |], None))
        dims 

    member shape.Dimensions = shape.DimensionsEliminatingFlex

    override shape.ToString() = 
        let dims, flexvar = shape.DimensionsWithFlexVar
        sprintf "shape %s" (String.concat " x " [ for i in dims -> i.ToString() ]) 
        + (if flexvar.IsSome then "x.." else "")

    member shape.AsTFShape() = TFShape(shape.DimensionsEliminatingFlex |> Array.map (fun dim -> int64 dim.Value))

    member shape.AsTFTensor() = shape.AsTFShape().AsTensor()

    static member Known dims = ShapeImpl(dims, None)

    static member Flex dims = ShapeImpl(dims, Some (InferenceVar()))

    static member Inferred with get() = Shape.Flex [| |]
    
    static member D with get() = Shape.Known [| DimKnown 1 |]
    
    static member DV with get() = Shape.Known [| Dim.Inferred |]
    
    static member DM with get() = Shape.Known [| Dim.Inferred; Dim.Inferred |]

    static member internal FlexN n = Shape.Flex [| for i in 1 .. n -> Dim.Inferred |]

    static member internal MinDimensions op (shape: Shape) dim = 
        let dims, flexvar = shape.DimensionsWithFlexVar
        if dim > dims.Length then 
            match flexvar with 
            | None -> 
                failwithf "shape %A must have at least %d dimensions for operator %s" shape dim op
            | Some v -> 
                v.Solve (Shape.FlexN dim)

    // TODO: update this for row inference
    static member internal  Unify op (actual: Shape) (expected: Shape) = 

        let rec loop (s1: Shape) (s2: Shape) =

            let dims1, flexvar1 = s1.DimensionsWithFlexVar
            let dims2, flexvar2 = s2.DimensionsWithFlexVar

            // Unify those in common - note relies on Seq.iter2 only iterating up to equal length
            (dims1, dims2) ||> Seq.iter2 (fun dim1 dim2 ->
                match Dim.UnifyInner op dim1 dim2 with 
                | Ok () -> ()
                | Error msg -> failwithf "mismatched shapes: expected %A but got %A for operator %s (expected dimension %s but got %s - %s) " expected actual op (dim2.ToString()) (dim1.ToString()) msg
             )

            let n = min dims1.Length dims2.Length
            if n > 0 then
                // Drop front dimensions - shapes smaller
                loop (ShapeImpl(dims1.[n..], flexvar1)) (ShapeImpl(dims2.[n..], flexvar2))

            elif dims1.Length > 0 then
                assert (dims2.Length = 0)
                match flexvar2 with 
                | Some v2 -> 
                    v2.Solve (Shape.FlexN dims1.Length) 
                    // expected now expanded and will have 'n' in common
                    loop s1 s2 
                | None -> 
                    Error ()

            elif dims2.Length > 0 then
                assert (dims1.Length = 0)
                match flexvar1 with 
                | Some v1 -> 
                    v1.Solve (Shape.FlexN dims2.Length) 
                    // actual now expanded and will have 'n' in common
                    loop s1 s2 
                | None -> 
                    Error ()

            else

                match flexvar1, flexvar2 with 
                | Some v1, Some v2 when Object.ReferenceEquals(v1,v2) -> Ok ()
                | Some v1, _ -> v1.Solve (ShapeImpl([| |], flexvar2)); Ok()
                | _, Some v2 -> v2.Solve (ShapeImpl([| |], flexvar1)); Ok()
                | None, None -> Ok()

        match loop actual expected with 
        | Ok () -> ()
        | Error () -> failwithf "mismatched shapes: expected %A but got %A for operator %s" expected actual op

    static member internal EquivShapes op (actual: Shape) (expected: Shape) = 
        Shape.Unify op actual expected
        actual

[<AutoOpen>]
module ShapeHelpers = 

    /// Create a non-inferred shape
    let shape (ints: seq<int>) = 
        ints |> Array.ofSeq |> Array.map (fun n -> if n = -1 then Dim.Inferred else DimKnown n) |> Shape.Known

    let memoize (dict: Dictionary<_,_>) key f = 
        match dict.TryGetValue (key) with 
        | true, res -> res
        | _ -> 
            let res = f ()
            dict.[key] <- res
            res

/// Represents a context for turning differentiable tensors into a TensorFlow graph
type internal Ctxt = 
    { Graph: TFGraph 
      Nodes: Dictionary<DT, TFOutput> // ensure unique nodes from unique DT values
      MomentNodes: Dictionary<DT, TFOutput * TFOutput> // ensure unique nodes from unique DT values
      AddGradientNodes: Dictionary<DT * DT[] * DT option, TFOutput[]> // ensure unique nodes from unique DT values
      Values: Map<string,DT> }

/// Represents a differentiable tensor value, which later corresponds to a node in a TensorFlow graph
and DT internal (shape: Shape, eval: (Ctxt -> TFOutput)) = 

    member internal dt.Apply(ctxt: Ctxt) = 
        memoize ctxt.Nodes dt (fun () -> eval ctxt)

    /// Get the inferred shape of the differentiable tensor 
    member __.Shape = shape

type internal WithScopeDisposable(name:string) = 
    interface IDisposable with 
        member __.Dispose() = ()
    member __.Name = name        



/// Represents a differentiable tensor value
type DT<'T> internal (shape: Shape, eval: (Ctxt -> TFOutput)) =
    inherit DT(shape, eval)

    static member AddN (vs: DT<'T>[]) : DT<'T> = 
        let outputShape = vs.[0].Shape 
        for v in vs do Shape.Unify "AddN" outputShape v.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.AddN(vs |> Array.map (fun v -> v.Apply ctxt)))

    static member (+) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "(+)" v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Add(v1.Apply ctxt, v2.Apply ctxt))

    static member (-) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "(-)" v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Sub(v1.Apply ctxt, v2.Apply ctxt))

    /// Pointwise multiplication
    static member ( * ) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "(*)" v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Mul(v1.Apply ctxt, v2.Apply ctxt))

    /// Pointwise multiplication
    static member ( ~- ) (v1: DT<'T>) : DT<'T> = 
        let outputShape = v1.Shape 
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Neg (v1.Apply ctxt))

    static member ( *! ) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let n1,m,n2 = Dim.Inferred, Dim.Inferred, Dim.Inferred 
        Shape.Unify "MatMul"  v1.Shape (Shape.Known [| n1; m |])
        Shape.Unify "MatMul" v2.Shape (Shape.Known [| m; n2 |])
        let outputShape = Shape.Known [| n1; n2 |]
        DT<'T> (outputShape, fun ctxt -> ctxt.Graph.MatMul(v1.Apply ctxt, v2.Apply ctxt))

    static member (/) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "(/)" v1.Shape v2.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Div(v1.Apply ctxt, v2.Apply ctxt))

    static member internal ReduceOp keep_dims (axis: int[] option) (v: DT<'T>) f : DT<'T> = 
        let outputShape = 
            match keep_dims, axis with
            | Some true, _ -> v.Shape 
            | _, None -> Shape.D
            | _, Some axis -> 
                // TODO: flex here
                let inputDims = v.Shape.DimensionsEliminatingFlex
                let outputDims = inputDims |> Array.indexed |> Array.filter (fun (idx, _) -> not (Array.contains idx axis)) |> Array.map snd
                if outputDims.Length = 0 then Shape.D else Shape.Known outputDims

        DT<_> (outputShape, fun ctxt -> 
            let axis = axis |> Option.map (fun axis -> ctxt.Graph.Const(new TFTensor(axis)))
            f ctxt axis (v.Apply ctxt))

    static member Sum (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v 
            (fun ctxt axis vnode -> ctxt.Graph.ReduceSum(vnode, ?axis=axis, ?keep_dims=keep_dims))

    static member Mean (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v
            (fun ctxt axis vnode -> ctxt.Graph.ReduceMean(vnode, ?axis=axis, ?keep_dims=keep_dims))

    static member Prod (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v 
            (fun ctxt axis vnode -> ctxt.Graph.ReduceProd(vnode, ?axis=axis, ?keep_dims=keep_dims))

    static member Min (v: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        let outputShape = if keep_dims = Some true then v.Shape else Shape.D
        DT<_> (outputShape, fun ctxt -> 
           let vnode = v.Apply ctxt
           ctxt.Graph.Min(vnode, ctxt.Graph.ReduceDims(vnode), ?keep_dims=keep_dims))

    static member Max (v: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        let outputShape = if keep_dims = Some true then v.Shape else Shape.D
        DT<_> (outputShape, fun ctxt -> 
           let vnode = v.Apply ctxt
           ctxt.Graph.Max(vnode, ctxt.Graph.ReduceDims(vnode), ?keep_dims=keep_dims))

    static member Norm (v: DT<'T>, ?axis, ?keep_dims: bool) : DT<'T> = 
        DT.Sqrt(DT.Sum(v * v, ?axis=axis, ?keep_dims= keep_dims))

    // TODO : generalize beyond vectors
    member v.Item 
        with get (n: int) : DT<'T> = 
            Shape.Unify "Item (index notaion)" v.Shape Shape.DV
            let outputShape = Shape.D
            DT<'T>(outputShape, fun ctxt -> 
               let vnode = v.Apply ctxt
               let graph = ctxt.Graph
               graph.Squeeze(graph.Slice(vnode, graph.Const(new TFTensor( [| n |])),graph.Const(new TFTensor( [| 1 |]))), [| 0L |])) // y = xv1

    // TODO : generalize beyond vectors
    member v.Item 
        with get (n1: int, n2: int) : DT<'T> = 
            Shape.Unify "Item (index notation)" v.Shape Shape.DV
            let outputShape = Shape.D
            DT<'T>(outputShape, fun ctxt -> 
               let vnode = v.Apply ctxt
               let graph = ctxt.Graph
               graph.Squeeze(graph.Slice(vnode, graph.Const(new TFTensor( [| n1; n2 |])),graph.Const(new TFTensor( [| 1; 1 |]))), [| 0L; 1L |])) // y = xv1

    // TODO : generalize beyond vectors
    member v.GetSlice(startIndex: int option, endIndex: int option) =
        Shape.Unify "GetSlice" v.Shape Shape.DV
        // TODO attach a constraint to the dimension that the endIndex is in-bounds
        let startIndex = defaultArg startIndex 0
        if endIndex.IsNone then failwith "end index must be specified"
        let len = endIndex.Value - startIndex + 1
        let outputShape = Shape.Known [| DimKnown len |]
        DT<'T>(outputShape, fun ctxt -> 
            let vnode = v.Apply ctxt
            let graph = ctxt.Graph
            graph.Slice(vnode, graph.Const(new TFTensor( [| startIndex |])),graph.Const(new TFTensor( [| len |])))) // y = xv1

    static member Sqrt (v: DT<'T>) : DT<'T> = 
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Sqrt(v.Apply ctxt))

    static member Square (v: DT<'T>) : DT<'T> = 
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Square(v.Apply ctxt))

    static member Exp (v: DT<'T>) : DT<'T> = 
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Exp(v.Apply ctxt))

    static member Reverse (v: DT<'T>) : DT<'T> = 
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.ReverseV2(v.Apply ctxt, ctxt.Graph.Const (new TFTensor( [| 0 |]))))

    static member DiagPart (v: DT<'T>) : DT<'T> = 
        let dims = v.Shape.DimensionsEliminatingFlex
        let n = dims.Length
        if n % 2 <> 0 then invalidArg "DiagPart: v" "expected a tensor with even rank"
        for i in 0 .. n - 1 do 
            Dim.Unify "DiagPart" dims.[i] dims.[n/2 + i]
        let outputShape = Shape.Known (dims.[0 .. n/2 - 1 ])
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.DiagPart(v.Apply ctxt))

    static member Trace v = DT.Sum (DT.DiagPart v)

    //static member Concat (concat_dim: int, vs: seq<DT<'T>>) : DT<'T> = 
    //    let vs = Seq.toArray vs
    //    if vs.Length = 0 then failwith "Vec: zero elements in vector"
    //    let actual = vs.[0].Shape
    //    let outputShape = Shape [| yield! actual.DimensionsEliminatingFlex; yield Dim (vs.Length) |]
    //    DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Concat(ctxt.Graph.Const(new TFTensor(concat_dim)), v.Apply ctxt))

    static member Stack (vs: seq<DT<'T>>, ?axis: int) : DT<'T> = 
        let vs = Seq.toArray vs
        if vs.Length = 0 then failwith "Stack: zero elements in vector"
        let axis = defaultArg axis 0
        let inputShape = vs.[0].Shape
        for v in vs do Shape.Unify "Stack" inputShape v.Shape
        Shape.MinDimensions "Stack" inputShape axis
        let inputDims = inputShape.DimensionsEliminatingFlex
        let outputShape = 
            Shape.Known
                [| yield! inputDims.[0 .. axis - 1]
                   yield DimKnown vs.Length 
                   yield! inputDims.[axis..] |]
        DT<'T>(outputShape, fun ctxt -> 
            let values = vs |> Array.map (fun v -> v.Apply(ctxt))
            ctxt.Graph.Stack(values, axis=axis))

    static member Sinh (v: DT<'T>) : DT<'T> = 
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Sinh(v.Apply ctxt))

    static member Sin (v: DT<'T>) : DT<'T> =  
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Sin(v.Apply ctxt))

    static member Cosh (v: DT<'T>) : DT<'T> = 
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Cosh(v.Apply ctxt))

    static member Cos (v: DT<'T>) : DT<'T> =  
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Cos(v.Apply ctxt))

    static member Tanh (v: DT<'T>) : DT<'T> = 
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Tanh(v.Apply ctxt))

    static member Tan (v: DT<'T>) : DT<'T> =  
        let outputShape = v.Shape
        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.Tan(v.Apply ctxt))

    override dt.ToString() = dt.Shape.ToString()

    /// Add partial deriviatives of loss function
    static member internal AddGradients (y: (* D *) DT, (* D, DV, DM, ...  *) xs: DT[], (* D *) ?dy: DT) =  
        Shape.Unify "AddGradients" y.Shape Shape.D
        let key = (y,xs,dy)
        xs |> Array.mapi (fun i x -> 
            let outputShape = x.Shape
            (outputShape, (fun (ctxt: Ctxt) -> 
                let dynodes = 
                    memoize ctxt.AddGradientNodes key (fun () -> 
                        let xnodes = xs |> Array.map (fun x -> x.Apply ctxt)
                        let ynode = y.Apply ctxt
                        let dynodesIn = match dy with None -> None | Some z -> Some [| z.Apply ctxt |]
                        let dynodes = ctxt.Graph.AddGradients([| ynode |], xnodes, ?dy=dynodesIn)
                        dynodes)
                dynodes.[i]))
             )

    static member Const (value: double, ?shape: Shape) : DT<'T> = 
        let shape = shape |> Option.defaultWith (fun () -> Shape.Inferred)
        DT<'T>(shape, fun ctxt -> ctxt.Graph.Reshape(ctxt.Graph.Const(new TFTensor(value)), ctxt.Graph.Const(shape.AsTFTensor())))

    static member ConstArray (value: 'T[], ?shape: Shape) : DT<'T> = 
        let shape = shape |> Option.defaultWith (fun () -> Shape.Inferred)
        DT<'T>(shape, fun ctxt -> ctxt.Graph.Reshape(ctxt.Graph.Const(new TFTensor(value)), ctxt.Graph.Const(shape.AsTFTensor())))

    static member TruncatedNormal (shape: Shape) : DT<double> = 
        DT<double> (shape, fun ctxt -> ctxt.Graph.TruncatedNormal(ctxt.Graph.Const(shape.AsTFTensor()), TFDataType.Float64 ))

    static member Variable (value: DT<'T>, ?name: string) : DT<'T> = 
        let outputShape = value.Shape
        DT<'T>(outputShape, fun ctxt -> 
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
        let _F1, _F2, C2, COut = filtersShape.AsRank4()
        Dim.Unify "Conv2D" C C2
        let outputShape = Shape.Known [| N; H/stride; W/stride; COut |]
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
        Dim.Unify "Conv2DBackpropInput" out_channels out_channels2
        let input_shape = Shape.Known [| N; out_height*stride; out_width*stride; in_channels |]
        DT<'T>(input_shape, fun ctxt -> 
           let input_sizes = ctxt.Graph.Const(input_shape.AsTFTensor())
           ctxt.Graph.Conv2DBackpropInput(input_sizes, filters.Apply ctxt, out_backprop.Apply ctxt, strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    /// Clips tensor values to a specified min and max.
    static member ClipByValue (input: DT<'T>, low: DT<'T>, high: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "ClipByValue" (Shape.EquivShapes "ClipByValue" input.Shape low.Shape) high.Shape
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
        let outputShape = Shape.Known [| Dim.Inferred; Dim.Inferred; DimKnown channels |]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.DecodeJpeg(contents=contents.Apply ctxt, channels=3L))

    static member Cast<'T, 'T2>(input: DT<'T>) : DT<'T2> = 
        let outputShape = input.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Cast(input.Apply ctxt, TFDataType.FromType(typeof<'T2>)))

    //static member map<'T, 'T2>(input: DT<'T>) : DT<'T2> = 
    //    let outputShape = input.Shape
    //    DT<_> (outputShape, fun ctxt -> TBD)

    static member WithScope(name: string) : IDisposable = 
        new WithScopeDisposable(name) :> _

    static member UsingWithScope (name: string) (f: unit -> DT<'T>) : DT<'T> = 
        let dt = f()
        let outputShape = dt.Shape
        DT<'T>(outputShape, fun ctxt -> use _scope = ctxt.Graph.NameScope(name) in dt.Apply ctxt)

    // TODO: broadcast
    static member CreateString(value: byte[]) : DT<string> = 
        let outputShape = shape [ 1 ]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Const(TFTensor.CreateString(value)))

    // TODO: handle expansion along multiple arbitrary dimensions
    static member ExpandDims(value: DT<'T>, ?dim: int) : DT<'T> = 
        let dim = defaultArg dim 0
        let inputShape = value.Shape
        // TODO: flex here?
        let inputDims = inputShape.DimensionsEliminatingFlex

        // Although the docs say "insert a dimension of 1" in practice the consumer expands/broadcasts to
        // arbitrary 'n'
        //
        // TODO check that this broadcasting always happens, perhaps Reshape is needed
        let outputShape = Shape.Known [| yield! inputDims.[0 .. dim-1]; yield Dim.Inferred; yield! inputDims.[dim..] |]

        DT<'T>(outputShape, fun ctxt -> ctxt.Graph.ExpandDims(value.Apply ctxt, ctxt.Graph.Const(new TFTensor( [| dim |] ))))

    static member RunTFTensors(values: DT[], ?weights: seq<string * DT>) : TFTensor[] = 
        let sess = new TFSession()
        let graph = sess.Graph
        let ctxt = 
            { Graph = graph
              MomentNodes = Dictionary(HashIdentity.Reference)
              AddGradientNodes = Dictionary(HashIdentity.Structural)
              Nodes = Dictionary(HashIdentity.Reference)
              Values = Map.ofSeq (defaultArg weights Seq.empty)}
        let nodes = values |> Array.map (fun value -> value.Apply ctxt)
        sess.Run([||],[||],nodes)

    static member RunTFTensor(value: DT<'T>, ?weights: seq<string * DT>) : TFTensor = 
        DT.RunTFTensors([| value |], ?weights=weights).[0]

    static member Run(value: DT<'T>, ?weights: seq<string * DT>) : obj = 
        if livecheck then 
            obj()
        else
            DT.RunTFTensor(value, ?weights=weights).GetValue() 

    static member Run(values: DT[], ?weights: seq<string * DT>) : obj[] = 
        if livecheck then 
            [| for v in values -> obj() |]
        else
            let results = DT.RunTFTensors(values, ?weights=weights)
            [| for res in results -> res.GetValue() |]

    static member RunScalar(value: DT<'T>, ?weights: seq<string * DT>) : 'T = 
        if livecheck then 
            Unchecked.defaultof<'T>
        else
            DT.Run(value, ?weights=weights) :?> 'T

    static member RunArrayAndScalar(values: DT<'T>[], ?weights: seq<string * DT>) : 'T[] * 'T = 
        if livecheck then 
            [| for v in values -> Unchecked.defaultof<'T> |], Unchecked.defaultof<'T>
        else
            let values = [| for v in values -> v :> DT |]
            let results = DT.Run(values, ?weights=weights)
            (results.[0] :?> 'T[]), (results.[1] :?> 'T)

    static member RunScalarPair(value1: DT<'T>, value2: DT<'T>, ?weights: seq<string * DT>) : 'T * 'T = 
        if livecheck then 
            Unchecked.defaultof<'T>, Unchecked.defaultof<'T>
        else
            let values = [| (value1 :> DT); (value2 :> DT) |]
            let results = DT.Run(values, ?weights=weights)
            (results.[0] :?> 'T), (results.[1] :?> 'T)

    static member RunScalars(values: DT<'T>[], ?weights: seq<string * DT>) : 'T[] = 
        if livecheck then 
            [| for v in values -> Unchecked.defaultof<'T> |]
        else
            let results = DT.RunTFTensors([| for v in values -> (v :> DT) |], ?weights=weights)
            [| for r in results -> r.GetValue() :?> 'T |]

    static member RunArray(value: DT<'T>, ?weights: seq<string * DT>) : 'T[] = 
        if livecheck then 
            [| Unchecked.defaultof<'T> |]
        else
            DT.Run(value, ?weights=weights) :?> 'T[]

    static member RunArray2D(value: DT<'T>, ?weights: seq<string * DT>) : 'T[,] = 
        if livecheck then 
            array2D [| [| Unchecked.defaultof<'T> |] |]
        else
            DT.Run(value, ?weights=weights) :?> 'T[,]

    static member RunArray3D(value: DT<'T>, ?weights: seq<string * DT>) : 'T[,,] = 
        if livecheck then 
            Array3D.init 1 1 1 (fun _ _ _ -> Unchecked.defaultof<'T>)
        else  
            DT.Run(value, ?weights=weights) :?> 'T[,,]

    static member RunArray4D(value: DT<'T>, ?weights: seq<string * DT>) : 'T[,,,] = 
        if livecheck then 
            Array4D.init 1 1 1 1 (fun _ _ _ _ -> Unchecked.defaultof<'T>)
        else
            DT.Run(value, ?weights=weights) :?> 'T[,,,]

/// Forward and reverse differentiation operations module (automatically opened)
module DT =

    /// Alias for a differentiable scalar.
    type D<'T> = DT<'T>

    /// Alias for a differentiable vector.
    type DV<'T> = DT<'T>

    /// Alias for a differentiable matrix
    type DM<'T> = DT<'T>

    /// Differential changes in scalar `y` with respect to differentials of `xs`. 
    let gradients (y: D<'T>) (xs: DT<'T>[]) = 
        DT.AddGradients (y, xs |> Array.map (fun x -> x :> DT)) 
            |> Array.map (fun (shape, f) -> DT<'T>(shape, f))

    /// Differential change in scalar `y` with respect to differentials of `x`. 
    let gradient (y: D<'T>) (x: DT<'T>) = 
        (gradients y [| x |]).[0]

    /// Original value and first derivative of a tensor-to-scalar function `f`, at point `x`.
    let evalAndDiff (f: DT<'T> -> D<'T>) (x: DT<'T>) = 
        let y = f x
        y, gradient y x

    /// First derivative of a scalar-to-scalar function `f`, at point `x`.
    let diff (f: D<'T> -> D<'T>) x = evalAndDiff f x |> snd

    /// Second derivative of a scalar-to-scalar function `f`, at point `x`.
    let diff2 (f: D<'T> -> D<'T>) x  : D<'T> =
        diff (diff f) x

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`, at point `x`.
    let evalAndDiffAndDiff2 (f: D<'T> -> D<'T>) x : D<'T> * D<'T> * D<'T> =
        let v, d = evalAndDiff f x
        let d2 = diff2 f x
        (v, d, d2)

    /// Original value and second derivative of a scalar-to-scalar function `f`, at point `x`.
    let diffAndDiff2 (f: D<'T> -> D<'T>) x  : D<'T> * D<'T> =
        evalAndDiffAndDiff2 f x |> (fun (a,_,c) -> a,c)

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`.
    let diffN n (f: D<'T> -> D<'T>) x  : D<'T> =
        if n < 0 then invalidArg "n" "must be positive"
        elif n = 0 then f x
        else
            let rec d n f =
                match n with
                | 1 -> diff f
                | _ -> d (n - 1) (diff f)
            x |> d n f

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`, at point `x`.
    let evalAndDiffN n (f: D<'T> -> D<'T>) x  : D<'T> * D<'T> =
        (x |> f, diffN n f x)

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let evalAndGrad (f: DV<'T> -> D<'T>) (x: DV<'T>) : D<'T> * DV<'T> = 
        Shape.Unify "evalAndGrad" x.Shape Shape.DV
        let y = f x
        let dy = gradient y x
        y, dy

    /// Gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let grad (f: DV<'T> -> D<'T>) x : DV<'T> =
        evalAndGrad f x |> snd

(*
    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`.
    let gradv' (f: DV<'T> -> D<'T>) x (v: DV<'T>) : D<'T> * D<'T> =
        let yv = f v
        let y = f x
        let dyv = DT.AddGradients (y, x, yv)
        y, dyv

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`.
    let gradv (f: DV<'T> -> D<'T>) x v : D<'T> =
        gradv' f x v |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`.
    let jacobianv' (f: DV<'T> -> DV<'T>) (x: DV<'T>) (v: DV<'T>) : DV<'T> * DV<'T> =
        Shape.Unify x.Shape Shape.DV
        Shape.Unify v.Shape Shape.DV
        let yv = f v
        let y = f x
        let ysize = 
            match y.Shape.DimensionsEliminatingFlex.[0].TryValue() with 
            | None -> failwith "unknown vector output size in jacobian"
            | Some d -> d
        let dyv = DT.Stack [| for i in 0 .. ysize-1 -> DT.AddGradients (y.[i], x, yv.[i]) |]
        y, dyv

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`.
    let jacobianv (f: DV<'T> -> DV<'T>) x v : DV<'T> =
        jacobianv' f x v |> snd
*)
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let evalAndJacobian (f: DV<'T> -> DV<'T>) (x:DV<'T>) : DV<'T> * DM<'T> =
        let y = f x
        let ysize = 
            match y.Shape.DimensionsEliminatingFlex.[0].TryValue() with 
            | None -> failwith "unknown vector output size in jacobian"
            | Some d -> d
        let jydx = DT.Stack [| for i in 0 .. ysize - 1 -> gradient y.[i] x |]
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
        let gv, hv = evalAndGrad (fun xx -> gradv f xx v) x
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

    let trace v = DT.Sum (DT.DiagPart v)

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let evalAndLaplacian (f: DV<'T> -> D<'T>) x : D<'T> * D<'T> = // TODO: reimplement faster
        let v, h = evalAndHessian f x
        (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let laplacian (f: DV<'T> -> D<'T>) x : D<'T> =
        evalAndLaplacian f x |> snd

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let evalAndCurl (f: DV<'T> -> DV<'T>) x =
        let v, j = evalAndJacobian f x
        //if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        v, DT.Stack [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let curl (f: DV<'T> -> DV<'T>) x : DV<'T> =
        evalAndCurl f x |> snd

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let evalAndDivergence (f: DV<'T> -> DV<'T>) x =
        let v, j = evalAndJacobian f x
        //if j.Rows <> j.Cols then ErrorMessages.InvalidArgDiv()
        v, DT.Trace j

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let divergence (f: DV<'T> -> DV<'T>) x : D<'T> =
        evalAndDivergence f x |> snd

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let evalAndCurlAndDivergence (f: DV<'T> -> DV<'T>) x =
        let v, j = evalAndJacobian f x
        //if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        v, DT.Stack [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
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

    let shape (ints: int list) = Shape.Known (Array.map DimKnown (Array.ofSeq ints))

    let scalar (d:double) : DT<double> = 
        let shape = Shape.Inferred 
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(d))))

    let v d = scalar d
    let vec (d:seq<double>) : DT<double> = 
        let d = Array.ofSeq d
        let shape = shape [ d.Length ]
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(d))))

    let batchScalar d = vec d

    let matrix (d: seq< #seq<'T>>) : DT<'T> = 
        let d = array2D d 
        let shape = shape [ d.GetLength(0); d.GetLength(1)  ]
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(d))))

    let batchVec d = matrix d 

    let matrix3 (d: seq< #seq< #seq<'T>>>) : DT<'T> = 
        let d = d |> Array.ofSeq |> Array.map array2D
        let ds = Array3D.init d.Length (d.[0].GetLength(0)) (d.[0].GetLength(1)) (fun i j k -> d.[i].[j,k])
        let shape = shape [ d.Length; d.[0].GetLength(0); d.[0].GetLength(1)  ]
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(ds))))

    let image d = matrix3 d

    let matrix4 (d: seq< #seq< #seq< #seq<'T>>>>) : DT<'T> = 
        let d = d |> array2D |> Array2D.map array2D
        let r1,r2,r3,r4 = (d.GetLength(0), d.GetLength(1), d.[0,0].GetLength(0),d.[0,0].GetLength(1))
        let ds = Array4D.init r1 r2 r3 r4 (fun i j k m -> d.[i,j].[k,m])
        let shape = shape [ r1; r2; r3; r4 ]
        DT<_> (shape, (fun ctxt -> ctxt.Graph.Const(new TFTensor(ds))))

    let batchImage d = matrix4 d 
    
    let inline relu (x: ^T) : ^T = 
        (^T: (static member Relu : ^T -> ^T) (x))

    let sum (x: DT<'T>) : DT<'T> = DT.Sum(x)

    let prod (x: DT<'T>) : DT<'T> = DT.Prod(x)

    let mean (x: DT<'T>) : DT<'T> = DT.Mean(x)

    let max (x: DT<'T>) : DT<'T> = DT.Max(x)

    let min (x: DT<'T>) : DT<'T> = DT.Min(x)

    let norm (x: DT<'T>) : DT<'T> = DT.Norm(x)

    /// Extend the value in the batch dimension
    let batchExtend (v: DT<'T>) = DT.ExpandDims v

    let variable value name = DT.Variable (value, name)

[<AttributeUsage(AttributeTargets.Field ||| AttributeTargets.Property ||| AttributeTargets.Method)>]
type LiveCheckAttribute() =
    inherit Attribute()

[<AttributeUsage(AttributeTargets.Field ||| AttributeTargets.Property ||| AttributeTargets.Method)>]
type LiveTestAttribute() =
    inherit Attribute()