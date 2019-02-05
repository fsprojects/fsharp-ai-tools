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
        if dims.Length = 0 then 
            "scalar" 
            + (if flexvar.IsSome then " (can expand)" else "")
        else
            sprintf "shape %s" (String.concat " x " [ for i in dims -> i.ToString() ]) 
            + (if flexvar.IsSome then "x.." else "")

    member shape.AsTFShape() = TFShape(shape.DimensionsEliminatingFlex |> Array.map (fun dim -> int64 dim.Value))

    member shape.AsTFTensor() = shape.AsTFShape().AsTensor()

    static member Known dims = ShapeImpl(dims, None)

    static member Flex dims = ShapeImpl(dims, Some (InferenceVar()))

    static member internal PossibleFlex (flex: bool) dims = if flex then Shape.Flex dims else Shape.Known dims

    static member Inferred with get() = Shape.Flex [| |]
    
    static member UserSpecified (ints: seq<int>) = 
        ints 
        |> Array.ofSeq 
        |> Array.map (fun i -> if i = -1 then Dim.Inferred else DimKnown i)
        |> Shape.Known 

    static member FromTFShapeArray (shape: int64[], ?flex: bool) = 
        let flex = defaultArg flex false
        let dims = shape |> Array.map (fun i -> if i = -1L then Dim.Inferred else DimKnown (int32 i))
        Shape.PossibleFlex flex dims

    static member FromTFShape (shape: TFShape) = 
        shape.Dims |> Shape.FromTFShapeArray

    static member D with get() = Shape.Known [| DimKnown 1 |]
    
    static member DV with get() = Shape.Known [| Dim.Inferred |]
    
    static member DM with get() = Shape.Known [| Dim.Inferred; Dim.Inferred |]

    /// At least 'n' dimensions, possible more
    static member internal FlexN n = Shape.Flex [| for i in 1 .. n -> Dim.Inferred |]

    static member internal MinDimensions op (shape: Shape) dim = 
        let dims, flexvar = shape.DimensionsWithFlexVar
        if dim > dims.Length then 
            match flexvar with 
            | None -> 
                failwithf "shape %A must have at least %d dimensions for operator %s" shape dim op
            | Some v -> 
                v.Solve (Shape.FlexN dim)

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

    let memoize (dict: Dictionary<_,_>) key f = 
        match dict.TryGetValue (key) with 
        | true, res -> res
        | _ -> 
            let res = f ()
            dict.[key] <- res
            res

type internal WithScopeDisposable(name:string) = 
    interface IDisposable with 
        member __.Dispose() = ()
    member __.Name = name        


/// Represents a context for turning differentiable tensors into a TensorFlow graph
type internal Ctxt = 
    { Graph: TFGraph 
      Nodes: Dictionary<DT, TFOutput> // ensure unique nodes from unique DT values
      MomentNodes: Dictionary<DT, TFOutput * TFOutput> // ensure unique nodes from unique DT values
      AddGradientNodes: Dictionary<DT * DT[] * DT option, TFOutput[]> // ensure unique nodes from unique DT values
      Values: Map<string,DT> }

/// Represents a differentiable tensor value, which later corresponds to a node in a TensorFlow graph
and DT internal (shape: Shape, cost: int, makeNode: (Ctxt -> TFOutput), asTFTensor: (unit -> TFTensor) option) = 

    member internal dt.MakeNode(ctxt: Ctxt) = 
        memoize ctxt.Nodes dt (fun () -> makeNode ctxt)

    /// Get the inferred shape of the differentiable tensor 
    member __.Shape = shape

    /// Get the inferred shape of the differentiable tensor 
    member internal __.Cost = cost 

    /// A quick check to see if this is a constant tensor, so we don't have to create a graph to
    /// view or analyze it.
    member internal __.TryAsConstTFTensor() = match asTFTensor with None -> None | Some f -> Some (f())

    static member RunTFTensors(values: DT[], ?weights: seq<string * DT>) : TFTensor[] = 
        let sess = new TFSession()
        let graph = sess.Graph
        let ctxt = 
            { Graph = graph
              MomentNodes = Dictionary(HashIdentity.Reference)
              AddGradientNodes = Dictionary(HashIdentity.Structural)
              Nodes = Dictionary(HashIdentity.Reference)
              Values = Map.ofSeq (defaultArg weights Seq.empty)}
        let nodes = values |> Array.map (fun value -> value.MakeNode ctxt)
        sess.Run([||],[||],nodes)

    static member RunTFTensor(value: DT, ?weights: seq<string * DT>) : TFTensor = 
        match value.TryAsConstTFTensor() with 
        | None -> DT.RunTFTensors([| value |], ?weights=weights).[0]
        | Some t -> t

    static member Run(value: DT, ?weights: seq<string * DT>) : obj = 
        if livecheck then 
            // TODO: give a better dummy value back here
            obj()
        else
            DT.RunTFTensor(value, ?weights=weights).GetValue() 

    static member Run(values: DT[], ?weights: seq<string * DT>) : obj[] = 
        if livecheck then 
            // TODO: give a better dummy value back here
            [| for v in values -> obj() |]
        else
            let results = DT.RunTFTensors(values, ?weights=weights)
            [| for res in results -> res.GetValue() |]

    /// A method to transform this object to a formattable object, used by F# interactive
    static member PrintTransform(value: DT) = 
        // cost = 0 implies constant, e.g. result from Eval
        match value.TryAsConstTFTensor() with 
        | Some t -> t.GetValue()
        | None -> 
            if value.Cost < 10 then 
                let v = DT.Run(value)
                v
            else
                box (sprintf "%A" value.Shape + " (unevaluated)")

    /// Display constants as data and delayed nodes as shapes
    override dt.ToString() = 
        // cost = 0 implies constant, e.g. result from Eval
        match dt.TryAsConstTFTensor() with 
        | Some t -> sprintf "%A" (t.GetValue())
        | None -> sprintf "%A" dt.Shape + " (unevaluated)"


/// Represents a differentiable tensor value
type DT<'T> internal (shape: Shape, cost: int, eval: (Ctxt -> TFOutput), ?asTFTensor: (unit -> TFTensor)) =

    inherit DT(shape, cost, eval, asTFTensor = asTFTensor)

    static member AddN (vs: DT<'T>[]) : DT<'T> = 
        let outputShape = vs.[0].Shape 
        let cost : int = (vs |> Array.sumBy (fun (v: DT<'T>) -> v.Cost)) + 1
        for v in vs do Shape.Unify "AddN" outputShape v.Shape
        DT<'T> (outputShape, cost, fun ctxt -> ctxt.Graph.AddN(vs |> Array.map (fun v -> v.MakeNode ctxt)))

    static member (+) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "(+)" v1.Shape v2.Shape
        let cost = v1.Cost + v2.Cost + 1
        DT<_> (outputShape, cost, fun ctxt -> ctxt.Graph.Add(v1.MakeNode ctxt, v2.MakeNode ctxt))

    static member (-) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "(-)" v1.Shape v2.Shape
        let cost = v1.Cost + v2.Cost + 1
        DT<_> (outputShape, cost, fun ctxt -> ctxt.Graph.Sub(v1.MakeNode ctxt, v2.MakeNode ctxt))

    /// Pointwise multiplication
    static member ( * ) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "(*)" v1.Shape v2.Shape
        let cost = v1.Cost + v2.Cost + 1
        DT<_> (outputShape, cost, fun ctxt -> ctxt.Graph.Mul(v1.MakeNode ctxt, v2.MakeNode ctxt))

    /// Pointwise multiplication
    static member ( ~- ) (input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape 
        let cost = input.Cost + 1 
        DT<_> (outputShape, cost, fun ctxt -> ctxt.Graph.Neg (input.MakeNode ctxt))

    static member ( *! ) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let n1,m,n2 = Dim.Inferred, Dim.Inferred, Dim.Inferred 
        Shape.Unify "MatMul"  v1.Shape (Shape.Known [| n1; m |])
        Shape.Unify "MatMul" v2.Shape (Shape.Known [| m; n2 |])
        let outputShape = Shape.Known [| n1; n2 |]
        let cost = v1.Cost + v2.Cost + 1
        DT<'T> (outputShape, cost, fun ctxt -> ctxt.Graph.MatMul(v1.MakeNode ctxt, v2.MakeNode ctxt))

    static member (/) (v1: DT<'T>, v2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "(/)" v1.Shape v2.Shape
        let cost = v1.Cost + v2.Cost + 1
        DT<_> (outputShape, cost, fun ctxt -> ctxt.Graph.Div(v1.MakeNode ctxt, v2.MakeNode ctxt))

    static member internal ReduceOp keep_dims (axis: int[] option) (input: DT<'T>) f : DT<'T> = 
        let outputShape = 
            match keep_dims, axis with
            | Some true, _ -> input.Shape 
            | _, None -> Shape.D
            | _, Some axis -> 
                // TODO: flex here
                let inputDims = input.Shape.DimensionsEliminatingFlex
                let outputDims = inputDims |> Array.indexed |> Array.filter (fun (idx, _) -> not (Array.contains idx axis)) |> Array.map snd
                if outputDims.Length = 0 then Shape.D else Shape.Known outputDims

        let cost = input.Cost + 1
        DT<_> (outputShape, cost, fun ctxt -> 
            let axis = axis |> Option.map (fun axis -> ctxt.Graph.Const(new TFTensor(axis)))
            f ctxt axis (input.MakeNode ctxt))

    static member Sum (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v 
            (fun ctxt axis vnode -> ctxt.Graph.ReduceSum(vnode, ?axis=axis, ?keep_dims=keep_dims))

    static member Mean (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v
            (fun ctxt axis vnode -> ctxt.Graph.ReduceMean(vnode, ?axis=axis, ?keep_dims=keep_dims))

    static member Prod (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v 
            (fun ctxt axis vnode -> ctxt.Graph.ReduceProd(vnode, ?axis=axis, ?keep_dims=keep_dims))

    static member Min (input: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        let outputShape = if keep_dims = Some true then input.Shape else Shape.D
        let cost = input.Cost + 1
        DT<_> (outputShape, cost, fun ctxt -> 
           let vnode = input.MakeNode ctxt
           ctxt.Graph.Min(vnode, ctxt.Graph.ReduceDims(vnode), ?keep_dims=keep_dims))

    static member Max (input: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        let outputShape = if keep_dims = Some true then input.Shape else Shape.D
        let cost = input.Cost + 1
        DT<_> (outputShape, cost, fun ctxt -> 
           let vnode = input.MakeNode ctxt
           ctxt.Graph.Max(vnode, ctxt.Graph.ReduceDims(vnode), ?keep_dims=keep_dims))

    static member Norm (v: DT<'T>, ?axis, ?keep_dims: bool) : DT<'T> = 
        DT.Sqrt(DT.Sum(v * v, ?axis=axis, ?keep_dims= keep_dims))

    // TODO : generalize beyond vectors
    member input.Item 
        with get (n: int) : DT<'T> = 
            Shape.Unify "Item (index notaion)" input.Shape Shape.DV
            let outputShape = Shape.D
            let cost = input.Cost
            DT<'T>(outputShape, cost, fun ctxt -> 
               let vnode = input.MakeNode ctxt
               let graph = ctxt.Graph
               graph.Squeeze(graph.Slice(vnode, graph.Const(new TFTensor( [| n |])),graph.Const(new TFTensor( [| 1 |]))), [| 0L |])) // y = xv1

    // TODO : generalize beyond vectors
    member v.Item 
        with get (n1: int, n2: int) : DT<'T> = 
            Shape.Unify "Item (index notation)" v.Shape Shape.DV
            let outputShape = Shape.D
            DT<'T>(outputShape, cost, fun ctxt -> 
               let vnode = v.MakeNode ctxt
               let graph = ctxt.Graph
               graph.Squeeze(graph.Slice(vnode, graph.Const(new TFTensor( [| n1; n2 |])),graph.Const(new TFTensor( [| 1; 1 |]))), [| 0L; 1L |])) // y = xv1

    // TODO : generalize beyond vectors
    member input.GetSlice(startIndex: int option, endIndex: int option) =
        Shape.Unify "GetSlice" input.Shape Shape.DV
        // TODO attach a constraint to the dimension that the endIndex is in-bounds
        let startIndex = defaultArg startIndex 0
        if endIndex.IsNone then failwith "end index must be specified"
        let len = endIndex.Value - startIndex + 1
        let outputShape = Shape.Known [| DimKnown len |]
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> 
            let vnode = input.MakeNode ctxt
            let graph = ctxt.Graph
            graph.Slice(vnode, graph.Const(new TFTensor( [| startIndex |])),graph.Const(new TFTensor( [| len |])))) // y = xv1

    static member Sqrt (input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Sqrt(input.MakeNode ctxt))

    static member Square (input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Square(input.MakeNode ctxt))

    static member Exp (input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Exp(input.MakeNode ctxt))

    static member Reverse (input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.ReverseV2(input.MakeNode ctxt, ctxt.Graph.Const (new TFTensor( [| 0 |]))))

    static member DiagPart (input: DT<'T>) : DT<'T> = 
        let dims = input.Shape.DimensionsEliminatingFlex
        let n = dims.Length
        if n % 2 <> 0 then invalidArg "DiagPart: v" "expected a tensor with even rank"
        for i in 0 .. n - 1 do 
            Dim.Unify "DiagPart" dims.[i] dims.[n/2 + i]
        let outputShape = Shape.Known (dims.[0 .. n/2 - 1 ])
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.DiagPart(input.MakeNode ctxt))

    static member Trace v = DT.Sum (DT.DiagPart v)

    //static member Concat (concat_dim: int, vs: seq<DT<'T>>) : DT<'T> = 
    //    let vs = Seq.toArray vs
    //    if vs.Length = 0 then failwith "Vec: zero elements in vector"
    //    let actual = vs.[0].Shape
    //    let outputShape = Shape [| yield! actual.DimensionsEliminatingFlex; yield Dim (vs.Length) |]
    //    DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Concat(ctxt.Graph.Const(new TFTensor(concat_dim)), v.Apply ctxt))

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
        let cost = (vs |> Array.sumBy (fun v -> v.Cost)) + 1
        DT<'T>(outputShape, cost, fun ctxt -> 
            let values = vs |> Array.map (fun v -> v.MakeNode(ctxt))
            ctxt.Graph.Stack(values, axis=axis))

    static member Sinh (input: DT<'T>) : DT<'T> = 
        let cost = input.Cost + 1
        let outputShape = input.Shape
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Sinh(input.MakeNode ctxt))

    static member Sin (input: DT<'T>) : DT<'T> =  
        let cost = input.Cost + 1
        let outputShape = input.Shape
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Sin(input.MakeNode ctxt))

    static member Cosh (input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Cosh(input.MakeNode ctxt))

    static member Cos (input: DT<'T>) : DT<'T> =  
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Cos(input.MakeNode ctxt))

    static member Tanh (input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Tanh(input.MakeNode ctxt))

    static member Tan (input: DT<'T>) : DT<'T> =  
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Tan(input.MakeNode ctxt))

    /// Add partial deriviatives of loss function
    static member internal AddGradients (y: (* D *) DT<'T>, (* D, DV, DM, ...  *) xs: DT[], (* D *) ?dy: DT<'T>) =  
        Shape.Unify "AddGradients" y.Shape Shape.D
        let key = ((y :> DT),xs,(match dy with None -> None | Some d -> Some (d :> DT)))
        xs |> Array.mapi (fun i x -> 
            let outputShape = x.Shape
            (outputShape, (fun (ctxt: Ctxt) -> 
                let dynodes = 
                    memoize ctxt.AddGradientNodes key (fun () -> 
                        let xnodes = xs |> Array.map (fun x -> x.MakeNode ctxt)
                        let ynode = y.MakeNode ctxt
                        let dynodesIn = match dy with None -> None | Some z -> Some [| z.MakeNode ctxt |]
                        let dynodes = ctxt.Graph.AddGradients([| ynode |], xnodes, ?dy=dynodesIn)
                        dynodes)
                dynodes.[i]))
             )

    static member internal MakeConst (dims, asTFTensor, flex: bool option) : DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex dims
        let cost = 0
        DT<'T>(shape, cost, 
              (fun ctxt -> 
                let node = ctxt.Graph.Const(asTFTensor())
                if flex then ctxt.Graph.Reshape(node, ctxt.Graph.Const(shape.AsTFTensor())) else node),
              asTFTensor = asTFTensor)

    static member ConstInner (obj: obj, ?flex: bool) : DT<'T> = 
        let shape = Shape.Inferred 
        let cost = 0
        match obj with 
        | :? single
        | :? double 
        | :? int64
        | :? int32 -> () 
        | _ -> failwithf "invalid scalar type %A" (typeof<'T>)
        DT<'T> (shape, cost, (fun ctxt -> 
            let t = 
                match obj with 
                | :? single as d -> new TFTensor(d)
                | :? double as d -> new TFTensor(d)
                | :? int32 as d -> new  TFTensor(d)
                | :? int64 as d -> new  TFTensor(d)
                | _ -> failwith "unreachable"
            ctxt.Graph.Const(t)))
    
    static member inline Const (value: 'T1, ?flex: bool) : DT<'T1> = 
        (fun () -> double value) |> ignore // places constraints on 'T without execution
        DT.ConstInner(box value, ?flex=flex)

    static member ConstArray (value: 'T[], ?flex: bool) : DT<'T> = 
        let dims = [| Dim.Known value.Length |]
        DT.MakeConst (dims, (fun () -> new TFTensor(value)), flex)

    static member ConstArray2D (value: 'T[,], ?flex: bool) : DT<'T> = 
        let dims = [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1))|]
        DT.MakeConst (dims, (fun () -> new TFTensor(value)), flex)

    static member ConstArray3D (value: 'T[,,], ?flex: bool) : DT<'T> = 
        let dims = [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1)); Dim.Known (value.GetLength(2))|]
        DT.MakeConst (dims, (fun () -> new TFTensor(value)), flex)

    static member ConstArray4D (value: 'T[,,,], ?flex: bool) : DT<'T> = 
        let dims = [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1)); Dim.Known (value.GetLength(2)); Dim.Known (value.GetLength(3))|]
        DT.MakeConst (dims, (fun () -> new TFTensor(value)), flex)

    static member internal ConstTensor (tensor: TFTensor, shape: Shape) : DT<'T> = 
        let cost = 0
        DT<'T>(shape, cost, 
              (fun ctxt -> 
                let node = ctxt.Graph.Const(tensor)
                ctxt.Graph.Reshape(node, ctxt.Graph.Const(shape.AsTFTensor()))),
              asTFTensor = (fun () -> tensor))

    static member TruncatedNormal (?shape: Shape) : DT<double> = 
        let shape = defaultArg shape Shape.Inferred
        let cost = 1
        DT<double> (shape, cost, fun ctxt -> ctxt.Graph.TruncatedNormal(ctxt.Graph.Const(shape.AsTFTensor()), TFDataType.Float64 ))

    static member Variable (value: DT<'T>, ?name: string) : DT<'T> = 
        let outputShape = value.Shape
        let cost = 100
        DT<'T>(outputShape, cost, fun ctxt -> 
                     let name2 = defaultArg name ""
                     match ctxt.Values.TryFind name2 with 
                     | None -> 
                         printfn "variable nodes not yet supported, and weight '%s' not found in Values, assuming constant" name2
                         //ctxt.Graph.Variable(value.Apply ctxt,name=name2).Read
                         value.MakeNode ctxt
                     | Some t -> 
                         match t with 
                         | :? DT<'T> as vt -> vt.MakeNode ctxt
                         | _ -> 
                         printfn "incorrect type in values, got '%A' expected '%A', assuming variable node is constant" (t.GetType()) (typeof<DT<'T>>)
                         value.MakeNode ctxt
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
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Conv2D(input.MakeNode ctxt, filters.MakeNode ctxt,strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

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
        let cost = out_backprop.Cost + 100
        DT<'T>(input_shape, cost, fun ctxt -> 
           let input_sizes = ctxt.Graph.Const(input_shape.AsTFTensor())
           ctxt.Graph.Conv2DBackpropInput(input_sizes, filters.MakeNode ctxt, out_backprop.MakeNode ctxt, strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    /// Clips tensor values to a specified min and max.
    static member ClipByValue (input: DT<'T>, low: DT<'T>, high: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "ClipByValue" (Shape.EquivShapes "ClipByValue" input.Shape low.Shape) high.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.ClipByValue(input.MakeNode ctxt, low.MakeNode ctxt, high.MakeNode ctxt))

    static member Moments(input: DT<'T>, ?axes: seq<int>) : DT<'T> * DT<'T> = 
        // Note: keep_dims = true
        let outputShape = input.Shape
        let compute (ctxt: Ctxt) = 
            memoize ctxt.MomentNodes (upcast input) (fun () -> 
                let axes = match axes with None -> None | Some v -> Some (ctxt.Graph.Const(Shape.UserSpecified(v).AsTFTensor()))
                ctxt.Graph.Moments(input.MakeNode ctxt, ?axes=axes,keep_dims=true))
        let cost = input.Cost + 1

        DT<'T>(outputShape, cost, fun ctxt -> fst (compute ctxt)),
        DT<'T>(outputShape, cost, fun ctxt -> snd (compute ctxt))

    static member Relu(input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.Relu(input.MakeNode ctxt))

    static member DecodeJpeg(contents:DT<string>, ?channels: int) : DT<int> = // V[int,H,W,C]
        let channels = defaultArg channels 3 // CHECK ME
        let outputShape = Shape.Known [| Dim.Inferred; Dim.Inferred; DimKnown channels |]
        let cost = 1
        DT<_> (outputShape, cost, fun ctxt -> ctxt.Graph.DecodeJpeg(contents=contents.MakeNode ctxt, channels=3L))

    static member Cast<'T, 'T2>(input: DT<'T>) : DT<'T2> = 
        let outputShape = input.Shape
        let cost = input.Cost + 1
        DT<_> (outputShape, cost, fun ctxt -> ctxt.Graph.Cast(input.MakeNode ctxt, TFDataType.FromType(typeof<'T2>)))

    //static member map<'T, 'T2>(input: DT<'T>) : DT<'T2> = 
    //    let outputShape = input.Shape
    //    DT<_> (outputShape, cost, fun ctxt -> TBD)

    static member WithScope(name: string) : IDisposable = 
        new WithScopeDisposable(name) :> _

    static member UsingWithScope (name: string) (f: unit -> DT<'T>) : DT<'T> = 
        let input = f()
        let outputShape = input.Shape
        let cost = input.Cost
        DT<'T>(outputShape, cost, fun ctxt -> use _scope = ctxt.Graph.NameScope(name) in input.MakeNode ctxt)

    // TODO: broadcast
    static member CreateString(value: byte[]) : DT<string> = 
        let outputShape = Shape.D 
        let cost = 1
        DT<_> (outputShape, cost, fun ctxt -> ctxt.Graph.Const(TFTensor.CreateString(value)))

    // TODO: handle expansion along multiple arbitrary dimensions
    static member ExpandDims(input: DT<'T>, ?dim: int) : DT<'T> = 
        let dim = defaultArg dim 0
        let inputShape = input.Shape
        // TODO: flex here?
        let inputDims = inputShape.DimensionsEliminatingFlex

        // Although the docs say "insert a dimension of 1" in practice the consumer expands/broadcasts to
        // arbitrary 'n'
        //
        // TODO check that this broadcasting always happens, perhaps Reshape is needed
        let outputShape = Shape.Known [| yield! inputDims.[0 .. dim-1]; yield Dim.Inferred; yield! inputDims.[dim..] |]
        let cost = input.Cost + 1

        DT<'T>(outputShape, cost, fun ctxt -> ctxt.Graph.ExpandDims(input.MakeNode ctxt, ctxt.Graph.Const(new TFTensor( [| dim |] ))))

    member value.GetLength(dim: int) = value.Shape.[dim].Value

    // TODO: improve this
    member value.ToScalar () : 'T = 
        if livecheck then 
            Unchecked.defaultof<'T>
        else
            DT.Run(value) :?> 'T

    static member Eval (value: DT<'T>, ?weights: seq<string * DT>) : DT<'T> = 
        if livecheck then 
            value
        else
            let tensor = DT.RunTFTensor(value, ?weights=weights)
            DT.ConstTensor(tensor, value.Shape)

    static member Eval2 (value1: DT<'T1>, value2: DT<'T2>, ?weights: seq<string * DT>) : DT<'T1> * DT<'T2> = 
        if livecheck then 
            value1, value2
        else
            let values = [| (value1 :> DT); (value2 :> DT) |]
            let tensors = DT.RunTFTensors(values, ?weights=weights)
            DT.ConstTensor(tensors.[0], value1.Shape), DT.ConstTensor(tensors.[1], value2.Shape)

    static member Eval3 (value1: DT<'T1>, value2: DT<'T2>, value3: DT<'T3>,  ?weights: seq<string * DT>) : DT<'T1> * DT<'T2> * DT<'T3> = 
        if livecheck then 
            value1, value2, value3
        else
            let values = [| (value1 :> DT); (value2 :> DT); (value3 :> DT) |]
            let tensors = DT.RunTFTensors(values, ?weights=weights)
            DT.ConstTensor(tensors.[0], value1.Shape), DT.ConstTensor(tensors.[1], value2.Shape), DT.ConstTensor(tensors.[2], value3.Shape)

(*
    static member RunScalars(values: DT<'T>[], ?weights: seq<string * DT>) : 'T[] = 
        if livecheck then 
            [| for v in values -> Unchecked.defaultof<'T> |]
        else
            let results = DT.RunTFTensors([| for v in values -> (v :> DT) |], ?weights=weights)
            [| for r in results -> r.GetValue() :?> 'T |]
*)

    member value.GetValue() : obj = 
        DT.Run(value) 

    member value.ToArray() : 'T[] = 
        if livecheck then 
            let dim1 = value.Shape.AsRank1()
            Array.zeroCreate dim1 .Value
        else
            DT.Run(value) :?> 'T[]

    member value.ToArray2D() : 'T[,] = 
        if livecheck then 
            let dim1, dim2 = value.Shape.AsRank2()
            Array2D.zeroCreate dim1.Value dim2.Value
        else
            DT.Run(value) :?> 'T[,]

    member value.ToArray3D() : 'T[,,] = 
        if livecheck then 
            let dim1, dim2, dim3 = value.Shape.AsRank3()
            Array3D.zeroCreate dim1.Value dim2.Value dim3.Value
        else  
            DT.Run(value) :?> 'T[,,]

    member value.ToArray4D() : 'T[,,,] = 
        if livecheck then 
            let dim1, dim2, dim3, dim4 = value.Shape.AsRank4()
            Array4D.zeroCreate dim1.Value dim2.Value dim3.Value dim4.Value
        else
            DT.Run(value) :?> 'T[,,,]

/// Forward and reverse differentiation operations module (automatically opened)
module DT =

    /// Alias for a differentiable scalar.
    type D<'T> = DT<'T>

    /// Alias for a differentiable vector.
    type DV<'T> = DT<'T>

    /// Alias for a differentiable matrix
    type DM<'T> = DT<'T>

    /// Differential changes in scalar `y` with respect to differentials of `xs`. 
    let gradients (y: D<'T>) (xs: DT[]) = 
        DT.AddGradients (y, xs) |> Array.map (fun (shape, f) -> 
            let cost = 100 in DT<'T>(shape, cost, f))

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
    let evalAndLaplacian (f: DV<'T> -> D<'T>) x : D<'T> * D<'T> = 
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

    /// Convert the input to an array if possible
    let toArray (input: DT<'T>) = input.ToArray()
    
    /// Convert the input to an array if possible
    let toArray2D (input: DT<'T>) = input.ToArray2D()
    
    /// Convert the input to an array if possible
    let toArray3D (input: DT<'T>) = input.ToArray3D()
    
    /// Convert the input to an array if possible
    let toArray4D (input: DT<'T>) = input.ToArray4D()
    
    /// Convert the input to a scalar if possible
    let toScalar (input: DT<'T>) = input.ToScalar()

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

    let shape (ints: int list) = 
        ints 
        |> Array.ofSeq 
        |> Array.map (fun i -> if i = -1 then Dim.Inferred else DimKnown i)
        |> Shape.Known 

    /// Create a scalar node (with implicit broadcast)
    let inline scalar (value:'T) : DT<'T> = DT.Const value 

    /// Create a scalar node (with implicit broadcast)
    let inline v x = scalar x

    /// Create a vector from raw data
    let vec (data:seq<double>) : DT<double> = 
        let d = Seq.toArray data
        DT.ConstArray(d, flex=false)

    /// Create a vector from existing differentiable tensors
    let vecOfScalars (xs:seq<DT<'T>>) : DT<'T> = 
        DT.Stack xs

    /// Extend the scalar node, adding a batch dimension
    let batchOfScalars d = vec d

    /// Create a matrix from raw data
    let matrix (data: seq< #seq<'T>>) : DT<'T> = 
        let data = array2D data 
        DT.ConstArray2D(data, flex=false)

    /// Create a matrix by stacking existing vectors of differentiable tensors
    let matrixOfVecs (ds:seq<DT<'T>>) : DT<'T> = 
        DT.Stack ds

    /// Extend the vector node, adding a batch dimension
    let batchOfVecs vecs = matrix vecs 

    let array3D data = 
        let data = data |> Array.ofSeq |> Array.map array2D
        Array3D.init data.Length (data.[0].GetLength(0)) (data.[0].GetLength(1)) (fun i j k -> data.[i].[j,k])

    let array4D data = 
        let data = data |> array2D |> Array2D.map array2D
        let r1,r2,r3,r4 = (data.GetLength(0), data.GetLength(1), data.[0,0].GetLength(0),data.[0,0].GetLength(1))
        Array4D.init r1 r2 r3 r4 (fun i j k m -> data.[i,j].[k,m])

    /// Create a rank-3 tensor from raw data
    let tensor3 (data: seq< #seq< #seq<'T>>>) : DT<'T> = 
        DT.ConstArray3D(array3D data, flex=false)

    let image data = 
        DT.ConstArray3D(array3D data, flex=true)

    /// Create a rank-4 tensor from raw data
    let tensor4 (data: seq< #seq< #seq< #seq<'T>>>>) : DT<'T> = 
        DT.ConstArray4D(array4D data, flex=false)

    let batchOfImages d = tensor4 d 
    
    let video data = 
        DT.ConstArray4D(array4D data, flex=true)

    //let batchOfVideos d = tensor5 d 
    
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