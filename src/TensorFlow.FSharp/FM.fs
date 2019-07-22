namespace TensorFlow.FSharp.DSL

open System
open System.Reflection
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

    /// One dimension is a divisor of another, striding semantics
    | DimDivInt of Dim * int

    /// The dimension is a variable, possibly solved
    | DimVar of InferenceVar<Dim>

    /// The dimension is named
    | DimNamed of string * Dim

    /// The dimension is known
    | DimKnown of int

    member internal dim.StripSolutions() = 
        match dim with 
        | DimNamed (name, dim2) -> 
            DimNamed (name, dim2.StripSolutions())
        | DimVar var -> 
            match var.Solution with 
            | Unsolved -> dim
            | Solved v -> v.StripSolutions()
        | _ -> dim

    /// Try to get the solved value for the dimension.
    // Note this is lossy on the name
    member dim.TryValue(ndeep) = 
        let rec loop ndeep (dim: Dim) = 
            if ndeep > 20 then None else
            match dim.StripSolutions() with 
            | DimNamed (_name, dim2) -> loop (ndeep+1) dim2 //.TryValue()
            | DimMulInt (expected,n) -> match loop (ndeep+1) expected (* .TryValue() *) with None -> None | Some dimv -> Some (dimv*n) 
            | DimDivInt (expected,n) -> match loop (ndeep+1) expected (* .TryValue() *) with None -> None | Some dimv -> Some (dimv/n + (if dimv % n > 0 then 1 else 0)) 
            | DimKnown n -> Some n 
            | DimVar v -> None 
        loop ndeep dim

    member internal dim.FreeVarsAcc(acc: HashSet<_>) = 
        match dim.StripSolutions() with 
        | DimNamed (_name, dim2) -> dim2.FreeVarsAcc(acc)
        | DimMulInt (dim2,n) -> dim2.FreeVarsAcc(acc)
        | DimDivInt (dim2,n) -> dim2.FreeVarsAcc(acc)
        | DimKnown n -> ()
        | DimVar v -> acc.Add(v)  |> ignore

    member dim.TryName() = 
        match dim.StripSolutions() with 
        | DimNamed (nm, dim2) -> Some (nm, dim2)
        | _ -> None

    member dim.HasName = dim.TryName().IsSome 

    member internal dim.IsSolved = dim.TryValue(0).IsSome
    
    member internal dim.ValueOrMinusOne = 
        match dim.TryValue(0) with 
        | Some value -> value
        | None -> -1

    member internal dim.ValueOrZero = 
        match dim.TryValue(0) with 
        | Some value -> value
        | None -> 0

    member dim.Value = 
        match dim.TryValue(0) with 
        | Some value -> value
        | None -> failwith "the value for the dimension could not be inferred"

    member internal dim.AsTFNode(graph: TFGraph, subst: IDictionary<InferenceVar<Dim>, TFOutput>) = 
        let rec loop (d: Dim) = 
            match d.StripSolutions() with 
            | DimMulInt (dim2, n) -> graph.Mul(loop dim2, graph.Const(new TFTensor(n)))
            | DimDivInt (dim2, n) -> graph.FloorDiv(graph.Add(loop dim2, graph.Const(new TFTensor(n-1))), graph.Const(new TFTensor(n)))
            | DimKnown n -> graph.Const(new TFTensor(n))
            | DimNamed (_, dim2) -> loop dim2
            | DimVar v -> 
               if subst.ContainsKey(v) then subst.[v] 
               else 
                   //printfn "Dim.AsTFNode: didn't find instantiation for variable dimension in %A, assuming 1" dim
                   graph.Const(new TFTensor(1))
        loop dim

    member internal dim.Subst(subst: IDictionary<InferenceVar<Dim>, Dim>) = 
        match dim.StripSolutions() with 
        | DimMulInt (dim2, n) -> DimMulInt (dim2.Subst(subst), n)
        | DimDivInt (dim2, n) -> DimDivInt (dim2.Subst(subst), n) 
        | DimKnown n -> DimKnown n
        | DimNamed (nm, dim2) -> DimNamed (nm, dim2.Subst(subst))
        | DimVar v -> 
            if subst.ContainsKey(v) then subst.[v] 
            else 
                //printfn "Dim.Subst: didn't find instantiation for variable dimension in %A, assuming unchanged" dim
                dim

    static member ( * ) (dim: Dim, stride: int) = if stride = 1 then dim else DimMulInt (dim, stride)

    static member ( / ) (dim: Dim, stride: int) = if stride = 1 then dim else DimDivInt (dim, stride)

    static member Known value = DimKnown value

    /// A dimension with an inferred value
    static member Inferred = DimVar (InferenceVar())

    /// A named dimension with a known value
    static member Named name value = DimNamed (name, Dim.Known value)

    /// A named dimension with an variable value
    static member Var name = DimNamed (name, DimVar (InferenceVar()))

    static member Unify op (actual: Dim) (expected: Dim) = 
        match Dim.UnifyInner op actual expected with
        | Ok () -> ()
        | Error msg -> failwithf "mismatched dimensions for operator %s: expected '%s' but got '%s' (%s)" op (expected.ToString())  (actual.ToString()) msg

    static member UnifyInner op (actual: Dim) (expected: Dim) = 
        match actual.TryValue(0), expected.TryValue(0) with 
        | Some v1, Some v2 -> if v1 <> v2 then Error "unequal values" else Ok()
        | _ -> 
        match actual.StripSolutions(), expected.StripSolutions() with 
        // check for identical variables
        | DimVar var1, DimVar var2 when Object.ReferenceEquals(var1,var2) -> Ok ()
        // solve
        | DimVar var1, _ -> 
            var1.Solve expected
            Ok()
        | _, DimVar var2 -> 
            var2.Solve actual
            Ok()
        | DimKnown _d1, DimKnown _d2 -> failwith "unreachable - each dimension had value"
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
        | DimNamed (name1, d1), DimNamed(name2, d2) when name1 = name2 -> 
            if name1 <> name2 then 
                printfn "named dimension '%s' was equated with named dimension '%s'" name1 name2
            Dim.UnifyInner op d1 d2
        | DimNamed (_, d1), d2 
        | d1, DimNamed (_, d2) -> 
            Dim.UnifyInner op d1 d2
        | _ -> 
            match actual.TryValue(0), expected.TryValue(0) with 
            | None, _ | _, None -> Error "incomplete dimension"
            | _ -> Ok () // equal, see above

    override dim.ToString() = 
        let rec loop ndeep (dim: Dim) = 
            if ndeep > 20 then "" else 
            match dim.TryName() with 
            | Some (name, dim2) -> 
                let slntext = loop (ndeep+1) dim2 // .ToString()
                name + (if slntext = "?" then "" else " (=" + slntext + ")")
            | None -> 
            // Check if it is a computed constant
            // We currently prefer this to showing symbolic names, e.g. N/2 or N/2*2
            match dim.TryValue(ndeep) with 
            | Some v -> string v
            | None ->  
            match dim.StripSolutions() with 
            | DimMulInt (expected, n) -> loop (ndeep+1) expected (* .ToString()  *) + "*" + string n 
            | DimDivInt (expected, n) -> loop (ndeep+1) expected (* .ToString() *) + "/" + string n 
            | DimKnown n -> string n 
            | DimNamed _ -> failwith "unreachable" 
            | DimVar _ -> "?"
        loop 0 dim

/// Represents an inferred shape
type Shape internal (flex: InferenceVar<Shape> option, suffix: Dim[]) =

    static let empty = Shape (None, [| |])

    member internal __.DimensionsWithFlexVar = 
        match flex with 
        | None -> None, suffix
        | Some v -> 
            match v.Solution with 
            | Unsolved -> Some v, suffix
            | Solved sln -> let flex, dims2 = sln.DimensionsWithFlexVar in flex, Array.append dims2 suffix

    member internal shape.DimensionsEliminatingFlex = 
        let flexvar, dims = shape.DimensionsWithFlexVar
        match flexvar with 
        | None -> ()
        | Some var -> var.Solve empty
        dims 

    member internal shape.FreeVarsAcc(acc: HashSet<_>) = 
        let _flexvar, dims = shape.DimensionsWithFlexVar
        for dim in dims do dim.FreeVarsAcc(acc)

    static member internal FreeVars(shapes: Shape[]) = 
        let acc = HashSet(HashIdentity.Reference)
        for shape in shapes do shape.FreeVarsAcc(acc)
        acc |> Seq.toArray

    /// Get the final inferred rank 
    member shape.Rank = shape.DimensionsEliminatingFlex.Length

    /// Lookup an inferred dimension
    member shape.Item 
        with get idx = 
            let dims = shape.DimensionsEliminatingFlex 
            if idx < dims.Length then 
                dims.[idx]
            else
                failwithf "tensor has insufficient dimensions, at least %d required but only %d available" idx dims.Length

    /// Get the shape as a Rank-1 shape
    member shape.AsRank1() = 
        let dims = shape.DimensionsEliminatingFlex 
        if dims.Length <> 1 then invalidArg "AsRank1" "not a rank 1 shape"
        dims.[0]

    /// Get the shape as a Rank-2 shape
    member shape.AsRank2() =
        let dims = shape.DimensionsEliminatingFlex 
        if dims.Length <> 2 then invalidArg "AsRank2" "not a rank 2 shape"
        dims.[0], dims.[1]

    /// Get the shape as a Rank-3 shape
    member shape.AsRank3() = 
        let dims = shape.DimensionsEliminatingFlex 
        if dims.Length <> 3 then invalidArg "AsRank3" "not a rank 3 shape"
        dims.[0], dims.[1], dims.[2]

    /// Get the shape as a Rank-4 shape
    member shape.AsRank4() = 
        let dims = shape.DimensionsEliminatingFlex 
        if dims.Length <> 4 then invalidArg "AsRank4" "not a rank 4 shape"
        dims.[0], dims.[1], dims.[2], dims.[3]

    member internal shape.AsFlexRank3() = 
        let flex, dims = shape.DimensionsWithFlexVar
        let dimsBefore, dims = dims |> Array.splitAt (dims.Length - 3)
        flex, dimsBefore, dims.[dims.Length-3], dims.[dims.Length-2], dims.[dims.Length-1]

    /// Get the dimensions of the shape
    member shape.Dimensions = shape.DimensionsEliminatingFlex

    /// Get the shape as a TensorFlow shape
    member shape.AsTFShape() = 
        TFShape(shape.DimensionsEliminatingFlex |> Array.map (fun dim -> int64 dim.ValueOrMinusOne))

    /// Get the shape as a TensorFlow node
    member internal shape.IsSolved = 
        shape.DimensionsEliminatingFlex |> Array.forall (fun dim -> dim.IsSolved)

    /// Get the shape as a TensorFlow node
    member internal shape.Subst(subst: IDictionary<InferenceVar<Dim>, Dim>) = 
        let dims = shape.DimensionsEliminatingFlex
        Shape.NoFlex [| for dim in dims -> dim.Subst(subst) |]

    /// Get the shape as a TensorFlow node
    member internal shape.AsTFNode(graph: TFGraph, subst: IDictionary<InferenceVar<Dim>, TFOutput>) = 
        if shape.IsSolved then 
            graph.Const(shape.AsTFShape().AsTensor())
        else
            let dims = shape.DimensionsEliminatingFlex
            if dims.Length = 0 then 
                graph.Const(shape.AsTFShape().AsTensor())
            else
                graph.Pack([| for dim in dims -> dim.AsTFNode(graph, subst) |])

    /// Copy the shape returning a map of old variables to new.  The new shape is then unified against another shape, giving a solution for those variables
    member internal shape.Match(shape2) = 
        let acc = Dictionary()
        let dims = shape.DimensionsEliminatingFlex
        let rec freshen (dim: Dim) = 
            match dim.StripSolutions() with 
            | DimMulInt (dim2, n) -> DimMulInt (freshen dim2, n) 
            | DimDivInt (dim2, n) -> DimDivInt (freshen dim2, n)
            | DimKnown _ -> dim
            | DimNamed (nm, dim2) -> DimNamed (nm, freshen dim2)
            | DimVar var -> 
                if acc.ContainsKey(var) then 
                    DimVar acc.[var] 
                else 
                    let newVar = InferenceVar()
                    acc.[var] <- newVar
                    DimVar newVar

        let dimsCopied = dims |> Array.map freshen 
        let shapeCopied = Shape.NoFlex dimsCopied
        //printfn "matching, shapeCopied = %A, shape2 = %A, shape = %A, #acc = %d" shapeCopied shape2 shape acc.Count
        Shape.Unify "match" shapeCopied shape2
        //printfn "after matching, shapeCopied = %A, shape2 = %A, shape = %A, #acc = %d" shapeCopied shape2 shape acc.Count
        dict [| for (KeyValue(v1, v2)) in acc do
                    match v2.Solution with 
                    | Unsolved -> failwith "the input shape didn't solve a shape variable"
                    | Solved d -> yield (v1, d) |]

    /// Create a shape with the given dimension information. The shape does not support broadcasting.
    static member NoFlex dims = Shape(None, dims)

    /// Create a shape with the given dimension information. The shape does not support broadcasting.
    new (dims) = Shape(None, Array.ofSeq dims)

    /// Create a shape with the given dimension information. The shape supports broadcasting to further initial dimensions.
    static member Flex dims = Shape(Some (InferenceVar()), dims)

    static member internal PossibleFlex (flex: bool) dims = if flex then Shape.Flex dims else Shape.NoFlex dims

    /// Create a new fully inferred shape 
    static member Inferred with get() = Shape.Flex [| |]
    
    /// Create a shape with the given dimensions. Same as 'shape [...]'.
    static member Known (ints: seq<int>) = 
        ints 
        |> Array.ofSeq 
        |> Array.map (fun i -> if i = -1 then Dim.Inferred else DimKnown i)
        |> Shape.NoFlex 

    /// Create a shape from a TensorFlow array of int64 values
    static member FromTFShapeArray (shape: int64[], ?flex: bool) = 
        let flex = defaultArg flex false
        let dims = shape |> Array.map (fun i -> if i = -1L then Dim.Inferred else DimKnown (int32 i))
        Shape.PossibleFlex flex dims

    /// Create a shape from a TensorFlow shape
    static member FromTFShape (shape: TFShape) = 
        shape.ToLongArray() |> Shape.FromTFShapeArray

    static member internal D = Shape [| |]
    
    static member internal DV = Shape [| Dim.Inferred |]
    
    static member internal DM = Shape [| Dim.Inferred; Dim.Inferred |]

    /// At least 'n' dimensions, possible more
    static member internal FlexN n = Shape.Flex [| for i in 1 .. n -> Dim.Inferred |]

    static member internal MinDimensions op (shape: Shape) dim = 
        let flexvar, dims = shape.DimensionsWithFlexVar
        if dim > dims.Length then 
            match flexvar with 
            | None -> 
                failwithf "shape %A must have at least %d dimensions for operator %s" shape dim op
            | Some v -> 
                v.Solve (Shape.FlexN (dim - dims.Length))

    static member internal  Unify op (actual: Shape) (expected: Shape) = 

        let rec loop (s1: Shape) (s2: Shape) =

            let flexvar1, dims1 = s1.DimensionsWithFlexVar
            let flexvar2, dims2 = s2.DimensionsWithFlexVar

            // How many dimensions in common?
            let n = min dims1.Length dims2.Length
            let dims1A, dims1B = dims1 |> Array.splitAt (dims1.Length-n)
            let dims2A, dims2B = dims2 |> Array.splitAt (dims2.Length-n)
            
            // Unify the prefix
            let prefixRes = 
                if n > 0 then
                    // Drop front dimensions - shapes smaller
                    loop (Shape(flexvar1, dims1A)) (Shape(flexvar2, dims2A))

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
                    | Some v1, _ -> v1.Solve (Shape(flexvar2, [| |])); Ok()
                    | _, Some v2 -> v2.Solve (Shape(flexvar1, [| |])); Ok()
                    | None, None -> Ok()
            match prefixRes with 

            | Error () -> Error()
            | Ok() -> 
                // Unify the common sufix
                (dims1B, dims2B) ||> Array.iter2 (fun dim1 dim2 ->
                    match Dim.UnifyInner op dim1 dim2 with 
                    | Ok () -> ()
                    | Error msg -> failwithf "mismatched shapes for operator %s: expected %A but got %A (size %s did not match %s, %s) " op expected actual (dim2.ToString()) (dim1.ToString()) msg)
                Ok()

        match loop actual expected with 
        | Ok () -> ()
        | Error () -> failwithf "mismatched shapes: expected %A but got %A for operator %s" expected actual op

    static member internal EquivShapes op (actual: Shape) (expected: Shape) = 
        Shape.Unify op actual expected
        // Preserve names from either
        let v1, dims1 = actual.DimensionsWithFlexVar
        let _v2, dims2 = expected.DimensionsWithFlexVar 
        let dims = (dims1, dims2) ||> Array.map2 (fun dim1 dim2 -> if dim2.HasName then dim2 else dim1)
        Shape(v1, dims)

    /// Convert the shape to a string
    override shape.ToString() = 
        let flexvar, dims = shape.DimensionsWithFlexVar
        if dims.Length = 0 then 
            "scalar" 
            + (if flexvar.IsSome then " (can broadcast)" else "")
        elif dims.Length = 1 then 
            "vector " + dims.[0].ToString()
            + (if flexvar.IsSome then " (can broadcast)" else "")
        elif dims.Length = 2 then 
            "matrix " + dims.[0].ToString() + " x " + dims.[1].ToString()
            + (if flexvar.IsSome then " (can broadcast)" else "")
        else
            sprintf "shape %s" (String.concat " x " [ for i in dims -> i.ToString() ]) 
            + (if flexvar.IsSome then "x.." else "")

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
type internal  TFCtxt = 
    { Graph: TFGraph 
      DimVarNodes: IDictionary<InferenceVar<Dim>, TFOutput>
      ConstNodes: Dictionary<obj, TFOutput * bool> // ensure unique nodes from unique DT values
      MomentNodes: Dictionary<DT, TFOutput * TFOutput> // ensure unique nodes from unique DT values
      AddGradientNodes: Dictionary<DT * DT[] * DT option, TFOutput[]> // ensure unique nodes from unique DT values
      Values: Map<string,DT> }

/// Represents a differentiable tensor value, which later corresponds to a node in a TensorFlow graph
and DT internal (shape: Shape, nodeCount: int, 
                 makeNode: (TFCtxt * bool -> TFOutput * bool), 
                 asTFTensor: (unit -> TFTensor) option) = 

    /// Used for operations that can produce a shape that doesn't need broadcasting
    static member internal ProducesCorrectShape f (ctxt:TFCtxt, _canProduceSmallerShape: bool) : TFOutput * bool =
        f ctxt, false

    /// Used for operations that can produce a smaller shape and may need broadcasting
    static member internal ProducesSmallerShape (outputShape: Shape) f (ctxt: TFCtxt, canProduceSmallerShape: bool) = 
        let prelimOutputNode, prelimOutputHasSmallerShape = f ctxt
        let outputHasSmallerShape = canProduceSmallerShape && prelimOutputHasSmallerShape
        let insertBroadcast  = not canProduceSmallerShape && prelimOutputHasSmallerShape
        let outputNode = 
            if insertBroadcast then 
                ctxt.Graph.BroadcastTo(prelimOutputNode, outputShape.AsTFNode(ctxt.Graph, ctxt.DimVarNodes))
            else
                prelimOutputNode
        outputNode, outputHasSmallerShape

    member internal dt.MakeNode(ctxt: TFCtxt, canProduceSmallerShape) = 
        makeNode (ctxt, canProduceSmallerShape)

    member internal dt.MakeNodeOfCorrectShape(ctxt: TFCtxt) = 
        dt.MakeNode (ctxt, false) |> fst

    static member internal MakeNodesOfCorrectShape(ctxt: TFCtxt, dts: (#DT)[]) = 
        dts |> Array.map (fun dt -> (dt :> DT).MakeNodeOfCorrectShape(ctxt))

    /// Get the inferred shape of the differentiable tensor 
    member __.Shape = shape

    /// Get the inferred shape of the differentiable tensor 
    member internal __.NodeCount = nodeCount 

    /// A quick check to see if this is a constant tensor, so we don't have to create a graph to
    /// view or analyze it.
    member internal __.TryAsConstTFTensor() = 
        if livecheck then 
            failwith "can't evaluate tensor during LiveCheck"
        match asTFTensor with 
        | None -> None 
        | Some f -> Some (f())


    /// Execute a computed DT<'T> value, returning a constant DT<'T> value.
    static member internal PreprocessCore (code: (DT[] -> DT[]), inputShapes: (Type * Shape)[], ?weights: seq<string * DT>)  = 
        if livecheck then 
            failwith "can't compile during LiveCheck"
        let session = new TFSession()
        let graph = session.Graph
        let freeDimVars = Shape.FreeVars(Array.map snd inputShapes) 
        let dimVarPlaceholders = [| for v in freeDimVars -> (v, graph.Placeholder(TFDataType.Int32)) |]
        let ctxt = 
            { Graph = graph
              DimVarNodes = dict dimVarPlaceholders
              MomentNodes = Dictionary(HashIdentity.Reference)
              AddGradientNodes = Dictionary(HashIdentity.Structural)
              ConstNodes = Dictionary(HashIdentity.Reference)
              Values = Map.ofSeq (defaultArg weights Seq.empty)}
        let placeholders = inputShapes |> Array.map (fun (ty,shape) -> DT.Placeholder(ty, shape))
        let placeholderNodes = DT.MakeNodesOfCorrectShape(ctxt, placeholders)
        let outputs = code placeholders
        let outputNodes = outputs |> Array.map (fun value -> value.MakeNodeOfCorrectShape(ctxt))
        dimVarPlaceholders, placeholderNodes, outputs, outputNodes, session

    static member Placeholder (ty, shape) : DT = 
        let nodeCount = 100
        
        // Unfortunately we have to use reflection to make an object of the correct specific type
        let arg1, arg2, arg3, arg4 = shape, nodeCount, DT.ProducesCorrectShape (fun ctxt -> ctxt.Graph.Placeholder(TFDataType.FromType ty, shape.AsTFShape())), None
        let args = [| box arg1; box arg2; box arg3; box arg4 |]

        // check the invocation against a dummy
        (fun () -> DT<int64>(arg1, arg2, arg3, ?asTFTensor=arg4)) |> ignore

        let gty = typedefof<DT<_>>.MakeGenericType(ty)
        System.Activator.CreateInstance(gty, BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance, null, args, culture=null)
          :?> DT

    static member RunTFTensors(values: DT[], ?weights: seq<string * DT>) : TFTensor[] = 
        let _dimVarPlaceholders, _placeholderNodes, _outputs, outputNodes, session = 
            DT.PreprocessCore((fun _ -> values), [| |], ?weights=weights)
        session.Run([||], [||], outputNodes)

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
        // nodeCount = 0 implies constant, e.g. result from Eval
        match value.TryAsConstTFTensor() with 
        | Some t -> t.GetValue()
        | None -> 
            if value.NodeCount < 10 then 
                DT.Run(value)
            else
                box (sprintf "%A" value.Shape + " (unevaluated)")

    static member Cast<'T>(input: DT) : DT<'T> = 
        let outputShape = input.Shape
        let nodeCount = input.NodeCount + 1
        DT<'T> (outputShape, nodeCount, 
                fun (ctxt, canProduceSmallerShape) -> 
                    let inputNode, inputHasSmallerShape = input.MakeNode(ctxt, canProduceSmallerShape)
                    let outputNode = ctxt.Graph.Cast(inputNode, TFDataType.FromType(typeof<'T>))
                    let outputHasSmallerShape = inputHasSmallerShape
                    outputNode, outputHasSmallerShape)

    /// Display constants as data and delayed nodes as shapes
    override dt.ToString() = 
        if livecheck then 
            dt.Shape.ToString()
        else
            // nodeCount = 0 implies constant, e.g. result from Eval
            match dt.TryAsConstTFTensor() with 
            | Some t -> sprintf "%A" (t.GetValue())
            | None -> sprintf "%A" dt.Shape + " (unevaluated)"

/// Represents a differentiable tensor value
and [<Sealed>] DT<'T> internal
         (shape: Shape, 
          nodeCount: int, 
          eval: (TFCtxt * bool -> TFOutput * bool), 
          ?asTFTensor: (unit -> TFTensor)) =

    inherit DT(shape, nodeCount, eval, asTFTensor)

    static member inline internal Unop f (input: DT<'T>) : DT<'T> =  
        let outputShape = input.Shape
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
            let inputNode, inputHasSmallerShape = input.MakeNode (ctxt, canProduceSmallerShape)
            let outputNode = f ctxt.Graph inputNode
            let outputHasSmallerShape = inputHasSmallerShape
            outputNode, outputHasSmallerShape)

    static member inline internal Binop opName f scalarf (input1: DT<'T>) (input2: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes opName input1.Shape input2.Shape
        let nodeCount = input1.NodeCount + input2.NodeCount + 1
        DT<_> (outputShape, nodeCount, 
               DT.ProducesSmallerShape outputShape (fun ctxt  -> 
                    // Eliminate BroadcastTo nodes when used in the context of a operator that intrinsically implements the broadcast
                   let inputNode1, inputHasSmallerShape1 = input1.MakeNode (ctxt, true)
                   let inputNode2, inputHasSmallerShape2 = input2.MakeNode (ctxt, true)
                   let prelimOutputNode = f ctxt.Graph inputNode1 inputNode2
                   let prelimOutputHasSmallerShape = inputHasSmallerShape1 && inputHasSmallerShape2
                   prelimOutputNode, prelimOutputHasSmallerShape))

    static member inline internal ReduceOp keep_dims (axis: int[] option) (input: DT<'T>) f : DT<'T> = 
        let outputShape = 
            match keep_dims, axis with
            | Some true, _ -> input.Shape 
            | _, None -> Shape.D
            | _, Some axis -> 
                // TODO: flex here?
                let inputDims = input.Shape.DimensionsEliminatingFlex
                let outputDims = inputDims |> Array.indexed |> Array.filter (fun (idx, _) -> not (Array.contains idx axis)) |> Array.map snd
                Shape.NoFlex outputDims

        let nodeCount = input.NodeCount + 1
        DT<_> (outputShape, 
               nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                   let axis = axis |> Option.map (fun axis -> ctxt.Graph.Const(new TFTensor(axis)))
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   f ctxt axis inputNode))

    static member AddN (inputs: DT<'T>[]) : DT<'T> = 
        let outputShape = inputs.[0].Shape 
        let nodeCount = (inputs |> Array.sumBy (fun (v: DT<'T>) -> v.NodeCount)) + 1
        for v in inputs do Shape.Unify "AddN" outputShape v.Shape
        DT<'T> (outputShape, 
                nodeCount, 
                DT.ProducesCorrectShape (fun ctxt -> 
                    let inputNodes = DT.MakeNodesOfCorrectShape (ctxt, inputs)
                    ctxt.Graph.AddN(inputNodes)))

    /// Pointwise negation
    static member ( ~- ) (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Neg node) input

    /// Pointwise addition
    static member (+) (input1: DT<'T>, input2: DT<'T>) : DT<'T> = 
        // TODO: should this be AddV2
        DT.Binop "(+)" (fun graph node1 node2 -> graph.Add(node1, node2)) (+) input1 input2

    /// Pointwise addition with implicit lift to broadcastable tensor
    static member (+) (input1: DT<double>, input2: double) : DT<double> = 
        input1 + DT<double>.Const (input2, flex=true)

    /// Pointwise addition with implicit lift to broadcastable tensor
    static member (+) (input1: DT<double>, input2: int) : DT<double> = 
        input1 + double input2

    /// Pointwise addition with implicit lift to broadcastable tensor
    static member (+) (input1: double, input2: DT<double>) : DT<double> = 
        DT<double>.Const (input1, flex=true) + input2

    /// Pointwise addition with implicit lift to broadcastable tensor
    static member (+) (input1: int, input2: DT<double>) : DT<double> = 
        double input1 + input2

    /// Pointwise subtraction
    static member (-) (input1: DT<'T>, input2: DT<'T>) : DT<'T> = 
        DT.Binop "(-)" (fun graph node1 node2 -> graph.Sub(node1, node2)) (-) input1 input2

    /// Pointwise subtraction with implicit lift to broadcastable tensor
    static member (-) (input1: DT<double>, input2: double) : DT<double> = 
        input1 - DT<double>.Const (input2, flex=true)

    /// Pointwise subtraction with implicit lift to broadcastable tensor
    static member (-) (input1: DT<double>, input2: int) : DT<double> = 
        input1 - double input2

    /// Pointwise subtraction with implicit lift to broadcastable tensor
    static member (-) (input1: double, input2: DT<double>) : DT<double> = 
        DT<double>.Const (input1, flex=true) - input2

    /// Pointwise subtraction with implicit lift to broadcastable tensor
    static member (-) (input1: int, input2: DT<double>) : DT<double> = 
        double input1 - input2

    /// Pointwise multiplication
    static member (*) (input1: DT<'T>, input2: DT<'T>) : DT<'T> = 
        DT.Binop "(*)" (fun graph node1 node2 -> graph.Mul(node1, node2)) ( * ) input1 input2

    /// Pointwise multiplication with implicit lift to broadcastable tensor
    static member (*) (input1: DT<double>, input2: double) : DT<double> = 
        input1 * DT<double>.Const (double input2, flex=true)

    /// Pointwise multiplication with implicit lift to broadcastable tensor
    static member (*) (input1: DT<double>, input2: int) : DT<double> = 
        input1 * double input2

    /// Pointwise multiplication with implicit lift to broadcastable tensor
    static member (*) (input1: double, input2: DT<double>) : DT<double> = 
        DT<double>.Const (input1, flex=true) * input2

    /// Pointwise multiplication with implicit lift to broadcastable tensor
    static member (*) (input1: int, input2: DT<double>) : DT<double> = 
        double input1 * input2

    /// Pointwise division
    static member (/) (input1: DT<'T>, input2: DT<'T>) : DT<'T> = 
        DT.Binop "(/)" (fun graph node1 node2 -> graph.Div(node1, node2)) (/) input1 input2

    /// Pointwise division with implicit lift to broadcastable tensor
    static member (/) (input1: DT<double>, input2: double) : DT<double> = 
        input1 / DT<double>.Const (input2, flex=true)

    /// Pointwise division with implicit lift to broadcastable tensor
    static member (/) (input1: DT<double>, input2: int) : DT<double> = 
        input1 / double input2

    /// Matrix 'MatMul' math multiplication
    static member ( *! ) (input1: DT<'T>, input2: DT<'T>) : DT<'T> = 
        let n1,m,n2 = Dim.Inferred, Dim.Inferred, Dim.Inferred 
        Shape.Unify "MatMul"  input1.Shape (Shape.NoFlex [| n1; m |])
        Shape.Unify "MatMul" input2.Shape (Shape.NoFlex [| m; n2 |])
        let outputShape = Shape.NoFlex [| n1; n2 |]
        let nodeCount = input1.NodeCount + input2.NodeCount + 1
        DT<'T> (outputShape, nodeCount, 
                DT.ProducesCorrectShape (fun ctxt -> 
                    let inputNode1 = input1.MakeNodeOfCorrectShape(ctxt)
                    let inputNode2 = input2.MakeNodeOfCorrectShape(ctxt)
                    ctxt.Graph.MatMul(inputNode1, inputNode2)))

    /// Pointwise absolute-value
    static member Abs (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Abs node) input

    /// Pointwise arc-cosine
    static member Acos (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Acos node) input

    /// Pointwise hyperbolic arc-cosine
    static member Acosh (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Acosh node) input

    /// Pointwise arc-sine
    static member Asin (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Asin node) input

    /// Pointwise cosine
    static member Cos (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Cos node) input

    /// Pointwise hyperbolic cosine
    static member Cosh (input: DT<'T>) : DT<'T> =  
        DT.Unop (fun graph node -> graph.Cosh node) input

    /// Pointwise sine
    static member Sin (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Sin node) input

    /// Pointwise hyperbolic sine
    static member Sinh (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Sinh node) input

    /// Pointwise sqrt
    static member Sqrt (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Sqrt node) input

    /// Pointwise square
    static member Square (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Square node) input

    /// Pointwise exponential
    static member Exp (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Exp node) input

    /// Pointwise relu
    static member Relu(input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Relu node) input

    /// Pointwise tangent 
    static member Tan (input: DT<'T>) : DT<'T> = 
        DT.Unop (fun graph node -> graph.Tan node) input

    /// Pointwise hyperbolic tangent
    static member Tanh (input: DT<'T>) : DT<'T> =  
        DT.Unop (fun graph node -> graph.Tanh node) input

    /// Sum along a particular axis. The default axis is zero.
    static member Sum (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v 
            (fun ctxt axis vnode -> ctxt.Graph.ReduceSum(vnode, ?axis=axis, ?keep_dims=keep_dims))

    /// Take an average along a particular axis. The default axis is zero.
    static member Mean (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v
            (fun ctxt axis vnode -> ctxt.Graph.ReduceMean(vnode, ?axis=axis, ?keep_dims=keep_dims))

    /// Take a product along a particular axis. The default axis is zero.
    static member Prod (v: DT<'T>, ?axis: int[], ?keep_dims: bool) : DT<'T> = 
        DT.ReduceOp keep_dims axis v 
            (fun ctxt axis vnode -> ctxt.Graph.ReduceProd(vnode, ?axis=axis, ?keep_dims=keep_dims))

    /// Take a minimum across all values in the tensor.
    static member Min (input: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        let outputShape = if keep_dims = Some true then input.Shape else Shape.D
        let nodeCount = input.NodeCount + 1
        DT<_> (outputShape, nodeCount, 
                   // TODO: can we propagate canProduceSmallerShape here?
               DT.ProducesCorrectShape (fun ctxt -> 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   // TODO: why do we need ReduceDims here?
                   ctxt.Graph.Min(inputNode, ctxt.Graph.ReduceDims(inputNode), ?keep_dims=keep_dims)))

    /// Take a maximum across all values in the tensor
    static member Max (input: DT<'T>, ?keep_dims: bool) : DT<'T> = 
        let outputShape = if keep_dims = Some true then input.Shape else Shape.D
        let nodeCount = input.NodeCount + 1
        DT<_> (outputShape, nodeCount, 
               // TODO: can we propagate canProduceSmallerShape here?
               DT.ProducesCorrectShape (fun ctxt -> 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   // TODO: why do we need ReduceDims here?
                   ctxt.Graph.Max(inputNode, ctxt.Graph.ReduceDims(inputNode), ?keep_dims=keep_dims)))

    // TODO: take the dimension along which to reverse
    static member Reverse (input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                   // TODO: can we propagate canProduceSmallerShape here?
                    let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                    ctxt.Graph.ReverseV2(inputNode, ctxt.Graph.Const (new TFTensor( [| 0 |])))))

    static member DiagPart (input: DT<'T>) : DT<'T> = 
        let dims = input.Shape.DimensionsEliminatingFlex
        let n = dims.Length
        if n % 2 <> 0 then invalidArg "DiagPart: v" "expected a tensor with even rank"
        for i in 0 .. n - 1 do 
            Dim.Unify "DiagPart" dims.[i] dims.[n/2 + i]
        let outputShape = Shape.NoFlex (dims.[0 .. n/2 - 1 ])
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                    let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                    ctxt.Graph.DiagPart(inputNode)))

    static member Norm (v: DT<'T>, ?axis, ?keep_dims: bool) : DT<'T> = 
        DT.Sqrt(DT.Sum(v * v, ?axis=axis, ?keep_dims= keep_dims))

    static member Trace v = 
        DT.Sum (DT.DiagPart v)

    static member TruncatedNormal(): DT<'T> = 
        let shape = Shape.Inferred
        let nodeCount = 1
        DT<'T> (shape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
            let graph = ctxt.Graph 
            let shapeNode = shape.AsTFNode(graph, ctxt.DimVarNodes)
            let outputNode = graph.TruncatedNormal(shapeNode, TFDataType.FromType typeof<'T>)
            outputNode, false)

    member input.Rank = input.Shape.Rank

    member input.Length = input.Shape.[0].Value

    member input.GetLength(n) = input.Shape.[n].Value

    member input.Slice (_begin: int[], size: int[]) : DT<'T> = 
        let inputShape = Shape.NoFlex [| for _dim in _begin -> Dim.Inferred |]
        Shape.Unify "Slice (input)" input.Shape inputShape
        if _begin.Length <> size.Length then failwith "Slice: begin and size arrays are different"
        let dims = Array.zip3 _begin size input.Shape.DimensionsEliminatingFlex 
        for b, sz, dim in dims do 
           match dim.TryValue (0) with 
           | Some d when b < 0 || b >= d || (sz <> -1 && (b+sz < 0 || (b+sz > d))) ->
               failwith "out of bounds in slice" 
           | _ -> ()
        let outputShape = 
            Shape.NoFlex 
                [| for b, sz, dim in dims -> 
                    if sz = -1 then 
                        match dim.TryValue(0) with 
                        // TODO: these Dim.Inferred are wrong in the case where the input size is unknown
                        | None -> Dim.Inferred 
                        | Some d -> Dim.Known (d - b) 
                    else 
                        Dim.Known sz |]
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                    let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                    ctxt.Graph.Slice(inputNode, ctxt.Graph.Const(new TFTensor(_begin)), ctxt.Graph.Const(new TFTensor(size)))))

    member input.Squeeze (squeeze_dims: int[]) : DT<'T> = 
        let inputShape = Shape.NoFlex [| for i, dim in Array.indexed input.Shape.DimensionsEliminatingFlex -> if Array.contains i squeeze_dims then Dim.Known 1 else dim |]
        let outputShape = Shape.NoFlex [| for i, dim in Array.indexed input.Shape.DimensionsEliminatingFlex do if not (Array.contains i squeeze_dims) then yield dim |]
        Shape.Unify "Squeeze (input)" input.Shape inputShape
        let squeezes = Array.map int64 squeeze_dims
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape  (fun ctxt -> 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   ctxt.Graph.Squeeze(inputNode, squeezes)))

    member input.GetItem (indexes: int[]) : DT<'T> = 
        input.Slice(indexes, [| for i in indexes -> 1 |]).Squeeze([| for (pos, _) in Array.indexed indexes -> pos |])

    member input.GetSlice(choices: Choice<int, int option * int option>[]) =
        let choices = Array.indexed choices

        let inputShape = Shape.NoFlex [| for c in choices -> Dim.Inferred |]
        Shape.Unify "GetSlice" input.Shape inputShape
        let inputDims = inputShape.DimensionsEliminatingFlex

        let indexes, sizes = 
            [| for i,c in choices do 
                match c with 
                | Choice1Of2 i -> yield (i,1)
                | Choice2Of2 (startIndex, endIndex) -> 
                    let startIndex = defaultArg startIndex 0
                    let endIndex = defaultArg endIndex -1
                    // TODO: this -1 loses information in the case where the input size is unknown
                    let len = if endIndex = -1 then (match inputDims.[i].TryValue(0) with None ->  -1 | Some n -> n) else endIndex - startIndex + 1
                    yield startIndex, len |]
             |> Array.unzip
               
        let squeeze_dims = [| for i,c in choices do match c with Choice1Of2 _ -> yield i | Choice2Of2 _ -> () |]
        let slice = input.Slice(indexes, sizes)
        if squeeze_dims.Length = 0 then 
            slice else 
        slice.Squeeze(squeeze_dims)

    /// Supports notation `input.[n]`. Index an element in a vector tensor.
    member input.Item 
        with get (n: int) : DT<'T> = 
            input.GetSlice [| Choice1Of2 n |]

    /// input.[n1,n2]
    member input.Item 
        with get (n1: int, n2: int) : DT<'T> = 
            input.GetSlice [| Choice1Of2 n1; Choice1Of2 n2 |]

    /// input.[n1,n2,n3]
    member input.Item 
        with get (n1: int, n2: int,n3: int) : DT<'T> = 
            input.GetSlice [| Choice1Of2 n1; Choice1Of2 n2; Choice1Of2 n3 |]

    /// input.[n1,n2,n3,n4]
    member input.Item 
        with get (n1: int, n2: int, n3: int, n4: int) : DT<'T> = 
            input.GetSlice [| Choice1Of2 n1; Choice1Of2 n2; Choice1Of2 n3 ; Choice1Of2 n4 |]

    /// input.[n1..n2] and input.[n1..] and input.[..n2]
    member input.GetSlice(startIndex: int option, endIndex: int option) =
        input.GetSlice [| Choice2Of2 (startIndex, endIndex) |]

    /// input.[n,*] and input.[n,n1..n2]
    member input.GetSlice(idx1: int, startIndex2: int option, endIndex2: int option) =
        input.GetSlice [| Choice1Of2 idx1; Choice2Of2 (startIndex2, endIndex2) |]

    /// input.[*,n] and input.[n1..n2,n]
    member input.GetSlice(startIndex1: int option, endIndex1: int option, idx2: int) =
        input.GetSlice [| Choice2Of2 (startIndex1, endIndex1); Choice1Of2 idx2 |]

    /// input.[n,*,*] and input.[n,n1..n2,m1..m2]
    member input.GetSlice(idx1: int, startIndex2: int option, endIndex2: int option, startIndex3: int option, endIndex3: int option) =
        input.GetSlice [| Choice1Of2 idx1; Choice2Of2 (startIndex2, endIndex2); Choice2Of2 (startIndex3, endIndex3) |]

    /// input.[n,*,*,*] and input.[n,n1..n2,m1..m2,p1..p2]
    member input.GetSlice(idx1: int, startIndex2: int option, endIndex2: int option, startIndex3: int option, endIndex3: int option, startIndex4: int option, endIndex4: int option) =
        input.GetSlice [| Choice1Of2 idx1; Choice2Of2 (startIndex2, endIndex2); Choice2Of2 (startIndex3, endIndex3); Choice2Of2 (startIndex4, endIndex4) |]

    // TODO: add the remaining slice operations (currently only slicing on first dimension)

    // TODO: handle expansion along multiple arbitrary dimensions
    static member ExpandDims(input: DT<'T>, ?dim: int) : DT<'T> = 
        let dim = defaultArg dim 0
        let inputShape = input.Shape
        // TODO: flex here?
        let inputDims = inputShape.DimensionsEliminatingFlex

        // Although the docs say "insert a dimension of 1" in practice the consumer expands/broadcasts to
        // arbitrary 'n'
        //
        // TODO check that this broadcasting always happens, perhaps BroadcastTo is needed
        let outputShape = Shape.NoFlex [| yield! inputDims.[0 .. dim-1]; yield Dim.Inferred; yield! inputDims.[dim..] |]
        let nodeCount = input.NodeCount + 1

        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt  -> 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   ctxt.Graph.ExpandDims(inputNode, ctxt.Graph.Const(new TFTensor( [| dim |] )))))

    //static member Concat (concat_dim: int, vs: seq<DT<'T>>) : DT<'T> = 
    //    let vs = Seq.toArray vs
    //    if vs.Length = 0 then failwith "Vec: zero elements in vector"
    //    let actual = vs.[0].Shape
    //    let outputShape = Shape [| yield! actual.DimensionsEliminatingFlex; yield Dim (vs.Length) |]
    //    DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> ctxt.Graph.Concat(ctxt.Graph.Const(new TFTensor(concat_dim)), v.Apply ctxt))

    static member Pack (inputs: seq<DT<'T>>, ?axis: int) : DT<'T> = 
        let inputs = Seq.toArray inputs
        if inputs.Length = 0 then failwith "Pack: zero elements in vector"
        let axis = defaultArg axis 0
        let inputShape = inputs.[0].Shape
        for v in inputs do Shape.Unify "Pack" inputShape v.Shape
        Shape.MinDimensions "Pack" inputShape axis
        let inputDims = inputShape.DimensionsEliminatingFlex
        let outputShape = 
            Shape.NoFlex
                [| yield! inputDims.[0 .. axis - 1]
                   yield DimKnown inputs.Length 
                   yield! inputDims.[axis..] |]
        let nodeCount = (inputs |> Array.sumBy (fun v -> v.NodeCount)) + 1
        DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
            let inputNodes = DT.MakeNodesOfCorrectShape (ctxt, inputs)
            let outputNode = ctxt.Graph.Pack(inputNodes, axis= int64 axis)
            outputNode, false)

    static member Reshape shape (input: DT<'T>) : DT<'T> = 
        let nodeCount = input.NodeCount + 1
        DT<'T>(shape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt  -> 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   let graph = ctxt.Graph
                   graph.Reshape(inputNode, shape.AsTFNode(graph, ctxt.DimVarNodes))))

    static member BroadcastTo (input: DT<'T>, outputShape: Shape) : DT<'T> = 
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesSmallerShape outputShape (fun ctxt -> 
                let prelimOutputNode = input.MakeNodeOfCorrectShape(ctxt)
                let sizesSmaller = (input.Shape.AsTFShape().ToLongArray() <> outputShape.AsTFShape().ToLongArray())
                prelimOutputNode, sizesSmaller))

    static member AssertShape (expectedShape: Shape) (input: DT<'T>) : DT<'T> = 
        Shape.Unify "AssertShape" input.Shape expectedShape 
        input

    static member internal MakeConstWithBroadcast (shape, asTFTensor: unit -> TFTensor) : DT<'T> = 
        let nodeCount = 0
        DT<'T>(shape, nodeCount, 
               DT.ProducesSmallerShape shape (fun ctxt -> 
                  let graph = ctxt.Graph
                  memoize ctxt.ConstNodes (box asTFTensor) (fun () -> 
                      let tensor = asTFTensor()
                      let sizesSmaller = (tensor.Shape <> shape.AsTFShape().ToLongArray())
                      graph.Const(tensor), sizesSmaller)))

    static member internal MakeConstWithReshape (shape, asTFTensor) : DT<'T> = 
        let nodeCount = 0
        DT<'T>(shape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                  let graph = ctxt.Graph
                  let tensor = asTFTensor()
                  let prelimOutputNode = graph.Const(tensor)
                  graph.Reshape(prelimOutputNode, shape.AsTFNode(graph, ctxt.DimVarNodes))),
               asTFTensor = asTFTensor)

    static member internal FromTFTensor (tensor: TFTensor) : DT<'T> = 
        let shape = Shape.FromTFShapeArray(tensor.Shape)
        DT.MakeConstWithBroadcast(shape, (fun () -> tensor))

    static member internal MakeScalarFromObj (obj: obj, ?flex: bool) : DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| |] // broadcast or scalar
        match obj with 
        | :? bool
        | :? byte
        | :? sbyte
        | :? int16
        | :? uint16
        | :? int32
        | :? int64
        | :? single
        | :? double -> () 
        | _ -> failwithf "invalid scalar type %A" (typeof<'T>)
        let asTFTensor () = 
            match obj with 
            | :? bool as d -> new TFTensor(d)
            | :? byte as d -> new  TFTensor(d)
            | :? sbyte as d -> new  TFTensor(d)
            | :? int16 as d -> new  TFTensor(d)
            | :? uint16 as d -> new  TFTensor(d)
            | :? int32 as d -> new  TFTensor(d)
            | :? int64 as d -> new  TFTensor(d)
            | :? single as d -> new TFTensor(d)
            | :? double as d -> new TFTensor(d)
            | _ -> failwith "unreachable"
        DT.MakeConstWithBroadcast(shape, asTFTensor)
    
    static member Const (value: 'T, ?flex: bool) : DT<'T> = 
        DT.MakeScalarFromObj(box value, ?flex=flex)

    static member ConstArray (value: System.Array, ?shape: Shape) : DT = 
        let shape = shape |> Option.defaultWith (fun () -> Shape.NoFlex [| for i in 1 .. value.Rank -> Dim.Known (value.GetLength(i-1)) |])

        // Unfortunately we have to use reflection to make an object of the correct specific type
        // based on the Array's element type
        let arg1, arg2 = (shape, (fun () -> new TFTensor(value)))
        
        // check the invocation against a dummy
        (fun () -> DT<double>.MakeConstWithReshape (arg1, arg2)) |> ignore
        
        let gty = typedefof<DT<_>>.MakeGenericType(value.GetType().GetElementType())
        gty.InvokeMember("MakeConstWithReshape", BindingFlags.InvokeMethod ||| BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Static, null, null, [| box arg1; box arg2 |], culture=null)
           :?> DT

    static member ConstArray1D (value: 'T[], ?flex: bool) : DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| Dim.Known value.Length |]
        let asTFTensor() = new TFTensor(value)
        DT.MakeConstWithBroadcast (shape, asTFTensor)

    static member ConstArray2D (value: 'T[,], ?flex: bool) : DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1))|]
        let asTFTensor() = new TFTensor(value)
        DT.MakeConstWithBroadcast (shape, asTFTensor)

    static member ConstArray3D (value: 'T[,,], ?flex: bool) : DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1)); Dim.Known (value.GetLength(2))|]
        let asTFTensor() = new TFTensor(value)
        DT.MakeConstWithBroadcast (shape, asTFTensor)

    static member ConstArray4D (value: 'T[,,,], ?flex: bool) : DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1)); Dim.Known (value.GetLength(2)); Dim.Known (value.GetLength(3))|]
        let asTFTensor() = new TFTensor(value)
        DT.MakeConstWithBroadcast (shape, asTFTensor)

   /// Add partial deriviatives of loss function
    static member internal AddGradients (y: DT<'T>, xs: DT[], ?dy: DT<'T>) =  
        Shape.Unify "AddGradients" y.Shape Shape.D
        let key = ((y :> DT), xs, (match dy with None -> None | Some d -> Some (d :> DT)))
        xs |> Array.mapi (fun i x -> 
            let outputShape = x.Shape
            (outputShape, 
             (fun ctxt -> 
                let dynodes = 
                    memoize ctxt.AddGradientNodes key (fun () -> 
                        let xnodes = DT.MakeNodesOfCorrectShape(ctxt, xs)
                        let ynode = y.MakeNodeOfCorrectShape(ctxt)
                        let dynodesIn = match dy with None -> None | Some z -> Some [| z.MakeNodeOfCorrectShape(ctxt) |]
                        let dynodes = ctxt.Graph.AddGradients([| ynode |], xnodes, ?dy=dynodesIn)
                        dynodes)
                dynodes.[i])))

    static member Variable (value: double, ?name: string) : DT<double> = 
        DT.Variable (DT.Const(value, flex=true), ?name=name)

    static member Variable (value: int, ?name: string) : DT<double> = 
        DT.Variable (double value, ?name=name)

    static member Variable (value: DT<'T>, ?name: string) : DT<'T> = 
        let outputShape = value.Shape
        let nodeCount = 100
        DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
                     let name2 = defaultArg name ""
                     match ctxt.Values.TryFind name2 with 
                     | None -> 
                         printfn "variable nodes not yet supported, and weight '%s' not found in Values, assuming constant" name2
                         //ctxt.Graph.Variable(value.Apply ctxt,name=name2).Read
                         value.MakeNode(ctxt, canProduceSmallerShape)
                     | Some t -> 
                         match t with 
                         | :? DT<'T> as vt -> vt.MakeNode(ctxt, canProduceSmallerShape)
                         | _ -> 
                         printfn "incorrect type in values, got '%A' expected '%A', assuming variable node is constant" (t.GetType()) (typeof<DT<'T>>)
                         value.MakeNode(ctxt, canProduceSmallerShape)
                         )

    /// Works on 3D or 4D (4D = batch-of-3D) tensors
    //input: V[N,H,W,C], filters: V[F1;F2;C;COut]) -> output:V[N,H,W,COut] 
    //input: V[H,W,C], filters: V[F1;F2;C;COut]) -> output:V[H,W,COut] 
    static member Conv2D (input: DT<'T>, filters: DT<'T>, out_channels: int, ?stride: int, ?padding: string, ?filter_size: int) : DT<'T> = 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let filtersShape = filters.Shape
        Shape.Unify "Conv2D (input)" input.Shape (Shape.Flex [| Dim.Inferred; Dim.Inferred; Dim.Inferred |])
        Shape.Unify "Conv2D (filters)" filtersShape 
            (Shape.NoFlex [| (match filter_size with None -> Dim.Inferred | Some sz -> Dim.Known sz)
                             (match filter_size with None -> Dim.Inferred | Some sz -> Dim.Known sz) 
                             Dim.Inferred
                             Dim.Known out_channels |])
        let flex, inputFlexDims, H, W, C = input.Shape.AsFlexRank3()
        let F1, F2, C2, COut = filtersShape.AsRank4()
        if F1.ValueOrMinusOne = -1 || F2.ValueOrMinusOne = -1 then failwith "The filter size for the convolution could not be inferred. Please specify the shape of the filters."
        Dim.Unify "Conv2D" C C2
        let outputShape = Shape (flex, [| yield! inputFlexDims; yield H/stride; yield W/stride; yield COut |]) // use the same flex in the output shape
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
            let graph = ctxt.Graph
            let dims = input.Shape.DimensionsEliminatingFlex
            let is3D = (dims.Length = 3)
            let inputNode = input.MakeNodeOfCorrectShape(ctxt)
            let inputNode = if is3D then graph.ExpandDims(inputNode, ctxt.Graph.Const(new TFTensor( [| 0 |] ))) else inputNode
            // TODO: consider the 1 stride on the channels - is it always 1
            let strides = [|1L;int64 stride;int64 stride;1L|]
            let filtersNode = filters.MakeNodeOfCorrectShape(ctxt)
            let outputNode = graph.Conv2D(inputNode, filtersNode,strides = strides, padding=padding)
            let outputNode = if is3D then graph.Squeeze(outputNode, [| 0L |]) else outputNode
            outputNode, false)

    /// Works on 3D or 4D (4D = batch-of-3D) tensors
    //
    // filter: 4-D with shape [filter_height, filter_width, in_channels, out_channels].
    // out_backprop: 3-D or 4-D with shape [batch, out_height, out_width, out_channels]. Gradients w.r.t. the output of the convolution.
    // input_sizes: An integer vector representing the shape of input, where input is a 4-D [batch, in_height, in_width, in_channels] tensor.
    // Output: 3-D of 4-D with shape [batch, in_height, in_width, in_channels]. Gradient w.r.t. the input of the convolution.
    // TODO: this doesn't yet allow for fully variable input shapes
    //
    //input: V[N,H,W,C], filters: V[F1;F2;C;COut]) -> output:V[N,H,W,COut] 
    //input: V[H,W,C], filters: V[F1;F2;C;COut]) -> output:V[H,W,COut] 
    static member Conv2DBackpropInput(filters: DT<'T>, out_backprop: DT<'T>, input_channels: int, ?stride: int, ?padding: string, ?filter_size: int) : DT<'T> = 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let filtersShape = filters.Shape
        Shape.Unify "Conv2DBackpropInput (input)" out_backprop.Shape (Shape.Flex [| Dim.Inferred; Dim.Inferred; Dim.Inferred |])
        Shape.Unify "Conv2D (filters)" filtersShape 
            (Shape.NoFlex [| (match filter_size with None -> Dim.Inferred | Some sz -> Dim.Known sz)
                             (match filter_size with None -> Dim.Inferred | Some sz -> Dim.Known sz) 
                             Dim.Known input_channels 
                             Dim.Inferred  |])
        let flex, outputFlexDims, outputHeight, outputWidth, out_channels = out_backprop.Shape.AsFlexRank3()
        //printfn "out_backprop.Shape = %A" out_backprop.Shape
        let F1, F2, C2, COut = filtersShape.AsRank4()
        if F1.ValueOrMinusOne = -1 || F2.ValueOrMinusOne = -1 then failwith "The filter size for the convolution could not be inferred. Please specify the shape of the filters."
        Dim.Unify "Conv2DBackpropInput" out_channels COut
        let inputShape = Shape (flex, [| yield! outputFlexDims; yield outputHeight*stride; yield outputWidth*stride; yield C2 |])
        let nodeCount = out_backprop.NodeCount + 100
        DT<'T>(inputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
            let graph = ctxt.Graph
            let dims = inputShape.DimensionsEliminatingFlex
            let is3D = (dims.Length = 3)
            let outputBackpropNode = out_backprop.MakeNodeOfCorrectShape(ctxt)
            let outputBackpropNodeExp = if is3D then graph.ExpandDims(outputBackpropNode, ctxt.Graph.Const(new TFTensor( [| 0 |] ))) else outputBackpropNode
            let inputShapeExp = if is3D then Shape.NoFlex [| yield Dim.Known 1;  yield! inputShape.DimensionsEliminatingFlex |] else inputShape 
            let inputSizesExp = inputShapeExp.AsTFNode(graph, ctxt.DimVarNodes)
            let strides = [|1L;int64 stride;int64 stride;1L|]
            let filtersNode = filters.MakeNodeOfCorrectShape(ctxt)
            let inputNode = ctxt.Graph.Conv2DBackpropInput(inputSizesExp, filtersNode, outputBackpropNodeExp, strides=strides, padding=padding)
            let inputNode = if is3D then graph.Squeeze(inputNode, [| 0L |]) else inputNode
            inputNode, false)

    /// Clips tensor values to a specified min and max.
    static member ClipByValue (input: DT<'T>, low: DT<'T>, high: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes "ClipByValue" (Shape.EquivShapes "ClipByValue" input.Shape low.Shape) high.Shape
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   let lowNode = low.MakeNodeOfCorrectShape(ctxt)
                   let highNode = low.MakeNodeOfCorrectShape(ctxt)
                   ctxt.Graph.ClipByValue(inputNode, lowNode, highNode)))

    /// Calculate the mean and variance of <c>input</c>
    static member Moments(input: DT<'T>, ?axes: int list) : DT<'T> * DT<'T> = 
        // Note: keep_dims = true
        let outputShape = input.Shape
        let compute (ctxt: TFCtxt) = 
            memoize ctxt.MomentNodes (upcast input) (fun () -> 
                let axes = match axes with None -> None | Some v -> Some (ctxt.Graph.Const(new TFTensor(Array.ofList v)))
                let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                ctxt.Graph.Moments(inputNode, ?axes=axes,keep_dims=true))
        let nodeCount = input.NodeCount + 1

        DT<'T>(outputShape, nodeCount, DT.ProducesCorrectShape (compute >> fst)),
        DT<'T>(outputShape, nodeCount, DT.ProducesCorrectShape (compute >> snd))

    /// <summary>
    ///    Decode a JPEG-encoded image to a uint8 tensor.
    /// </summary>
    /// <param name="contents">
    ///    0-D.  The JPEG-encoded image.
    /// </param>
    /// <param name="channels">
    ///    Optional argument. Number of color channels for the decoded image.
    /// </param>
    static member DecodeJpeg(input:DT<string>, ?channels: int) : DT<int> = // V[int,H,W,C]
        let channels = defaultArg channels 3 // CHECK ME
        let outputShape = Shape.NoFlex [| Dim.Inferred; Dim.Inferred; DimKnown channels |]
        let nodeCount = 1
        DT<_> (outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   ctxt.Graph.DecodeJpeg(contents=inputNode, channels=3L)))

    static member WithScope(name: string) : IDisposable = 
        new WithScopeDisposable(name) :> _

    static member UsingWithScope (name: string) (f: unit -> DT<'T>) : DT<'T> = 
        let input = f()
        let outputShape = input.Shape
        let nodeCount = input.NodeCount
        DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> use _scope = ctxt.Graph.NameScope(name) in input.MakeNode(ctxt, canProduceSmallerShape))

    static member CreateString(value: byte[]) : DT<string> = 
        let shape = Shape.Flex [| |]
        DT.MakeConstWithBroadcast(shape, (fun () -> TFTensor.CreateString(value)))

    member value.ToScalar () : 'T = 
        if livecheck then 
            Unchecked.defaultof<'T>
        else
            DT.Run(value) :?> 'T

    static member Preprocess (model : (DT<'T1> -> DT<'T2>), inputShape: Shape, ?weights: seq<string * DT>) : (DT<'T1> -> DT<'T2>) = 
        let dimVarPlaceholders, placeholderNodes, outputValues, resultNodes, session = 
            DT.PreprocessCore((fun inputs -> [| model (inputs.[0] :?> DT<'T1>) :> DT |]), [| (typeof<'T1>, inputShape) |], ?weights=weights)
        (fun input -> 
            let inputTensor = DT.RunTFTensor(input)
            let placeholders = Array.append (Array.map snd dimVarPlaceholders) placeholderNodes
            let dimVarSubst = inputShape.Match(input.Shape)
            let inputs = 
                [| for (dimVar,_) in dimVarPlaceholders do 
                       if dimVarSubst.ContainsKey dimVar then 
                           let dim = dimVarSubst.[dimVar] 
                           yield new TFTensor(dim.Value)
                       else
                           failwith "the input shape didn't give a value for shape variable"
                   yield inputTensor |]
            let outputTensors = session.Run(placeholders, inputs, resultNodes)
            let outputShape = outputValues.[0].Shape.Subst(dimVarSubst)
            DT.FromTFTensor outputTensors.[0] |> DT.AssertShape outputShape)

    /// Execute a computed DT<'T> value, returning a constant DT<'T> value.
    static member Eval (value: DT<'T>, ?weights: seq<string * DT>) : DT<'T> = 
        if livecheck then 
            value
        else
            let tensor = DT.RunTFTensor(value, ?weights=weights)
            DT.FromTFTensor tensor |> DT.AssertShape value.Shape

    /// Execute a pair of DT<'T> values, returning constant DT<'T> values
    static member Eval2 (value1: DT<'T1>, value2: DT<'T2>, ?weights: seq<string * DT>) : DT<'T1> * DT<'T2> = 
        if livecheck then 
            value1, value2
        else
            let values = [| (value1 :> DT); (value2 :> DT) |]
            let tensors = DT.RunTFTensors(values, ?weights=weights)
            DT.FromTFTensor tensors.[0]  |> DT.AssertShape value1.Shape, 
            DT.FromTFTensor tensors.[1]  |> DT.AssertShape value2.Shape

    /// Execute a triple of DT<'T> values, returning triple of DT<'T> values
    static member Eval3 (value1: DT<'T1>, value2: DT<'T2>, value3: DT<'T3>,  ?weights: seq<string * DT>) : DT<'T1> * DT<'T2> * DT<'T3> = 
        if livecheck then 
            value1, value2, value3
        else
            let values = [| (value1 :> DT); (value2 :> DT); (value3 :> DT) |]
            let tensors = DT.RunTFTensors(values, ?weights=weights)
            DT.FromTFTensor tensors.[0]  |> DT.AssertShape value1.Shape, 
            DT.FromTFTensor tensors.[1]  |> DT.AssertShape value2.Shape, 
            DT.FromTFTensor tensors.[2]  |> DT.AssertShape value3.Shape

    /// Execute a DT<'T> value and get its value as an object
    member value.GetValue() : obj = 
        if livecheck then 
            // TODO: give a better dummy value back here
            obj()
        else
            DT.Run(value) 

    /// Execute a DT<'T> value and get its value as an array of scalars
    member value.ToArray() : 'T[] = 
        if livecheck then 
            let dim1 = value.Shape.AsRank1()
            Array.zeroCreate dim1.ValueOrZero
        else
            DT.Run(value) :?> 'T[]

    /// Execute a DT<'T> value and get its value as a 2D array of scalars
    member value.ToArray2D() : 'T[,] = 
        if livecheck then 
            let dim1, dim2 = value.Shape.AsRank2()
            Array2D.zeroCreate dim1.ValueOrZero dim2.ValueOrZero
        else
            DT.Run(value) :?> 'T[,]

    /// Execute a DT<'T> value and get its value as a 3D array of scalars
    member value.ToArray3D() : 'T[,,] = 
        if livecheck then 
            let dim1, dim2, dim3 = value.Shape.AsRank3()
            Array3D.zeroCreate dim1.ValueOrZero dim2.ValueOrZero dim3.ValueOrZero
        else  
            DT.Run(value) :?> 'T[,,]

    /// Execute a DT<'T> value and get its value as a 4D array of scalars
    member value.ToArray4D() : 'T[,,,] = 
        if livecheck then 
            let dim1, dim2, dim3, dim4 = value.Shape.AsRank4()
            Array4D.zeroCreate dim1.ValueOrZero dim2.ValueOrZero dim3.ValueOrZero dim4.ValueOrZero
        else
            DT.Run(value) :?> 'T[,,,]

    /// Get a DT<'T> value representing zeros
    static member Zero : DT<'T> = 
        DT.MakeScalarFromObj(box (Unchecked.defaultof<'T>), flex=true)

    /// Get a dummy value with the given shape for use in live checking
    static member Dummy(shape: Shape) : DT<'T> = 
        DT.MakeConstWithBroadcast(shape, (fun () -> failwith "dummy nodes should not be evaluated during live checking"))

/// Alias for a tensor scalar.
type Scalar<'T> = DT<'T>

/// Alias for a tensor vector.
type Vector<'T> = DT<'T>

/// Alias for a tensor matrix
type Matrix<'T> = DT<'T>

/// Alias for a 3-dimensional tensor 
type Tensor3<'T> = DT<'T>

/// Alias for a 4-dimensional tensor 
type Tensor4<'T> = DT<'T>

/// Alias for a 5-dimensional tensor 
type Tensor5<'T> = DT<'T>

/// Alias for a single, batch or other shape of scalars (a vector).
type Scalars<'T> = DT<'T>

/// Alias for a single, batch or other shape of vectors.
type Vectors<'T> = DT<'T>

/// Alias for a single, batch or other shape of matrices.
type Matrices<'T> = DT<'T>

/// Alias for a 3-dimensional tensor 
type Tensor<'T> = DT<'T>

/// Alias for a tensor scalar.
type Scalar = Scalars<double>

/// Alias for a batch of scalars.
type Scalars = Scalars<double>

/// Alias for a tensor vector.
type Vector = Vector<double>

/// Alias for a single, batch or other shape of vectors.
type Vectors = Vectors<double>

/// Alias for a matrix.
type Matrix = Matrix<double>

/// Alias for a single, batch or other shape of matrices.
type Matrices = Matrix<double>

/// Alias for a 3-dimensional tensor 
type Tensor3 = Tensor3<double>

/// Alias for a 4-dimensional tensor 
type Tensor4 = Tensor4<double>

/// Alias for a 5-dimensional tensor 
type Tensor5 = Tensor5<double>

type Tensor = Tensor<double>


/// F#-style module of operations for tensor values
module DT =

    /// Differential changes in scalar or batch-of-scalars `y` with respect to differentials of `xs`. 
    let gradients (y: Scalars<'T>) (xs: DT[]) = 
        DT.AddGradients (y, xs) |> Array.map (fun (shape, f) -> 
            let nodeCount = 100 
            DT<'T>(shape, nodeCount, DT.ProducesCorrectShape f))

    /// Differential change in scalars `y` with respect to differentials of `x`. 
    let gradient (y: Scalars<'T>) (x: DT<'T>) = 
        (gradients y [| x |]).[0]

    /// Original value and first derivative of a scalars-to-scalar function `f`, at point `x`.
    let evalAndDiff (f: DT<'T> -> Scalars<'T>) (x: DT<'T>) = 
        let y = f x
        y, gradient y x

    /// First derivative of a scalars-to-scalar function `f`, at point `x`.
    let diff (f: DT<'T> -> Scalars<'T>) x = evalAndDiff f x |> snd

    /// Second derivative of a scalars-to-scalar function `f`, at point `x`.
    let diff2 (f: DT<'T> -> Scalars<'T>) x  : Scalars<'T> =
        diff (diff f) x

    /// Original value, first derivative, and second derivative of a scalars-to-scalar function `f`, at point `x`.
    let evalAndDiffAndDiff2 (f: DT<'T> -> Scalars<'T>) x : Scalars<'T> * DT<'T> * DT<'T> =
        let v, d = evalAndDiff f x
        let d2 = diff2 f x
        (v, d, d2)

    /// Original value and second derivative of a scalars-to-scalar function `f`, at point `x`.
    let diffAndDiff2 (f: DT<'T> -> Scalars<'T>) x  : DT<'T> * DT<'T> =
        evalAndDiffAndDiff2 f x |> (fun (a,_,c) -> a,c)

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`.
    let diffN n (f: DT<'T> -> Scalars<'T>) x  : DT<'T> =
        if n < 0 then invalidArg "n" "must be positive"
        elif n = 0 then f x
        else
            let rec d n f =
                match n with
                | 1 -> diff f
                | _ -> d (n - 1) (diff f)
            x |> d n f

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`, at point `x`.
    let evalAndDiffN n (f: DT<'T> -> Scalars<'T>) x  : Scalars<'T> * DT<'T> =
        (x |> f, diffN n f x)

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let evalAndGrad (f: DT<'T> -> Scalars<'T>) (x: DT<'T>) : Scalars<'T> * DT<'T> = 
        let y = f x
        let dy = gradient y x
        y, dy

    /// Gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let grad (f: DT<'T> -> Scalars<'T>) x : DT<'T> =
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
            match y.Shape.DimensionsEliminatingFlex.[0].TryValue(0) with 
            | None -> failwith "unknown vector output size in jacobian"
            | Some d -> d
        let dyv = DT.Pack [| for i in 0 .. ysize-1 -> DT.AddGradients (y.[i], x, yv.[i]) |]
        y, dyv

    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`.
    let jacobianv (f: DV<'T> -> DV<'T>) x v : DV<'T> =
        jacobianv' f x v |> snd
*)
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let evalAndJacobian (f: DT<'T> -> Vectors<'T>) (x:DT<'T>) : Vectors<'T> * Matrices<'T> =
        let y = f x
        let ysize = 
            let dims = y.Shape.DimensionsEliminatingFlex
            match dims.[dims.Length-1].TryValue(0) with 
            | None -> failwith "unknown vector output size in jacobian"
            | Some d -> d
        let jydx = DT.Pack [| for i in 0 .. ysize - 1 -> gradient y.[i] x |]
        y, jydx

    /// Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let jacobian f x = 
        evalAndJacobian f x |> snd

    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let gradAndHessian f x = 
        evalAndJacobian (grad f) x

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let evalAndGradAndHessian (f: Vector<'T> -> Scalars<'T>) x : Scalars<'T> * Vector<'T> * Matrix<'T> =
        let g, h = gradAndHessian f x
        (f x, g, h)

(*
    // NOTE: not supported on TensorFlow due to use of C++ AddGradients only supporting
    // first-order differentials 
    // Message: TensorFlow.FSharp.TFException : No gradient defined for op: OnesLike. 

    /// Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let hessian (f: DT<'T> -> Scalars<'T>) x : Matrices<'T> =
        jacobian (grad f) x

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let evalAndHessian (f: Vector<'T> -> Scalars<'T>) x : Scalars<'T> * Matrix<'T> =
        (x |> f, hessian f x)


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

(*
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let evalAndLaplacian (f: Vector<'T> -> Scalars<'T>) x : Scalars<'T> * Scalars<'T> = 
        let v, h = evalAndHessian f x
        (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let laplacian (f: Vector<'T> -> Scalars<'T>) x : Scalars<'T> =
        evalAndLaplacian f x |> snd
*)

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let evalAndCurl (f: Vector<'T> -> Vector<'T>) x =
        let v, j = evalAndJacobian f x
        //if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        v, DT.Pack [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let curl (f: Vector<'T> -> Vector<'T>) x : Vector<'T> =
        evalAndCurl f x |> snd

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let evalAndDivergence (f: Vector<'T> -> Vector<'T>) x =
        let v, j = evalAndJacobian f x
        //if j.Rows <> j.Cols then ErrorMessages.InvalidArgDiv()
        v, DT.Trace j

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let divergence (f: Vector<'T> -> Vector<'T>) x : Scalars<'T> =
        evalAndDivergence f x |> snd

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let evalAndCurlAndDivergence (f: Vector<'T> -> Vector<'T>) x =
        let v, j = evalAndJacobian f x
        //if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        v, DT.Pack [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let curlAndDivergence (f: Vector<'T> -> Vector<'T>) x : Vector<'T> * Scalars<'T> =
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
type Model = ReflectedDefinitionAttribute

/// fm { ... }  is a computational DSL (not yet using ReflectedDefinition) that does a layer of shape inference 
/// in the first runtime phase, and then the actual graph construction in the second runtime phase.   
type FMBuilder() =
    member x.Return(v: DT<'T>) = v
    
    /// Supports the use of `use _ = ...` in tf expressions
    member x.Using(v: IDisposable, f: (unit -> DT<'T>)) = 
        match v with 
        | :? WithScopeDisposable as w -> DT.UsingWithScope w.Name f
        | _ -> use x = v in f()

[<AutoOpen>]
module FMHelpers = 

    /// Create a concrete tensor shape. -1 can be used for unknown (inferred) dimensions
    let fm = FMBuilder()

    /// Create a concrete tensor shape. -1 can be used for unknown (inferred) dimensions
    let shape (ints: int list) = Shape.Known ints

    /// Create a scalar node (with implicit broadcast)
    let scalar (value:'T) : DT<'T> = DT.Const (value, flex=true)

    /// Create a scalar node (with implicit broadcast)
    let v x = scalar x

    /// Create a vector from raw data
    let vec (data:seq<'T>) : DT<'T> = 
        let d = Seq.toArray data
        DT.ConstArray1D(d, flex=false)

    /// Create a vector from existing differentiable tensors
    let vecOfScalars (xs:seq<DT<'T>>) : DT<'T> = 
        DT.Pack xs

    /// Extend the scalar node, adding a batch dimension
    let batchOfScalars d = vec d

    /// Create a matrix from raw data
    let matrix (data: seq< #seq<'T>>) : DT<'T> = 
        let data = array2D data 
        DT.ConstArray2D(data, flex=false)

    /// Create a matrix by stacking existing vectors of differentiable tensors
    let matrixOfVecs (ds:seq<DT<'T>>) : DT<'T> = 
        DT.Pack ds

    /// Extend the vector node, adding a batch dimension
    let batchOfVecs vecs = matrix vecs 

    /// Create a non-jagged 3D array from jagged data
    let array3D data = 
        let data = data |> Array.ofSeq |> Array.map array2D
        let r1, r2, r3 = data.Length, data.[0].GetLength(0), data.[0].GetLength(1)
        if (r1 <> r2) || r2 <> r3 then invalidArg "data" (sprintf "jagged input: %d x %d x %d" r1 r2 r3)
        Array3D.init r1 r2 r3 (fun i j k -> data.[i].[j,k])

    /// Create a non-jagged 4D array from jagged data
    let array4D data = 
        let data = data |> array2D |> Array2D.map array2D
        let r1,r2,r3,r4 = (data.GetLength(0), data.GetLength(1), data.[0,0].GetLength(0),data.[0,0].GetLength(1))
        if (r1 <> r2) || r2 <> r3 || r3 <> r4 then invalidArg "data" (sprintf "jagged input: %d x %d x %d x %d" r1 r2 r3 r4)
        Array4D.init r1 r2 r3 r4 (fun i j k m -> data.[i,j].[k,m])

    /// Create a rank-3 tensor from raw data
    let tensor3 (data: seq< #seq< #seq<'T>>>) : DT<'T> = 
        DT.ConstArray3D(array3D data, flex=false)

    /// Makes a tensor from a 1D array representing a pixel of an image. The inferred tensor shape
    /// may be larger if the tensor value is used in a construct where broadcasting is required.
    let pixel data = 
        DT.ConstArray1D(data, flex=true)

    /// Makes a tensor from a 3D array representing the pixels of an image as input. The inferred tensor shape
    /// may be larger if the tensor value is used in a construct where broadcasting is required.
    let image data = 
        DT.ConstArray3D(array3D data, flex=true)

    /// Create a rank-4 tensor from raw data
    let tensor4 (data: seq< #seq< #seq< #seq<'T>>>>) : DT<'T> = 
        DT.ConstArray4D(array4D data, flex=false)

    let batchOfImages d = tensor4 d 
    
    /// Makes a tensor from a 4D array representing video frames as input. The inferred tensor shape
    /// may be larger if the tensor value is used in a construct where broadcasting is required.
    let video data = 
        DT.ConstArray4D(array4D data, flex=true)

    //let batchOfVideos d = tensor5 d 
    
    let moments (x: DT<'T>) axes = 
        DT.Moments(x, axes=axes)

    /// The pointwise relu function of the elements of a tensor
    let relu (x: DT<'T>) : DT<'T> = 
        DT.Relu(x)
        // We can't use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Relu : DT<'T> -> DT<'T>) (x))

    /// The sum of the elements of a tensor
    let sum (x: DT<'T>) : DT<'T> = 
        DT.Sum(x)
        // We can't use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Sum : DT<'T> -> DT<'T>) (x))

    /// The product of the elements of a tensor
    let prod (x: DT<'T>) : DT<'T> = 
        DT.Prod(x)
        //(DT<'T>: (static member Prod : DT<'T> -> DT<'T>) (x))
        // We can't use this because of reflection issues for the live check interpreter

    /// The average value of the elements of a tensor
    let mean (x: DT<'T>) : DT<'T> = 
        DT.Mean(x)
        // We can't use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Mean : DT<'T> -> DT<'T>) (x))

    /// The max value of the elements of a tensor
    let maxValue (x: DT<'T>) : DT<'T> = 
        DT.Max(x)
        // We can't use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Max : DT<'T> -> DT<'T>) (x))

    /// The min value of the elements of a tensor
    let minValue (x: DT<'T>) : DT<'T> = 
        DT.Min(x)
        // We can't use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Min : DT<'T> -> DT<'T>) (x))


    /// The norm of a tensor
    let norm (x: DT<'T>) : DT<'T> = 
        DT.Norm(x)
        // We can't use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Norm : DT<'T> -> DT<'T>) (x))

    let inline sqr x = x * x

    /// The global random number generator
    let rnd = new System.Random()

    /// Prepare a randome number using the global generator
    let rand() = rnd.NextDouble()

    let crossEntropy (x:DT<_>) (y:DT<_>) : DT<double> = failwith "fail"
    //    -(x |> DM.toCols |> Seq.mapi (fun i v -> 
    //        (DV.standardBasis v.Length (int (float y.[0, i]))) * log v) |> Seq.sum) / x.Cols

    /// Change a 3D-array to friendly notation
    let friendly3D (d : 'T[,,]) =
        [| for i in 0..Array3D.length1 d - 1 -> [| for j in 0..Array3D.length2 d - 1 -> [| for k in 0..Array3D.length3 d - 1 -> d.[i,j,k]  |]|]|]
        |> Array.map array2D

    /// Change an 4D-array to friendly notation
    let friendly4D (d : 'T[,,,]) =
        [| for i in 0..Array4D.length1 d - 1 -> [| for j in 0..Array4D.length2 d - 1 -> [| for k in 0..Array4D.length3 d - 1 -> [| for m in 0..Array4D.length4 d - 1 -> d.[i,j,k,m]  |]|]|]|]
        |> array2D |> Array2D.map array2D

    /// Extend the value in the batch dimension
    let batchExtend (v: DT<'T>) = DT.ExpandDims v

    /// Create a batch of values
    let batch  (vs: seq<DT<'T>>) = DT.Pack vs

    /// Create a variable placeholder node
    let variable (value: DT<'T>) name = DT.Variable (value, name)

[<AttributeUsage(AttributeTargets.Field ||| AttributeTargets.Property ||| AttributeTargets.Method)>]
type LiveCheckAttribute() =
    inherit Attribute()

[<AttributeUsage(AttributeTargets.Field ||| AttributeTargets.Property ||| AttributeTargets.Method)>]
type LiveTestAttribute() =
    inherit Attribute()
