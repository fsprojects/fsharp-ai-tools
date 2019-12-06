namespace FSAI.Tools.DSL

open System
open System.Reflection
open System.Collections.Generic
open Tensorflow
open NumSharp
open System.Numerics
open System.Runtime.InteropServices
open FSAI.Tools.NNImpl

[<AutoOpen>]
module Binding = 
    let tf = Tensorflow.Binding.tf
    //let mutable invocation = 0
    //let mutable indent = ""
    //let enter msg = 
    //   let i = invocation
    //   //printfn "%sEnter %s #%d" indent msg i
    //   indent <- indent + " "
    //   invocation <- invocation + 1
    //   { new IDisposable with 
    //       member __.Dispose() = 
    //           indent <- indent.[0..indent.Length - 2]
    //           //printfn "%sExit %s #%d" indent msg i 
    //   }

#nowarn "9"

type TFGraph = Graph

type TFOutput = Tensorflow.Tensor

type ops = Tensorflow.Operations.gen_ops

[<AutoOpen>]
module LiveChecking = 
    let mutable livecheck = 
        try 
            match System.Environment.GetEnvironmentVariable("LIVECHECK") with null | "0" -> false | _ -> true 
        with _ -> false
    let WithLiveCheck() = 
       livecheck <- true
       { new IDisposable with member __.Dispose() = livecheck <- false } 

type internal InferenceVarSoln<'T> =
    | Solved of 'T
    | Unsolved
    
module Array3D = 
    /// Not properly tested, probably slow, not to be used beyond a temporary work around
    let flatten(xsss:'a[,,]) = 
      [| for x in 0..(xsss.GetLength(0)-1) do 
          for y in 0..(xsss.GetLength(1)-1) do 
            for z in 0..(xsss.GetLength(2)-1) -> xsss.[x,y,z]|]

// NOTE: This should be moved elsewhere
[<AutoOpen>]
module Utils = 
    let private convert<'T when 'T : (new : unit -> 'T) and 'T: struct and 'T :> ValueType>(x: NDArray) =
        let xs = x.Data<'T>() 
        match x.shape with
        | [||] -> x.Data<'T>().[0] |> box
        | [|_|] -> x.Data<'T>().ToArray() |> box
        | [|d1;d2|] -> 
            Array2D.init d1 d2 (fun a b -> xs.[a * d2 + b]) |> box
        | [|d1;d2;d3|] -> 
            Array3D.init d1 d2 d3 (fun a b c -> xs.[a * d2 * d3 + b * d3 + c]) |> box
        | [|d1;d2;d3;d4|] -> 
            Array4D.init d1 d2 d3 d4 (fun a b c d -> xs.[a * d2 * d3 * d4 + b * d3 * d4 + c * d4 + d]) |> box
        | x -> failwithf "Shape %A is unsupported at this time" x

    type NDArray with

        member this.ToArrayOrSingle() = 
            match this.dtype.ToString() with
            | "System.Byte" -> convert<byte> this
            | "System.SByte" -> convert<sbyte> this
            | "System.Int16" -> convert<int16> this
            | "System.UInt16" -> convert<uint16> this
            | "System.Int32" -> convert<int32> this
            | "System.UInt32" -> convert<uint32> this
            | "System.Int64" -> convert<int64> this
            | "System.UInt64" -> convert<uint64> this
            | "System.Double" -> convert<double> this
            | "System.Single" -> convert<single> this
            | x -> failwithf "Type %s is unsupported at this time" x
    
    open Microsoft.FSharp.NativeInterop

    type TF_Status = IntPtr
    type TF_Graph = IntPtr
    type TF_Output = IntPtr

    type TF_DataType with
        /// <summary>
        /// Converts a system type to a <see cref="TFDataType"/>.
        /// </summary>
        /// <param name="t">The system type to be converted.</param>
        /// <returns>The <see cref="TFDataType"/> corresponding to the given type.</returns>
        static member FromType (t:Type): TF_DataType = 
            //if true then TFDataType.Float32 else TFDataType.Unknown
            if   t = typeof<single>     then TF_DataType.TF_FLOAT
            elif t = typeof<double>    then TF_DataType.TF_DOUBLE
            elif t = typeof<int>       then TF_DataType.TF_INT32
            elif t = typeof<byte>      then TF_DataType.TF_UINT8
            elif t = typeof<int16>     then TF_DataType.TF_UINT16
            elif t = typeof<sbyte>     then TF_DataType.TF_INT8
            elif t = typeof<string>    then TF_DataType.TF_STRING
            elif t = typeof<int64>     then TF_DataType.TF_INT32
            elif t = typeof<bool>      then TF_DataType.TF_BOOL
            elif t = typeof<uint16>    then TF_DataType.TF_UINT16
            elif t = typeof<Complex>   then TF_DataType.TF_COMPLEX128
            else raise(ArgumentOutOfRangeException ("t", sprintf "The given type %A could not be mapped to an existing TFDataType." t))

    [<DllImport ("tensorflow")>]
    extern void TF_AddGradients (TF_Graph graph, TF_Output* ys, int ny, TF_Output* xs, int nx, TF_Output* dx, TF_Status status, TF_Output* dy)

    /// <summary>
    /// Adds a gradient: the operations needed to compute the partial derivatives of sum of <paramref name="y"/>` wrt to <paramref name="x"/>.
    /// </summary>
    /// <returns>The partial derivatives, the size of the array is the same as the length of the <paramref name="y"/> array.</returns>
    /// <param name="y">The y elements.</param>
    /// <param name="x">The x elements.</param>
    /// <param name="dy">Initial gradients, which represent the symbolic partial derivatives of some loss function `L` w.r.t. <paramref name="y"/> ).   
    /// If the parameter is null, the implementation will use dx for 'OnesLike' for all shapes in <paramref name="y"/></param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// d(y[0] + y[1]+ ...)/dx[0], d(y[0] + y[1] + ...)/dx[1]z...
    /// </remarks>
    type gen_ops =
        // TODO this should be updated when the Status functions progress through TensorFlow.Net
        static member add_gradients(y: TFOutput [], x: TFOutput [], ?dy: TFOutput []): TFOutput [] =
          if y = null then raise(ArgumentNullException ("y"))
          if x = null then raise(ArgumentNullException ("x"))
          dy |> Option.iter (fun dy -> 
              if dy.Length <> y.Length then 
                  raise(ArgumentException ("If dy is not null, the size of the gradients must match the size of y", "dy")))
          let ret = Array.zeroCreate<TF_Output> x.Length //new Output [x.Length]
          use pret = fixed &ret.[0]
          let f xs = xs |> Array.map (fun x -> TFOutput.op_Implicit(x): IntPtr)
          let y = f y 
          let x = f x 
          use py = fixed &y.[0]
          use px = fixed &x.[0] 
          let graph_handle = Graph.op_Implicit(tf.get_default_graph()): IntPtr
          use status = new Status()
          match dy with
          | None ->
              TF_AddGradients (graph_handle, py, y.Length, px, x.Length, NativePtr.ofNativeInt IntPtr.Zero, Status.op_Implicit(status), pret)
          | Some(dx) ->
              let dx = f dx
              use pdx = fixed &dx.[0]
              TF_AddGradients (graph_handle, py, y.Length, px, x.Length, pdx, Status.op_Implicit(status), pret)
          if status.Code <> TF_Code.TF_OK then
            failwith status.Message
          else
            [|for x in ret -> new TFOutput(x)|]

type internal InferenceVar<'T>(canSolve: bool) = 
    let mutable solution: InferenceVarSoln<'T> = Unsolved
    
    member __.IsSolved = match solution with Solved _ -> true | Unsolved -> false
    member __.CanSolve = canSolve

    member __.Solve sln = 
        if canSolve then 
            solution <- Solved sln
        else 
            failwith "can't solve"

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

    member internal dim.AsTFNode(subst: IDictionary<InferenceVar<Dim>, TFOutput>) = 
        let rec loop (d: Dim) = 
            match d.StripSolutions() with 
            | DimMulInt (dim2, n) -> tf.multiply(loop dim2, tf.constant(n))
            | DimDivInt (dim2, n) -> ops.floor_div(ops.add(loop dim2, tf.constant(n-1)), tf.constant(n))
            //| DimDivInt (dim2, n) -> tf.divide(tf.add(loop dim2, tf.constant(n-1)), tf.constant(n))
            | DimKnown n -> tf.constant(n)

            | DimNamed (_, dim2) -> loop dim2
            | DimVar v -> 
               if subst.ContainsKey(v) then subst.[v] 
               else 
                   //printfn "Dim.AsTFNode: didn'T find instantiation for variable dimension in %A, assuming 1" dim
                   tf.constant(1)
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
                //printfn "Dim.Subst: didn'T find instantiation for variable dimension in %A, assuming unchanged" dim
                dim

    static member ( * ) (dim: Dim, stride: int) = if stride = 1 then dim else DimMulInt (dim, stride)

    static member ( / ) (dim: Dim, stride: int) = if stride = 1 then dim else DimDivInt (dim, stride)

    static member Known value = DimKnown value

    /// A dimension with an inferred value
    static member Inferred = DimVar (InferenceVar(true))

    /// A named dimension with a known value
    static member Named name value = DimNamed (name, Dim.Known value)

    /// A dimension variable that gets inferred statically
    static member InferredVar name = DimNamed (name, DimVar (InferenceVar(true)))

    /// A dimension variable that is always variable and part of the input, for example
    static member Var name = DimNamed (name, DimVar (InferenceVar(false)))
    
    // A dimension variable that gets solved on the graph
    static member internal ExistentialVar name = let v = (InferenceVar(false)) in v, DimNamed (name, DimVar v)

    static member Unify op (actual: Dim) (expected: Dim) = 
        match Dim.UnifyInner op actual expected with
        | Ok () -> ()
        | Error msg -> failwithf "mismatched dimensions for operator %s: expected '%s' but got '%s' (%s)" op (expected.ToString())  (actual.ToString()) msg

    static member private occurs (v: InferenceVar<Dim>) (soln: Dim) = 
        match soln.StripSolutions() with 
        | DimVar v2 when Object.ReferenceEquals(v,v2) -> true
        | DimMulInt (d, _) -> Dim.occurs v d
        | DimDivInt (d, _) -> Dim.occurs v d
        | DimNamed (_, d) -> Dim.occurs v d
        | DimVar _ -> false
        | DimKnown _ -> false

    static member private solve (v: InferenceVar<Dim>) (vexp: Dim) (soln: Dim) = 
        if Dim.occurs v soln then 
            Error (sprintf "dimension expression '%s = %s' would be infinite" (vexp.ToString()) (soln.ToString()))
        else
            v.Solve soln
            Ok()

    static member UnifyInner op (actual: Dim) (expected: Dim) = 
        //use _holder = enter "Dim - UnifyInner"
        match actual.TryValue(0), expected.TryValue(0) with 
        | Some v1, Some v2 -> if v1 <> v2 then Error "unequal values" else Ok()
        | _ -> 
        match actual.StripSolutions(), expected.StripSolutions() with 
        // check for identical variables
        | DimVar var1, DimVar var2 when Object.ReferenceEquals(var1,var2) -> Ok ()
        // solve
        | DimVar var1, _ when var1.CanSolve -> 
            Dim.solve var1 actual expected

        | _, DimVar var2 when var2.CanSolve -> 
            Dim.solve var2 expected actual

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
            //if name1 <> name2 then 
            //    printfn "named dimension '%s' was equated with named dimension '%s'" name1 name2
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
        | Some var when var.CanSolve -> var.Solve empty
        | _ -> ()
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

    member shape.AsRank3OrMore() = 
        let _flex, dimsBefore, a, b, c = shape.AsFlexRank3()
        dimsBefore, a, b, c

    /// Get the dimensions of the shape
    member shape.Dimensions = shape.DimensionsEliminatingFlex

    /// Get the shape as a TensorFlow shape
    member shape.AsTFShape() = 
        TensorShape(shape.DimensionsEliminatingFlex |> Array.map (fun dim -> int dim.ValueOrMinusOne))

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
            tf.constant(shape.AsTFShape().dims) // NOTE: we're assuming tf.constant can handle shapes
        else
            let dims = shape.DimensionsEliminatingFlex
            if dims.Length = 0 then 
                tf.constant(shape.AsTFShape().dims) // NOTE: we're assuming tf.constant can handle shapes
            else
                let dimExprs = [| for dim in dims -> dim.AsTFNode(subst) |]
                ops.pack dimExprs

    /// Copy the shape returning a map of old variables to new.  The new shape is then unified against another shape, giving a solution for those variables
    member internal shape.Match(shape2) = 
        //use _holder = enter "Shape - Match"
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
                    let newVar = InferenceVar(true)
                    acc.[var] <- newVar
                    DimVar newVar

        let dimsCopied = dims |> Array.map freshen 
        let shapeCopied = Shape.NoFlex dimsCopied
        //printfn "matching, shapeCopied = %A, shape2 = %A, shape = %A, #acc = %d" shapeCopied shape2 shape acc.Count
        Shape.Unify "match" shapeCopied shape2
        //printfn "after matching, shapeCopied = %A, shape2 = %A, shape = %A, #acc = %d" shapeCopied shape2 shape acc.Count
        [| for (KeyValue(v1, v2)) in acc do
                    match v2.Solution with 
                    | Unsolved -> failwith "the input shape didn'T solve a shape variable"
                    | Solved d -> yield (v1, d) |]

    /// Create a shape with the given dimension information. The shape does not support broadcasting.
    static member NoFlex dims = Shape(None, dims)

    /// Create a shape with the given dimension information. The shape does not support broadcasting.
    new (dims) = Shape(None, Array.ofSeq dims)
    new (dims:int[]) = Shape(None, Array.ofSeq (dims |> Array.map (function | -1 -> Dim.Inferred | n -> Dim.Known(n))))

    /// Create a shape with the given dimension information. The shape supports broadcasting to further initial dimensions.
    static member Flex dims = Shape(Some (InferenceVar(true)), dims)

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
    static member FromTFShapeArray (shape: int[], ?flex: bool) = 
        let flex = defaultArg flex false
        let dims = shape |> Array.map (fun i -> if i = -1 then Dim.Inferred else DimKnown (int32 i))
        Shape.PossibleFlex flex dims

    /// Create a shape from a TensorFlow shape
    static member FromTFShape (shape: TensorShape) = 
        shape.dims |> Shape.FromTFShapeArray

    static member internal D = Shape Array.empty<Dim>
    
    static member internal DV = Shape [| Dim.Inferred |]
    
    static member internal DM = Shape [| Dim.Inferred; Dim.Inferred |]

    /// At least 'n' dimensions, possible more
    static member internal FlexN n = Shape.Flex [| for i in 1 .. n -> Dim.Inferred |]

    static member internal MinDimensions op (shape: Shape) dim = 
        let flexvar, dims = shape.DimensionsWithFlexVar
        if dim > dims.Length then 
            match flexvar with 
            | Some v when v.CanSolve -> 
                v.Solve (Shape.FlexN (dim - dims.Length))
            | _ -> 
                failwithf "shape %A must have at least %d dimensions for operator %s" shape dim op

    static member internal  Unify op (actual: Shape) (expected: Shape) = 

        let rec loop (s1: Shape) (s2: Shape) =
            //use _holder = enter "Shape - Unify - loop"

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
                    | Some v2 when v2.CanSolve -> 
                        v2.Solve (Shape.FlexN dims1.Length) 
                        // expected now expanded and will have 'n' in common
                        loop s1 s2 
                    | _ -> 
                        Error ()

                elif dims2.Length > 0 then
                    assert (dims1.Length = 0)
                    match flexvar1 with 
                    | Some v1 when v1.CanSolve -> 
                        v1.Solve (Shape.FlexN dims2.Length) 
                        // actual now expanded and will have 'n' in common
                        loop s1 s2 
                    | _ -> 
                        Error ()

                else

                    match flexvar1, flexvar2 with 
                    | Some v1, Some v2 when Object.ReferenceEquals(v1,v2) -> Ok ()
                    | Some v1, _ when v1.CanSolve -> v1.Solve (Shape(flexvar2, [| |])); Ok()
                    | _, Some v2 when v2.CanSolve -> v2.Solve (Shape(flexvar1, [| |])); Ok()
                    | None, None -> Ok()
                    | _ -> Error ()
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

/// Represents a context for turning DT into a TensorFlow graph
type internal  TFCtxt = 
    { Graph: TFGraph 
      
      // The tensor expressions for inferred shape dimensions
      DimVarNodes: Dictionary<InferenceVar<Dim>, TFOutput>
      
      // Ensure unique nodes from unique DT values
      Nodes: Dictionary<obj, TFOutput * bool> 

      // Ensure unique nodes from unique DT values for linked moment nodes
      MomentNodes: Dictionary<DT, TFOutput * TFOutput> 

      // Ensure unique nodes from unique DT values for linked AddGradient nodes
      AddGradientNodes: Dictionary<DT * DT[] * DT option, TFOutput[]>

      // The values to use for variable nodes on pretrained models
      Values: Map<string,DT> }

/// Represents a differentiable tensor value, which later corresponds to a node in a TensorFlow graph
and DT internal (shape: Shape, nodeCount: int, 
                 makeNode: (TFCtxt * bool -> TFOutput * bool), 
                 asNDArray: (unit -> NDArray) option) = 

    /// Used for operations that can produce a shape that doesn'T need broadcasting
    static member internal ProducesCorrectShape f (ctxt:TFCtxt, _canProduceSmallerShape: bool): TFOutput * bool =
        f ctxt, false

    /// Used for operations that can produce a smaller shape and may need broadcasting
    static member internal ProducesSmallerShape (outputShape: Shape) f (ctxt: TFCtxt, canProduceSmallerShape: bool) = 
        let prelimOutputNode, prelimOutputHasSmallerShape = f ctxt
        let insertBroadcast  = not canProduceSmallerShape && prelimOutputHasSmallerShape
        let outputNode = 
            if insertBroadcast then 
                ops.broadcast_to(prelimOutputNode, outputShape.AsTFNode(ctxt.Graph, ctxt.DimVarNodes))
            else
                prelimOutputNode
        let outputHasSmallerShape = canProduceSmallerShape && prelimOutputHasSmallerShape
        outputNode, outputHasSmallerShape

    member internal dt.MakeNode(ctxt: TFCtxt, canProduceSmallerShape) = 
        ////use _holder = enter "MakeNode"
        memoize ctxt.Nodes (box dt) (fun () -> 
            makeNode (ctxt, canProduceSmallerShape))

    member internal dt.MakeNodeOfCorrectShape(ctxt: TFCtxt) = 
        ////use _holder = enter "MakeNodeOfCorrectShape"
        dt.MakeNode (ctxt, false) |> fst

    static member internal MakeNodesOfCorrectShape(ctxt: TFCtxt, dts: (#DT)[]) = 
        dts |> Array.map (fun dt -> (dt :> DT).MakeNodeOfCorrectShape(ctxt))

    /// Get the inferred shape of the differentiable tensor 
    member __.Shape = shape

    /// Get the inferred shape of the differentiable tensor 
    member internal __.NodeCount = nodeCount 

    /// A quick check to see if this is a constant tensor, so we don'T have to create a graph to
    /// view or analyze it.
    member internal __.TryAsConstNDArray() = 
        if livecheck then 
            failwith "can'T evaluate tensor during LiveCheck"
        match asNDArray with 
        | None -> None 
        | Some f -> Some (f())

    static member Placeholder (ty, shape): DT = 
        let nodeCount = 100
        
        // Unfortunately we have to use reflection to make an object of the correct specific type
        let arg1, arg2, arg3, arg4 = 
            shape,
            nodeCount, 
            DT.ProducesCorrectShape (fun ctxt -> 
                //use _holder = enter "Placeholder - makeNode" 
                ops.placeholder(TF_DataType.FromType ty (* , shape.AsTFShape() *))), 
            None
        let args = [| box arg1; box arg2; box arg3; box arg4 |]

        // check the invocation against a dummy
        (fun () -> DT<int64>(arg1, arg2, arg3, ?asNDArray=arg4)) |> ignore

        let gty = typedefof<DT<_>>.MakeGenericType(ty)
        System.Activator.CreateInstance(gty, BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Instance, null, args, culture=null)
          :?> DT
    /// A method to transform this object to a formattable object, used by F# interactive
    /// TODO double check that `obj` is in fact the desired return type
    static member PrintTransform(value: DT): obj = 
        // nodeCount = 0 implies constant, e.g. result from Eval
        match value.TryAsConstNDArray() with 
        | Some t -> box t
        | None -> 
            if value.NodeCount < 10 then 
                box (DT.Run(value))
            else
                box (sprintf "%A" value.Shape + " (unevaluated)")

    static member Cast<'T>(input: DT): DT<'T> = 
        let outputShape = input.Shape
        let nodeCount = input.NodeCount + 1
        DT<'T> (outputShape, nodeCount, 
                fun (ctxt, canProduceSmallerShape) -> 
                    //use _holder = enter "Cast - makeNode"
                    let inputNode, inputHasSmallerShape = input.MakeNode(ctxt, canProduceSmallerShape)
                    let outputNode = tf.cast(inputNode, TF_DataType.FromType(typeof<'T>))
                    let outputHasSmallerShape = inputHasSmallerShape
                    outputNode, outputHasSmallerShape)

    /// Display constants as data and delayed nodes as shapes
    override dt.ToString() = 
        if livecheck then 
            dt.Shape.ToString()
        else
            // nodeCount = 0 implies constant, e.g. result from Eval
            match dt.TryAsConstNDArray() with 
            | Some t -> sprintf "%A" (t)
            | None -> sprintf "%A" dt.Shape + " (unevaluated)"

/// Represents a differentiable tensor value
and [<Sealed>] DT<'T> internal
         (shape: Shape, 
          nodeCount: int, 
          eval: (TFCtxt * bool -> TFOutput * bool), 
          ?asNDArray: (unit -> NDArray)) =

    inherit DT(shape, nodeCount, eval, asNDArray)

    static member inline internal Unop f (input: DT<'T>): DT<'T> =  
        let outputShape = input.Shape
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
            fun (ctxt, canProduceSmallerShape) -> 
                //use _holder = enter "Unop - makeNode"
                let inputNode, inputHasSmallerShape = input.MakeNode (ctxt, canProduceSmallerShape)
                let outputNode = f ctxt.Graph inputNode
                let outputHasSmallerShape = inputHasSmallerShape
                outputNode, outputHasSmallerShape)

    static member inline internal Binop opName f scalarf (input1: DT<'T>) (input2: DT<'T>): DT<'T> = 
        let outputShape = Shape.EquivShapes opName input1.Shape input2.Shape
        let nodeCount = input1.NodeCount + input2.NodeCount + 1
        DT<_> (outputShape, nodeCount, 
               DT.ProducesSmallerShape outputShape (fun ctxt  -> 
                   //use _holder = enter "Binop - makeNode" 
                    // Eliminate BroadcastTo nodes when used in the context of a operator that intrinsically implements the broadcast
                   let inputNode1, inputHasSmallerShape1 = input1.MakeNode (ctxt, true)
                   let inputNode2, inputHasSmallerShape2 = input2.MakeNode (ctxt, true)
                   let prelimOutputNode = f ctxt.Graph inputNode1 inputNode2
                   let prelimOutputHasSmallerShape = inputHasSmallerShape1 && inputHasSmallerShape2
                   prelimOutputNode, prelimOutputHasSmallerShape))

    static member inline internal ReduceOp keep_dims (axis: int[] option) (input: DT<'T>) (f: TFCtxt -> int[] option -> TFOutput-> TFOutput): DT<'T> = 
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
                   //use _holder = enter "Reduceop - makeNode" 
                   //let axis = axis |> Option.map (fun axis -> tf.constant(new NDArray(axis)))
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   f ctxt axis inputNode))

    static member AddN (inputs: DT<'T>[]): DT<'T> = 
        let outputShape = inputs.[0].Shape 
        let nodeCount = (inputs |> Array.sumBy (fun (v: DT<'T>) -> v.NodeCount)) + 1
        for v in inputs do Shape.Unify "AddN" outputShape v.Shape
        DT<'T> (outputShape, 
                nodeCount, 
                DT.ProducesCorrectShape (fun ctxt -> 
                    //use _holder = enter "AddN - makeNode" 
                    let inputNodes = DT.MakeNodesOfCorrectShape (ctxt, inputs)
                    ops.add_n(inputNodes)))

    /// Pointwise negation
    static member ( ~- ) (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.neg node) input

    /// Pointwise addition
    static member (+) (input1: DT<'T>, input2: DT<'T>): DT<'T> = 
        // TODO: should this be AddV2
        DT.Binop "(+)" (fun graph node1 node2 -> ops.add(node1, node2)) (+) input1 input2

    ///// Pointwise addition with implicit lift to broadcastable tensor
    //static member (+) (input1: DT<double>, input2: double): DT<double> = 
    //    input1 + DT<double>.Const (input2, flex=true)

    ///// Pointwise addition with implicit lift to broadcastable tensor
    //static member (+) (input1: DT<double>, input2: int): DT<double> = 
    //    input1 + double input2

    ///// Pointwise addition with implicit lift to broadcastable tensor
    //static member (+) (input1: double, input2: DT<double>): DT<double> = 
    //    DT<double>.Const (input1, flex=true) + input2

    ///// Pointwise addition with implicit lift to broadcastable tensor
    //static member (+) (input1: int, input2: DT<double>): DT<double> = 
    //    double input1 + input2

    /// Pointwise subtraction
    static member (-) (input1: DT<'T>, input2: DT<'T>): DT<'T> = 
        DT.Binop "(-)" (fun graph node1 node2 -> ops.sub(node1, node2)) (-) input1 input2

    ///// Pointwise subtraction with implicit lift to broadcastable tensor
    //static member (-) (input1: DT<double>, input2: double): DT<double> = 
    //    input1 - DT<double>.Const (input2, flex=true)

    ///// Pointwise subtraction with implicit lift to broadcastable tensor
    //static member (-) (input1: DT<double>, input2: int): DT<double> = 
    //    input1 - double input2

    ///// Pointwise subtraction with implicit lift to broadcastable tensor
    //static member (-) (input1: double, input2: DT<double>): DT<double> = 
    //    DT<double>.Const (input1, flex=true) - input2

    ///// Pointwise subtraction with implicit lift to broadcastable tensor
    //static member (-) (input1: int, input2: DT<double>): DT<double> = 
    //    double input1 - input2

    /// Pointwise multiplication
    static member (*) (input1: DT<'T>, input2: DT<'T>): DT<'T> = 
        DT.Binop "(*)" (fun graph node1 node2 -> ops.mul(node1, node2)) ( * ) input1 input2

    ///// Pointwise multiplication with implicit lift to broadcastable tensor
    //static member (*) (input1: DT<double>, input2: double): DT<double> = 
    //    input1 * DT<double>.Const (double input2, flex=true)

    ///// Pointwise multiplication with implicit lift to broadcastable tensor
    //static member (*) (input1: DT<double>, input2: int): DT<double> = 
    //    input1 * double input2

    ///// Pointwise multiplication with implicit lift to broadcastable tensor
    //static member (*) (input1: double, input2: DT<double>): DT<double> = 
    //    DT<double>.Const (input1, flex=true) * input2

    ///// Pointwise multiplication with implicit lift to broadcastable tensor
    //static member (*) (input1: int, input2: DT<double>): DT<double> = 
    //    double input1 * input2

    /// Pointwise division
    static member (/) (input1: DT<'T>, input2: DT<'T>): DT<'T> = 
        DT.Binop "(/)" (fun graph node1 node2 -> ops.div(node1, node2)) (/) input1 input2

    ///// Pointwise division with implicit lift to broadcastable tensor
    //static member (/) (input1: DT<double>, input2: double): DT<double> = 
    //    input1 / DT<double>.Const (input2, flex=true)

    ///// Pointwise division with implicit lift to broadcastable tensor
    //static member (/) (input1: DT<double>, input2: int): DT<double> = 
    //    input1 / double input2

    /// Matrix 'MatMul' math multiplication
    static member ( *! ) (input1: DT<'T>, input2: DT<'T>): DT<'T> = 
        let n1,m,n2 = Dim.Inferred, Dim.Inferred, Dim.Inferred 
        Shape.Unify "MatMul"  input1.Shape (Shape.NoFlex [| n1; m |])
        Shape.Unify "MatMul" input2.Shape (Shape.NoFlex [| m; n2 |])
        let outputShape = Shape.NoFlex [| n1; n2 |]
        let nodeCount = input1.NodeCount + input2.NodeCount + 1
        DT<'T> (outputShape, nodeCount, 
                DT.ProducesCorrectShape (fun ctxt -> 
                    //use _holder = enter "*! - makeNode" 
                    let inputNode1 = input1.MakeNodeOfCorrectShape(ctxt)
                    let inputNode2 = input2.MakeNodeOfCorrectShape(ctxt)
                    ops.mat_mul(inputNode1, inputNode2)))

    /// Pointwise absolute-value
    static member Abs (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.abs node) input

    /// Pointwise arc-cosine
    static member Acos (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.acos node) input

    /// Pointwise hyperbolic arc-cosine
    static member Acosh (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.acosh node) input

    /// Pointwise arc-sine
    static member Asin (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.asin node) input

    /// Pointwise cosine
    static member Cos (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.cos node) input

    /// Pointwise hyperbolic cosine
    static member Cosh (input: DT<'T>): DT<'T> =  
        DT.Unop (fun graph node -> ops.cosh node) input

    /// Pointwise sine
    static member Sin (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.sin node) input

    /// Pointwise hyperbolic sine
    static member Sinh (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.sinh node) input

    /// Pointwise sqrt
    static member Sqrt (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.sqrt node) input

    /// Pointwise square
    static member Square (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.square node) input

    /// Pointwise exponential
    static member Exp (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.exp node) input

    /// Pointwise relu
    static member Relu(input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.relu node) input

    /// Pointwise tangent 
    static member Tan (input: DT<'T>): DT<'T> = 
        DT.Unop (fun graph node -> ops.tan node) input

    /// Pointwise hyperbolic tangent
    static member Tanh (input: DT<'T>): DT<'T> =  
        DT.Unop (fun graph node -> ops.tanh node) input

    /// Sum along a particular axis. The default axis is zero.
    static member Sum (v: DT<'T>, ?axis: int[], ?keep_dims: bool): DT<'T> = 
        DT.ReduceOp keep_dims axis v 
            (fun ctxt axis vnode -> 
              //use _holder = enter "Sum - makeNode" 
              // workaround
              let axis = match axis with | Some(x) -> tf.constant(x) | _ -> null
              match keep_dims with
              | Some(keep_dims) -> math_ops.reduce_sum(vnode,axis,keepdims=keep_dims)
              | None -> math_ops.reduce_sum(vnode, axis)
              )

    /// Take an average along a particular axis. The default axis is zero.
    static member Mean (v: DT<'T>, ?axis: int[], ?keep_dims: bool): DT<'T> = 
        DT.ReduceOp keep_dims axis v
            (fun ctxt axis vnode -> 
              //use _holder = enter "Mean - makeNode" 
              match axis,keep_dims with
              | Some(axis),Some(keep_dims) -> math_ops.reduce_mean(vnode, axis=axis,keepdims=keep_dims)
              | Some(axis),None -> math_ops.reduce_mean(vnode, axis=axis)
              | None,Some(keep_dims) -> math_ops.reduce_mean(vnode, keepdims=keep_dims)
              | None,None -> math_ops.reduce_mean(vnode)
              ) 

    /// Take a product along a particular axis. The default axis is zero.
    static member Prod (v: DT<'T>, ?axis: int[], ?keep_dims: bool): DT<'T> = 
        DT.ReduceOp keep_dims axis v 
            (fun ctxt (axis: int[] option) vnode -> 
              //use _holder = enter "Prod - makeNode" 
              match axis,keep_dims with
              | Some(axis),Some(keep_dims) -> math_ops.reduce_prod(vnode, axis=axis,keepdims=keep_dims)
              | Some(axis),None -> math_ops.reduce_prod(vnode, axis=axis)
              | None,Some(keep_dims) -> math_ops.reduce_prod(vnode, keepdims=keep_dims)
              | None,None -> math_ops.reduce_prod(vnode)
              ) 

    /// Take a minimum across all values in the tensor.
    static member Min (input: DT<'T>, ?keep_dims: bool): DT<'T> = 
        let outputShape = if keep_dims = Some true then input.Shape else Shape.D
        let nodeCount = input.NodeCount + 1
        DT<_> (outputShape, nodeCount, 
                   // TODO: can we propagate canProduceSmallerShape here?
               DT.ProducesCorrectShape (fun ctxt -> 
                   //use _holder = enter "Min - makeNode" 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   let reduce_dims = ops.reduce_dims(inputNode)
                   ops.min(inputNode, reduce_dims, keep_dims= (keep_dims |> Option.toNullable))))

    /// Take a maximum across all values in the tensor
    static member Max (input: DT<'T>, ?keep_dims: bool): DT<'T> = 
        let outputShape = if keep_dims = Some true then input.Shape else Shape.D
        let nodeCount = input.NodeCount + 1
        DT<_> (outputShape, nodeCount, 
               // TODO: can we propagate canProduceSmallerShape here?
               DT.ProducesCorrectShape (fun ctxt -> 
                   //use _holder = enter "Max - makeNode" 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   ops.max(inputNode, ops.reduce_dims(inputNode), keep_dims= (keep_dims |> Option.toNullable))))

    // TODO: take the dimension along which to reverse
    static member Reverse (input: DT<'T>): DT<'T> = 
        let outputShape = input.Shape
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                    //use _holder = enter "Reverse - makeNode" 
                   // TODO: can we propagate canProduceSmallerShape here?
                    let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                    ops.reverse_v2(inputNode, tf.constant([|0|]))))

    static member DiagPart (input: DT<'T>): DT<'T> = 
        let dims = input.Shape.DimensionsEliminatingFlex
        let n = dims.Length
        if n % 2 <> 0 then invalidArg "DiagPart: v" "expected a tensor with even rank"
        for i in 0 .. n - 1 do 
            Dim.Unify "DiagPart" dims.[i] dims.[n/2 + i]
        let outputShape = Shape.NoFlex (dims.[0 .. n/2 - 1 ])
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                    //use _holder = enter "DiagPart - makeNode" 
                    let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                    ops.diag_part(inputNode)))

    static member Norm (v: DT<'T>, ?axis, ?keep_dims: bool): DT<'T> = 
        DT.Sqrt(DT.Sum(v * v, ?axis=axis, ?keep_dims= keep_dims))

    static member Trace v = 
        DT.Sum (DT.DiagPart v)

    static member TruncatedNormal(): DT<'T> = 
        let shape = Shape.Inferred
        let nodeCount = 1
        DT<'T> (shape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
            //use _holder = enter "TruncatedNormal - makeNode"
            let graph = ctxt.Graph 
            let shapeNode = shape.AsTFNode(graph, ctxt.DimVarNodes)
            let outputNode = ops.truncated_normal(shapeNode, TF_DataType.FromType typeof<'T>)
            outputNode, false)

    member input.Rank = input.Shape.Rank

    member input.Length = input.Shape.[0].Value

    member input.GetLength(n) = input.Shape.[n].Value

    member input.Slice (_begin: int[], size: int[]): DT<'T> = 
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
                    //use _holder = enter "Slice - makeNode" 
                    let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                    ops.slice(inputNode, tf.constant(_begin), tf.constant(size))))

    member input.Squeeze (squeeze_dims: int[]): DT<'T> = 
        let inputShape = Shape.NoFlex [| for i, dim in Array.indexed input.Shape.DimensionsEliminatingFlex -> if Array.contains i squeeze_dims then Dim.Known 1 else dim |]
        let outputShape = Shape.NoFlex [| for i, dim in Array.indexed input.Shape.DimensionsEliminatingFlex do if not (Array.contains i squeeze_dims) then yield dim |]
        Shape.Unify "Squeeze (input)" input.Shape inputShape
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape  (fun ctxt -> 
                   //use _holder = enter "Squeeze - makeNode" 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   ops.squeeze(inputNode, squeeze_dims)))

    member input.GetItem (indexes: int[]): DT<'T> = 
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
            slice 
        else 
            slice.Squeeze(squeeze_dims)

    /// Supports notation `input.[n]`. Index an element in a vector tensor.
    member input.Item 
        with get (n: int): DT<'T> = 
            input.GetSlice [| Choice1Of2 n |]

    /// input.[n1,n2]
    member input.Item 
        with get (n1: int, n2: int): DT<'T> = 
            input.GetSlice [| Choice1Of2 n1; Choice1Of2 n2 |]

    /// input.[n1,n2,n3]
    member input.Item 
        with get (n1: int, n2: int,n3: int): DT<'T> = 
            input.GetSlice [| Choice1Of2 n1; Choice1Of2 n2; Choice1Of2 n3 |]

    /// input.[n1,n2,n3,n4]
    member input.Item 
        with get (n1: int, n2: int, n3: int, n4: int): DT<'T> = 
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
    static member ExpandDims(input: DT<'T>, ?dim: int): DT<'T> = 
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
                   //use _holder = enter "ExpandDims - makeNode" 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   ops.expand_dims(inputNode, tf.constant([| dim |]))))

    //static member Concat (concat_dim: int, vs: seq<DT<'T>>): DT<'T> = 
    //    let vs = Seq.toArray vs
    //    if vs.Length = 0 then failwith "Vec: zero elements in vector"
    //    let actual = vs.[0].Shape
    //    let outputShape = Shape [| yield! actual.DimensionsEliminatingFlex; yield Dim (vs.Length) |]
    //    DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> ctxt.Graph.Concat(ctxt.tf.constant(new NDArray(concat_dim)), v.Apply ctxt))

    static member Pack (inputs: seq<DT<'T>>, ?axis: int): DT<'T> = 
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
            //use _holder = enter "Pack - makeNode"
            let inputNodes = DT.MakeNodesOfCorrectShape (ctxt, inputs)
            let outputNode = ops.pack(inputNodes, axis = Nullable(axis))
            outputNode, false)

    static member Reshape shape (input: DT<'T>): DT<'T> = 
        let nodeCount = input.NodeCount + 1
        DT<'T>(shape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt  -> 
                   //use _holder = enter "Reshape - makeNode" 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   let graph = ctxt.Graph
                   ops.reshape(inputNode, shape.AsTFNode(graph, ctxt.DimVarNodes))))

    static member BroadcastTo (input: DT<'T>, outputShape: Shape): DT<'T> = 
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesSmallerShape outputShape (fun ctxt -> 
                //use _holder = enter "BroadcastTo - makeNode" 
                let prelimOutputNode = input.MakeNodeOfCorrectShape(ctxt)
                let outputHasSmallerShape = (input.Shape.AsTFShape().dims.Length < outputShape.AsTFShape().dims.Length)
                prelimOutputNode, outputHasSmallerShape))

    static member AssertShape (expectedShape: Shape) (input: DT<'T>): DT<'T> = 
        Shape.Unify "AssertShape" input.Shape expectedShape 
        input

    static member internal MakeConstWithBroadcast (shape, asNDArray: unit -> NDArray): DT<'T> = 
        let nodeCount = 0
        DT<'T>(shape, nodeCount, 
               DT.ProducesSmallerShape shape (fun ctxt -> 
                    let tensor = asNDArray()
                    //printfn "tensor.shape = %A" tensor.shape
                    //printfn "shape.AsTFShape().dims = %A" (shape.AsTFShape().dims)
                    let outputHasSmallerShape = (tensor.shape.Length < shape.AsTFShape().dims.Length)
                    //let dtype = TF_DataType.FromType(typeof<'T>)
                    //let tdtype = TF_DataType.FromType(tensor.dtype)
                    let node = tf.constant(tensor)
                    //let node2 = 
                    //    if dtype <> tdtype then 
                    //        tf.cast(node, dtype)
                    //    else 
                    //       node
                    node, outputHasSmallerShape))

    static member internal MakeConstWithReshape (shape, asNDArray): DT<'T> = 
        let nodeCount = 0
        DT<'T>(shape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                  let graph = ctxt.Graph
                  let tensor = asNDArray(): NDArray
                  let prelimOutputNode = tf.constant(tensor) //tf.constant(tensor)
                  ops.reshape(prelimOutputNode, shape.AsTFNode(graph, ctxt.DimVarNodes))),
               asNDArray = asNDArray)

    /// MM) opening this up temporarily try to do a workaround
    static member (*internal*) FromNDArray (tensor: NDArray): DT<'T> = 
        let shape = Shape.FromTFShapeArray(tensor.shape)
        DT.MakeConstWithBroadcast(shape, (fun () -> tensor))

    static member internal MakeScalarFromObj (obj: obj, ?flex: bool): DT<'T> = 
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
        let asNDArray () = 
            match obj with 
            | :? bool as d -> NDArray.Scalar(d)
            | :? byte as d -> NDArray.Scalar(d)
            | :? sbyte as d -> NDArray.Scalar(d)
            | :? int16 as d -> NDArray.Scalar(d)
            | :? uint16 as d -> NDArray.Scalar(d)
            | :? int32 as d -> NDArray.Scalar(d)
            | :? int64 as d -> NDArray.Scalar(d)
            | :? single as d -> NDArray.Scalar(d)
            | :? double as d -> NDArray.Scalar(d)
            | _ -> failwith "unreachable"
        DT.MakeConstWithBroadcast(shape, asNDArray)
    
    static member Const (value: 'T, ?flex: bool): DT<'T> = 
        DT.MakeScalarFromObj(box value, ?flex=flex)

    static member ConstArray (value: System.Array, ?shape: Shape): DT = 
        let shape = shape |> Option.defaultWith (fun () -> Shape.NoFlex [| for i in 1 .. value.Rank -> Dim.Known (value.GetLength(i-1)) |])

        // Unfortunately we have to use reflection to make an object of the correct specific type
        // based on the Array's element type
        let arg1, arg2 = (shape, (fun () -> new NDArray(value)))
        
        // check the invocation against a dummy
        (fun () -> DT<double>.MakeConstWithReshape (arg1, arg2)) |> ignore
        
        let gty = typedefof<DT<_>>.MakeGenericType(value.GetType().GetElementType())
        gty.InvokeMember("MakeConstWithReshape", BindingFlags.InvokeMethod ||| BindingFlags.Public ||| BindingFlags.NonPublic ||| BindingFlags.Static, null, null, [| box arg1; box arg2 |], culture=null)
           :?> DT

    static member ConstArray1D (value: 'T[], ?flex: bool): DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| Dim.Known value.Length |]
        let asNDArray() = new NDArray(value)
        DT.MakeConstWithBroadcast (shape, asNDArray)

    static member ConstArray2D (value: 'T[,], ?flex: bool): DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1))|]
        let asNDArray() = new NDArray(value)
        DT.MakeConstWithBroadcast (shape, asNDArray)

    static member ConstArray3D (value: 'T[,,], ?flex: bool): DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1)); Dim.Known (value.GetLength(2))|]
        let asNDArray() = new NDArray(value |> Array3D.flatten, NumSharp.Shape(value.GetLength(0),value.GetLength(1),value.GetLength(2)))
        DT.MakeConstWithBroadcast (shape, asNDArray)

    static member ConstArray4D (value: 'T[,,,], ?flex: bool): DT<'T> = 
        let flex = defaultArg flex false
        let shape = Shape.PossibleFlex flex [| Dim.Known (value.GetLength(0)); Dim.Known (value.GetLength(1)); Dim.Known (value.GetLength(2)); Dim.Known (value.GetLength(3))|]
        let asNDArray() = new NDArray(value)
        DT.MakeConstWithBroadcast (shape, asNDArray)

   /// Add partial deriviatives of loss function
    static member internal AddGradients (y: DT<'T>, xs: DT[], ?dy: DT<'T>) =  
        Shape.Unify "AddGradients" y.Shape Shape.D
        let key = ((y :> DT), xs, (match dy with None -> None | Some d -> Some (d :> DT)))
        xs |> Array.mapi (fun i x -> 
            let outputShape = x.Shape
            (outputShape, 
             (fun ctxt -> 
                let dynodes = 
                    // memoize ensures linked tf.gradients output nodes
                    memoize ctxt.AddGradientNodes key (fun () -> 
                        let xnodes: Tensor[] = DT.MakeNodesOfCorrectShape(ctxt, xs)
                        let ynodes: Tensor[] = [| y.MakeNodeOfCorrectShape(ctxt) |]
                        let dynodesIn: Tensor[] option = match dy with None -> None | Some z -> Some [| z.MakeNodeOfCorrectShape(ctxt) |]
                        let dynodes = 
                            match dynodesIn with 
                            | None -> tf.gradients(ys = ynodes, xs=xnodes)
                            | Some gys -> tf.gradients(ys = ynodes, xs=xnodes, grad_ys=gys)
                        dynodes)
                dynodes.[i])))

    //static member Variable (value: double, ?name: string): DT<double> = 
    //    DT.Variable (DT.Const(value, flex=true), ?name=name)

    //static member Variable (value: int, ?name: string): DT<double> = 
    //    DT.Variable (double value, ?name=name)

    static member Variable (value: DT<'T>, ?name: string): DT<'T> = 
        let outputShape = value.Shape
        let nodeCount = 100
        DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
                     //use _holder = enter "Variable makeNode"
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
    static member Conv2D (input: DT<'T>, filters: DT<'T>, out_channels: int, ?stride: int, ?padding: string, ?filter_size: int): DT<'T> = 
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
            //use _holder = enter "Conv2D - makeNode"
            let graph = ctxt.Graph
            let dims = input.Shape.DimensionsEliminatingFlex
            let is3D = (dims.Length = 3)
            let inputNode = input.MakeNodeOfCorrectShape(ctxt)
            let inputNode = if is3D then ops.expand_dims(inputNode, tf.constant([| 0 |])) else inputNode
            // TODO: consider the 1 stride on the channels - is it always 1
            let strides = [|1;stride;stride;1|]
            let filtersNode = filters.MakeNodeOfCorrectShape(ctxt)
            let outputNode = ops.conv2d(inputNode, filtersNode,strides = strides, padding=padding)
            let outputNode = if is3D then ops.squeeze(outputNode, [| 0 |]) else outputNode
            outputNode, false)

    /// Works on 3D or 4D (4D = batch-of-3D) tensors
    //
    // filter: 4-D with shape [filter_height, filter_width, in_channels, out_channels].
    // out_backprop: 3-D or 4-D with shape [batch, out_height, out_width, out_channels]. Gradients w.r.t. the output of the convolution.
    // input_sizes: An integer vector representing the shape of input, where input is a 4-D [batch, in_height, in_width, in_channels] tensor.
    // Output: 3-D of 4-D with shape [batch, in_height, in_width, in_channels]. Gradient w.r.t. the input of the convolution.
    // TODO: this doesn'T yet allow for fully variable input shapes
    //
    //input: V[N,H,W,C], filters: V[F1;F2;C;COut]) -> output:V[N,H,W,COut] 
    //input: V[H,W,C], filters: V[F1;F2;C;COut]) -> output:V[H,W,COut] 
    static member Conv2DBackpropInput(filters: DT<'T>, out_backprop: DT<'T>, input_channels: int, ?stride: int, ?padding: string, ?filter_size: int): DT<'T> = 
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
            let outputBackpropNodeExp = if is3D then ops.expand_dims(outputBackpropNode, tf.constant([| 0 |])) else outputBackpropNode
            
            let inputShapeExp = if is3D then Shape.NoFlex [| yield Dim.Known 1;  yield! inputShape.DimensionsEliminatingFlex |] else inputShape 
            let inputSizesExp = inputShapeExp.AsTFNode(graph, ctxt.DimVarNodes)
            let strides = [|1;stride;stride;1|]
            let filtersNode = filters.MakeNodeOfCorrectShape(ctxt)
            //printfn "is3D = %b, inputShape = %A, filters.Shape = %A, outputBackpropNodeExp.shape = %A, out_backprop.Shape = %A, strides=%A, padding=%A"
            //   is3D inputShape filters.Shape outputBackpropNodeExp.shape out_backprop.Shape strides padding
            let inputNode = ops.conv2d_backprop_input(input_sizes=inputSizesExp, filter=filtersNode, out_backprop=outputBackpropNodeExp, strides=strides, padding=padding)
            let inputNode = if is3D then ops.squeeze(inputNode, [| 0 |]) else inputNode
            inputNode, false)

    /// Clips tensor values to a specified min and max.
    static member ClipByValue (input: DT<'T>, low: DT<'T>, high: DT<'T>): DT<'T> = 
        let outputShape = Shape.EquivShapes "ClipByValue" (Shape.EquivShapes "ClipByValue" input.Shape low.Shape) high.Shape
        let nodeCount = input.NodeCount + 1
        DT<'T>(outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   let lowNode = low.MakeNodeOfCorrectShape(ctxt)
                   let highNode = high.MakeNodeOfCorrectShape(ctxt)
                   ops.clip_by_value(inputNode, lowNode, highNode)))

    /// Calculate the mean and variance of <c>input</c>
    static member Moments(input: DT<'T>, axes: int list): DT<'T> * DT<'T> = 
        // Note: keep_dims = true
        let outputShape = input.Shape
        let compute (ctxt: TFCtxt) = 
            // memoize ensures linked moment output nodes
            memoize ctxt.MomentNodes (input :> DT) (fun () -> 
                //let axes = match axes with None -> None | Some v -> Some (tf.constant(Array.ofList v))
                let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                let a,b = tf.nn.moments(inputNode, axes=(axes |> List.toArray),keep_dims=true).ToTuple()
                // TODO: don't put in these cast nodes if they aren't needed
                let a = tf.cast(a, TF_DataType.FromType(typeof<'T>))
                let b = tf.cast(b, TF_DataType.FromType(typeof<'T>))
                a,b
                )
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
    static member DecodeJpeg(input:DT<string>, ?channels: int): DT<int> = // V[int,H,W,C]
        let channels = defaultArg channels 3 // CHECK ME
        let (hv, H), (wv, W) = Dim.ExistentialVar "H", Dim.ExistentialVar "W"
        let outputShape = Shape.NoFlex [| H; W; DimKnown channels |]
        let nodeCount = 1
        DT<_> (outputShape, nodeCount, 
               DT.ProducesCorrectShape (fun ctxt -> 
                   
                   let inputNode = input.MakeNodeOfCorrectShape(ctxt)
                   let outputNode = ops.decode_jpeg(contents=inputNode, channels=Nullable(channels))
                   let outputNodeShape = ops.shape(outputNode,out_type=Nullable TF_DataType.TF_INT32 )
                   // Record the solutions to the H and W sizes as shapes available at runtime
                   ctxt.DimVarNodes.Add(hv, ops.squeeze(ops.slice(outputNodeShape, tf.constant([| 0 |]), tf.constant([| 1 |]), name="slice44"), [| 0 |]))
                   ctxt.DimVarNodes.Add(wv, ops.squeeze(ops.slice(outputNodeShape, tf.constant([| 1 |]), tf.constant([| 1 |]), name="slice45"), [| 0 |]))
                   outputNode))

    static member WithScope(name: string): IDisposable = 
        new WithScopeDisposable(name) :> _

    static member UsingWithScope (name: string) (f: unit -> DT<'T>): DT<'T> = 
        let input = f()
        let outputShape = input.Shape
        let nodeCount = input.NodeCount
        DT<'T>(outputShape, nodeCount, fun (ctxt, canProduceSmallerShape) -> 
            //use _holder = enter "UsingWithScope - makeNode" 
            use _scope = tf.name_scope(name)
            input.MakeNode(ctxt, canProduceSmallerShape))

    static member CreateString(bytes: byte[]): DT<string> = 
        let shape = Shape.Flex [| |]
        DT.MakeConstWithBroadcast(shape, (fun () -> NDArray(bytes)))

    member value.ToScalar (): 'T = 
        if livecheck then 
            Unchecked.defaultof<'T>
        else
            DT.Run(value) :?> 'T

    /// Execute a computed DT<'T> value, returning a constant DT<'T> value.
    static member internal PreprocessCore (code: (DT[] -> DT[]), inputShapes: (Type * Shape)[], ?weights: seq<string * DT>)  = 
        if livecheck then 
            failwith "can'T compile during LiveCheck"
        let session = new Session()
        let graph = session.graph
        let inputDimVars = Shape.FreeVars(Array.map snd inputShapes) 
        //printfn "inputSHapes = %A" (Array.map snd inputShapes)
        //use _holder = enter "PreprocessCore: Create ctxt"
        let dimVarNodes = Dictionary()
        let inputDimVarPlaceholders = [| for v in inputDimVars -> (v, ops.placeholder(TF_DataType.TF_INT32)) |]
        for v, ph in inputDimVarPlaceholders do
           dimVarNodes.Add (v, ph)

        let ctxt = 
            { Graph = graph
              DimVarNodes = dimVarNodes
              MomentNodes = Dictionary(HashIdentity.Reference)
              AddGradientNodes = Dictionary(HashIdentity.Structural)
              Nodes = Dictionary(HashIdentity.Reference)
              Values = Map.ofSeq (defaultArg weights Seq.empty)}

        let placeholders = inputShapes |> Array.map (fun (ty,shape) -> DT.Placeholder(ty, shape))
        let placeholderNodes = DT.MakeNodesOfCorrectShape(ctxt, placeholders)

        //printfn "PreprocessCore: Run code"
        let outputs = code placeholders

        // Additional dimension variable solutions may have been recorded
        let outputNodes = outputs |> Array.map (fun value -> value.MakeNodeOfCorrectShape(ctxt))
        let outputDimVarNodes = [| for (KeyValue(v,d)) in ctxt.DimVarNodes -> (v,d) |]

        //printfn "PreprocessCore: Create output nodes, #inputDimVarPlaceholders = %d" inputDimVarPlaceholders.Length
        inputDimVarPlaceholders, placeholderNodes, outputs, outputNodes, outputDimVarNodes, session

    static member Preprocess (model: (DT<'T1> -> DT<'T2>), inputShape: Shape, ?weights: seq<string * DT>): (DT<'T1> -> DT<'T2>) = 
        //use _holder = enter "Preprocess phase one start"
        let inputDimVarPlaceholders, placeholderNodes, outputValues, outputNodes, outputDimVarNodes, session = 
            DT.PreprocessCore((fun inputs -> [| model (inputs.[0] :?> DT<'T1>) :> DT |]), [| (typeof<'T1>, inputShape) |], ?weights=weights)
        (fun input -> 
            //use _holder = enter "Preprocess phase two start"
            let inputTensor = DT.RunNDArray(input)
            let placeholders = Array.append (Array.map snd inputDimVarPlaceholders) placeholderNodes
            //printfn "inputTensor = %A, #inputDimVarPlaceholders = %d" (inputTensor.ToMuliDimArray()) inputDimVarPlaceholders.Length
            let inputVarSubst = inputShape.Match(input.Shape)
            let inputVarTable = dict inputVarSubst
            let inputs = 
                [| 
                   for (dimVar,_) in inputDimVarPlaceholders do 
                       if inputVarTable.ContainsKey dimVar then 
                           let dim = inputVarTable.[dimVar] 
                           //printfn "dimVar %A --> dim %A" dimVar dim
                           yield new NDArray([|dim.Value|])
                       else
                           failwith "the input shape didn'T give a value for shape variable"
                   yield inputTensor |]
            /// Error: System.Exception: 'You must feed a value for placeholder tensor 'Placeholder_166' with dtype float and shape [?,?,?] [[{{node Placeholder_166}}]]'
            /// Stepping through it appears that the inputs are fed correctly to the given placeholders. 
            /// Judging by the number of placeholders created in the graph it is possible that these are the wrong placeholders for the requested output
            /// Closer examination is needed as to why 166 are created when only 4 should be needed.
            /// Given that the shape is often derived from the input placeholder it is usual to only have 1 placeholder
            /// Alternatively it could be surfacing a bug in NDArray which gives the wrong dtype
            let feed_dict = [| for k,v in (placeholders, inputs) ||> Array.zip -> FeedItem(k,v) |]
            
            // Include the solved shape expressions in the output
            let allOutputNodes = [| yield! outputNodes; for v, d in outputDimVarNodes do yield d |]
            
            let outputTensors = session.run(allOutputNodes,  feed_dict = feed_dict)
            let outputTensor = outputTensors.[0]
            let outputDimVarTensors = outputTensors.[1..]
            
            // Substitute the solved shape expressions through the symbolic expected output expression
            let expectedOutputShape = outputValues.[0].Shape
            let dimVarSubst = dict [| for (v, _d), dt in (Array.zip outputDimVarNodes outputDimVarTensors) -> v, Dim.Known (dt.GetInt32()) |]
            let concreteExpectedOutputShape = expectedOutputShape.Subst(dimVarSubst)

            // Check the shape matches
            DT.FromNDArray outputTensor
               |> DT.AssertShape concreteExpectedOutputShape
           )


    static member RunNDArrays(values: DT[], ?weights: seq<string * DT>): NDArray[] = 
        let _inputDimVarPlaceholders, _placeholderNodes, _outputs, outputNodes, _outputDimVarNodes, session = 
            DT.PreprocessCore((fun _ -> values), [| |], ?weights=weights)
        let outputs = session.run(outputNodes)
        outputs

    static member RunNDArray(value: DT, ?weights: seq<string * DT>): NDArray = 
        match value.TryAsConstNDArray() with 
        | None -> DT.RunNDArrays([| value |], ?weights=weights).[0]
        | Some t -> t

    static member Run(value: DT, ?weights: seq<string * DT>): obj = 
        if livecheck then 
            // TODO: give a better dummy value back here
            obj()
        else
            DT.RunNDArray(value, ?weights=weights).ToArrayOrSingle() |> box

    static member Run(values: DT[], ?weights: seq<string * DT>): obj[] = 
        if livecheck then 
            // TODO: give a better dummy value back here
            [| for v in values -> obj() |]
        else
            let results = DT.RunNDArrays(values, ?weights=weights)
            [| for res in results -> res.ToArrayOrSingle() |]

    /// Execute a computed DT<'T> value, returning a constant DT<'T> value.
    static member Eval (value: DT<'T>, ?weights: seq<string * DT>): DT<'T> = 
        if livecheck then 
            value
        else
            let tensor = DT.RunNDArray(value, ?weights=weights)
            DT.FromNDArray tensor |> DT.AssertShape value.Shape

    /// Execute a pair of DT<'T> values, returning constant DT<'T> values
    static member Eval2 (value1: DT<'T1>, value2: DT<'T2>, ?weights: seq<string * DT>): DT<'T1> * DT<'T2> = 
        if livecheck then 
            value1, value2
        else
            let values = [| (value1 :> DT); (value2 :> DT) |]
            let tensors = DT.RunNDArrays(values, ?weights=weights)
            DT.FromNDArray tensors.[0]  |> DT.AssertShape value1.Shape, 
            DT.FromNDArray tensors.[1]  |> DT.AssertShape value2.Shape

    /// Execute a triple of DT<'T> values, returning triple of DT<'T> values
    static member Eval3 (value1: DT<'T1>, value2: DT<'T2>, value3: DT<'T3>,  ?weights: seq<string * DT>): DT<'T1> * DT<'T2> * DT<'T3> = 
        if livecheck then 
            value1, value2, value3
        else
            let values = [| (value1 :> DT); (value2 :> DT); (value3 :> DT) |]
            let tensors = DT.RunNDArrays(values, ?weights=weights)
            DT.FromNDArray tensors.[0]  |> DT.AssertShape value1.Shape, 
            DT.FromNDArray tensors.[1]  |> DT.AssertShape value2.Shape, 
            DT.FromNDArray tensors.[2]  |> DT.AssertShape value3.Shape

    /// Execute a DT<'T> value and get its value as an object
    member value.GetValue(): obj = 
        if livecheck then 
            // TODO: give a better dummy value back here
            obj()
        else
            DT.Run(value) 

    /// Execute a DT<'T> value and get its value as an array of scalars
    member value.ToArray(): 'T[] = 
        if livecheck then 
            let dim1 = value.Shape.AsRank1()
            Array.zeroCreate dim1.ValueOrZero
        else
            DT.Run(value) :?> 'T[]

    /// Execute a DT<'T> value and get its value as a 2D array of scalars
    member value.ToArray2D(): 'T[,] = 
        if livecheck then 
            let dim1, dim2 = value.Shape.AsRank2()
            Array2D.zeroCreate dim1.ValueOrZero dim2.ValueOrZero
        else
            DT.Run(value) :?> 'T[,]

    /// Execute a DT<'T> value and get its value as a 3D array of scalars
    member value.ToArray3D(): 'T[,,] = 
        if livecheck then 
            let dim1, dim2, dim3 = value.Shape.AsRank3()
            Array3D.zeroCreate dim1.ValueOrZero dim2.ValueOrZero dim3.ValueOrZero
        else  
            DT.Run(value) :?> 'T[,,]

    /// Execute a DT<'T> value and get its value as a 4D array of scalars
    member value.ToArray4D(): 'T[,,,] = 
        if livecheck then 
            let dim1, dim2, dim3, dim4 = value.Shape.AsRank4()
            Array4D.zeroCreate dim1.ValueOrZero dim2.ValueOrZero dim3.ValueOrZero dim4.ValueOrZero
        else
            DT.Run(value) :?> 'T[,,,]

    /// Get a DT<'T> value representing zeros
    static member Zero: DT<'T> = 
        DT.MakeScalarFromObj(box (Unchecked.defaultof<'T>), flex=true)

    /// Get a dummy value with the given shape for use in live checking
    static member Dummy(shape: Shape): DT<'T> = 
        DT.MakeConstWithBroadcast(shape, (fun () -> 
            if livecheck then 
                failwith "dummy nodes should not be evaluated during live checking"
            else 
                failwith "livechecking is not enabled but dummy nodes have been used"))

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
    let diff2 (f: DT<'T> -> Scalars<'T>) x : Scalars<'T> =
        diff (diff f) x

    /// Original value, first derivative, and second derivative of a scalars-to-scalar function `f`, at point `x`.
    let evalAndDiffAndDiff2 (f: DT<'T> -> Scalars<'T>) x: Scalars<'T> * DT<'T> * DT<'T> =
        let v, d = evalAndDiff f x
        let d2 = diff2 f x
        (v, d, d2)

    /// Original value and second derivative of a scalars-to-scalar function `f`, at point `x`.
    let diffAndDiff2 (f: DT<'T> -> Scalars<'T>) x : DT<'T> * DT<'T> =
        evalAndDiffAndDiff2 f x |> (fun (a,_,c) -> a,c)

    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`.
    let diffN n (f: DT<'T> -> Scalars<'T>) x : DT<'T> =
        if n < 0 then invalidArg "n" "must be positive"
        elif n = 0 then f x
        else
            let rec d n f =
                match n with
                | 1 -> diff f
                | _ -> d (n - 1) (diff f)
            x |> d n f

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`, at point `x`.
    let evalAndDiffN n (f: DT<'T> -> Scalars<'T>) x : Scalars<'T> * DT<'T> =
        (x |> f, diffN n f x)

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let evalAndGrad (f: DT<'T> -> Scalars<'T>) (x: DT<'T>): Scalars<'T> * DT<'T> = 
        let y = f x
        let dy = gradient y x
        y, dy

    /// Gradient of a vector-to-scalar function `f`, at point `x`. Reverse AD.
    let grad (f: DT<'T> -> Scalars<'T>) x: DT<'T> =
        evalAndGrad f x |> snd

(*
    /// Original value and gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`.
    let gradv' (f: DV<'T> -> D<'T>) x (v: DV<'T>): D<'T> * D<'T> =
        let yv = f v
        let y = f x
        let dyv = DT.AddGradients (y, x, yv)
        y, dyv

    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`.
    let gradv (f: DV<'T> -> D<'T>) x v: D<'T> =
        gradv' f x v |> snd

    /// Original value and Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`.
    let jacobianv' (f: DV<'T> -> DV<'T>) (x: DV<'T>) (v: DV<'T>): DV<'T> * DV<'T> =
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
    let jacobianv (f: DV<'T> -> DV<'T>) x v: DV<'T> =
        jacobianv' f x v |> snd
*)
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`. Forward or reverse AD, depending on input and output dimensions.
    let evalAndJacobian (f: DT<'T> -> Vectors<'T>) (x:DT<'T>): Vectors<'T> * Matrices<'T> =
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
    let evalAndGradAndHessian (f: Vector<'T> -> Scalars<'T>) x: Scalars<'T> * Vector<'T> * Matrix<'T> =
        let g, h = gradAndHessian f x
        (f x, g, h)

(*
    // NOTE: not supported on TensorFlow due to use of C++ AddGradients only supporting
    // first-order differentials 
    // Message: FSAI.Tools.TFException: No gradient defined for op: OnesLike. 

    /// Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let hessian (f: DT<'T> -> Scalars<'T>) x: Matrices<'T> =
        jacobian (grad f) x

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`. Forward-on-reverse AD.
    let evalAndHessian (f: Vector<'T> -> Scalars<'T>) x: Scalars<'T> * Matrix<'T> =
        (x |> f, hessian f x)


    /// Original value, gradient-vector product (directional derivative), and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let gradAndHessianv' (f: DV<'T> -> D<'T>) x v =
        let gv, hv = evalAndGrad (fun xx -> gradv f xx v) x
        (x |> f, gv, hv)

    /// Gradient-vector product (directional derivative) and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let gradAndHessianv (f: DV<'T> -> D<'T>) x v: D<'T> * DV<'T> =
        gradAndHessianv' f x v |> (fun (_,b,c) -> b,c)

    /// Original value and Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let hessianv' (f: DV<'T> -> D<'T>) x v =
        gradAndHessianv' f x v |> (fun (a,_,c) -> a,c)

    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`. Reverse-on-forward AD.
    let hessianv (f: DV<'T> -> D<'T>) x v: DV<'T> =
        hessianv' f x v |> snd
*)

    let trace v = DT.Sum (DT.DiagPart v)

(*
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let evalAndLaplacian (f: Vector<'T> -> Scalars<'T>) x: Scalars<'T> * Scalars<'T> = 
        let v, h = evalAndHessian f x
        (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`. Reverse-on-forward AD.
    let laplacian (f: Vector<'T> -> Scalars<'T>) x: Scalars<'T> =
        evalAndLaplacian f x |> snd
*)

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let evalAndCurl (f: Vector<'T> -> Vector<'T>) x =
        let v, j = evalAndJacobian f x
        //if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        v, DT.Pack [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let curl (f: Vector<'T> -> Vector<'T>) x: Vector<'T> =
        evalAndCurl f x |> snd

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let evalAndDivergence (f: Vector<'T> -> Vector<'T>) x =
        let v, j = evalAndJacobian f x
        //if j.Rows <> j.Cols then ErrorMessages.InvalidArgDiv()
        v, DT.Trace j

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let divergence (f: Vector<'T> -> Vector<'T>) x: Scalars<'T> =
        evalAndDivergence f x |> snd

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let evalAndCurlAndDivergence (f: Vector<'T> -> Vector<'T>) x =
        let v, j = evalAndJacobian f x
        //if (j.Rows, j.Cols) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        v, DT.Pack [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let curlAndDivergence (f: Vector<'T> -> Vector<'T>) x: Vector<'T> * Scalars<'T> =
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
    let scalar (value:'T): DT<'T> = DT.Const (value, flex=true)

    /// Create a scalar node (with implicit broadcast)
    let v x = scalar x

    /// Create a vector from raw data
    let vec (data:seq<'T>): DT<'T> = 
        let d = Seq.toArray data
        DT.ConstArray1D(d, flex=false)

    /// Create a vector from existing differentiable tensors
    let vecOfScalars (xs:seq<DT<'T>>): DT<'T> = 
        DT.Pack xs

    /// Extend the scalar node, adding a batch dimension
    let batchOfScalars d = vec d

    /// Create a matrix from raw data
    let matrix (data: seq< #seq<'T>>): DT<'T> = 
        let data = array2D data 
        DT.ConstArray2D(data, flex=false)

    /// Create a matrix by stacking existing vectors of differentiable tensors
    let matrixOfVecs (ds:seq<DT<'T>>): DT<'T> = 
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
    let tensor3 (data: seq< #seq< #seq<'T>>>): DT<'T> = 
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
    let tensor4 (data: seq< #seq< #seq< #seq<'T>>>>): DT<'T> = 
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
    let relu (x: DT<'T>): DT<'T> = 
        DT.Relu(x)
        // We can'T use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Relu: DT<'T> -> DT<'T>) (x))

    /// The sum of the elements of a tensor
    let sum (x: DT<'T>): DT<'T> = 
        DT.Sum(x)
        // We can'T use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Sum: DT<'T> -> DT<'T>) (x))

    /// The product of the elements of a tensor
    let prod (x: DT<'T>): DT<'T> = 
        DT.Prod(x)
        //(DT<'T>: (static member Prod: DT<'T> -> DT<'T>) (x))
        // We can'T use this because of reflection issues for the live check interpreter

    /// The average value of the elements of a tensor
    let mean (x: DT<'T>): DT<'T> = 
        DT.Mean(x)
        // We can'T use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Mean: DT<'T> -> DT<'T>) (x))

    /// The max value of the elements of a tensor
    let maxValue (x: DT<'T>): DT<'T> = 
        DT.Max(x)
        // We can'T use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Max: DT<'T> -> DT<'T>) (x))

    /// The min value of the elements of a tensor
    let minValue (x: DT<'T>): DT<'T> = 
        DT.Min(x)
        // We can'T use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Min: DT<'T> -> DT<'T>) (x))


    /// The norm of a tensor
    let norm (x: DT<'T>): DT<'T> = 
        DT.Norm(x)
        // We can'T use this because of reflection issues for the live check interpreter
        //(DT<'T>: (static member Norm: DT<'T> -> DT<'T>) (x))

    let inline sqr x = x * x

    /// The global random number generator
    let rnd = new System.Random()

    /// Prepare a randome number using the global generator
    let rand() = rnd.NextDouble()

    //let crossEntropy (x:DT<_>) (y:DT<_>): DT<double> = failwith "fail"
    //    -(x |> DM.toCols |> Seq.mapi (fun i v -> 
    //        (DV.standardBasis v.Length (int (float y.[0, i]))) * log v) |> Seq.sum) / x.Cols

    /// Change a 3D-array to friendly notation
    let friendly3D (d: 'T[,,]) =
        [| for i in 0..Array3D.length1 d - 1 -> [| for j in 0..Array3D.length2 d - 1 -> [| for k in 0..Array3D.length3 d - 1 -> d.[i,j,k]  |]|]|]
        |> Array.map array2D

    /// Change an 4D-array to friendly notation
    let friendly4D (d: 'T[,,,]) =
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
