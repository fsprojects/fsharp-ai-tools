namespace TensorFlow.FSharp 

#I __SOURCE_DIRECTORY__
#r "netstandard"
#r "../../tests/bin/Debug/net472/TensorFlow.FSharp.dll"

open System
open TensorFlow.FSharp

type Dim =
    | DimMulInt of Dim * int
    | DimDivInt of Dim * int
    | DimVar of Dim option ref
    | Dim of int
    override dim.ToString() = 
        match dim with 
        | DimMulInt (dim, n) -> 
            match dim.TryValue with 
            | Some v -> string (v*n)
            | None -> dim.ToString() + "*" + string n 
        | DimDivInt (dim, n) -> 
            match dim.TryValue with 
            | Some v -> string v
            | None -> dim.ToString() + "/" + string n 
        | Dim n -> string n 
        | DimVar v -> 
            match v.Value with 
            | None -> "?" 
            | Some v -> v.ToString()

    member dim.TryValue = 
        match dim with 
        | DimMulInt (dim,n) -> match dim.TryValue with None -> None | Some dimv -> Some (dimv*n) 
        | DimDivInt (dim,n) -> match dim.TryValue with None -> None | Some dimv -> Some (dimv/n) 
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

    static member UnifyShapes (shape1: Shape) (shape2: Shape) = 
        match shape1, shape2 with 
        | ShapeVar ({ contents = None } as v1), ShapeVar ({ contents = None } as v2) when Object.ReferenceEquals(v1,v2) -> ()
        | ShapeVar ({ contents = None } as v1), _ -> v1 := Some shape2
        | _, ShapeVar ({ contents = None } as v2) -> v2 := Some shape1
        | ShapeVar { contents = Some soln1}, _ -> Shape.UnifyShapes soln1 shape2
        | _, ShapeVar { contents = Some soln2} -> Shape.UnifyShapes shape1 soln2
        | Shape dims1, Shape dims2 -> 
            if dims1.Length <> dims2.Length then 
                failwithf "mismatched shapes: %A and %A, dims1.Length = %d, dims2.Length = %d" shape1 shape2 dims1.Length dims2.Length
            (dims1, dims2) ||> Array.iter2 (fun d1 d2 -> Dim.Unify d1 d2)

    static member EquivShapes (shape1: Shape) (shape2: Shape) = 
        Shape.UnifyShapes shape1 shape2
        shape1

[<AutoOpen>]
module ShapeHelpers = 

    let shape (ints: seq<int>) = Shape(Array.map Dim (Array.ofSeq ints))

type DT() = class end

/// A differentiable tensor value
type Ctxt = 
    { Graph: TFGraph 
      Values: Map<string,DT> }

type DT<'T>(shape: Shape, eval: (Ctxt -> TFOutput)) =
    inherit DT()

    member __.Apply(ctxt) = eval ctxt

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

    member __.Shape : Shape = shape

    override dt.ToString() = dt.Shape.ToString()

type WithScopeDisposable(name:string) = 
    interface IDisposable with 
        member __.Dispose() = ()
    member __.Name = name        

type DT with

    static member Const (value: 'T, ?shape: Shape) : DT<'T> = 
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
    //input: V[N,H,W,C], filters: V[F;F;C;COut]) -> output:V[N,H,W,COut] 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let inputShape = input.Shape
        let filtersShape = filters.Shape
        let N, H, W, C = inputShape.[0], inputShape.[1], inputShape.[2], inputShape.[3]
        let F = filtersShape.[0]
        let COut = filtersShape.[3]
        Shape.UnifyShapes filtersShape (Shape [| F; F; C; COut |])
        let outputShape = Shape [| N; H/stride; W/stride; COut |]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Conv2D(input.Apply(ctxt), filters.Apply(ctxt),strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    static member Conv2DBackpropInput(filters: DT<'T>, out_backprop: DT<'T>, ?stride: int, ?padding: string) : DT<'T> = 
        let stride = defaultArg stride 1
        let padding = defaultArg padding "SAME"
        let inputShape = out_backprop.Shape
        let filtersShape = filters.Shape
        let N, H, W, C = inputShape.[0], inputShape.[1], inputShape.[2], inputShape.[3]
        let F = filtersShape.[0]
        let COut = filtersShape.[3]
        Shape.UnifyShapes filtersShape (Shape [| F; F; C; COut |])
        let outputShape = Shape [| N; H*stride; W*stride; COut |]
        DT<_> (outputShape, fun ctxt -> 
           let input_sizes = ctxt.Graph.Const(outputShape.AsTFTensor())
           ctxt.Graph.Conv2DBackpropInput(input_sizes, filters.Apply(ctxt), out_backprop.Apply(ctxt), strides = [|1L;int64 stride;int64 stride;1L|], padding=padding))

    static member ClipByValue (input: DT<'T>, low: DT<'T>, high: DT<'T>) : DT<'T> = 
        let outputShape = Shape.EquivShapes (Shape.EquivShapes input.Shape low.Shape) high.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.ClipByValue(input.Apply(ctxt), low.Apply(ctxt), high.Apply(ctxt)))

    static member Moments(dims: seq<int>, input: DT<'T>) : DT<'T> * DT<'T> = 
        // Note: keep_dims = true
        let outputShape = input.Shape
        // Note: avoid making two Moments here
        let mutable cache = None
        let compute (ctxt: Ctxt) = 
            match cache with 
            | Some (ctxt2, res) when Object.ReferenceEquals(ctxt, ctxt2) -> res
            | _ -> 
                let res = ctxt.Graph.Moments(input.Apply(ctxt), ctxt.Graph.Const((shape dims).AsTFTensor()),keep_dims=true)
                cache <- Some (ctxt, res)
                res

        DT<_> (outputShape, fun ctxt -> fst (compute ctxt)),
        DT<_> (outputShape, fun ctxt -> snd (compute ctxt))

    static member Relu(input: DT<'T>) : DT<'T> = 
        let outputShape = input.Shape
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Relu(input.Apply(ctxt)))

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
        DT<_> (dt.Shape, fun ctxt -> use _scope = ctxt.Graph.WithScope(name) in dt.Apply(ctxt))

    static member CreateString(value: byte[]) : DT<string> = 
        let outputShape = shape [ 1 ]
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.Const(TFTensor.CreateString(value)))

    static member ExpandDims(value: DT<'T>, [<ParamArray>] dims: int[]) : DT<'T> = 
        let outputShape = shape dims
        DT<_> (outputShape, fun ctxt -> ctxt.Graph.ExpandDims(value.Apply(ctxt), ctxt.Graph.Const(outputShape.AsTFTensor())))
    //TF.ExpandDims[Dim](value: V[shape]) : V[Expanded(Dim,shape)]

    static member Run(value: DT<'T>, ?weights: seq<string * DT>) : TFTensor = 
        let sess = new TFSession()
        let graph = sess.Graph
        let ctxt = { Graph = graph; Values = Map.ofSeq (defaultArg weights Seq.empty)}
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
