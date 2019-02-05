namespace TensorFlow.FSharp

open System

//
//[<AutoOpen>]
//module Global =
//
//    let mutable defaultSession : TFSession option = None
//
//    let mutable defaultGraph : TFGraph option = None
//
//    let resetSession() =
//        let sess = new TFSession()
//        defaultSession <- Some(sess)
//        defaultGraph <- Some(sess.Graph)
//
//type TF() =
//
//    static member DefaultGraph = 
//        match defaultGraph with 
//        | Some(x) -> x
//        | None -> resetSession(); defaultGraph.Value
//
//    static member DefaultSession = 
//        match defaultSession with 
//        | Some(x) -> x
//        | None -> resetSession(); defaultSession.Value
//
//    static member WithGraphScope(graph:TFGraph) =
//        let oldGraph = defaultGraph
//        defaultGraph <- Some(graph)
//        {new IDisposable with member this.Dispose() = defaultGraph <- oldGraph}
//
//    static member WithScope(nameScopeDesc : string) = TF.DefaultGraph.WithScope(nameScopeDesc)
//
//    static member MakeName(name : string, ?userName : string) = TF.DefaultGraph.MakeName(name, userName |> Option.defaultValue "")
//
//    static member GetShape(x:TFOutput, ?status:TFStatus) = TF.DefaultGraph.GetShape(x,?status=status)
//
//    static member SetTensorShape(x:TFOutput,?dims : int64[], ?status:TFStatus) = TF.DefaultGraph.SetTensorShape(x,?dims=dims, ?status=status)
//
//    static member GetTensorShape(x:TFOutput,?status:TFStatus) : TFShape = TF.DefaultGraph.GetTensorShape(x,?status=status)
//
//    static member WithDependencies([<ParamArray>] xs : TFOperation[]) = TF.DefaultGraph.WithDependencies(xs)
//
//    static member GetTensorNumDims(x:TFOutput, ?status:TFStatus) = TF.DefaultGraph.GetTensorNumDims(x, ?status=status)
//
