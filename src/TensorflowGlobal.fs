namespace TensorFlow

open System


[<AutoOpen>]
module Global =

    let mutable defaultSession : Session option = None

    let mutable defaultGraph : Graph option = None

    let resetSession() =
        let sess = new Session()
        defaultSession <- Some(sess)
        defaultGraph <- Some(sess.Graph)

type TF() =
    static member DefaultGraph = 
        match defaultGraph with 
        | Some(x) -> x
        | None -> resetSession(); defaultGraph.Value

    static member DefaultSession = 
        match defaultSession with 
        | Some(x) -> x
        | None -> resetSession(); defaultSession.Value

    static member WithGraphScope(graph:Graph) =
        let oldGraph = defaultGraph
        defaultGraph <- Some(graph)
        {new IDisposable with member this.Dispose() = defaultGraph <- oldGraph}

    static member WithScope(nameScopeDesc : string) = TF.DefaultGraph.WithScope(nameScopeDesc)

    static member MakeName(name : string, ?userName : string) = TF.DefaultGraph.MakeName(name, userName |> Option.defaultValue "")

    static member GetShape(x:Output, ?status:TFStatus) = TF.DefaultGraph.GetShape(x,?status=status)

    static member SetTensorShape(x:Output,?dims : int64[], ?status:TFStatus) = TF.DefaultGraph.SetTensorShape(x,?dims=dims, ?status=status)

    static member GetTensorShape(x:Output,?status:TFStatus) : Shape = TF.DefaultGraph.GetTensorShape(x,?status=status)

    static member WithDependencies([<ParamArray>] xs : Operation[]) = TF.DefaultGraph.WithDependencies(xs)

    static member GetTensorNumDims(x:Output, ?status:TFStatus) = TF.DefaultGraph.GetTensorNumDims(x, ?status=status)
