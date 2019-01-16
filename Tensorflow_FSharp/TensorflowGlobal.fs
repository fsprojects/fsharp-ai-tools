namespace Tensorflow
open System


[<AutoOpen>]
module Global =
    // TODO this may not be optimum behavior
    let mutable defaultSession : Session = new Session()
    let mutable defaultGraph : Graph = defaultSession.Graph
    let mutable currentDependencies : Operation[] = [||]

type TF() =

    static member WithGraphScope(graph:Graph) =
        let oldGraph = defaultGraph
        defaultGraph <- graph
        {new IDisposable with member this.Dispose() = defaultGraph <- oldGraph}

    static member WithScope(nameScopeDesc : string) = defaultGraph.WithScope(nameScopeDesc)
    static member MakeName(name : string, ?userName : string) = defaultGraph.MakeName(name, userName |> Option.orDefault "")
    static member GetShape(x:Output, ?status:TFStatus) = defaultGraph.GetShape(x,?status=status)
    static member SetTensorShape(x:Output,?dims : int64[], ?status:TFStatus) = defaultGraph.SetTensorShape(x,?dims=dims, ?status=status)
    static member GetTensorShape(x:Output,?status:TFStatus) : Shape = defaultGraph.GetTensorShape(x,?status=status)
    static member WithDependencies([<ParamArray>] xs : Operation[]) = defaultGraph.WithDependencies(xs)
    static member GetTensorNumDims(x:Output, ?status:TFStatus) = defaultGraph.GetTensorNumDims(x, ?status=status)