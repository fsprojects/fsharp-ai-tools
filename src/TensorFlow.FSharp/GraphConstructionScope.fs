namespace TensorFlow.FSharp

/// NOTE: The use of a graph construction scope should clean up state timelines and allow easy access to outer contexts which would 
///       otherwise be rather difficult

type Context = obj
type OpSpecification = obj

type GraphConstructionScope = {
    graph : TFGraph
    nameScope : string
    device : string
    deviceFunction : OpSpecification -> string
    /// NOTE: Here we may need to use TFOperation instead of TFOutput
    colocationOps : Set<TFOutput> 
    attributes : Map<string,obj>
    container : string
    controlFlowContext : Context option
    outerContext : GraphConstructionScope option
}

