/// Contains helper functions and classes for creating and dealing with Varaible objects.
module rec TensorFlow.FSharp.Ops
/// TODO The aim of this is to enable calling get_variable(...)
///      Then to emulate the rest of tensorflow_scala api/ops/variables

open System
open TensorFlow.FSharp.Operations


//    [<AbstractClass>]
//    type ScalingMode() =
//        abstract scale : float32 *  TFShape -> float32
//        member this.computeFans(shape : TFShape) : (int64*int64) = 
//            let xs = shape.ToLongArray() 
//            match xs.Length with
//            | 0 -> 0L,0L
//            | 1 -> xs.[0], xs.[0]
//            | 2 -> xs.[0], xs.[1]
//            | _ ->
//                // Assuming convolution kernels (2D, 3D, or more) with shape [..., inputDepth, depth]
//                let last = xs.Length - 1
//                let receptiveFieldSize = xs.[0..last - 2] |> Array.fold (*) 1L
//                xs.[last - 2] * receptiveFieldSize, xs.[last - 1] * receptiveFieldSize
//
//    type FanInScalingMode() =
//        inherit ScalingMode()
//        override this.scale(initialScale : float32, shape : TFShape) = 
//            initialScale / float32 (Math.Max(1L, fst(this.computeFans(shape))))
//
//    type FanOutScalingMode() =
//        inherit ScalingMode()
//        override this.scale(initialScale : float32, shape : TFShape) = 
//            initialScale / float32 (Math.Max(1L, snd(this.computeFans(shape))))
//
//    type FanAverageScalingMode() =
//        inherit ScalingMode()
//        override this.scale(initialScale : float32, shape : TFShape) = 
//            initialScale / (Math.Max(1.0f, float32(this.computeFans(shape) ||> (+)) / 2.0f))

type PartitionInformation = obj

type ScalingMode = 
| FanInScalingMode
| FanOutScalingMode
| FanAverageScalingMode
with 
    member this.scale(initialScale : float32, shape : TFShape) : float32 = 
        let computeFans(shape : TFShape) : (int64*int64) =
            let xs = shape.ToLongArray() 
            match xs.Length with
            | 0 -> 0L,0L
            | 1 -> xs.[0], xs.[0]
            | 2 -> xs.[0], xs.[1]
            | _ ->
                // Assuming convolution kernels (2D, 3D, or more) with shape [..., inputDepth, depth]
                let last = xs.Length - 1
                let receptiveFieldSize = xs.[0..last - 2] |> Array.fold (*) 1L
                xs.[last - 2] * receptiveFieldSize, xs.[last - 1] * receptiveFieldSize
        let (fanIn,fanOut) = computeFans(shape)
        let d = 
            match this with
            | FanInScalingMode -> float32 fanIn
            | FanOutScalingMode ->  float32 fanOut
            | FanAverageScalingMode -> float32(fanIn + fanOut) / 2.0f
        initialScale / Math.Max(1.0f,d)

type Distribution =
| NormalDistribution
| UniformDistribution
with 
    member this.initialValue(graph : TFGraph, scale : float32, shape : TFShape, ?seed : int) = 
        match this with 
        | NormalDistribution ->
            graph.RandomTruncatedNormal(shape = shape, mean = 0.f, stddev = Math.Sqrt(scale))
        | UniformDistribution ->
            let limit = Math.Sqrt(3.0f * scale)
            graph.RandomUniform(shape, minval = -limit, maxval = limit, ?seed = seed)

            //graph.RandomTruncatedNormal(shape = shape, mean = 0.f, stddev = Math.Sqrt(scale))
            //failwith "todo"
            



[<AbstractClass>]
type Initializer(graph:TFGraph) =
    member this.Apply(shape : TFShape, datatype : TFDataType, partionInfo : PartitionInformation) =
        failwith "todo initializationScope" // probably need a graph here
    /// Generates an initial value op
    abstract initialValue : TFShape*TFDataType*PartitionInformation -> TFOutput

type ZerosInitializer(graph:TFGraph) =
    inherit Initializer(graph)
    override this.initialValue(shape : TFShape, datatype : TFDataType, _partitionInfo : PartitionInformation) = 
        graph.Zeros(shape,datatype)


        

//    type Distribution = 
//        abstract member initialValue(scale : float32, shape : TFShape, ?seed : int) : TFOutput
//
//    type NormalDistribution() = 
//        interface Distribution with
//            member initialValue(scale : float32, shape : TFShape, ?seed : int) = failwith "todo"
//        
//    type NormalDistribution() = 
//        interface Distribution with
//            member initialValue(scale : float32, shape : TFShape, ?seed : int) = failwith "todo"


/// <summary>
/// A variable regularizer is simply a function that takes a tensor representing the variable value as input, and
/// returns another tensor representing the regularizer value as output.
/// </summary>
[<AbstractClass>]
type Regularizer() = 
    abstract apply : TFOutput -> TFOutput

type VariableScopeStore() =
    static member current : VariableScopeStore = failwith "todo"
    member this.scope 
        with get() : VariableScope = failwith "todo"
        and set(x : VariableScope) = failwith "todo"

    member this.enterVariableScope(name : string) = failwith "todo"
    member this.closeVariableSubScopes(name : string) = failwith "todo"

type VariableGetter = { x : int }

type VariableScope = {
    graph : TFGraph
    reuse : Reuse
    name : string
    initializer : Initializer option
    regularizer : Regularizer option
    cachingDevice : (OpSpecification -> string) option
    nameScope : string
    underlyingGetter : VariableGetter option
} with
    /// TODO Docs
    member this.getVariable(store : VariableStore, name : string, dataType : TFDataType, shape : TFShape, ?initializer : Initializer, 
                            ?regularizer : Regularizer, ?trainable : bool, ?reuse : Reuse, ?collections : Set<string>,
                            ?cachingDevice  : OpSpecification -> string) =
        let initializer = initializer |> Option.orElse this.initializer
        let regularizer = regularizer |> Option.orElse this.regularizer
        let reuse       = defaultArg reuse this.reuse
        let collections = defaultArg collections Set.empty
        let cachingDevice = cachingDevice |> Option.orElse this.cachingDevice
        let fullName = if name <> "" then sprintf "%s/%s" this.name name else name
        let trainable = defaultArg trainable true

        // Variable names only depend on the variable scope and not on the name scope, so we reset it blow for the time of
        // variable creation
        use _ns = this.graph.NameScope("")
        store.getVariable(fullName, dataType, shape, ?initializer = initializer, ?regularizer = regularizer, trainable = trainable, reuse=reuse, collections=collections, ?cachingDevice = cachingDevice)

    /// TODO Docs
    static member scope(graph : TFGraph, name : string, ?reuse : Reuse, ?initializer : Initializer, ?regularizer : Regularizer,
                        ?cachingDevice : (OpSpecification -> string), ?underlyingGetter : VariableGetter, 
                        ?isDefaultName : bool, ?isPure : bool)  =
        fun (block : unit -> TFOutput) ->
            let reuse = defaultArg reuse ReuseOrCreateNew
            let isDefaultName = defaultArg isDefaultName false
            let isPure = defaultArg isPure false
            if reuse = ReuseExistingOnly && isDefaultName then
                invalidArg "isDefaultName" "'reuse' cannot be set to 'ReuseExistingOnly' with 'isDefaultName' set to 'true'."
            let variableScopeStore = VariableScopeStore.current
            let oldVariableScope = variableScopeStore.scope
            let newName =
                let uniqueName = if isDefaultName then VariableScope.unique(name) else name
                if oldVariableScope.name <> "" then 
                    sprintf "%s%s" oldVariableScope.name uniqueName
                else uniqueName
            variableScopeStore.enterVariableScope(newName)
            let newVariableScope : VariableScope = failwith "todo"
            variableScopeStore.scope <- newVariableScope
            let result = 
                if isPure then block()
                else
                    use _ns = graph.NameScope(name)
                    block()
            variableScopeStore.closeVariableSubScopes(newName)
            variableScopeStore.scope <- oldVariableScope
            result

        /// I'm not sure how this is supposed to be unique w/o a reference to the graph
    static member unique(name : string) = failwith "todo"



/// Initializer capable of adapting its scale to the shape of weights tensors.
/// 
/// With the Normal distribution option, samples are drawn from a truncated Normal distribution centered on zero, and 
/// with standard deviation equal to `sqrt(initialScale / n)`, where `n` is:
/// 
///  - the number of input units in the weight tensor, if `mode = FanInScalingMode`,
///  - the nubmer of output units, if `mode = FanOutScalingMode`, or 
///  - the average of the numbers of input and output units, if `mode = FanAverageScalingMode`


//type VarianceScalingInitializer(?initialScale : float32, scalingMode )


let defaultInitializer(graph : TFGraph,name : string, dataType : TFDataType) : Initializer = 
//    TODO
//    if dataType.IsFloat then GlorotUniformInitializer()
//    elif dataType.IsInteger || dataType.IsUnsigned || dataType.IsBoolean then ZerosInitializer
//    else raise (IllegalArgument(sprintf "A default initializer for variable '%s' of type '%o' is required." name dataType))
    upcast(ZerosInitializer(graph))


type Collection = obj//Graph.Key<Varaible<obj>>
type OpSpecification = obj

// TODO Ops.initializationScope
let initilizationScope() = {new IDisposable with member this.Dispose() = printfn "no-op for now, initialization scope needs to be fixed"}

type Variable() = 
    let x = 10
    member this.shape : TFShape = failwith "todo"
    member this.dataType : TFDataType = failwith "todo"
    member this.Op : TFOperation = failwith "todo"
    member this.value : TFOutput = failwith "todo"

// Variable Store

/// Variable store that carries a number of named variables.
type VariableStore(graph : TFGraph) =
    let mutable variables : Map<string,Variable> = Map.empty
    
    /// <summary>
    /// Gets or creates a variable.
    /// </summary>
    /// <param name="name">Variable name.</param>
    /// <param name="dataType">Variable dataType.</param>
    /// <param name="shape">Variable shape</param>
    /// <param name="initializer">Variable initializer. If `initializer` is `None` (the default), the default 
    /// initializer passed in the consturctor is used. If that one is `None` as well, then we use a new
    /// `glorotUniformInitializer`. The initializer will be called for each part of the partitioned variable 
    /// separely.
    /// </param>
    /// <param name="regularizer"> Variable regularizer.</param>
    /// <param name="trainable"> If `true`, the default, the variable is added to the graph collection 
    /// `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables
    /// to use by the optimizers.  </param>
    /// <param name="reuse"> Reuse value indication whether to re-use an existing variable with the same name, create
    /// a new variable, or do either. </param>
    /// <param name="collections">Set of graph collection keys. The variable is added to these collectionos. Defaults to 
    /// `Set(Graph.Keys.GLOBAL_VARIABLES)`.</param>
    /// <param name="cachingDevice> Device specificaiton describing where the variable shold be cached for reading. Defautls to the variable's device. Typical use is to cache on the device where the ops using the variable reside, to deduplicate copying through `Switch` and other conditional statements.</param>
    member this.getVariable(name : string, dataType : TFDataType, ?shape : TFShape, 
                            ?initializer : Initializer, ?regularizer : Regularizer,
                            ?trainable : bool, ?reuse : Reuse, ?collections : Set<string>,
                            ?cachingDevice : OpSpecification -> string) = 
        let trainable = defaultArg trainable true
        let reuse = defaultArg reuse ReuseOrCreateNew
        let collections = defaultArg collections (Set([GraphKeys.GLOBAL_VARIABLES]))
        //let illError(msg : string) =  // todo pull out errors
        // Single variable case.
        if variables.Contains(sprintf "%s/part_0" name) then
            let msg = "No partitioner was provided but a partitioned version of the variable ('$name/part_0') was found in" +
                      "the variable store. Perhaps a variable of the same name was already created with partitioning?"
            invalidArg "name" msg
        match variables.TryFind(name) with
        | Some(foundVariable) -> 
            // Here we handle the case of returning an existing variable.
            if (reuse = CreateNewOnly) then
                sprintf "Variable `%s` already exists, but variable scope re-use was set to `CreatenewOnly`." name
                |> invalidArg "name" 
            shape |> Option.iter(fun shape -> 
                if not(shape.IsCompatibleWith(foundVariable.shape)) then
                    let msg = sprintf "Trying to share variable '%s', but the specified shape '%O' is not compatible with the " name shape +
                              sprintf "existing variable shape '%O'." foundVariable.shape 
                    raise (ShapeMismatchException(msg)))
            if dataType <> foundVariable.dataType then
                sprintf "Trying to share variable '%s', but the specified data type '%O' is not compatible with the existing variable data type '%O'" name dataType foundVariable.dataType
                |> invalidArg "dataType"
            foundVariable
        | None ->
            // Here we handle the case of creating a new variable.
            if reuse = ReuseExistingOnly then
                invalidArg "reuse" (sprintf "Variable '%s' does not exist, but variable scope re-use was set to 'ReuseExistingOnly'." name)
            if shape.IsSome && not shape.Value.IsFullyDefined then
                invalidArg "shape" (sprintf "The shape of a new varaible ('%s') must be fully defined but instead it was set to %O" name shape.Value)
            let actualInitializer = 
                use _iScope = initilizationScope()
                initializer |> Option.defaultWith (fun _ -> defaultInitializer(graph,name,dataType))
            let variable = 
                failwith "todo"
                //makeGetter()(name, dataType, shape, actualInitializer, regularizer, trainable, reuse, collections, cachingDevice, None)
            variables <- variables.Add(name,variable)
            // TODO Logging
            // Run the regularizer if specified and save the resulting loss.
            match regularizer with
            | None -> ()
            | Some(regularizer) -> 
                failwith "todo - currently unsupported"
//                use _colo = graph.ColocateWith(Set [variabe.Op], ignoreExisting = true)
//                use _ns = graph.NameScope(sprintf "%s/Regularizer" name)
//                let loss = regularizer.apply(variable.value)
//                graph.AddToCollection(GraphKeys.REGULARIZATION_LOSSES, loss)
            variable
            
