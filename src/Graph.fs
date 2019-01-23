namespace TensorFlow.FSharp

open System
open System.Runtime.InteropServices
open System.Collections.Generic
open System.Text
open FSharp.NativeInterop
open TensorFlow.FSharp.Utils

#nowarn "9"

[<StructLayout (LayoutKind.Sequential)>]
[<Struct>]
type TFWhileParams = {
    ninputs : int  
    cond_graph : TF_Graph
    cond_inputs : nativeptr<TF_Output> 
    cond_output : TF_Output 
    body_graph : TF_Graph 
    body_inputs : nativeptr<TF_Output>
    body_outputs : nativeptr<TF_Output>
    mutable charPtrName : IntPtr
}



/// Device type
type DeviceType = 
    /// The device is the Central Processing Unit (CPU)
    | CPU
    /// The device is a Graphics Processing Unit (GPU)
    | GPU
    /// The device is a TFTensor Processing Unit (TPU)
    | TPU

/// Describes the device attributes 
type DeviceAttributes internal (name:string, deviceType:DeviceType, memoryLimitBytes:int64) =
        /// <summary>
        /// The full name of the device (e.g. /job:worker/replica:0/...)
        /// </summary>
        member this.Name = name 

        /// <summary>
        /// Gets the type of the device.
        /// </summary>
        /// <value>The type of the device.</value>
        member this.DeviceType = deviceType

        /// <summary>
        /// The amount of memory associated with a given device.
        /// </summary>
        /// <value>The memory limit bytes.</value>
        member this.MemoryLimitBytes = memoryLimitBytes

module TFImportGraphDefOptionsExternal = 
    // extern TF_ImportGraphDefOptions * TF_NewImportGraphDefOptions ()
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern TF_ImportGraphDefOptions TF_NewImportGraphDefOptions ()

/// Contains options that are used to control how graph importing works.
type TFImportGraphDefOptions() =
    inherit TFDisposable(TFImportGraphDefOptionsExternal.TF_NewImportGraphDefOptions ())

    // extern void TF_DeleteImportGraphDefOptions (TF_ImportGraphDefOptions *opts)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteImportGraphDefOptions (TF_ImportGraphDefOptions opts)

    // extern void TF_ImportGraphDefOptionsSetPrefix (TF_ImportGraphDefOptions *opts, const char *prefix)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsSetPrefix (TF_ImportGraphDefOptions opts, string prefix)

    // extern void TF_ImportGraphDefOptionsAddInputMapping (TF_ImportGraphDefOptions *opts, const char* src_name, int src_index, TF_Output dst)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsAddInputMapping (TF_ImportGraphDefOptions opts, string src_name, int src_index, TF_Output dst)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsAddControlDependency (TF_ImportGraphDefOptions opts, TF_Operation oper)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsAddReturnOutput (TF_ImportGraphDefOptions opts, string oper_name, int index)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_ImportGraphDefOptionsNumReturnOutputs (TF_ImportGraphDefOptions opts)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsRemapControlDependency (TF_ImportGraphDefOptions opts, string srcName, TF_Operation dst)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsSetUniquifyNames (TF_ImportGraphDefOptions opts, byte uniquify)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsSetUniquifyPrefix (TF_ImportGraphDefOptions opts, byte uniquify_prefix)

    override this.NativeDispose (handle : IntPtr) = TF_DeleteImportGraphDefOptions (handle)

    member this.SetPrefix (prefix : string) =
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsSetPrefix (this.Handle, prefix)


    /// <summary>
    /// Adds an input mapping from a source name and index to a destination output
    /// </summary>
    /// <param name="srcName">Source name.</param>
    /// <param name="srcIndex">Source index (in the source).</param>
    /// <param name="dst">Replacement value for the srcName:srcIndex.</param>
    /// <remarks>
    /// Set any imported nodes with input `src_name:src_index` to have that input
    /// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
    /// `dst` references a node already existing in the graph being imported into.
    /// </remarks>
    member this.AddInputMapping (srcName : string, srcIndex : int, dst : TFOutput) =
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsAddInputMapping (this.Handle, srcName, srcIndex, dst.Struct)


    /// <summary>
    /// Cause the imported graph to have a control dependency on the provided operation.
    /// </summary>
    /// <param name="operation">This operation should exist in the graph being imported to.</param>
    member this.AddControlDependency (operation : TFOperation) =
        if box operation = null then raise(ArgumentNullException ("operation"))
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsAddControlDependency (this.Handle, operation.Handle)


    /// <summary>
    /// Add an output in the graph definition to be returned via the return outputs parameter.
    /// </summary>
    /// <param name="operName">Operation name.</param>
    /// <param name="index">Operation index.</param>
    /// <remarks>
    /// If the output is remapped via an input
    /// mapping, the corresponding existing tensor in graph will be returned.
    /// </remarks>
    member this.AddReturnOutput (name: string, index : int) =
        if name = null then raise(ArgumentNullException ("name"))
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsAddReturnOutput (this.Handle, name, index)


    /// <summary>
    /// Gets the number return outputs added via AddReturnOutput.
    /// </summary>
    /// <value>The number return outputs.</value>
    member this.NumReturnOutputs  
        with get() : int = 
            if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
            TF_ImportGraphDefOptionsNumReturnOutputs (this.Handle)

    /// <summary>
    /// Sets any imported nodes with a given control input to have it replaced with an operation
    /// </summary>
    /// <param name="srcName">Node in the graph to be imported.</param>
    /// <param name="destination">References an operation that already exists in the graph being imported.</param>
    /// <remarks>
    /// Set any imported nodes with control input <paramref name="srcName"/> to have that input
    /// replaced with <paramref name="destination"/>. 
    /// </remarks>
    member this.RemapControlDependency (srcName : string, destination : TFOperation) =
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        if srcName = null then raise(ArgumentNullException ("srcName"))
        if box destination = null then raise(ArgumentNullException ("destination"))
        if destination.Handle = IntPtr.Zero then raise(ObjectDisposedException ("destination"))
        TF_ImportGraphDefOptionsRemapControlDependency (this.Handle, srcName, destination.Handle)


    /// <summary>
    /// Set whether to uniquify imported operation names.
    /// </summary>
    /// <param name="uniquifyNames">If set to <c>true</c> imported operation names will be modified if their name already exists in the graph.
    /// If set to <c>false</c> conflicting names will be treated as an error.
    /// </param>
    /// <remarks>
    ///  Note that this option has no effect if a prefix is set, since the prefix will guarantee all names are
    ///  Defaults to false.
    /// </remarks>
    member this.SetUniquifyNames (uniquifyNames : bool) = 
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsSetUniquifyNames (this.Handle, if uniquifyNames then 1uy else 0uy)

    /// <summary>
    /// Sets the uniquify prefix.  This option has no effect if no prefix is specified.
    /// </summary>
    /// <param name="uniquifyPrefix">If set to <c>true</c> the specified prefix will be modified if it already exists as an
    /// operation name or prefix in the graph. 
    /// If set to <c>false</c> a conflicting prefix will be treated as an error.
    /// </param>
    member this.SetUniquifyPrefix (uniquifyPrefix : bool) = 
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsSetUniquifyPrefix (this.Handle, if uniquifyPrefix then 1uy else 0uy)

module GraphExternal =
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern void TF_GraphSetTensorShape (TF_Graph graph, TF_Output output, IntPtr dims, int num_dims, TF_Status status)

/// <summary>
/// Signature of the method that will be invoked by the Graph.While method to construct a while loop
/// </summary>
/// <remarks>
/// <para>
/// The method should build up the condition on the conditionGraph and the body of the while 
/// loop in the provided bodyGraph.   It should set the condOutput to the value used as the
/// condition output and the array of values in bodyOutputs to the final outputs as well as the
/// name to be used, if not set, one will be assigned.
/// </para>
/// <para>
/// The conditionGraph represents the while condition and the inputs are the current values of the
/// input variables (condInputs).   The output should be a scalar boolean.
/// </para>
/// <para>
/// The loop body graph is in bodyGraph, The inputs are the current values of the loop
/// variables. The outputs are the updated values of the loop variables.
/// </para>
/// <para>
/// You can use the passed status record problems with it.
/// </para>
/// </remarks>
type WhileConstructor = delegate of conditionGraph : TFGraph * condInputs : TF_Output [] * [<Out>] condOutput : TF_Output  * 
                        bodyGraph : TFGraph * bodyInputs : TF_Output [] * bodyOutpus : TF_Output [] * [<Out>] name : string  -> unit


//
// A Graph that will not release the undelying handle, this is used
// when we want to surface a Graph that we do not own, so we do not
// want to delete the handle when this object is collected
//
and GraphUnowned internal (handle:IntPtr) = 
    inherit TFGraph (handle)

    // Nothing, we do not own the handle
    override this.NativeDispose (handle : TF_Status) = ()

/// <summary>
/// Represents a computation graph.  Graphs may be shared between sessions and are thread safe.
/// </summary>
/// <remarks>
/// <para>
/// Graphs consist of operations (represented by Operation objects), these can be named, or 
/// the runtime will automatically assign a name.
/// </para> 
/// <para>
/// For debugging purposes, you might want to group operations together, for this, call the
/// WithScope method with your new scope, which will create a new namespace for your object names.
/// </para>
/// <para>
/// For example, if you call WithScope ("demo"), and add an operation named "add" inside the
/// scope, the full name of the operation will be "demo/add", if you create a new scope inside, say
/// "hot", and add a "sub" operation there the result will be "demo/hot/sub".
/// </para>
/// </remarks>
and TFGraph internal (handle) =
    inherit TFDisposableThreadSafe(handle)

    let mutable currentNameScope = ""
    let mutable currentDependencies = Array.empty<TFOperation>
    let values = new DictionaryCount<string> ()

    /// <summary>
    /// Gets or sets the graph random seed, see remarks for details.
    /// </summary>
    /// <value>The seed.</value>
    /// <remarks>
    ///  Operations that rely on a random seed actually derive it from two seeds:
    ///  the graph-level and operation-level seeds.This sets the graph-level seed.
    ///
    /// Its interactions with operation-level seeds is as follows:
    /// 1. If neither the graph-level nor the operation seed is set:
    ///    A random seed is used for this op.
    /// 2. If the graph-level seed is set, but the operation seed is not:
    ///    The system deterministically picks an operation seed in conjunction
    ///    with the graph-level seed so that it gets a unique random sequence.
    /// 3. If the graph-level seed is not set, but the operation seed is set:
    ///    A default graph-level seed and the specified operation seed are used to
    ///    determine the random sequence.
    /// 4. If both the graph-level and the operation seed are set:
    ///    Both seeds are used in conjunction to determine the random sequence.
    /// </remarks>
    let mutable seed = 87654321

    let pending_init_variables : List<TFOperation> = List<TFOperation>()
    let trainable_variables : List<TFVariable> = List<TFVariable>()

    let mutable lastId = 0

    // extern TF_Graph * TF_NewGraph ()
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Graph TF_NewGraph ()

    // extern void TF_DeleteGraph (TF_Graph *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteGraph (TF_Graph graph)

    // extern void TF_GraphSetTensorShape (TF_Graph *graph, TF_Output output, const int64_t *dims, const int num_dims, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphSetTensorShape (TF_Graph graph, TF_Output output, int64 [] dims, int num_dims, TF_Status status)

    // extern int TF_GraphGetTensorNumDims (TF_Graph *graph, TF_Output output, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_GraphGetTensorNumDims (TF_Graph graph, TF_Output output, TF_Status status)

    // extern void TF_GraphGetTensorShape (TF_Graph *graph, TF_Output output, int64_t *dims, int num_dims, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphGetTensorShape (TF_Graph graph, TF_Output output, int64 [] dims, int num_dims, TF_Status status)

    // extern void TF_GraphToGraphDef (TF_Graph *graph, TF_Buffer *output_graph_def, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphToGraphDef (TF_Graph graph, LLBuffer* output_graph_def, TF_Status status)

    // extern void TF_GraphImportGraphDef (TF_Graph *graph, const TF_Buffer *graph_def, const TF_ImportGraphDefOptions *options, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphImportGraphDef (TF_Graph graph, LLBuffer* graph_def, TF_ImportGraphDefOptions options, TF_Status status)

    // extern TF_Operation * TF_GraphOperationByName (TF_Graph *graph, const char *oper_name)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Operation TF_GraphOperationByName (TF_Graph graph, string oper_name)

    // extern TF_Operation * TF_GraphNextOperation (TF_Graph *graph, size_t *pos)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Operation TF_GraphNextOperation (TF_Graph graph, IntPtr& token) // ref token

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphImportGraphDefWithReturnOutputs (TF_Graph graph, LLBuffer *graph_def, TF_ImportGraphDefOptions options, TF_Output *return_outputs, int num_return_outputs, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TFWhileParams TF_NewWhile (TF_Graph g, TF_Output [] inputs, int ninputs, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AbortWhile (TFWhileParams& pars) // ref

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_FinishWhile (TFWhileParams& pars, TF_Status status, TF_Output *outputs) // ref

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddGradients (TF_Graph graph, TF_Output* ys, int ny, TF_Output* xs, int nx, TF_Output* dx, TF_Status status, TF_Output* dy)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddGradientsWithPrefix (TF_Graph graph, string prefix, TF_Output* ys, int ny, TF_Output* xs, int nx, TF_Output* dx, TF_Status status, TF_Output* dy)
   
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphCopyFunction (TF_Graph graph, TF_Function func, TF_Function grad, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_GraphToFunction (TF_Graph body, string fn_name, byte append_hash_to_fn_name, int num_opers, IntPtr opers, int ninputs, TF_Output [] inputs, int noutputs, TF_Output [] ouputs, string [] output_names, IntPtr options, string description, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphVersions (TF_Graph graph, LLBuffer *output_version_def, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_GraphNumFunctions (TF_Graph graph)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_GraphGetFunctions (TF_Graph graph, IntPtr funcs, int max_func, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern bool TF_TryEvaluateConstant (TF_Graph graph, TF_Output output, IntPtr& result, TF_Status status) // ref result

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern string TF_GraphDebugString (TF_Graph graph, IntPtr& len) //[<Out>]


    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.Graph"/> class.
    /// </summary>
    new () = new TFGraph (TF_NewGraph ())

    override this.NativeDispose (handle : IntPtr) = TF_DeleteGraph (handle)

    member this.PendingInitVariables with get() = pending_init_variables
    member this.TrainingVariables with get() = trainable_variables


    /// <summary>
    /// Returns the graph and local seeds based on an optionally set incoming seed value.
    /// </summary>
    /// <param name="operationSeed">The seed value that might be set.</param>
    /// <param name="graphSeed">Returned graph seed.</param>
    /// <param name="localSeed">Returned local seed.</param>
    /// <remarks>
    /// This helper function returns two seeds derived from graph-level and op-level seeds.
    /// Many random operations internally use the two seeds to allow user to change 
    /// the seed globally for a graph, or for only specific operations.
    /// </remarks>
    member this.GetRandomSeeds (?operationSeed : int) = (seed, operationSeed |> Option.defaultValue lastId)

    /// <summary>
    /// Sets the tensor shape of the tensor referenced by <paramref name="output"/> to the shape described by <paramref name="dims"/>.
    /// </summary>
    /// <param name="output">The tensor on which this method will operate in the graph.</param>
    /// <param name="dims">The tensor shape, specified as an array of dimensions.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.SetTensorShape (output : TFOutput, ?dims : int64 [], ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        match dims with
        | None -> 
            GraphExternal.TF_GraphSetTensorShape (handle, output.Struct, IntPtr.Zero, 0, cstatus.Handle)
        | Some(dims) ->
            TF_GraphSetTensorShape (handle, output.Struct, dims, dims.Length, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore


    /// <summary>
    /// Returns the number of dimensions of the TFTensor referenced by output
    /// </summary>
    /// <returns>The number of dimensions of the tensor.</returns>
    /// <param name="output">The tensor to probe.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.GetTensorNumDims (output : TFOutput, ?status : TFStatus) = 
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let code = TF_GraphGetTensorNumDims (handle, output.Struct, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        code


    /// <summary>
    /// Returns the shape of a tensor specified in <paramref name="output"/>.
    /// </summary>
    ///
    /// <returns>The tensor shape.    If the number of dimensions in the shape is unknown or the shape is, a scalar, the values in the array will be zero. Otherwise, each element of will be set corresponding to the size of the dimension. An  unknown dimension is represented by -1.</returns>
    /// <param name="output">The tensor that you want to look up.  </param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.GetTensorShape (output : TFOutput, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let n = TF_GraphGetTensorNumDims (handle, output.Struct, cstatus.Handle)
        if (not (cstatus.CheckMaybeRaise (?incoming=status, last = false))) then TFShape.Unknown
        elif n = -1 then TFShape.Unknown
        else
            let dims = Array.zeroCreate<int64> n
            TF_GraphGetTensorShape (handle, output.Struct, dims, dims.Length, cstatus.Handle)
            cstatus.CheckMaybeRaise (?incoming=status) |> ignore
            new TFShape (dims)

    /// <summary>
    /// Write out a serialized representation of the graph (as a GraphDef protocol buffer message) into <paramref name="outputGraphDef"/>.
    /// </summary>
    /// <param name="outputGraphDef">Target buffer where the graphs is serialized into.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.ToGraphDef (outputGraphDef : TFBuffer, ?status : TFStatus) = 
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box outputGraphDef = null then raise(ArgumentNullException ("outputGraphDef"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_GraphToGraphDef (handle, outputGraphDef.LLBuffer, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status)

    /// <summary>
    /// Import a serialized graph into this graph, using the specified prefix.
    /// </summary>
    /// <returns>The import.</returns>
    /// <param name="graphDef">A buffer containing the serialized graph.</param>
    /// <param name="prefix">A prefix that will be prepended to names of nodes in the <paramref name="graphDef"/> when they are imported into the graph.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.Import (graphDef : TFBuffer, ?prefix : string, ?status : TFStatus) = 
        let prefix = defaultArg prefix ""
        if handle = IntPtr.Zero then raise(ObjectDisposedException("handle"))
        if box graphDef = null then raise(ArgumentNullException ("graphDef"))
        if box prefix =null then raise (ArgumentNullException ("prefix"))
        use options = new TFImportGraphDefOptions()
        options.SetPrefix (prefix)
        this.Import (graphDef, options, ?status=status)

    /// <summary>
    /// Import a serialized graph into this graph, using the specified importing options.
    /// </summary>
    /// <returns>The import.</returns>
    /// <param name="graphDef">A buffer containing the serialized graph.</param>
    /// <param name="options">Importing graph options.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.Import (graphDef : TFBuffer, options : TFImportGraphDefOptions, ?status : TFStatus) =
        if (handle = IntPtr.Zero) then raise(ObjectDisposedException ("handle"))
        if (box graphDef = null) then raise(ArgumentNullException ("graphDef"))
        if (box options = null) then raise(ArgumentNullException ("options"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_GraphImportGraphDef (handle, graphDef.LLBuffer, options.Handle, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore

    /// <summary>
    /// Import a serialized graph held in a byte array into this graph, using the specified prefix.
    /// </summary>
    /// <returns>The import.</returns>
    /// <param name="buffer">A byte array containing the serialized graph.</param>
    /// <param name="prefix">A prefix that will be prepended to names of nodes in the graph when they are imported into the graph.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>		
    member this.Import (buffer : byte [], ?prefix : string, ?status : TFStatus) = 
        let prefix = defaultArg prefix ""
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box buffer = null then raise(ArgumentNullException ("buffer"))
        if box prefix = null then raise(ArgumentNullException ("prefix"))
        use options = new TFImportGraphDefOptions()
        options.SetPrefix prefix
        this.Import (buffer, options, ?status=status)

    /// <summary>
    /// Import a serialized graph held in a byte array into this graph, using the specified import options.
    /// </summary>
    /// <returns>The import.</returns>
    /// <param name="buffer">A byte array containing the serialized graph.</param>
    /// <param name="options">Importing graph options.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    ///   If you are tryig to load a file stored using the SavedModel file format, you should use the <see cref="T:TensorFlow.Session.FromSavedModel"/> API instead.
    /// </remarks>
    member this.Import (buffer : byte [], options : TFImportGraphDefOptions, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box buffer = null then raise(ArgumentNullException ("buffer"))
        if box options = null then raise(ArgumentNullException ("options"))
        let cstatus = TFStatus.Setup (?incoming=status)
        use  tb = new TFBuffer (buffer, 0, buffer.Length)
        this.Import (tb, options, ?status=status)
        cstatus.CheckMaybeRaise (?incoming=status)

    /// <summary>
    /// Gets the <see cref="T:TensorFlow.Graph"/> with the specified name, or None if the named operation does not exist in the graph.
    /// </summary>
    /// <param name="name">Name to lookup.</param>
    member this.TryGet (name : string) = 
            if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
            let h = TF_GraphOperationByName (handle, name)
            if h = IntPtr.Zero then None
            else Some(new TFOperation (h))

    /// <summary>
    /// Gets the <see cref="T:TensorFlow.Graph"/> with the specified name, or null if the named operation does not exist in the graph.
    /// </summary>
    /// <param name="name">Name to lookup.</param>
    member this.Item 
        with get(name : string) : TFOperation = this.TryGet(name) |> Option.orNull
    

    /// <summary>
    /// Returns the enumerator that returns all the Operations in a graph.
    /// </summary>
    /// <returns>The enumerator.</returns>
    member this.GetEnumerator () : IEnumerable<TFOperation> =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        let mutable token = IntPtr.Zero
        Seq.unfold (fun _ -> 
            match TF_GraphNextOperation (handle, &token) with
            | operll when operll = IntPtr.Zero -> None
            | operll -> Some(TFOperation(operll),())) ()

    /// <summary>
    ///  Returns the tensor shape for the specific output pparameters as an array of longs.
    /// </summary>
    /// <returns>null for single dimension, .</returns>
    /// <param name="output">The output operation to probe.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.GetShape (output : TFOutput, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let ndims = TF_GraphGetTensorNumDims (this.Handle, output.Struct, cstatus.Handle)
        if not(cstatus.CheckMaybeRaise (?incoming = status, last = false)) || 
            ndims = 0
        then box null :?> Int64 []
        else 
            let ret = Array.zeroCreate<int64> ndims 
            TF_GraphGetTensorShape (handle, output.Struct, ret, ndims, cstatus.Handle)
            cstatus.CheckMaybeRaise (?incoming=status) |> ignore
            ret

    /// <summary>
    /// Returns the current name scope in use, to change this, use the WithScope method.
    /// </summary>
    /// <value>The current name scope.</value>
    member this.CurrentNameScope 
        with get() = currentNameScope
        and internal set(x) = currentNameScope <- x

    /// <summary>
    /// Creates a new namescope by setting the scope to the description provided.
    /// </summary>
    /// <returns>A new scope that will remain in use until the return TFScope is disposed.</returns>
    /// <param name="nameScopeDesc">The namescope description, if the value is null, this
    /// will reset the toplevel namescope to be the empty value. </param>
    /// <remarks>
    /// <para>
    /// To more easily name your operations and group then, you can use the
    /// WithScope method to set a current name scope that alter the complete name
    /// of an operation added to the graph.
    /// </para>
    /// <para>
    /// The graph starts with a scope set to the empty string, you can introduce new
    /// scopes by calling WithScope, and can be conveniently used with the C# using
    /// statement, like this:
    /// </para>
    /// <code>
    /// Assert (graph.CurrentNamescope, "")
    /// using (var nested = graph.WithScope ("nested")){
    ///    Assert (graph.CurrentNameScope, "nested")
    ///    using (var inner = graph.WithScope ("inner")){
    ///        Assert (graph.CurrentNameScope, "nested/inner")
    ///    }
    /// }
    /// </code>
    /// </remarks>
    member this.WithScope (nameScopeDesc : string) =
        let prevScope = this.CurrentNameScope
        this.CurrentNameScope <- 
            match this.CurrentNameScope with
            | "" -> nameScopeDesc
            | _ -> currentNameScope + "/" + nameScopeDesc
        {new IDisposable with member x.Dispose() = this.CurrentNameScope <- prevScope}

    /// <summary>
    /// Returns the current variable dependencies in use. New tensors and operations will be created
    /// with an added input dependency to the operations specified in this property. To change this, 
    /// use the WithDependencies method.
    /// </summary>
    /// <value>The current input dependencies to be used for new tensors and operations.</value>
    member this.CurrentDependencies 
        with get() = currentDependencies 
        and internal set(deps) = currentDependencies <- deps

    /// <summary>
    /// Adds new dependencies for new tensors and operations created while the context is active.
    /// </summary>
    member this.WithDependencies ([<ParamArray>] dependencies : TFOperation []) = 
        let prevDeps = this.CurrentDependencies
        this.CurrentDependencies <- [|yield! prevDeps; yield! dependencies|] |> Array.distinct
        {new IDisposable with member x.Dispose() = this.CurrentDependencies <- prevDeps}

    member this.MakeUnique (name : string) = name + (string  <| values.GetThenIncrement(name))

    member this.MakeName (name : string, userName : string) = 
        match userName with 
        | "" -> 
            if this.CurrentNameScope = "" then name else this.CurrentNameScope + "/" + name
            |> this.MakeUnique
        | _ -> 
            if this.CurrentNameScope = "" then userName else this.CurrentNameScope + "/" + name 


    member internal this.GetNextId () = 
        let x = lastId
        lastId <- lastId + 1
        x

    /// <summary>
    /// Imports a graph serialized into the graph
    /// </summary>
    /// <param name="graphDef">Serialized graph definition (in protocol buffer format).</param>
    /// <param name="options">Import options.</param>
    /// <param name="returnOutputs">Array large enough to contain all the return options.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    ///   If you are tryig to load a file stored using the SavedModel file format, you should use the <see cref="T:TensorFlow.Session.FromSavedModel"/> API instead.
    /// </remarks>
    member this.ImportGraphDef (graphDef : TFBuffer, options :TFImportGraphDefOptions, returnOutputs : TFOutput [], ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box graphDef = null then raise(ArgumentNullException ("graphDef"))
        if box options = null then raise(ArgumentNullException ("options"))
        let cstatus = TFStatus.Setup (?incoming=status)
        if returnOutputs = null then
            TF_GraphImportGraphDefWithReturnOutputs (handle, graphDef.LLBuffer, options.Handle, NativePtr.ofNativeInt IntPtr.Zero, 0, cstatus.Handle)
        else 
            let returnOutputs = returnOutputs |> Array.map (fun x -> x.Struct)
            use first = fixed &returnOutputs.[0]
            TF_GraphImportGraphDefWithReturnOutputs (handle, graphDef.LLBuffer, options.Handle, first, returnOutputs.Length, cstatus.Handle)


    static member CopyFrom (ptr : nativeptr<TF_Output>, n : int) : TF_Output []   =
        let r = Array.zeroCreate<TF_Output> n
        for i = 0 to n - 1 do
            r.[i] <- NativePtr.get ptr i
        r


    /// <summary>
    /// Constructs a while loop with the specified inputs and a callback that composes the while loop
    /// </summary>
    /// <param name="inputs">Inputs.</param>
    /// <param name="constructor">Callback method that fills out the various while loop parameters.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <returns>
    /// An array of Outputs from creating the While loop, or null if there is an error creating the 
    /// while loop, or if the constructor raised an exception when it was invoked.
    /// </returns>
    member this.While(inputs : TFOutput [], constructor : WhileConstructor, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box inputs = null then raise(ArgumentNullException ("inputs"))
        if constructor = null then raise(ArgumentNullException ("constructor"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let mutable result = TF_NewWhile (handle, inputs |> Array.map (fun x -> x.Struct), inputs.Length, cstatus.Handle)
        if cstatus.Error then box null :?> TFOutput[]
        else
            try
                // 
                // Call constructor here
                // Wrap the various TF_graphs (with owns=false)
                // Marshal the condInputs, bodyInputs
                //
                let mutable name : string = box null :?> string
                let n = result.ninputs
                let bodyOutputs = Array.zeroCreate<TF_Output> n 
                let condGraph = new GraphUnowned (result.cond_graph) :> TFGraph
                let bodyGraph = new GraphUnowned (result.body_graph) :> TFGraph
                constructor.Invoke(condGraph, TFGraph.CopyFrom (result.cond_inputs, n), result.cond_output, bodyGraph, TFGraph.CopyFrom (result.body_inputs, n), bodyOutputs, name)

                let name = if box name = null || name = "" then this.MakeUnique ("while") else name
                // On return, copy the condOutput and bodyOututs
                let text = Encoding.UTF8.GetBytes (name)
                result.charPtrName <- Marshal.AllocHGlobal (text.Length + 1)
                Marshal.Copy (text, 0, result.charPtrName, text.Length)
                Marshal.WriteByte (result.charPtrName, text.Length, 0uy)
                for i = 0 to n - 1 do
                    NativePtr.set result.body_outputs i bodyOutputs.[i]

                let ret = Array.zeroCreate<TF_Output> n 
                use first = fixed &ret.[0]
                TF_FinishWhile (&result, cstatus.Handle, first)
                if (cstatus.CheckMaybeRaise (?incoming=status)) then 
                    ret |> Array.map TFOutput
                else null
            with
            | _ -> 
                TF_AbortWhile (&result)
                null

    /// <summary>
    /// Adds a gradient: the operations needed to compute the partial derivatives of sum of <paramref name="y"/>` wrt to <paramref name="x"/>.
    /// </summary>
    /// <returns>The partial derivatives, the size of the array is the same as the length of the <paramref name="y"/> array.</returns>
    /// <param name="y">The y elements.</param>
    /// <param name="x">The x elements.</param>
    /// <param name="dx">Initial gradients, which represent the symbolic partial derivatives of some loss function `L` w.r.t. <paramref name="y"/> ).   
    /// If the parameter is null, the implementation will use dx for 'OnesLike' for all shapes in <paramref name="y"/></param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// d(y[0] + y[1]+ ...)/dx[0], d(y[0] + y[1] + ...)/dx[1]z...
    /// </remarks>
    member this.AddGradients(y : TFOutput [], x : TFOutput [], ?dx : TFOutput [], ?status : TFStatus) : TFOutput [] =
        if y = null then raise(ArgumentNullException ("y"))
        if x = null then raise(ArgumentNullException ("x"))
        dx |> Option.iter (fun dx -> 
            if dx.Length <> y.Length then 
                raise(ArgumentException ("If dx is not null, the size of the gradients must match the size of y", "dx")))
        let cstatus = TFStatus.Setup (?incoming=status)
        let ret = Array.zeroCreate<TF_Output> x.Length //new Output [x.Length]
        use pret = fixed &ret.[0]
        let y = y |> Array.map (fun x -> x.Struct)
        let x = x |> Array.map (fun x -> x.Struct)
        use py = fixed &y.[0]
        use px = fixed &x.[0] 
        match dx with
        | None ->
            TF_AddGradients (handle, py, y.Length, px, x.Length, NativePtr.ofNativeInt IntPtr.Zero, cstatus.Handle, pret)
        | Some(dx) ->
            let dx = dx |> Array.map (fun x -> x.Struct)
            use pdx = fixed &dx.[0]
            TF_AddGradients (handle, py, y.Length, px, x.Length, pdx, cstatus.Handle, pret)
        if not(cstatus.CheckMaybeRaise (?incoming=status, last = false)) then
             null
        else
            ret |> Array.map TFOutput

    /// <summary>
    /// Adds a gradient: the operations needed to compute the partial derivatives of sum of <paramref name="y"/>` wrt to <paramref name="x"/>.
    /// </summary>
    /// <returns>The partial derivatives, the size of the array is the same as the length of the <paramref name="y"/> array.</returns>
    /// <param name="prefix">names the scope into which all gradients operations are being added.  This must be unique within 
    /// the provided graph otherwise this operation will fail.  If the value is null, the default prefixing behaviour takes
    /// place, see AddGradients for more details.
    /// </param>
    /// <param name="y">The y elements.</param>
    /// <param name="x">The x elements.</param>
    /// <param name="dx">Initial gradients, which represent the symbolic partial derivatives of some loss function `L` w.r.t. <paramref name="y"/> ).   
    /// If the parameter is null, the implementation will use dx for 'OnesLike' for all shapes in <paramref name="y"/></param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// d(y[0] + y[1]+ ...)/dx[0], d(y[0] + y[1] + ...)/dx[1]z...
    /// </remarks>
    member this.AddGradients (prefix : string,  y : TFOutput [], x : TFOutput [], ?dx : TFOutput [], ?status : TFStatus) : TFOutput [] =
        if y = null then raise(ArgumentNullException ("y"))
        if x = null then raise(ArgumentNullException ("x"))
        dx |> Option.iter (fun dx -> 
            if dx.Length <> y.Length then 
                raise(ArgumentException ("If dx is not null, the size of the gradients must match the size of y", "dx")))
        let cstatus = TFStatus.Setup (?incoming=status)
        let ret = Array.zeroCreate<TF_Output> x.Length
        use pret = fixed &ret.[0]
        let y = y |> Array.map (fun x -> x.Struct)
        let x = x |> Array.map (fun x -> x.Struct)
        use py = fixed &y.[0]
        use px = fixed &x.[0]
        match dx with
        | None -> 
            TF_AddGradientsWithPrefix (handle, prefix, py, y.Length, px, x.Length, NativePtr.ofNativeInt IntPtr.Zero, cstatus.Handle, pret)
        | Some(dx) -> 
            let dx = dx |> Array.map (fun x -> x.Struct)
            use pdx = fixed &dx.[0] 
            TF_AddGradientsWithPrefix (handle, prefix, py, y.Length, px, x.Length, pdx, cstatus.Handle, pret)
        if not(cstatus.CheckMaybeRaise (?incoming=status, last = false)) then
             null
        else
            ret |> Array.map TFOutput

    /// <summary>
    /// Creates a TFFunction from a Graph
    /// </summary>
    /// <returns>The function.</returns>
    /// <param name="functionName">Name of the new function.  Should match the operation name (OpDef.name) regexp [A-Z][A-Za-z0-9_.\\-/]*.  If appendHashToFunctioName is false, the name must be unique (at least those registered in graphs where this function will be used).</param>
    /// <param name="description">Optional, human readable description of this function.</param>
    /// <param name="operations">Array of operations to become the body of the function or null.  
    ///     If no array is given , all the
    ///     operations in function body will become part of the function
    ///     except operations referenced in inputs. These operations
    ///     must have a single output (these operations are typically
    ///     placeholders created for the sole purpose of representing
    ///     an input).
    /// 
    ///     If an array is given, all operations
    ///     in it will become part of the function. In particular, no
    ///     automatic skipping of dummy input operations is performed.
    /// </param>
    /// <param name="inputs">Array that specify the inputs to the function, or null.  The names used for function inputs are normalized
    ///     names of the operations (usually placeholders) pointed to by
    ///     inputs.  These operation names should start with a letter.
    ///     Normalization will convert all letters to lowercase and
    ///     non-alphanumeric characters to '_' to make resulting names match
    ///     the "[a-z][a-z0-9_]*" pattern for operation argument names.
    ///     `inputs` cannot contain the same tensor twice.</param>
    /// <param name="outputs">rray that specify the inputs to the function, or null.   This can contain the same tensor twice.</param>
    /// <param name="outputNames">The names of the function's outputs.   The array either has the same elements of outputs, or be null.   Names must match "[a-z][a-z0-9_]*" regexp, if null is passed, the names are generated automatically.</param>
    /// <param name="appendHashToFunctionName">If set to <c>true</c> appends hash to functionName, otherwise it will use the specified name in functionName.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// <para>
    ///   This method converts the graph whose operations (or a subset of its operations) will be converted
    ///   into a TFFunction.
    /// </para>
    /// <para>
    ///   Note that when the same TF_Output is listed as both an input and an output,
    ///   the corresponding function's output will equal to this input,
    ///   instead of the original node's output.
    /// </para>
    /// <para>
    /// Callers must also satisfy the following constraints:
    /// </para>
    /// <para>
    ///   <paramref name="inputs"/> cannot refer to Outputs within a control flow context. For
    ///   example, one cannot use the output of "switch" node as input.
    /// </para>
    /// <para>
    ///   <paramref name="inputs"/> and <paramref name="outputs"/> cannot have reference types. Reference types are
    ///   not exposed through C API and are being replaced with Resources. We support
    ///   reference types inside function's body to support legacy code. Do not
    ///   use them in new code.
    /// </para>
    /// <para>
    ///   Every node in the function's body must have all of its inputs (including
    ///   control inputs). In other words, for every node in the body, each input
    ///   must be either listed in <paramref name="inputs"/> or must come from another node in
    ///   the body. In particular, it is an error to have a control edge going from
    ///   a node outside of the body into a node in the body. This applies to control
    ///   edges going from nodes referenced in <paramref name="inputs"/> to nodes in the body when
    ///   the former nodes are not in the body (automatically skipped or not
    ///   included in explicitly specified body).
    /// </para>
    /// </remarks>
    member this.ToFunction (functionName : string,
                            description : string,
                            operations : TFOperation [],
                            inputs : TFOutput [],
                            outputs : TFOutput [],
                            outputNames : string [],
                            ?appendHashToFunctionName : bool, // false
                            ?status : TFStatus) : TFFunction  =
        let appendHashToFunctionName = defaultArg appendHashToFunctionName false
        if functionName = null then raise(ArgumentNullException ("functionName"))
        if outputs = null then
            if outputNames <> null then raise(ArgumentException ("outputs is null, but outputNames is not", "outputNames"))
        else
            if outputNames <> null && outputs.Length <> outputNames.Length then
                raise(ArgumentException ("the outputs and outputNames array are specified, but have different lengths"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let functionOptions = IntPtr.Zero
        let ops = IntPtr.Zero
        let nops, ops =
            if box operations = null then 0,IntPtr.Zero
            else
                let nops = operations.Length
                let ops = Marshal.AllocHGlobal (sizeof<IntPtr> * operations.Length)
                for i = 0 to nops - 1 do
                    Marshal.WriteIntPtr (ops, i * sizeof<IntPtr>, operations.[i].Handle)
                nops, ops

        let fnHandle = 
            TF_GraphToFunction (
                                handle, 
                                functionName, (if appendHashToFunctionName then 1uy else 0uy),
                                nops, ops,
                                (if inputs = null then 0 else inputs.Length), (inputs |> Array.map (fun x -> x.Struct)),
                                (if outputs = null then 0 else outputs.Length), (outputs |> Array.map (fun x -> x.Struct)),
                                outputNames,
                                functionOptions,
                                description,
                                cstatus.Handle)

        if ops <> IntPtr.Zero then Marshal.FreeHGlobal (ops)
        if not(cstatus.CheckMaybeRaise (?incoming=status, last = false)) then box null :?> TFFunction
        else new TFFunction (fnHandle)


    /// <summary>
    /// Returns the serialized VersionDef proto for this graph.
    /// </summary>
    /// <returns>The versions.</returns>
    /// <param name="outputVersionDef">The buffer where the serialized protocol buffer will be stored.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.Versions (outputVersionDef : TFBuffer, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box outputVersionDef = null then raise (ArgumentNullException ("outputVersionDef"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_GraphVersions (handle, outputVersionDef.LLBuffer, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status)

    /// <summary>
    /// Returns the number of TF_Functions registered in this graph.
    /// </summary>
    /// <value>The number functions.</value>
    member this.NumFunctions : int = TF_GraphNumFunctions (handle)

    /// <summary>
    /// Returns an the functions that have been defined in the graph.
    /// </summary>
    /// <value>The functions.</value>
    member this.Functions 
        with get() : TFFunction [] =
            if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
            let n = this.NumFunctions
            let size = sizeof<IntPtr>
            let buffer = Marshal.AllocHGlobal (n * size)
            let ret =
                use status = new TFStatus() 
                let num = TF_GraphGetFunctions (handle, buffer, n, status.Handle)
                if num > 0 && status.Ok  then
                    let ret = Array.zeroCreate<TFFunction> num
                    let mutable ofs = 0
                    for i = 0 to num - 1 do
                        let tfhandle = Marshal.ReadIntPtr (buffer, ofs)
                        ret.[i] <- new TFFunction (tfhandle)
                        ofs <- ofs + size
                    ret
                else box null :?> TFFunction[]
            Marshal.FreeHGlobal (buffer)
            ret


    /// <summary>
    /// Attempts to evaluate the <paramref name="output"/>.   This is only possible if <paramref name="output"/> does not
    /// depend on any graph inputs - the function is safe to call if this is not the case though.
    /// </summary>
    /// <returns><c>true</c>, if the evaluation is successful, in which case the result is returned in <paramref name="tensor"/>, <c>false</c> otherwise.</returns>
    /// <param name="output">Output.</param>
    /// <param name="tensor">TFTensor.</param>
    member this.TryEvaluateConstant (output : TFOutput, [<Out>] tensor : TFTensor byref) = 
        let cstatus = new TFStatus ()
        let mutable ptr = IntPtr.Zero
        let ret = TF_TryEvaluateConstant (handle, output.Struct, &ptr, cstatus.Handle)
        cstatus.Dispose ()
        if ret then
            tensor <- new TFTensor (ptr)
        else
            tensor <- box null :?> TFTensor
        ret
    
    override this.ToString () =
            let mutable len = IntPtr.Zero
            TF_GraphDebugString (this.Handle, &len)

module OperationDescNative = 
    // extern void TF_SetAttrShape (TF_OperationDescription *desc, const char *attr_name, const int64_t *dims, int num_dims)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, int64 [] dims, int num_dims)

/// <summary>
/// Low-level TensorFlow operation builder
/// </summary>
/// <remarks>
/// <para>This is the low-level API that is used to create operations by manually specificying all
/// the parameters of an operation (inputs, outputs, attribute descriptions) that can then
/// be attached into a graph.
/// </para>
/// <para>
/// Generally, you will instead be using the methods surfaced in <see cref="T:TensorFlow.Graph"/> 
/// that surfaces a C# high-level API that has already been bound to the built-in TensorFlow
/// nodes.
/// </para>
/// <para>
/// You create instances bound to a graph, add inputs, attributes and so on, and when you are done
/// you can call the <see cref="FinishOperation"/> method that will turn this TFOperationDesc 
/// into a <see cref="T:TensorFlow.Operation"/>.
/// </para>
/// </remarks>
type TFOperationDesc private (graph : TFGraph, opType : string, name : string, handle : IntPtr) =
    inherit TFDisposable(handle)
    let mutable handle = handle

    // extern TF_OperationDescription * TF_NewOperation (TF_Graph *graph, const char *op_type, const char *oper_name)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_OperationDescription TF_NewOperation (TF_Graph graph, string opType, string oper_name)

    // extern void TF_AddInputList (TF_OperationDescription *desc, const TF_Output *inputs, int num_inputs)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddInputList (TF_OperationDescription desc, TF_Output [] inputs, int num_inputs)

    // extern void TF_SetDevice (TF_OperationDescription *desc, const char *device)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetDevice (TF_OperationDescription desc, string device)

    // extern void TF_AddInput (TF_OperationDescription *desc, TF_Output input)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddInput (TF_OperationDescription desc, TF_Output input)

    // extern void TF_AddControlInput (TF_OperationDescription *desc, TF_Operation *input)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddControlInput (TF_OperationDescription desc, TF_Operation input)

    // extern void TF_SetAttrString (TF_OperationDescription *desc, const char *attr_name, const void *value, size_t length)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrString (TF_OperationDescription desc, string attr_name, IntPtr value, size_t length)

    // extern void TF_SetAttrStringList (TF_OperationDescription *desc, const char *attr_name, const void *const *values, const size_t *lengths, int num_values)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrStringList (TF_OperationDescription desc, string attr_name, IntPtr [] values, UIntPtr [] lengths, int num_values)

    // extern void TF_ColocateWith (TF_OperationDescription *desc, TF_Operation *op)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ColocateWith (TF_OperationDescription desc, TF_Operation op)

    // extern void TF_SetAttrInt (TF_OperationDescription *desc, const char *attr_name, int64_t value)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrInt (TF_OperationDescription desc, string attr_name, int64 value)
    // extern void TF_SetAttrIntList (TF_OperationDescription *desc, const char *attr_name, const int64_t *values, int num_values)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrIntList (TF_OperationDescription desc, string attr_name, int64 [] values, int num_values)
    // extern void TF_SetAttrFloat (TF_OperationDescription *desc, const char *attr_name, float32 value)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrFloat (TF_OperationDescription desc, string attr_name, float32 value)
    // extern void TF_SetAttrFloatList (TF_OperationDescription *desc, const char *attr_name, const float32 *values, int num_values)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrFloatList (TF_OperationDescription desc, string attr_name, float32 [] values, int num_values)

    // extern void TF_SetAttrBool (TF_OperationDescription *desc, const char *attr_name, unsigned char value)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrBool (TF_OperationDescription desc, string attr_name, byte value)

    // extern void TF_SetAttrBoolList (TF_OperationDescription *desc, const char *attr_name, const unsigned char *values, int num_values)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrBoolList (TF_OperationDescription desc, string attr_name, bool [] values, int num_values)

    // extern void TF_SetAttrType (TF_OperationDescription *desc, const char *attr_name, TF_DataType value)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrType (TF_OperationDescription desc, string attr_name, uint32 value)

    // extern void TF_SetAttrTypeList (TF_OperationDescription *desc, const char *attr_name, const TF_DataType *values, int num_values)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTypeList (TF_OperationDescription desc, string attr_name, uint32[] values, int num_values)


    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, IntPtr dims, int num_dims)

    // extern void TF_SetAttrShapeList (TF_OperationDescription *desc, const char *attr_name, const int64_t *const *dims, const int *num_dims, int num_shapes)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrShapeList (TF_OperationDescription desc, string attr_name, IntPtr[] dims, int[] num_dims, int num_shapes)

    // extern void TF_SetAttrTensorShapeProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTensorShapeProto (TF_OperationDescription desc, string attr_name, IntPtr proto, size_t proto_len, TF_Status status)

    // extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription *desc, const char *attr_name, const void *const *protos, const size_t *proto_lens, int num_shapes, TF_Status *status)
    //[DllImport (NativeBinding.TensorFlowLibrary)]
    //static extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription desc, string attr_name, void** protos, size_t* proto_lens, int num_shapes, TF_Status status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription desc, string attr_name, IntPtr[] protos, size_t* proto_lens, int num_shapes, TF_Status status)

    // extern void TF_SetAttrTensor (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *value, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTensor (TF_OperationDescription desc, string attr_name, TF_Tensor value, TF_Status status)

    // extern void TF_SetAttrTensorList (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *const *values, int num_values, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTensorList (TF_OperationDescription desc, string attr_name, IntPtr [] values, int num_values, TF_Status status)
    // extern void TF_SetAttrValueProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrValueProto (TF_OperationDescription desc, string attr_name, IntPtr proto, size_t proto_len, TF_Status status)

    // extern TF_Operation * TF_FinishOperation (TF_OperationDescription *desc, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Operation TF_FinishOperation (TF_OperationDescription desc, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrFuncName (TF_OperationDescription desc, string attr_name, string value, IntPtr len)

    new (graph : TFGraph, opType : string, name: string) =
        let handle = TF_NewOperation (graph.Handle, opType, name)
        new TFOperationDesc(graph, opType, name,handle)

    member this.OpType = opType
    override this.NativeDispose (handle : IntPtr) =
        // If you reach this, you never called FinishOperation
        printf "OperationDescription(%s,%s was never turned into an Operation" opType name

    /// <summary>
    /// Specifies the device for the operation, if one is not provided, the operation is unconstrained.
    /// </summary>
    /// <returns>This instance, allows for chaining operation invocations.</returns>
    /// <param name="device">The device to constraint to in this operation.</param>
    member this.SetDevice (device : string) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if device = null then raise(ArgumentNullException ("device"))
        TF_SetDevice (handle, device)
        this

    /// <summary>
    /// Adds the specified input to the operation
    /// </summary>
    /// <returns>The input.</returns>
    /// <param name="input">Input.</param>
    member this.AddInput (input : TFOutput) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        TF_AddInput (handle, input.Struct)
        this

        /// <summary>
        /// Adds a series of inputs to the operation.
        /// </summary>
        /// <param name="inputs">Inputs, this is a params array for your convenience.</param>
    member this.AddInputs([<ParamArray>] inputs : TFOutput []) =
            if (handle = IntPtr.Zero) then raise (ObjectDisposedException ("handle"))
            if not (box inputs = null || inputs.Length = 0) then
                TF_AddInputList (handle, inputs |> Array.map (fun x -> x.Struct), inputs.Length)
            this

    /// <summary>
    /// Ensure that the operation does not execute before the control operation does.
    /// </summary>
    /// <param name="control">Operation that must be executed before running this operation.</param>
    /// <remarks>
    /// <para>
    /// A control input is an Operation that must be executed before running the operation 
    /// currently being built.  
    /// </para>
    /// <para>
    /// For example, an Assert operation may be added as a control input for this operation. 
    /// The Assert now behaves as a pre-condition that will always verify itself before
    /// running the operation.
    /// </para>
    /// </remarks>
    member this.AddControlInput (control : TFOperation) =
        if (handle = IntPtr.Zero) then raise (ObjectDisposedException ("handle"))
        if (box control = null) then raise (ArgumentNullException ("input"))
        TF_AddControlInput (handle, control.Handle)
        this

    member this.ColocateWith (op : TFOperation) = 
        if (handle = IntPtr.Zero) then (raise (ObjectDisposedException ("handle")))
        if (box op = null) then raise (ArgumentNullException ("op"))
        TF_ColocateWith (handle, op.Handle)
        this

    member this.SetAttr (attrName : string, value : string) =
        if (handle = IntPtr.Zero) then raise (ObjectDisposedException ("handle"))
        if (box attrName = null) then raise (ArgumentNullException ("attrName"))
        let bytes = Encoding.UTF8.GetBytes (value)
        let buf = Marshal.AllocHGlobal (bytes.Length + 1)
        Marshal.Copy (bytes, 0, buf, bytes.Length)
        TF_SetAttrString (handle, attrName, buf, UIntPtr(uint64 bytes.Length))
        this


    member this.SetAttr (attrName : string, values : string []) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box attrName = null then raise (ArgumentNullException ("attrName"))
        if box values = null then raise (ArgumentNullException ("values"))
        let n = values.Length
        let unmanaged,lengths = 
            Array.init n (fun i ->
                let bytes = Encoding.UTF8.GetBytes (values.[i])
                let buf = Marshal.AllocHGlobal (bytes.Length + 1)
                let bc = bytes.Length
                Marshal.Copy (bytes, 0, buf, bc)
                (buf,UIntPtr(uint64 bc))) |> Array.unzip
        TF_SetAttrStringList (handle, attrName, unmanaged, lengths, n)
        this


    member this.SetAttr (attrName : string, value : int64) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("attrName"))
        TF_SetAttrInt (handle, attrName, value)
        this

    member this.SetAttr (attrName : string, values : int64[]) = 
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("attrName"))
        if box values = null then raise(ArgumentNullException ("values"))
        TF_SetAttrIntList (handle, attrName, values, values.Length)
        this

    member this.SetAttr (attrName : string, value : float32) =
        if handle = IntPtr.Zero then raise( ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        TF_SetAttrFloat (handle, attrName, value)
        this

    member this.SetAttr (attrName : string, values : float32[]) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("attrname"))
        if box values = null then raise(ArgumentNullException ("values"))
        TF_SetAttrFloatList (handle, attrName, values, values.Length)
        this


    member this.SetAttr (attrName : string, value : bool) =
        if handle = IntPtr.Zero then raise( ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        TF_SetAttrBool (handle, attrName, if value then 1uy else 0uy)
        this


    member this.SetAttr (attrName : string, values : bool[]) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("handle"))
        if box values = null then raise(ArgumentNullException ("values"))
        TF_SetAttrBoolList (handle, attrName, values, values.Length)
        this

    member this.SetAttr (attrName : string, value : TFDataType) =
        if handle = IntPtr.Zero then raise( ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        TF_SetAttrType (handle, attrName, uint32 value)
        this

    member this.SetAttr (attrName : string, [<ParamArray>] values : TFDataType[]) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("handle"))
        if box values = null then raise(ArgumentNullException ("values"))
        TF_SetAttrTypeList(handle, attrName, values |> Array.map uint32, values.Length)
        this

    member this.SetAttr (attrName : string, shape : TFShape) =
        if handle = IntPtr.Zero then raise( ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        if (box shape = null || box shape.Dims = null) then
            TF_SetAttrShape (handle, attrName, box null :?> IntPtr, -1)
        else
            OperationDescNative.TF_SetAttrShape (handle, attrName, shape.Dims, shape.Dims.Length)
        this

    member this.SetAttr (attrName : string, shapeList : TFShape []) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        if box shapeList = null then raise(ArgumentNullException ("shapeList"))
        let num_shapes = shapeList.Length
        let num_dims = Array.zeroCreate<int> shapeList.Length 
        let dims = Array.init num_shapes (fun i -> 
            num_dims.[i] <- shapeList.[i].NumDimensions
            let array = Marshal.AllocHGlobal(sizeof<int64> * shapeList.[i].Dims.Length)
            Marshal.Copy(shapeList.[i].Dims, 0, array, shapeList.[i].Dims.Length)
            array)
        TF_SetAttrShapeList (handle, attrName, dims, num_dims, num_shapes)
        this

    member this.SetAttrTensorShapeProto (attrName : string, proto : IntPtr, protoLen : size_t, ?status : TFStatus) = 
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_SetAttrTensorShapeProto (handle, attrName, proto, protoLen, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        this


    // WARN: untested
    member this.SetAttrShapeProtoList (attrName : string, protos : TFBuffer[], ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("attrName"))
        if box protos = null then raise(ArgumentNullException ("protos"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let lengths = Array.zeroCreate<size_t> protos.Length
        let unmanaged = Array.init protos.Length (fun i -> protos.[i].Handle)
        use protoLengths = fixed &lengths.[0]
        TF_SetAttrTensorShapeProtoList(handle, attrName, unmanaged, protoLengths, unmanaged.Length, cstatus.Handle)
        // prevent finalization of managed TFBuffer
        GC.KeepAlive(protos)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        this

    member this.SetAttr (attrName : string, tensor : TFTensor, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box attrName = null then raise (ArgumentNullException ("attrName"))
        if box tensor = null then raise (ArgumentNullException ("tensor"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_SetAttrTensor (handle, attrName, tensor.Handle, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        this


    member this.SetAttr (attrName : string, tensors : TFTensor [], ?status : TFStatus) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box attrName = null then raise (ArgumentNullException ("attrName"))
        if box tensors = null then raise (ArgumentNullException ("tensors"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let unmanaged = Array.init tensors.Length (fun i -> tensors.[i].Handle)
        TF_SetAttrTensorList (handle, attrName, unmanaged, unmanaged.Length, cstatus.Handle)
        // prevent finalization of managed Tensors
        GC.KeepAlive(tensors)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        this

    // WARN: untested
    // NOTE was proto originally had type TFProto
    member this.SetAttr(attrName : string, proto : TFBuffer, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
        if box attrName = null then raise(ArgumentNullException ("attrName"))
        if box proto = null then raise(ArgumentNullException ("proto"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let buff = proto.LLBuffer |> NativePtr.read
        TF_SetAttrValueProto(handle, attrName, buff.data, buff.length, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        this

    /// <summary>
    /// Turns the operation description into an actual operation in the graph.
    /// </summary>
    /// <returns>The operation on success, or null on error.</returns>
    /// <param name="status">Optional status, on failure the operation is not added to the graph.  If you pass null (the default), this operation throws on error conditions.</param>
    member this.FinishOperation (?status : TFStatus) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let h = TF_FinishOperation (handle, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        handle <- IntPtr.Zero
        GC.SuppressFinalize (this)
        match status with 
        | Some status when status.Error -> box null :?> TFOperation
        | _ -> new TFOperation (h)

    /// <summary>
    /// Sets an attribute on the function to the specified value.
    /// </summary>
    /// <param name="attrName">The attribute name.</param>
    /// <param name="value">The value for the attribute.</param>
    member this.SetAttribute (attrName : string, value : string) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box attrName = null then raise (ArgumentNullException ("attrName"))
        if box value = null then raise (ArgumentNullException ("value"))
        TF_SetAttrFuncName (handle, attrName, value, IntPtr(value.Length))
