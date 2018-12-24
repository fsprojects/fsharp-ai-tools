namespace Tensorflow
open System.Reflection

#nowarn "9"
open System
open System.Runtime.InteropServices
open System.Text
open System.Globalization
open System.Linq
open Utils
open Common
open Microsoft.FSharp.NativeInterop
open System.Numerics;
open System.Collections.Generic;
open System.Linq.Expressions;

// We use this TF_Xxx as the native "TF_Xxx *" as those are opaque
type TF_Status = System.IntPtr
type TF_SessionOptions = System.IntPtr
type TF_Graph = System.IntPtr
type TF_OperationDescription = System.IntPtr
type TF_Operation = System.IntPtr
type TF_Session = System.IntPtr
type TF_DeprecatedSession = System.IntPtr
type TF_Tensor = System.IntPtr
type TF_ImportGraphDefOptions = System.IntPtr
type TF_Library = System.IntPtr
type TF_BufferPtr = System.IntPtr
type TF_Function = System.IntPtr
type TF_DeviceList = System.IntPtr

type size_t = System.UIntPtr


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
    /// The device is a Tensor Processing Unit (TPU)
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
    // extern TF_ImportGraphDefOptions * TF_NewImportGraphDefOptions ();
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern TF_ImportGraphDefOptions TF_NewImportGraphDefOptions ();

/// Contains options that are used to control how graph importing works.
type TFImportGraphDefOptions() =
    inherit TFDisposable(TFImportGraphDefOptionsExternal.TF_NewImportGraphDefOptions ())

    // extern void TF_DeleteImportGraphDefOptions (TF_ImportGraphDefOptions *opts);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteImportGraphDefOptions (TF_ImportGraphDefOptions opts);

    // extern void TF_ImportGraphDefOptionsSetPrefix (TF_ImportGraphDefOptions *opts, const char *prefix);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsSetPrefix (TF_ImportGraphDefOptions opts, string prefix);

    // extern void TF_ImportGraphDefOptionsAddInputMapping (TF_ImportGraphDefOptions *opts, const char* src_name, int src_index, TF_Output dst);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsAddInputMapping (TF_ImportGraphDefOptions opts, string src_name, int src_index, TF_Output dst);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsAddControlDependency (TF_ImportGraphDefOptions opts, TF_Operation oper);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsAddReturnOutput (TF_ImportGraphDefOptions opts, string oper_name, int index);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_ImportGraphDefOptionsNumReturnOutputs (TF_ImportGraphDefOptions opts);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsRemapControlDependency (TF_ImportGraphDefOptions opts, string srcName, TF_Operation dst);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsSetUniquifyNames (TF_ImportGraphDefOptions opts, byte uniquify);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ImportGraphDefOptionsSetUniquifyPrefix (TF_ImportGraphDefOptions opts, byte uniquify_prefix);

    override this.NativeDispose (handle : IntPtr) = TF_DeleteImportGraphDefOptions (handle);

    member this.SetPrefix (prefix : string) =
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsSetPrefix (this.Handle, prefix);


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
    member this.AddInputMapping (srcName : string, srcIndex : int, dst : Output) =
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsAddInputMapping (this.Handle, srcName, srcIndex, dst.Struct);


    /// <summary>
    /// Cause the imported graph to have a control dependency on the provided operation.
    /// </summary>
    /// <param name="operation">This operation should exist in the graph being imported to.</param>
    member this.AddControlDependency (operation : Operation) =
        if box operation = null then raise(ArgumentNullException ("operation"))
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        TF_ImportGraphDefOptionsAddControlDependency (this.Handle, operation.Handle);


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
        TF_ImportGraphDefOptionsAddReturnOutput (this.Handle, name, index);


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
    member this.RemapControlDependency (srcName : string, destination : Operation) =
        if this.Handle = IntPtr.Zero then raise(ObjectDisposedException ("Handle"))
        if srcName = null then raise(ArgumentNullException ("srcName"))
        if box destination = null then raise(ArgumentNullException ("destination"))
        if destination.Handle = IntPtr.Zero then raise(ObjectDisposedException ("destination"))
        TF_ImportGraphDefOptionsRemapControlDependency (this.Handle, srcName, destination.Handle);


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

module TFGraphExternal =
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern void TF_GraphSetTensorShape (TF_Graph graph, TF_Output output, IntPtr dims, int num_dims, TF_Status status);

/// <summary>
/// Signature of the method that will be invoked by the TFGraph.While method to construct a while loop
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
// A TFGraph that will not release the undelying handle, this is used
// when we want to surface a TFGraph that we do not own, so we do not
// want to delete the handle when this object is collected
//
and TFGraphUnowned internal (handle:IntPtr) = 
    inherit TFGraph (handle)

    // Nothing, we do not own the handle
    override this.NativeDispose (handle : TF_Status) = ()

/// <summary>
/// Represents a computation graph.  Graphs may be shared between sessions and are thread safe.
/// </summary>
/// <remarks>
/// <para>
/// Graphs consist of operations (represented by TFOperation objects), these can be named, or 
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
    let mutable currentDependencies = Array.empty<Operation>
    let values = new DictionaryCount<string> ();

    let mutable lastId = 0

    // extern TF_Graph * TF_NewGraph ();
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Graph TF_NewGraph ();

    // extern void TF_DeleteGraph (TF_Graph *);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteGraph (TF_Graph graph);

    // extern void TF_GraphSetTensorShape (TF_Graph *graph, TF_Output output, const int64_t *dims, const int num_dims, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphSetTensorShape (TF_Graph graph, TF_Output output, int64 [] dims, int num_dims, TF_Status status);

    // extern int TF_GraphGetTensorNumDims (TF_Graph *graph, TF_Output output, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_GraphGetTensorNumDims (TF_Graph graph, TF_Output output, TF_Status status);

    // extern void TF_GraphGetTensorShape (TF_Graph *graph, TF_Output output, int64_t *dims, int num_dims, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphGetTensorShape (TF_Graph graph, TF_Output output, int64 [] dims, int num_dims, TF_Status status);

    // extern void TF_GraphToGraphDef (TF_Graph *graph, TF_Buffer *output_graph_def, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphToGraphDef (TF_Graph graph, LLBuffer* output_graph_def, TF_Status status);

    // extern void TF_GraphImportGraphDef (TF_Graph *graph, const TF_Buffer *graph_def, const TF_ImportGraphDefOptions *options, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphImportGraphDef (TF_Graph graph, LLBuffer* graph_def, TF_ImportGraphDefOptions options, TF_Status status);

    // extern TF_Operation * TF_GraphOperationByName (TF_Graph *graph, const char *oper_name);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Operation TF_GraphOperationByName (TF_Graph graph, string oper_name);

    // extern TF_Operation * TF_GraphNextOperation (TF_Graph *graph, size_t *pos);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Operation TF_GraphNextOperation (TF_Graph graph, IntPtr& token); // ref token

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphImportGraphDefWithReturnOutputs (TF_Graph graph, LLBuffer *graph_def, TF_ImportGraphDefOptions options, TF_Output *return_outputs, int num_return_outputs, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TFWhileParams TF_NewWhile (TF_Graph g, TF_Output [] inputs, int ninputs, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AbortWhile (TFWhileParams& pars); // ref

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_FinishWhile (TFWhileParams& pars, TF_Status status, TF_Output *outputs); // ref

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddGradients (TF_Graph graph, TF_Output* ys, int ny, TF_Output* xs, int nx, TF_Output* dx, TF_Status status, TF_Output* dy);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddGradientsWithPrefix (TF_Graph graph, string prefix, TF_Output* ys, int ny, TF_Output* xs, int nx, TF_Output* dx, TF_Status status, TF_Output* dy);
   
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphCopyFunction (TF_Graph graph, TF_Function func, TF_Function grad, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_GraphToFunction (TF_Graph body, string fn_name, byte append_hash_to_fn_name, int num_opers, IntPtr opers, int ninputs, TF_Output [] inputs, int noutputs, TF_Output [] ouputs, string [] output_names, IntPtr options, string description, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_GraphVersions (TF_Graph graph, LLBuffer *output_version_def, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_GraphNumFunctions (TF_Graph graph);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_GraphGetFunctions (TF_Graph graph, IntPtr funcs, int max_func, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern bool TF_TryEvaluateConstant (TF_Graph graph, TF_Output output, IntPtr& result, TF_Status status); // ref result

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern string TF_GraphDebugString (TF_Graph graph, [<Out>] IntPtr len);


    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.TFGraph"/> class.
    /// </summary>
    new () = new TFGraph (TF_NewGraph ())

    override this.NativeDispose (handle : IntPtr) = TF_DeleteGraph (handle);


    /// <summary>
    /// Sets the tensor shape of the tensor referenced by <paramref name="output"/> to the shape described by <paramref name="dims"/>.
    /// </summary>
    /// <param name="output">The tensor on which this method will operate in the graph.</param>
    /// <param name="dims">The tensor shape, specified as an array of dimensions.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.SetTensorShape (output : Output, ?dims : int64 [], ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status);
        match dims with
        | None -> 
            TFGraphExternal.TF_GraphSetTensorShape (handle, output.Struct, IntPtr.Zero, 0, cstatus.Handle);
        | Some(dims) ->
            TF_GraphSetTensorShape (handle, output.Struct, dims, dims.Length, cstatus.Handle);
        cstatus.CheckMaybeRaise (?incomingStatus=status);


    /// <summary>
    /// Returns the number of dimensions of the Tensor referenced by output
    /// </summary>
    /// <returns>The number of dimensions of the tensor.</returns>
    /// <param name="output">The tensor to probe.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.GetTensorNumDims (output : Output, ?status : TFStatus) = 
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let code = TF_GraphGetTensorNumDims (handle, output.Struct, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
        code


    /// <summary>
    /// Returns the shape of a tensor specified in <paramref name="output"/>.
    /// </summary>
    ///
    /// <returns>The tensor shape.    If the number of dimensions in the shape is unknown or the shape is, a scalar, the values in the array will be zero. Otherwise, each element of will be set corresponding to the size of the dimension. An  unknown dimension is represented by -1.</returns>
    /// <param name="output">The tensor that you want to look up.  </param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.GetTensorShape (output : Output, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let n = TF_GraphGetTensorNumDims (handle, output.Struct, cstatus.Handle)
        if (not (cstatus.CheckMaybeRaise (?incomingStatus=status, last = false))) then TFShape.Unknown;
        elif n = -1 then TFShape.Unknown;
        else
            let dims = Array.zeroCreate<int64> n
            TF_GraphGetTensorShape (handle, output.Struct, dims, dims.Length, cstatus.Handle);
            cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise (?incomingStatus=status)

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
        //this.Import (graphDef, options, status)
        failwith "todo"

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
        let cstatus = TFStatus.Setup (?incoming=status);
        TF_GraphImportGraphDef (handle, graphDef.LLBuffer, options.Handle, cstatus.Handle);
        cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore

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
        //this.Import (buffer, options, status)
        failwith "todo"

    /// <summary>
    /// Import a serialized graph held in a byte array into this graph, using the specified import options.
    /// </summary>
    /// <returns>The import.</returns>
    /// <param name="buffer">A byte array containing the serialized graph.</param>
    /// <param name="options">Importing graph options.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    ///   If you are tryig to load a file stored using the SavedModel file format, you should use the <see cref="T:TensorFlow.TFSession.FromSavedModel"/> API instead.
    /// </remarks>
    member this.Import (buffer : byte [], options : TFImportGraphDefOptions, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box buffer = null then raise(ArgumentNullException ("buffer"))
        if box options = null then raise(ArgumentNullException ("options"))
        let cstatus = TFStatus.Setup (?incoming=status)
        use  tb = new TFBuffer (buffer, 0, buffer.Length)
        this.Import (tb, options, ?status=status)
        cstatus.CheckMaybeRaise (?incomingStatus=status)

    /// <summary>
    /// Gets the <see cref="T:TensorFlow.TFGraph"/> with the specified name, or None if the named operation does not exist in the graph.
    /// </summary>
    /// <param name="name">Name to lookup.</param>
    member this.TryGet (name : string) = 
            if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
            let h = TF_GraphOperationByName (handle, name);
            if h = IntPtr.Zero then None
            else Some(new Operation (h))

    /// <summary>
    /// Gets the <see cref="T:TensorFlow.TFGraph"/> with the specified name, or null if the named operation does not exist in the graph.
    /// </summary>
    /// <param name="name">Name to lookup.</param>
    member this.Item 
        with get(name : string) : Operation = this.TryGet(name) |> Option.orNull
    

    /// <summary>
    /// Returns the enumerator that returns all the TFOperations in a graph.
    /// </summary>
    /// <returns>The enumerator.</returns>
    member this.GetEnumerator () : IEnumerable<Operation> =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        let mutable token = IntPtr.Zero;
        failwith "todo - token pointer needs to be address"
        Seq.unfold (fun _ -> 
            match TF_GraphNextOperation (handle, &token) with
            | operll when operll = IntPtr.Zero -> None
            | operll -> Some(Operation(operll),())) ()

    /// <summary>
    ///  Returns the tensor shape for the specific output pparameters as an array of longs.
    /// </summary>
    /// <returns>null for single dimension, .</returns>
    /// <param name="output">The output operation to probe.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.GetShape (output : Output, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status);
        let ndims = TF_GraphGetTensorNumDims (this.Handle, output.Struct, cstatus.Handle);
        if not(cstatus.CheckMaybeRaise (?incomingStatus = status, last = false)) || 
            ndims = 0
        then box null :?> Int64 []
        else 
            let ret = Array.zeroCreate<int64> ndims 
            TF_GraphGetTensorShape (handle, output.Struct, ret, ndims, cstatus.Handle);
            cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
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
    /// Assert (graph.CurrentNamescope, "");
    /// using (var nested = graph.WithScope ("nested")){
    ///    Assert (graph.CurrentNameScope, "nested");
    ///    using (var inner = graph.WithScope ("inner")){
    ///        Assert (graph.CurrentNameScope, "nested/inner");
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
    member this.WithDependencies ([<ParamArray>] dependencies : Operation []) = 
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
            // TODO make sure we don't need to make this unique as well
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
    ///   If you are tryig to load a file stored using the SavedModel file format, you should use the <see cref="T:TensorFlow.TFSession.FromSavedModel"/> API instead.
    /// </remarks>
    member this.ImportGraphDef (graphDef : TFBuffer, options :TFImportGraphDefOptions, returnOutputs : Output [], ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box graphDef = null then raise(ArgumentNullException ("graphDef"))
        if box options = null then raise(ArgumentNullException ("options"))
        let cstatus = TFStatus.Setup (?incoming=status);
        if returnOutputs = null then
            TF_GraphImportGraphDefWithReturnOutputs (handle, graphDef.LLBuffer, options.Handle, NativePtr.ofNativeInt IntPtr.Zero, 0, cstatus.Handle);
        else 
            let returnOutputs = returnOutputs |> Array.map (fun x -> x.Struct)
            use first = fixed &returnOutputs.[0]
            TF_GraphImportGraphDefWithReturnOutputs (handle, graphDef.LLBuffer, options.Handle, first, returnOutputs.Length, cstatus.Handle);


    static member CopyFrom (ptr : nativeptr<TF_Output>, n : int) : TF_Output []   =
        let r = Array.zeroCreate<TF_Output> n;
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
    member this.While(inputs : Output [], constructor : WhileConstructor, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box inputs = null then raise(ArgumentNullException ("inputs"))
        if constructor = null then raise(ArgumentNullException ("constructor"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let mutable result = TF_NewWhile (handle, inputs |> Array.map (fun x -> x.Struct), inputs.Length, cstatus.Handle);
        if cstatus.Error then box null :?> Output[]
        else
            try
                // 
                // Call constructor here
                // Wrap the various TF_graphs (with owns=false)
                // Marshal the condInputs, bodyInputs
                //
                let mutable name : string = box null :?> string
                let n = result.ninputs;
                let bodyOutputs = Array.zeroCreate<TF_Output> n 
                let condGraph = new TFGraphUnowned (result.cond_graph) :> TFGraph
                let bodyGraph = new TFGraphUnowned (result.body_graph) :> TFGraph
                constructor.Invoke(condGraph, TFGraph.CopyFrom (result.cond_inputs, n), result.cond_output, bodyGraph, TFGraph.CopyFrom (result.body_inputs, n), bodyOutputs, name);

                let name = if box name = null || name = "" then this.MakeUnique ("while") else name
                // On return, copy the condOutput and bodyOututs
                let text = Encoding.UTF8.GetBytes (name);
                result.charPtrName <- Marshal.AllocHGlobal (text.Length + 1);
                Marshal.Copy (text, 0, result.charPtrName, text.Length);
                Marshal.WriteByte (result.charPtrName, text.Length, 0uy);
                for i = 0 to n - 1 do
                    NativePtr.set result.body_outputs i bodyOutputs.[i]

                let ret = Array.zeroCreate<TF_Output> n 
                use first = fixed &ret.[0]
                TF_FinishWhile (&result, cstatus.Handle, first);
                if (cstatus.CheckMaybeRaise (?incomingStatus=status)) then 
                    ret |> Array.map Output
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
    member this.AddGradients(y : Output [], x : Output [], ?dx : Output [], ?status : TFStatus) : Output [] =
        if y = null then raise(ArgumentNullException ("y"))
        if x = null then raise(ArgumentNullException ("x"))
        dx |> Option.iter (fun dx -> 
            if dx.Length <> y.Length then 
                raise(ArgumentException ("If dx is not null, the size of the gradients must match the size of y", "dx")))
        let cstatus = TFStatus.Setup (?incoming=status)
        let ret = Array.zeroCreate<TF_Output> x.Length //new Output [x.Length];
        use pret = fixed &ret.[0]
        let y = y |> Array.map (fun x -> x.Struct)
        let x = x |> Array.map (fun x -> x.Struct)
        use py = fixed &y.[0]
        use px = fixed &x.[0] 
        match dx with
        | None ->
            TF_AddGradients (handle, py, y.Length, px, x.Length, NativePtr.ofNativeInt IntPtr.Zero, cstatus.Handle, pret);
        | Some(dx) ->
            let dx = dx |> Array.map (fun x -> x.Struct)
            use pdx = fixed &dx.[0]
            TF_AddGradients (handle, py, y.Length, px, x.Length, pdx, cstatus.Handle, pret);
        if not(cstatus.CheckMaybeRaise (?incomingStatus=status, last = false)) then
             null;
        else
            ret |> Array.map Output

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
    member this.AddGradients (prefix : string,  y : Output [], x : Output [], ?dx : Output [], ?status : TFStatus) : Output [] =
        if y = null then raise(ArgumentNullException ("y"))
        if x = null then raise(ArgumentNullException ("x"))
        dx |> Option.iter (fun dx -> 
            if dx.Length <> y.Length then 
                raise(ArgumentException ("If dx is not null, the size of the gradients must match the size of y", "dx")))
        let cstatus = TFStatus.Setup (?incoming=status);
        let ret = Array.zeroCreate<TF_Output> x.Length
        use pret = fixed &ret.[0]
        let y = y |> Array.map (fun x -> x.Struct)
        let x = x |> Array.map (fun x -> x.Struct)
        use py = fixed &y.[0]
        use px = fixed &x.[0]
        match dx with
        | None -> 
            TF_AddGradientsWithPrefix (handle, prefix, py, y.Length, px, x.Length, NativePtr.ofNativeInt IntPtr.Zero, cstatus.Handle, pret);
        | Some(dx) -> 
            let dx = dx |> Array.map (fun x -> x.Struct)
            use pdx = fixed &dx.[0] 
            TF_AddGradientsWithPrefix (handle, prefix, py, y.Length, px, x.Length, pdx, cstatus.Handle, pret);
        if not(cstatus.CheckMaybeRaise (?incomingStatus=status, last = false)) then
             null;
        else
            ret |> Array.map Output

    /// <summary>
    /// Creates a TFFunction from a TFGraph
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
                            operations : Operation [],
                            inputs : Output [],
                            outputs : Output [],
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
        let functionOptions = IntPtr.Zero;
        let ops = IntPtr.Zero;
        let nops, ops =
            if box operations = null then 0,IntPtr.Zero
            else
                let nops = operations.Length;
                let ops = Marshal.AllocHGlobal (sizeof<IntPtr> * operations.Length);
                for i = 0 to nops - 1 do
                    Marshal.WriteIntPtr (ops, i * sizeof<IntPtr>, operations.[i].Handle);
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
        if not(cstatus.CheckMaybeRaise (?incomingStatus=status, last = false)) then box null :?> TFFunction
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
        let cstatus = TFStatus.Setup (?incoming=status);
        TF_GraphVersions (handle, outputVersionDef.LLBuffer, cstatus.Handle);
        cstatus.CheckMaybeRaise (?incomingStatus=status);

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
            let n = this.NumFunctions;
            let size = sizeof<IntPtr>
            let buffer = Marshal.AllocHGlobal (n * size);
            let ret =
                use status = new TFStatus() 
                let num = TF_GraphGetFunctions (handle, buffer, n, status.Handle);
                if num > 0 && status.Ok  then
                    let ret = Array.zeroCreate<TFFunction> num
                    let mutable ofs = 0;
                    for i = 0 to num - 1 do
                        let tfhandle = Marshal.ReadIntPtr (buffer, ofs);
                        ret.[i] <- new TFFunction (tfhandle);
                        ofs <- ofs + size
                    ret
                else box null :?> TFFunction[]
            Marshal.FreeHGlobal (buffer);
            ret;


    /// <summary>
    /// Attempts to evaluate the <paramref name="output"/>.   This is only possible if <paramref name="output"/> does not
    /// depend on any graph inputs - the function is safe to call if this is not the case though.
    /// </summary>
    /// <returns><c>true</c>, if the evaluation is successful, in which case the result is returned in <paramref name="tensor"/>, <c>false</c> otherwise.</returns>
    /// <param name="output">Output.</param>
    /// <param name="tensor">Tensor.</param>
    member this.TryEvaluateConstant (output : Output, [<Out>] tensor : TFTensor byref) = // TODO ref?
        let cstatus = new TFStatus ()
        let mutable ptr = IntPtr.Zero;
        let ret = TF_TryEvaluateConstant (handle, output.Struct, &ptr, cstatus.Handle); // ref ptr
        cstatus.Dispose ();
        if ret then
            tensor <- new TFTensor (ptr);
        else
            tensor <- box null :?> TFTensor
        ret
    
    override this.ToString () =
            let mutable len = IntPtr.Zero
            TF_GraphDebugString (this.Handle, len); // ref len

