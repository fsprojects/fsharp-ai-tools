namespace Tensorflow 
open Microsoft.FSharp.NativeInterop
open System.Collections.Generic
open System
open System.Runtime.InteropServices

/// <summary>
/// Use the runner class to easily configure inputs, outputs and targets to be passed to the session runner.
/// </summary>
/// <remarks>
/// <para>
/// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
/// to construct the parameters that will be passed to the Session.Run method.
/// </para>
/// <para>
/// Instances of this class are created by calling the GetRunner method on the Session.
/// </para>
/// <para>
/// The various methods in this class return an instance to the Runner itsel, to allow
/// to easily construct chains of execution like this:
/// </para>
/// <code>
/// var result = session.GetRunner ().AddINput (myInput).Fetch (MyOutput).Run ()
/// </code>
/// <para>
/// You do not need to chain the operations, this works just the same:
/// </para>
/// <code>
/// runner = session.GetRunner ()
/// runner.AddInput(myInput)
/// runner.Fetch(myOutput)
/// var results = runner.Run()
/// </code>
/// </remarks>
/// 
type Runner internal (session : Session) =

    let inputs = new List<Output> () 
    let outputs = new List<Output> ()
    let inputValues = new List<Tensor> ()
    let targets = new List<Operation>()

    /// <summary>
    /// Adds an input to the session
    /// </summary>
    /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
    /// <param name="input">Incoming port.</param>
    /// <param name="value">Value to assing to the incoming port.</param>
    member this.AddInput (input : Output, value : Tensor) : Runner =
        if box value = null then  raise(ArgumentNullException("value"))
        inputs.Add (input)
        inputValues.Add (value)
        this

    /// <summary>
    /// Adds an input to the session specified by name, with an optional index in the operation (separated by a colon).
    /// </summary>
    /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
    /// <param name="input">Incoming port, with an optional index separated by a colon.</param>
    /// <param name="value">Value to assing to the incoming port.</param>
    member this.AddInput(input : string,  value : Tensor) : Runner  = 
        if box value = null then  raise(ArgumentNullException("value"))
        inputValues.Add (value)
        this

    /// <summary>
    /// Adds the specified operations as the ones to be retrieved.
    /// </summary>
    /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
    /// <param name="targets">One or more targets.</param>
    member this.AddTarget ([<ParamArray>] newTargets : Operation[]) : Runner =
        targets.AddRange(newTargets)
        this


    /// Parses user strings that contain both the operation name and an index.
    member this.ParseOutput (operation : string) = 
        // TODO, wrap this in an option
        match operation.Split(':') with
        | [|op;Integer(idx)|] -> session.Graph.[op].[idx]
        | [|op|] -> session.Graph.[operation].[0]
        | _ -> failwithf "error parsing %s" operation

    /// <summary>
    /// Adds the specified operation names as the ones to be retrieved.
    /// </summary>
    /// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
    /// <param name="targetNames">One or more target names.</param>
    member this.AddTarget ([<ParamArray>] targetNames : string [])  :  Runner =
        targets.AddRange(targetNames |> Array.map (fun name -> session.Graph.[name]))
        this

    /// <summary>
    /// Makes the Run method return the index-th output of the tensor referenced by operation.
    /// </summary>
    /// <returns>The instance of runner, to allow chaining operations.</returns>
    /// <param name="operation">The name of the operation in the graph.</param>
    /// <param name="index">The index of the output in the operation.</param>
    member this.Fetch (operation : string, index : int) : Runner =
        outputs.Add (session.Graph.[operation].[index])
        this

    /// <summary>
    /// Makes the Run method return the output of the tensor referenced by operation, the operation string can contain the output index.
    /// </summary>
    /// <returns>The instance of runner, to allow chaining operations.</returns>
    /// <param name="operation">The name of the operation in the graph, which might be a simple name, or it might be name:index, 
    /// where the index is the .</param>
    member this.Fetch(operation : string) : Runner = 
        outputs.Add (this.ParseOutput(operation))
        this

    /// <summary>
    /// Makes the Run method return the output of the tensor referenced by output
    /// </summary>
    /// <returns>The instance of runner, to allow chaining operations.</returns>
    /// <param name="output">The output referencing a specified tensor.</param>
    member this.Fetch (output : Output) : Runner =
        outputs.Add (output)
        this

    /// <summary>
    /// Makes the Run method return the output of all the tensor referenced by outputs.
    /// </summary>
    /// <returns>The instance of runner, to allow chaining operations.</returns>
    /// <param name="outputs">The outputs referencing a specified tensor.</param>
    member this.Fetch ([<ParamArray>] outputsToFetch : Output []) : Runner =
        outputs.AddRange(outputsToFetch)
        this

    /// <summary>
    /// Makes the Run method return the output of all the tensor referenced by outputs.
    /// </summary>
    /// <returns>The instance of runner, to allow chaining operations.</returns>
    /// <param name="outputs">The output sreferencing a specified tensor.</param>
    member this.Fetch ([<ParamArray>] outputsToFetch : string []) : Runner =
        outputs.AddRange(outputsToFetch |> Array.map this.ParseOutput)
        this

    /// <summary>
    /// Protocol buffer encoded block containing the metadata passed to the <see cref="M:TensorFlow.Session.Run"/> method.
    /// </summary>
    member val RunMetadata : TFBuffer option = None with get, set

    /// <summary>
    /// Protocol buffer encoded block containing the run options passed to the <see cref="M:TensorFlow.Session.Run"/> method.
    /// </summary>
    member val RunOptions : TFBuffer option = None with get, set

    /// <summary>
    ///  Execute the graph fragments necessary to compute all requested fetches.
    /// </summary>
    /// <returns>One Tensor for each call to Fetch that you made, in the order that you made them.</returns>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.Run(?status : TFStatus) : Tensor [] =
        session.Run (inputs.ToArray (), inputValues.ToArray (), outputs.ToArray (), targets.ToArray (), ?runMetadata=this.RunMetadata, ?runOptions=this.RunOptions, ?status=status)

    /// <summary>
    /// Run the specified operation, by adding it implicity to the output, single return value
    /// </summary>
    /// <param name="operation">The output of the operation.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// This method is a convenience method, and when you call it, it will clear any 
    /// calls that you might have done to Fetch() and use the specified operation to Fetch
    /// instead.
    /// </remarks>
    member this.Run(operation : Output, ?status : TFStatus) : Tensor =
        outputs.Clear ()
        this.Fetch (operation) |> ignore
        this.Run(?status=status).[0]


/// <summary>
/// Token returned from using one of the Partial Run Setup methods from <see cref="T:TensorFlow.Session"/>,
/// and use this token subsequently for other invocations.
/// </summary>
/// <remarks>
/// Calling Dispose on this object will release the resources associated with setting up 
/// a partial run.
/// </remarks>
and PartialRunToken(token:IntPtr) =

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeletePRunHandle (IntPtr partialRunHandle)

    member __.Dispose() =
        if token <> IntPtr.Zero then
            TF_DeletePRunHandle (token)

    member __.Token = token

/// <summary>
/// Drives the execution of a graph
/// </summary>
/// <remarks>
/// <para>
/// This creates a new context to execute a Graph.   You can use the 
/// constructor to create an empty session, or you can load an existing
/// model using the <see cref="FromSavedModel"/> static method in this class.
/// </para>
/// <para>
/// To execute operations with the graph, call the <see cref="GetRunner"/>  method
/// which returns an object that you can use to build the operation by providing
/// the inputs, requesting the operations that you want to execute and the desired outputs.
/// </para>
/// <para>
/// The <see cref="GetRunner"/> method is a high-level helper function that wraps a
/// call to the <see cref="Run"/> method which just takes too many parameters that must
/// be kept in sync.
/// </para>
/// </remarks>
and Session private (handle:IntPtr, graph : Graph,  ?status : TFStatus) =
    inherit TFDisposableThreadSafe(handle:IntPtr)
    // extern TF_Session * TF_NewSession (TF_Graph *graph, const TF_SessionOptions *opts, TF_Status *status)
    [<DllImport(NativeBinding.TensorFlowLibrary)>]
    static extern TF_Session TF_NewSession (TF_Graph graph, TF_SessionOptions opts, TF_Status status)

    // extern TF_Session * TF_LoadSessionFromSavedModel (const TF_SessionOptions *session_options, const TF_Buffer *run_options, const char *export_dir, const char *const *tags, int tags_len, TF_Graph *graph, TF_Buffer *meta_graph_def, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Session TF_LoadSessionFromSavedModel (TF_SessionOptions session_options, LLBuffer* run_options, string export_dir, string [] tags, int tags_len, TF_Graph graph, LLBuffer* meta_graph_def, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_DeviceList TF_SessionListDevices (TF_Session session, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_DeviceListCount (TF_DeviceList list)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_DeviceListName (TF_DeviceList list, int index, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_DeviceListType (TF_DeviceList list, int index, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int64 TF_DeviceListMemoryBytes (TF_DeviceList list, int index, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteDeviceList (TF_DeviceList list)

    // extern void TF_CloseSession (TF_Session *, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_CloseSession (TF_Session session, TF_Status status)

    // extern void TF_SessionPRunSetup (TF_Session, const TF_Output *inputs, int ninputs, const TF_Output *outputs, int noutputs, const TF_Operation *const *target_opers, int ntargets, const char **handle, TF_Status *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SessionPRunSetup (TF_Session session, TF_Output [] inputs, int ninputs, TF_Output [] outputs, int noutputs, TF_Operation [] target_opers, int ntargets, [<Out>] IntPtr returnHandle, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeletePRunHandle (IntPtr partialRunHandle)

    // extern void TF_SessionPRun (TF_Session *, const char *handle, const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs, const TF_Output *outputs, TF_Tensor **output_values, int noutputs, const TF_Operation *const *target_opers, int ntargets, TF_Status *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SessionPRun (TF_Session session, IntPtr partialHandle, TF_Output [] inputs, TF_Tensor [] input_values, int ninputs, TF_Output [] outputs, TF_Tensor [] output_values, int noutputs, TF_Operation [] target_opers, int ntargets, TF_Status status)

    // extern void TF_DeleteSession (TF_Session *, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteSession (TF_Session session, TF_Status status)

    // extern void TF_SessionRun (TF_Session *session, const TF_Buffer *run_options, const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs, const TF_Output *outputs, TF_Tensor **output_values, int noutputs, const TF_Operation *const *target_opers, int ntargets, TF_Buffer *run_metadata, TF_Status *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SessionRun (TF_Session session, LLBuffer* run_options, TF_Output [] inputs, TF_Tensor [] input_values, int ninputs, TF_Output [] outputs, TF_Tensor [] output_values, int noutputs, TF_Operation [] target_opers, int ntargets, LLBuffer* run_metadata, TF_Status status)


    /// <summary>
    /// Creates a new execution session associated with the specified session graph with some configuration options.
    /// </summary>
    /// <param name="graph">The Graph to which this session is associated.</param>
    /// <param name="sessionOptions">Session options.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    new (?graph : Graph, ?sessionOptions : SessionOptions , ?status : TFStatus ) =
        let graph = graph |> Option.orDefaultDelay (fun () -> new Graph())
        let cstatus = TFStatus.Setup (?incoming=status)
        let h = 
            match sessionOptions with 
            | Some(sessionOptions) -> TF_NewSession (graph.Handle, sessionOptions.Handle, cstatus.Handle)
            | None ->
                let empty = new SessionOptions()
                let res = TF_NewSession (graph.Handle, empty.Handle, cstatus.Handle)
                empty.Dispose()
                res
        new Session(h,graph,?status=status)

    member this.Graph : Graph = graph

    /// <summary>
    /// Lists available devices in this session.
    /// </summary>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.ListDevices(?status : TFStatus) : DeviceAttributes[] =
        let cstatus = TFStatus.Setup (?incoming=status)
        let rawDeviceList = TF_SessionListDevices (this.Handle, cstatus.Handle)
        let size = TF_DeviceListCount (rawDeviceList)
        let list = Array.init<DeviceAttributes> size (fun i ->
            let name = Marshal.PtrToStringAnsi (TF_DeviceListName (rawDeviceList, i, cstatus.Handle))
            let deviceType  = Enum.Parse (typeof<DeviceType>, Marshal.PtrToStringAnsi (TF_DeviceListType (rawDeviceList, i, cstatus.Handle))) :?> DeviceType
            let memory = TF_DeviceListMemoryBytes (rawDeviceList, i, cstatus.Handle)
            DeviceAttributes(name,deviceType,memory)
        )
        TF_DeleteDeviceList (rawDeviceList)
        list

    /// <summary>
    /// Creates a session and graph from a model stored in the SavedModel file format.
    /// </summary>
    /// <returns>On success, this populates the provided <paramref name="graph"/> with the contents of the graph stored in the specified model and <paramref name="metaGraphDef"/> with the MetaGraphDef of the loaded model.</returns>
    /// <param name="sessionOptions">Session options to use for the new session.</param>
    /// <param name="runOptions">Options to use to initialize the state (can be null).</param>
    /// <param name="exportDir">must be set to the path of the exported SavedModel.</param>
    /// <param name="tags">must include the set of tags used to identify one MetaGraphDef in the SavedModel.</param>
    /// <param name="graph">This must be a newly created graph.</param>
    /// <param name="metaGraphDef">On success, this will be populated on return with the contents of the MetaGraphDef (can be null).</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// <para>
    /// This function creates a new session using the specified <paramref name="sessionOptions"/> and then initializes
    /// the state (restoring tensors and other assets) using <paramref name="runOptions"/>.
    /// </para>
    /// <para>
    /// This function loads the data that was saved using the SavedModel file format, as described
    /// here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md
    /// </para>
    /// </remarks>
    member this.FromSavedModel (sessionOptions : SessionOptions, exportDir : string,tags :  string [], graph : Graph,?runOptions : TFBuffer,  ?metaGraphDef : TFBuffer, ?status : TFStatus) : Session option =
        if (box graph = null) then raise (ArgumentNullException("graph"))
        if (box tags = null) then raise (ArgumentNullException("tags"))
        if (box exportDir = null) then raise (ArgumentNullException ("exportDir"))
        if (box metaGraphDef = null) then raise (ArgumentNullException ("metaGraphDef"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let h = TF_LoadSessionFromSavedModel (sessionOptions.Handle, runOptions |> Option.mapOrNull (fun x -> x.LLBuffer), 
                                                exportDir, tags, tags.Length, graph.Handle, 
                                                metaGraphDef |> Option.mapOrNull (fun x -> x.LLBuffer) , cstatus.Handle)

        if cstatus.CheckMaybeRaise (?incoming=status) 
        then Some(new Session (h, graph))
        else None


    /// <summary>
    /// Closes the session.  Contacts any other processes associated with the session, if applicable.
    /// </summary>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// Can not be called after calling DeleteSession.
    /// </remarks>
    member this.CloseSession (?status : TFStatus) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_CloseSession (handle, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status)

    /// <summary>
    /// Deletes the session.
    /// </summary>
    /// <param name="status">Status.</param>
    member this.DeleteSession (?status : TFStatus) = 
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_DeleteSession (handle, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status)

    override this.NativeDispose (handle : IntPtr) =
        use s = new TFStatus()
        TF_DeleteSession (handle, s.Handle)


    /// <summary>
    /// Gets a new runner, this provides a simpler API to prepare the inputs to run on a session
    /// </summary>
    /// <returns>The runner.</returns>
    /// <remarks>
    /// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
    /// to construct the parameters that will be passed to the Session.Run method.
    /// 
    /// The Run method will return an array of Tensor values, one for each invocation to the Fetch method.
    /// </remarks>
    member this.GetRunner () : Runner = new Runner (this)

    /// <summary>
    /// Executes a pipeline given the specified inputs, inputValues, outputs, targetOpers, runMetadata and runOptions.   
    /// A simpler API is available by calling the <see cref="M:GetRunner"/> method which performs all the bookkeeping
    /// necessary.
    /// </summary>
    /// <returns>An array of tensors fetched from the requested outputs.</returns>
    /// <param name="inputs">Inputs nodes.</param>
    /// <param name="inputValues">Input values.</param>
    /// <param name="outputs">Output nodes.</param>
    /// <param name="targetOpers">Target operations to execute.</param>
    /// <param name="runMetadata">Run metadata, a buffer containing the protocol buffer encoded value for https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/core/protobuf/config.proto.</param>
    /// <param name="runOptions">Run options, a buffer containing the protocol buffer encoded value for https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/core/protobuf/config.proto.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member __.Run(inputs : Output [], inputValues : Tensor [], outputs : Output [] , ?targetOpers : Operation [], ?runMetadata : TFBuffer, ?runOptions : TFBuffer, ?status : TFStatus) : Tensor [] =
        if handle = IntPtr.Zero then raise (ObjectDisposedException("handle"))
        if box inputs = null then raise (ArgumentNullException("inputs"))
        if box inputValues = null then raise (ArgumentNullException "inputValues")
        if box outputs = null then raise (ArgumentNullException ("outputs"))
        let iLen = inputs.Length
        if iLen <> inputValues.Length then raise (ArgumentException ("inputs and inputValues have different lengths", "inputs"))
        let oLen = outputs.Length
        // runOptions and runMetadata might be null
        let cstatus = TFStatus.Setup (?incoming=status)

        // Create arrays for the unmanaged versions
        let ivals = inputValues |> Array.map (fun x -> x.Handle)

        // I believe this might not be necessary, the output values in TF_SessionRun looks like a write-only result
        let ovals = Array.zeroCreate<IntPtr> outputs.Length
        
        let topers, tLen =
            match targetOpers with
            | Some(targetOpers) -> targetOpers |> Array.map (fun x -> x.Handle), targetOpers.Length
            | None -> null, 0

        TF_SessionRun (handle, 
                        runOptions |> Option.mapOrNull (fun x -> x.LLBuffer), inputs |> Array.map (fun x -> x.Struct), ivals, iLen, 
                        outputs |> Array.map (fun x -> x.Struct), ovals, oLen, topers, tLen, 
                        runMetadata |> Option.mapOrNull (fun x -> x.LLBuffer), cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore

        // prevent finalization of managed Tensors
        GC.KeepAlive(inputValues)

        ovals |> Array.map (fun x -> new Tensor(x))

    /// <summary>
    /// Prepares the session for a partial run.
    /// </summary>
    /// <returns>A token that can be used to call <see cref="PartialRun"/> repeatedly.   To complete your partial run, you should call Dispose on the resulting method.</returns>
    /// <param name="inputs">Inputs.</param>
    /// <param name="outputs">Outputs.</param>
    /// <param name="targetOpers">Target operations to run.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member __.PartialRunSetup (inputs : Output [], outputs : Output [], targetOpers : Operation [], ?status : TFStatus) : PartialRunToken =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box inputs = null then raise (ArgumentNullException ("inputs"))
        if box outputs = null then raise (ArgumentNullException ("outputs"))
        if box targetOpers = null then raise (ArgumentNullException ("targetOpers"))
        let mutable returnHandle = IntPtr.Zero
        let cstatus = TFStatus.Setup (?incoming=status)
        let tLen = targetOpers.Length
        let topers = targetOpers |> Array.map (fun x -> x.Handle)
        TF_SessionPRunSetup (handle, inputs |> Array.map (fun x -> x.Struct), inputs.Length, outputs |> Array.map (fun x -> x.Struct), outputs.Length, topers, tLen, returnHandle, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        PartialRunToken(returnHandle)

    member __.PartialRun (token : PartialRunToken, inputs : Output [], inputValues : Tensor [], outputs : Output [], targetOpers : Operation [], ?status : TFStatus) : Tensor [] =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box inputs = null then raise(ArgumentNullException("inputs"))
        if box inputValues = null then raise (ArgumentNullException ("inputValues"))
        if box outputs = null then raise (ArgumentNullException ("outputs"))
        if box targetOpers = null then raise (ArgumentNullException ("targetOpers"))
        let iLen = inputs.Length
        if iLen <> inputValues.Length then raise (ArgumentException ("inputs and inputValues have different lengths", "inputs"))
        let oLen = outputs.Length

        // runOptions and runMetadata might be null
        let cstatus = TFStatus.Setup (?incoming=status)

        // Create arrays for the unmanaged versions
        let ivals = inputValues |> Array.map (fun x -> x.Handle) 
        let ovals = Array.zeroCreate<IntPtr> oLen
        let tLen = targetOpers.Length
        let topers = targetOpers |> Array.map (fun x -> x.Handle)
        TF_SessionPRun (handle, token.Token, inputs |> Array.map (fun x -> x.Struct), ivals, iLen, outputs |> Array.map (fun x -> x.Struct), ovals, oLen, topers, tLen, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore

        // prevent finalization of managed Tensors
        GC.KeepAlive(inputValues)

        ovals |> Array.map (fun x -> new Tensor(x))
