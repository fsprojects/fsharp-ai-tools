namespace TensorFlow.FSharp

open System
open System.Runtime.InteropServices
open TensorFlow.FSharp.Utils

[<StructLayout (LayoutKind.Sequential)>]
[<Struct>]
type TF_Input = {
    operation : TF_Operation
    index : int
}

[<StructLayout (LayoutKind.Sequential)>]
[<Struct>]
type TF_Output = {
    handle : TF_Operation 
    index  : int
}

#nowarn "9"

/// <summary>
/// Represents a specific input of an operation.
/// </summary>
type TFInput(handle : TF_Operation, index : int) =

    //extern TF_Output TF_OperationInput (TF_Input oper_in)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Output TF_OperationInput (TF_Input oper_in)

    //extern TF_DataType TF_OperationInputType (TF_Input oper_in)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TFDataType TF_OperationInputType (TF_Input oper_in)

    /// <summary>
    /// Initializes a new Output instance from a TF_Output struct
    /// </summary>
    /// <param name="value">The TF_Output struct</param>
    new (value:TF_Input) = new TFInput(value.operation, value.index)

    /// <summary>
    /// The operation that this input is for
    /// </summary>
    member __.Operation = handle

    /// <summary>
    /// The index of the output within the Operation
    /// </summary>
    member __.Index = index

    member __.GetOutput (operIn : TF_Input) : TFOutput = 
        let tfOut = TF_OperationInput operIn
        new TFOutput(tfOut.handle,tfOut.index)

    member __.TFDataType : TFDataType = TF_OperationInputType ({operation = handle; index = index})

    member internal __.Struct with get () : TF_Input = { operation = handle; index = index }

/// <summary>
/// Represents a specific output of an operation on a tensor.
/// </summary>
/// <remarks>
/// <para>
/// Output objects represent one of the outputs of an operation in the graph
/// (Graph).  Outputs have a data type, and eventually a shape that you can 
/// retrieve by calling the <see cref="M:TensorFlow.Graph.GetShape"/> method.
/// </para>
/// <para>
/// These can be passed as an input argument to a function for adding operations 
/// to a graph, or to the Session's Run and GetRunner method as values to be
/// fetched.
/// </para>
/// </remarks>
and TFOutput(handle: IntPtr, ?index : int) =

    // extern int TF_OperationOutputNumConsumers (TF_Output oper_out)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationOutputNumConsumers (TFOutput oper_out)

    // extern TF_DataType TF_OperationOutputType (TF_Output oper_out)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TFDataType TF_OperationOutputType (TF_Output oper_out)

    // extern int TF_OperationOutputConsumers (TF_Output oper_out, TF_Input *consumers, int max_consumers)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationOutputConsumers (TF_Output oper_out, TF_Input* consumers, int max_consumers)

    member internal __.Struct with get () = { handle = handle; index = index |> Option.defaultValue 0}

    //do if handle = null then raise(ArgumentNullException ("Outputs does not have a valid operation pointer"))
    member __.LLOperation = handle

    /// <summary>
    /// The index of the output within the operation.
    /// </summary>
    member __.Index = index.Value

    /// <summary>
    /// Gets the number consumers.
    /// </summary>
    /// <value>The number consumers.</value>
    /// <remarks>
    /// This number can change when new operations are added to the graph.
    /// </remarks>
    member this.NumConsumers = TF_OperationOutputNumConsumers (this)

    /// <summary>
    /// Gets the type of the output.
    /// </summary>
    /// <value>The type of the output.</value>
    member this.TFDataType = if this.LLOperation = IntPtr.Zero then TFDataType.Unknown else TF_OperationOutputType (this.Struct)
    //public TFDataType OutputType => LLOperation == IntPtr.Zero ? TFDataType.Unknown : TF_OperationOutputType (this)

    /// <summary>
    /// Initializes a new Output instance.
    /// </summary>
    /// <param name="operation">The operation to which to attach the output.</param>
    /// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
    new (operation : TFOperation, ?index : int) =
        if box operation = null then raise(ArgumentNullException ("operation"))
        TFOutput(operation.Handle, ?index=index)

    /// <summary>
    /// Initializes a new Output instance from another Output
    /// </summary>
    /// <param name="output">The other Output that is having its operation attached.</param>
    /// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
    new (output : TFOutput, ?index : int) = 
        if box output.LLOperation = null then raise(ArgumentNullException ("Outputs does not have a valid operation pointer"))
        TFOutput(output.LLOperation, ?index=index)

    /// <summary>
    /// Initializes a new Output instance from a TF_Output struct
    /// </summary>
    /// <param name="value">The TF_Output struct</param>
    new (value:TF_Output) = new TFOutput(value.handle, value.index)

    /// <summary>
    /// Get list of all current consumers of a specific output of an operation
    /// </summary>	
    /// <value>The output consumers.</value>
    /// <remarks>
    /// A concurrent modification of the graph can increase the number of consumers of
    /// an operation.
    /// This can return null if the Output does not point to a valid object.
    /// </remarks>
    member this.OutputConsumers 
        with get() : TFInput[] = 
            let result = Array.zeroCreate<TF_Input> this.NumConsumers
            use first = fixed &result.[0]
            TF_OperationOutputConsumers (this.Struct, first, result.Length) |> ignore
            result |> Array.map (fun x -> TFInput(x.operation,x.index))

    /// <summary>
    /// The associated operation.
    /// </summary>
    /// <value>The operation.</value>
    member this.Operation = new TFOperation (this.LLOperation)

    /// <summary>
    /// The associated operation.
    /// </summary>
    /// <value>The operation.</value>
    member this.Op = this.Operation

    member this.Name = sprintf "%s:%i" this.Operation.Name this.Index

    /// <summary>
    /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.Output"/>.
    /// </summary>
    /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.Output"/>.</returns>
    override this.ToString () =
        sprintf "[%O Index=%i Operation=%O (0x%i)]"  this.TFDataType this.Index this.Operation this.LLOperation 

    interface IComparable with 
        member this.CompareTo(x : obj) = 
            if (x.GetType() <> this.GetType()) then -1
            else (this :> IComparable<TFOutput>).CompareTo(x :?> TFOutput)

    interface IComparable<TFOutput> with 
        member this.CompareTo(other : TFOutput) =
            let left = this.Operation.Handle.ToInt64()
            let right = other.Operation.Handle.ToInt64()
            if left <> right then
                left.CompareTo(right)
            else this.Index.CompareTo(other.Index)

    override this.Equals(x:obj) = 
        match x with
        | :? TFOutput as other -> 
            (this.LLOperation = other.LLOperation) && (this.Index = other.Index)
        | _ -> false
    
    override this.GetHashCode() = (this.Operation.Handle , this.Index).GetHashCode()

[<AutoOpen>]
module TFOperationExtensions =

    type TFOperation with

        /// <summary>
        /// Returns the handle to the idx-th output of the operation.
        /// </summary>
        /// <param name="idx">Index of the output in the operation.</param>
        member this.Item with get (idx:int) : TFOutput = new TFOutput (this, idx)

        member this.Outputs = [|for i in 0..this.NumOutputs - 1 -> TFOutput(this.Handle,i)|]

        member this.Inputs  = [|for i in 0..this.NumInputs - 1 -> TFInput(this.Handle,i)|]

        member this.TryGetOutput(i:int) = if i >= 0 && i < this.NumOutputs then Some(this.[i]) else None