namespace Tensorflow
// TODO as we've remove graph from Operations we should figure out how to make sure we prevent the graph from disposing when operations are live

open System
open System.Runtime.InteropServices
open System.Text
open Utils
open Microsoft.FSharp.NativeInterop

#nowarn "9"

/// <summary>
/// Represents a computation node in the graph.  Tensorflow operations are attached to a <see cref="T:Tensorflow.Graph"/>.
/// </summary>
/// <remarks>
/// Operations are usually created by  invoking one of the methods in 
/// <see cref="T:Tensorflow.Graph"/>, but they can also be constructed
/// manually using the low-level <see cref="T:Tensorflow.OperationDesc"/> API.
/// </remarks>
type Operation((*graph : Graph,*) handle : IntPtr)  =

    let attrTypeToString(attrType : TFAttributeType, isList : bool) = 
        let name = 
            match attrType with 
            | TFAttributeType.String -> "String"
            | TFAttributeType.Int -> "Int"
            | TFAttributeType.Float -> "Float"
            | TFAttributeType.Bool -> "Boolean"
            | TFAttributeType.Type -> "DataType"
            | TFAttributeType.Shape -> "Shape"
            | TFAttributeType.Tensor -> "Tensor"
            | TFAttributeType.Placeholder -> "Placeholder"
            | TFAttributeType.Func -> "Function"
            | _ -> "Unknown"
        if isList then sprintf "List[%s]" name else name

    let checkAttrType(attrName : string, metadata : TFAttributeMetadata, _type : TFAttributeType, isList : bool) = 
        if metadata.Type <> _type || metadata.IsList <> isList then
            sprintf"Attribute '%s' is not a %s. It is a '%s', instead." 
                attrName (attrTypeToString(_type,isList)) (attrTypeToString(metadata.Type,metadata.IsList))
            |> Exception |> raise

    let voidToNativeInt x = x |> NativePtr.ofVoidPtr<int64> |> NativePtr.toNativeInt

    // extern const char * TF_OperationName (TF_Operation *oper);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_OperationName (TF_Operation oper);

    // extern const char * TF_OperationOpType (TF_Operation *oper);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_OperationOpType (TF_Operation oper);

    // extern const char * TF_OperationDevice (TF_Operation *oper);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_OperationDevice (TF_Operation oper);


    // extern int TF_OperationNumOutputs (TF_Operation *oper);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationNumOutputs (TF_Operation oper);

    // extern int TF_OperationOutputListLength (TF_Operation *oper, const char *arg_name, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationOutputListLength (TF_Operation oper, string arg_name, TF_Status status);

    // extern int TF_OperationNumInputs (TF_Operation *oper);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationNumInputs (TF_Operation oper);

    // extern int TF_OperationInputListLength (TF_Operation *oper, const char *arg_name, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationInputListLength (TF_Operation oper, string arg_name, TF_Status status);
    // extern int TF_OperationNumControlInputs (TF_Operation *oper);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationNumControlInputs (TF_Operation oper);

    // extern int TF_OperationGetControlInputs (TF_Operation *oper, TF_Operation **control_inputs, int max_control_inputs);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationGetControlInputs (TF_Operation oper, TF_Operation control_inputs, int max_control_inputs);

    // extern int TF_OperationNumControlOutputs (TF_Operation *oper);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationNumControlOutputs (TF_Operation oper);

    //extern int TF_OperationGetControlOutputs (TF_Operation *oper, TF_Operation **control_outputs, int max_control_outputs);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_OperationGetControlOutputs (TF_Operation oper, [<Out>] [<MarshalAs (UnmanagedType.LPArray, SizeParamIndex = 2s)>] IntPtr [] control_outputs, int max_control_outputs);

    // extern TF_AttrMetadata TF_OperationGetAttrMetadata (TF_Operation *oper, const char *attr_name, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TFAttributeMetadata TF_OperationGetAttrMetadata (TF_Operation oper, string attr_name, TF_Status status);

    // extern void TF_OperationGetAttrString (TF_Operation *oper, const char *attr_name, void *value, size_t max_length, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrString (TF_Operation oper, string attr_name, void* value, int max_length, TF_Status status);

    // extern void TF_OperationGetAttrStringList (TF_Operation *oper, const char *attr_name, void **values, size_t *lengths, int max_values, void *storage, size_t storage_size, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrStringList (TF_Operation oper, string attr_name, IntPtr* values, UIntPtr* lengths, int max_values, IntPtr storage, int storage_size, TF_Status status);

    // extern void TF_OperationGetAttrInt (TF_Operation *oper, const char *attr_name, int64_t *value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrInt (TF_Operation oper, string attr_name, int64* value, TF_Status status);

    // extern void TF_OperationGetAttrIntList (TF_Operation *oper, const char *attr_name, int64_t *values, int max_values, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrIntList (TF_Operation oper, string attr_name, int64* values, int max_values, TF_Status status);

    // extern void TF_OperationGetAttrFloat (TF_Operation *oper, const char *attr_name, float32 *value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrFloat (TF_Operation oper, string attr_name, float32* value, TF_Status status);

    // extern void TF_OperationGetAttrFloatList (TF_Operation *oper, const char *attr_name, float32 *values, int max_values, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrFloatList (TF_Operation oper, string attr_name, float32* values, int max_values, TF_Status status);

    // extern void TF_OperationGetAttrBool (TF_Operation *oper, const char *attr_name, unsigned char *value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrBool (TF_Operation oper, string attr_name, byte* value, TF_Status status);

    // extern void TF_OperationGetAttrBoolList (TF_Operation *oper, const char *attr_name, unsigned char *values, int max_values, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrBoolList (TF_Operation oper, string attr_name, byte* values, int max_values, TF_Status status);

    // extern void TF_OperationGetAttrType (TF_Operation *oper, const char *attr_name, TF_DataType *value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrType (TF_Operation oper, string attr_name, DType* value, TF_Status status);
    
    // extern void TF_OperationGetAttrTypeList (TF_Operation *oper, const char *attr_name, TF_DataType *values, int max_values, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrTypeList (TF_Operation oper, string attr_name, DType* values, int max_values, TF_Status status);

    // extern void TF_OperationGetAttrShape (TF_Operation *oper, const char *attr_name, int64_t *value, int num_dims, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrShape (TF_Operation oper, string attr_name, int64* value, int num_dims, TF_Status status);

    // extern void TF_OperationGetAttrShapeList (TF_Operation *oper, const char *attr_name, int64_t **dims, int *num_dims, int num_shapes, int64_t *storage, int storage_size, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrShapeList (TF_Operation oper, string attr_name, IntPtr* dims, int* num_dims, int num_shapes, int64* storage, int storage_size, TF_Status status);

    // extern void TF_OperationGetAttrTensorShapeProto (TF_Operation *oper, const char *attr_name, TF_Buffer *value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrTensorShapeProto (TF_Operation oper, string attr_name, LLBuffer* value, TF_Status status);

    // extern void TF_OperationGetAttrTensorShapeProtoList (TF_Operation *oper, const char *attr_name, TF_Buffer **values, int max_values, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrTensorShapeProtoList (TF_Operation oper, string attr_name, IntPtr* values, int max_values, TF_Status status); // LLBuffer** -> IntPtr*

    // extern void TF_OperationGetAttrTensor (TF_Operation *oper, const char *attr_name, TF_Tensor **value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrTensor (TF_Operation oper, string attr_name, TF_Tensor* value, TF_Status status);

    // extern void TF_OperationGetAttrTensorList (TF_Operation *oper, const char *attr_name, TF_Tensor **values, int max_values, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrTensorList (TF_Operation oper, string attr_name, TF_Tensor* values, int max_values, TF_Status status);

    // extern void TF_OperationGetAttrValueProto (TF_Operation *oper, const char *attr_name, TF_Buffer *output_attr_value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrValueProto (TF_Operation oper, string attr_name, LLBuffer* output_attr_value, TF_Status status);

    // extern void TF_OperationToNodeDef (TF_Operation *oper, TF_Buffer *output_node_def, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationToNodeDef (TF_Operation oper, LLBuffer* output_node_def, TF_Status status);

    /// <summary>
    /// Gets the handle to the unmanaged TF_Operation object.
    /// </summary>
    /// <value>The handle.</value>
    member this.Handle = handle;

    // Pointer to the graph, to keep it from collecting if there are Operations alive.
    // member this.graph = graph

    /// <summary>
    /// The name for this operation/
    /// </summary>
    /// <value>The name.</value>
    member this.Name = if handle = IntPtr.Zero then "<ObjectDisposed>" else (TF_OperationName (handle)).GetStr ()

    /// <summary>
    /// Gets the number of outputs on this operation.
    /// </summary>
    /// <value>The number outputs.</value>
    member this.NumOutputs = if handle = IntPtr.Zero then -1 else TF_OperationNumOutputs (handle)

    member this.OutputListLength (argName : string, ?status : TFStatus) = 
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
        let cstatus = TFStatus.Setup (?incoming=status);
        let res = TF_OperationOutputListLength (handle, argName, cstatus.Handle);
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        res;

    /// <summary>
    /// Gets the number of inputs for this operation.
    /// </summary>
    /// <value>The number inputs.</value>
    member this.NumInputs = TF_OperationNumInputs (handle);

    member this.InputListLength (argName :string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
        let cstatus = TFStatus.Setup (?incoming=status)
        let res = TF_OperationInputListLength (handle, argName, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        res

    /// <summary>
    /// Gets the number of control inputs oto an operation
    /// </summary>
    /// <value>The number control inputs.</value>
    member this.NumControlInputs = TF_OperationNumControlInputs (handle);

    /// <summary>
    /// Gets the number of operations that have this operation as a control input.
    /// </summary>
    member this.NumControlOutputs = TF_OperationNumControlOutputs (handle);

    /// <summary>
    /// Get the list of operations that have this operation as a control input.
    /// </summary>
    /// <value>The control outputs.</value>
    member this.ControlOutputs 
        with get() = 
            let n = this.NumControlOutputs;
            let arr = Array.zeroCreate<IntPtr> n
            TF_OperationGetControlOutputs (handle, arr, n) |> ignore
            Array.create n (fun i -> new Operation((*graph,*) arr.[i]))

    member this.Device = (TF_OperationDevice (handle)).GetStr ();

    member this.GetAttributeMetadata (attrName : string, ?status : TFStatus) : TFAttributeMetadata =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
        let cstatus = TFStatus.Setup (?incoming=status);
        let x = TF_OperationGetAttrMetadata (handle, attrName, cstatus.Handle);
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        x

    member this.GetAttrString (attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise (ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        if metadata.TotalSize < 0L then raise(Exception("Metadata Error"))
        checkAttrType(attrName, metadata, TFAttributeType.String, false)
        let buf = Marshal.AllocHGlobal((int) metadata.TotalSize + 1)
        let maxLength = int metadata.TotalSize
        TF_OperationGetAttrString(handle, attrName, buf.ToPointer() |> voidToNativeInt,  maxLength, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        let bytes = Array.zeroCreate<byte> maxLength 
        Marshal.Copy(buf, bytes, 0, bytes.Length)
        Encoding.UTF8.GetString(bytes)

    member this.GetAttrStringList (attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if attrName = null then raise (ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup (?incoming=status);
        let metadata = this.GetAttributeMetadata (attrName, cstatus);
        if metadata.TotalSize < 0L then raise (Exception("Metadata Error"))
        checkAttrType(attrName, metadata, TFAttributeType.String, true)
        let storageSize = (int) metadata.TotalSize + 1;
        let storage = Marshal.AllocHGlobal(storageSize);
        let lengths = Array.zeroCreate<UIntPtr> (int metadata.ListSize)
        let values = Array.zeroCreate<IntPtr> (int metadata.ListSize)
        use valuesF = fixed &values.[0]
        use lengthsF = fixed &lengths.[0]
        TF_OperationGetAttrStringList(handle, attrName, valuesF, lengthsF, int metadata.ListSize, storage, storageSize, cstatus.Handle) 
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        let returnValues = Array.create (int metadata.ListSize) (fun i ->
            let length = int lengths.[i]
            let bytes = Array.zeroCreate<byte> length 
            Marshal.Copy(values.[i],bytes,0,length)
            System.Text.Encoding.UTF8.GetString(bytes))
        Marshal.FreeHGlobal(storage);
        returnValues;

    member this.GetAttrInt (attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if (box attrName = null) then raise (ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let mutable value = 0L
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Int, false)
        TF_OperationGetAttrInt(handle, attrName, &&value, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incoming = status) |> ignore
        value;

    // WARN: untested
    member this.GetAttrIntList (attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Int, true)
        let value = Array.zeroCreate<int64> (int metadata.ListSize)
        use data = fixed &value.[0]
        TF_OperationGetAttrIntList(handle, attrName, data, (int) metadata.ListSize, cstatus.Handle)
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        value

    member this.GetAttrFloat(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrname"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus);
        checkAttrType(attrName, metadata, TFAttributeType.Float, false);
        let mutable value = 0.0f;
        TF_OperationGetAttrFloat(handle, attrName, &&value, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        value

    // WARN: untested
    member this.GetAttrFloatList(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Float, true)
        let value = Array.zeroCreate<float32>  (int metadata.ListSize)
        use data = fixed  &value.[0]
        TF_OperationGetAttrFloatList(handle, attrName, data, (int) metadata.ListSize, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        value

    member this.GetAttrBool(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Bool, false)
        let mutable value : byte = 0uy
        TF_OperationGetAttrBool(handle, attrName, &&value, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        Convert.ToBoolean(value)

    // WARN: untested
    member this.GetAttrBoolList(attrName : string, ?status : TFStatus) = 
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Bool, true)
        let values = Array.zeroCreate<byte> (int metadata.ListSize)
        use valuesF = fixed &values.[0]
        TF_OperationGetAttrBoolList(handle, attrName, valuesF, (int) metadata.ListSize, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        values |> Array.map (Convert.ToBoolean)

    member this.GetAttrType(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Type, false)
        let mutable value = DType.Unknown
        TF_OperationGetAttrType(handle, attrName, &&value, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        value

    member this.GetAttrTypeList(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Type, true)
        let values = Array.zeroCreate<DType> (int metadata.ListSize)
        use valuesF = fixed &values.[0]
        TF_OperationGetAttrTypeList(handle, attrName, valuesF, (int) metadata.ListSize, cstatus.Handle)
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        values

    member this.GetAttrShape(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        if metadata.TotalSize < 0L then raise(Exception("Metadata Error"))
        checkAttrType(attrName, metadata, TFAttributeType.Shape, false) |> ignore
        let value = Array.zeroCreate<int64> (int metadata.TotalSize)
        use data = fixed &value.[0]
        TF_OperationGetAttrShape(handle, attrName, data, (int) metadata.TotalSize, cstatus.Handle)
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        value

    member this.GetAttrShapeList(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrname"))
        let cstatus = TFStatus.Setup(?incoming=status);
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        if metadata.TotalSize < 0L then raise(Exception("Metadata Error"))
        checkAttrType(attrName, metadata, TFAttributeType.Shape, true)
        let numDims = Array.zeroCreate<int> (int metadata.ListSize)
        let storage = Array.zeroCreate<int64> (int metadata.TotalSize)
        let dims = Array.zeroCreate<IntPtr> (int metadata.ListSize)
        use dimsF = fixed &dims.[0]
        use numDimsF = fixed &numDims.[0]
        use storageF = fixed &storage.[0]
        TF_OperationGetAttrShapeList(handle, attrName, dimsF, numDimsF, (int)metadata.ListSize, storageF, (int)metadata.TotalSize, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incoming=status) |> ignore
        let returnValues = Array.zeroCreate<Shape> (int metadata.ListSize)
        let mutable offset = 0;
        for i = 0 to int metadata.ListSize - 1 do
            let xs = Array.zeroCreate<int64> numDims.[i]
            for j = 0 to numDims.[i] - 1 do
                xs.[j] <- storage.[offset + j]
            returnValues.[i] <- new Shape(xs)
            offset <- offset + numDims.[i]
        returnValues

    // WARN: We don't have an attribute metadata type associated with TensorShapeProto
    // WARN: untested
    member this.GetAttrTensorShapeProto(attrName : string, ?status : TFStatus) : TFBuffer =
       if handle = IntPtr.Zero then  TFDisposable.ObjectDisposedException()
       let cstatus = TFStatus.Setup(?incoming=status)
       let r = new TFBuffer()
       let metadata = this.GetAttributeMetadata(attrName, cstatus)
       if metadata.TotalSize < 0L then raise(Exception("Metadata Error"))
       TF_OperationGetAttrTensorShapeProto(handle, attrName, r.LLBuffer, cstatus.Handle)
       cstatus.CheckMaybeRaise(?incoming=status) |> ignore
       r

    // NOTE: Commented out for now as we don't have an attribute metadata type associated with TensorShapeProtoList
    // WARN: untested
    member this.GetAttrTensorShapeProtoList(attrName : string, ?status : TFStatus) =
       if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException();
       let cstatus = TFStatus.Setup(?incoming=status);
       let metadata = this.GetAttributeMetadata(attrName, cstatus)
       let rs = Array.init (int metadata.ListSize) (fun _ -> new TFBuffer())
       let ls = rs |> Array.map (fun x -> x.LLBuffer |> NativePtr.toNativeInt)
       use data = fixed &ls.[0]
       TF_OperationGetAttrTensorShapeProtoList(handle, attrName, data, (int)metadata.ListSize, cstatus.Handle)
       cstatus.CheckMaybeRaise(?incoming=status) |> ignore
       rs

    // member this.GetAttrTensor(attrName : string, ?status : TFStatus) =
    //     if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
    //     let cstatus = TFStatus.Setup (?incoming=status)
    //     let metadata = this.GetAttributeMetadata(attrName, cstatus)
    //     checkAttrType(attrName, metadata, TFAttributeType.Tensor, false)
    //     let tensor = IntPtr.Zero : TF_Tensor
    //     TF_OperationGetAttrTensor(handle, attrName, &&tensor, cstatus.Handle)
    //     cstatus.CheckMaybeRaise (?incoming=status)
    //     new Tensor(tensor)

    //     // WARN: untested
    //     public Tensor[] GetAttrTensorList(string attrName, TFStatus status = null)
    //     {
    //         if (handle == IntPtr.Zero)
    //             TFDisposable.ObjectDisposedException ();
    //         var cstatus = TFStatus.Setup (status);
    //         var metadata = GetAttributeMetadata(attrName, cstatus);
    //         if (metadata.TotalSize < 0)
    //             throw new Exception("Metadata Error");
    //         checkAttrType(attrName, metadata, TFAttributeType.Tensor, true);
    //         Tensor[] returnValue = new Tensor[metadata.ListSize];
    //         unsafe
    //         {
    //             TF_Tensor[] tensorPointers = new TF_Tensor[metadata.ListSize];
    //             fixed (TF_Tensor* tensorPointersF = &tensorPointers[0])
    //                 TF_OperationGetAttrTensorList(handle, attrName, tensorPointersF, (int) metadata.ListSize, cstatus.Handle);
    //             for (int i = 0; i < metadata.ListSize; i++)
    //             {
    //                 returnValue[i] = new Tensor(tensorPointers[i]);
    //             }
    //         }
    //         cstatus.CheckMaybeRaise (status);
    //         return returnValue;
    //     }
    //     // NOTE: Commented out for now as we don't have an attribute metadata type associated with Proto
    //     // WARN: untested
    //     //public TFBuffer GetAttrValueProto(string attrName, TFStatus status = null)
    //     //{
    //     //    if (handle == IntPtr.Zero)
    //     //        TFDisposable.ObjectDisposedException();
    //     //    var cstatus = TFStatus.Setup(status);
    //     //    var r = new TFBuffer();
    //     //    unsafe
    //     //    {
    //     //        TF_OperationGetAttrValueProto(handle, attrName, r.LLBuffer, cstatus.Handle);
    //     //    }
    //     //    cstatus.CheckMaybeRaise(status);
    //     //    return r;
    //     //}

    /// <summary>
    /// Encodes the Operation as a protocol buffer payload
    /// </summary>
    /// <returns>The buffer with the encoded operation in the protocol buffer format.</returns>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// </remarks>
    member this.ToNodeDef (?status : TFStatus) : TFBuffer = 
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
        let cstatus = TFStatus.Setup (?incoming=status)
        let r = new TFBuffer ()
        TF_OperationToNodeDef (handle, r.LLBuffer, cstatus.Handle)
        // No need to raise, we can return null in that case.
        if not(cstatus.Ok) then
            r.Dispose ();
            box null :?> TFBuffer // TODO consider taking this out
        else 
            r;

    interface IComparable
        with 
            member this.CompareTo(x : obj) = 
                if (x.GetType() <> this.GetType()) then -1
                else (this :> IComparable<Operation>).CompareTo(x :?> Operation);

    interface IComparable<Operation>
        with 
            member this.CompareTo(x : Operation) =
                if box x = null then -1
                else this.Name.CompareTo(x.Name);

    override this.Equals(x:obj) = 
        match x with
        | :? Operation as other -> this.Handle = other.Handle
        | _ -> false

