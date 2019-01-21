namespace Tensorflow

open System
open System.Runtime.InteropServices
open System.Text
open System.Globalization
open System.Linq
open Utils
open Microsoft.FSharp.NativeInterop
open System.Numerics
open System.Collections.Generic
open System.Linq.Expressions

#nowarn "9"

/// <summary>
/// TensorFlow Exception
/// </summary>
type TFException(message:string) =
    inherit Exception(message)

/// Status code for invoking a tensorflow operation
[<RequireQualifiedAccess>]
type TFCode = 

    /// Not an error; return on success
    | Ok = 0u

    /// The operation was cancelled (typically by the caller).
    | Cancelled = 1u

    /// Unknown error.  An example of where this error may be returned is
    /// if a Status value received from another address space belongs to
    /// an error-space that is not known in this address space.  Also
    /// errors raised by APIs that do not return enough error information
    /// may be converted to this error.
    | Unknown = 2u

    /// Client specified an invalid argument.  Note that this differs
    /// from FailedPrecondition.  InvalidArgumentindicates arguments
    /// that are problematic regardless of the state of the system
    /// (e.g., a malformed file name).
    | InvalidArgument = 3u

    /// Deadline expired before operation could complete.  For operations
    /// that change the state of the system, this error may be returned
    /// even if the operation has completed successfully.  For example, a
    /// successful response from a server could have been delayed long
    /// enough for the deadline to expire.
    | DeadlineExceeded = 4u

    /// Some requested entity (e.g., file or directory) was not found.
    /// For privacy reasons, this code may be returned when the client
    /// does not have the access right to the entity.
    | NotFound = 5u

    /// Some entity that we attempted to create (e.g., file or directory) already exists.
    | AlreadyExists = 6u

    /// The caller does not have permission to execute the specified
    /// operation.  PermissionDenied must not be used for rejections
    /// caused by exhausting some resource (use ResourceExhausted
    /// instead for those errors).  PermissionDeniedmust not be
    /// used if the caller can not be identified (use Unauthenticated
    /// instead for those errors).
    | PermissionDenied = 7u

    /// The request does not have valid authentication credentials for the
    /// operation.
    | Unauthenticated = 16u

    /// Some resource has been exhausted, perhaps a per-user quota, or
    /// perhaps the entire file system is out of space.
    | ResourceExhausted = 8u

    /// Operation was rejected because the system is not in a state
    /// required for the operation's execution.  For example, directory
    /// to be deleted may be non-empty, an rmdir operation is applied to
    /// a non-directory, etc.
    ///
    /// A litmus test that may help a service implementor in deciding
    /// between FailedPrecondition, Aborted, and Unavailable:
    /// 
    ///  (a) Use Unavailableif the client can retry just the failing call.
    ///  (b) Use Aborted if the client should retry at a higher-level
    ///      (e.g., restarting a read-modify-write sequence).
    ///  (c) Use FailedPrecondition if the client should not retry until
    ///      the system state has been explicitly fixed.  E.g., if an "rmdir"
    ///      fails because the directory is non-empty, FailedPrecondition
    ///      should be returned since the client should not retry unless
    ///      they have first fixed up the directory by deleting files from it.
    ///  (d) Use FailedPrecondition if the client performs conditional
    ///      REST Get/Update/Delete on a resource and the resource on the
    ///      server does not match the condition. E.g., conflicting
    ///      read-modify-write on the same resource.
    | FailedPrecondition = 9u

    /// The operation was aborted, typically due to a concurrency issue
    /// like sequencer check failures, transaction aborts, etc.
    ///
    /// See litmus test above for deciding between FailedPrecondition,
    /// Aborted and Unavailable
    | Aborted = 10u

    /// Operation tried to iterate past the valid input range.  E.g., seeking or
    /// reading past end of file.
    ///
    /// Unlike InvalidArgument, this error indicates a problem that may
    /// be fixed if the system state changes. For example, a 32-bit file
    /// system will generate InvalidArgument if asked to read at an
    /// offset that is not in the range [0,2^32-1], but it will generate
    /// OutOfRange if asked to read from an offset past the current
    /// file size.
    ///
    /// There is a fair bit of overlap between FailedPrecondition and
    /// OutOfRange.  We recommend using OutOfRane (the more specific
    /// error) when it applies so that callers who are iterating through
    /// a space can easily look for an OutOfRange error to detect when
    /// they are done.
    | OutOfRange = 11u

    /// Operation is not implemented or not supported/enabled in this service.
    | Unimplemented = 12u

    /// Internal errors.  Means some invariants expected by underlying
    /// system has been broken.  If you see one of these errors,
    /// something is very broken.
    | Internal = 13u

    /// The service is currently unavailable.  This is a most likely a
    /// transient condition and may be corrected by retrying with
    /// a backoff.
    ///
    /// See litmus test above for deciding between FailedPrecondition,
    /// Aborted, and Unavailable.
    | Unavailable = 14u

    /// Unrecoverable data loss or corruption.
    | DataLoss = 15u


module internal TFString = 
    // extern size_t TF_StringEncode (const char *src, size_t src_len, char *dst, size_t dst_len, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern size_t TF_StringEncode (byte* src, size_t src_len, sbyte* dst, size_t dst_len, TF_Status status)

    // extern size_t TF_StringDecode (const char *src, size_t src_len, const char **dst, size_t *dst_len, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern size_t TF_StringDecode (sbyte* src, size_t src_len, IntPtr* dst, size_t* dst_len, TF_Status status)

    // extern size_t TF_StringEncodedSize (size_t len)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern size_t TF_StringEncodedSize (size_t len)

/// <summary>
/// Used to track the result of TensorFlow operations.
/// </summary>
/// <remarks>
/// <para>
/// TFStatus is used to track the status of a call to some TensorFlow
/// operations.   Instances of this object are passed to various
/// TensorFlow operations and you can use the <see cref="P:TensorFlow.TFStatus.Ok"/>
/// to quickly check if the operation succeeded, or get more detail from the
/// <see cref="P:TensorFlow.TFStatus.StatusCode"/> and a human-readable text
/// using the <see cref="P:TensorFlow.TFStatus.StatusMessage"/> property.
/// </para>
/// <para>
/// The convenience <see cref="M:TensorFlow.TFStatus.Raise"/> can be used
/// to raise a <see cref="P:TensorFlow.TFException"/> if the status of the
/// operation did not succeed.
/// </para>
/// </remarks>
type TFStatus(handle) =
    inherit TFDisposable(handle)

    // extern TF_Status * TF_NewStatus ()
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Status TF_NewStatus ()

    // extern void TF_DeleteStatus (TF_Status *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteStatus (TF_Status status)

    // extern void TF_SetStatus (TF_Status *s, TF_Code code, const char *msg)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetStatus (TF_Status s, TFCode code, string msg)

    // extern TF_Code TF_GetCode (const TF_Status *s)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TFCode TF_GetCode (TF_Status s)

    // extern const char * TF_Message (const TF_Status *s)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_Message (TF_Status s)

    static let tfstatus = new System.Threading.ThreadLocal<_>(fun () -> new TFStatus())

    /// <summary>
    /// Per-thread global status that you can use if you do not need to create a new instance of this object.
    /// </summary>
    /// <remarks>
    /// This is provided as a convenience for APIs that take a TFStatus.   While the TFStatus is usually an
    /// optional parameter, when it is made optional, API calls that fail raise an exception.   Use this 
    /// property to pass a TFStatus without having to allocate a new one.   The problem with this of course
    /// is that you risk having multiple parts of your code override this thread-global variable.
    /// </remarks>
    static member Default = tfstatus.Value

    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.TFStatus"/> class.
    /// </summary>
    new() = new TFStatus(TF_NewStatus ())

    override this.NativeDispose(handle:IntPtr) = TF_DeleteStatus (handle)

    /// <summary>
    /// Sets the status code on this TFStatus.
    /// </summary>
    /// <param name="code">Code.</param>
    /// <param name="msg">Message.</param>
    member this.SetStatusCode (code : TFCode, msg : string) = TF_SetStatus (handle, code, msg)


    /// <summary>
    /// Gets the status code for the status code.
    /// </summary>
    /// <value>The status code as an enumeration.</value>
    member this.StatusCode 
        with get() = 
            if handle = IntPtr.Zero then
                raise(ObjectDisposedException ("TFStatus"))
            TF_GetCode (handle)

    /// <summary>
    /// Gets a human-readable status message.
    /// </summary>
    /// <value>The status message.</value>
    member this.StatusMessage = (TF_Message (handle)).GetStr ()

    /// <summary>
    /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFStatus"/>.
    /// </summary>
    /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFStatus"/>.</returns>
    override this.ToString () = 
        if handle = IntPtr.Zero then
            raise(ObjectDisposedException ("TFStatus"))
        sprintf "[TFStatus: StatusCode=%A, StatusMessage=%s]" this.StatusCode this.StatusMessage

    /// <summary>
    /// Gets a value indicating whether this <see cref="T:TensorFlow.TFStatus"/> state has been set to ok.
    /// </summary>
    /// <value><c>true</c> if ok; otherwise, <c>false</c>.</value>
    member this.Ok = this.StatusCode = TFCode.Ok

    /// <summary>
    /// Gets a value indicating whether this <see cref="T:TensorFlow.TFStatus"/> state has been set to an error.
    /// </summary>
    /// <value><c>true</c> if error; otherwise, <c>false</c>.</value>
    member this.Error = this.StatusCode <> TFCode.Ok

    /// <summary>
    /// Convenience method that raises an exception if the current status is an error.
    /// </summary>
    /// <remarks>
    /// You can use this method as a convenience to raise an exception after you
    /// invoke an operation if the operation did not succeed.
    /// </remarks>
    member this.Raise () =
        if (TF_GetCode (handle) <> TFCode.Ok) then
            raise (TFException (this.StatusMessage))

    // Utility function used to simplify implementing the idiom
    // where the user optionally provides a TFStatus, if it is provided,
    // the error is returned there;   If it is not provided, then an
    // exception is raised.
    member internal this.CheckMaybeRaise (?incoming : TFStatus, ?last : bool) = 
        let last = defaultArg last true
        match incoming with
        | None ->
            if handle = IntPtr.Zero then
                Console.WriteLine ("oops")
            if this.StatusCode <> TFCode.Ok  then
                let e = new TFException (this.StatusMessage)
                if last then
                    this.Dispose ()
                raise e
            if last then
                this.Dispose ()
            true
        | Some(_) ->
            this.StatusCode = TFCode.Ok

    static member (*internal*) Setup (?incoming : TFStatus) = 
        match incoming with | None -> new TFStatus() | Some(x) -> x


/// <summary>
/// The session options object holds configuration options that you want to use during your session, like the TensorFlow target or the configuration.
/// </summary>
type SessionOptions(handle) =
    inherit TFDisposable(handle)

    // extern TF_SessionOptions * TF_NewSessionOptions ()
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_SessionOptions TF_NewSessionOptions ()

    // extern void TF_DeleteSessionOptions (TF_SessionOptions *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteSessionOptions (TF_SessionOptions options)

    // extern void TF_SetTarget (TF_SessionOptions *options, const char *target)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetTarget (TF_SessionOptions options, string target)

    // extern void TF_SetConfig (TF_SessionOptions *options, const void *proto, size_t proto_len, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetConfig (TF_SessionOptions options, IntPtr proto, size_t proto_len, TF_Status status)

    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.SessionOptions"/> class.
    /// </summary>
    new() = new SessionOptions(TF_NewSessionOptions ())

    override this.NativeDispose (handle : IntPtr) = TF_DeleteSessionOptions (handle)

    /// <summary>
    /// Sets the target in options.
    /// </summary>
    /// <param name="target">target can be empty, a single entry, or a comma separated list of entries.
    /// Each entry is in one of the following formats: "local", ip:port, host:port.</param>
    member this.SetTarget (target : string) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("SessionOptions"))
        else TF_SetTarget (handle, target)

    /// <summary>
    /// Sets the configuration information for the session.
    /// </summary>
    /// <param name="protoData">Serialized protocol buffer for the tensorflow.ConfigProto message.</param>
    /// <param name="length">Length of the buffer.</param>
    /// <param name="status">If config was not parsed successfully as a ConfigProto, the error is recorded here.</param>
    /// <remarks>
    /// The configuration option is a Protocol Buffer representing the tensorflow.ConfigProto
    /// </remarks>
    member this.SetConfig (protoData : IntPtr, length : int, ?status : TFStatus) = 
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("SessionOptions"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_SetConfig (handle, protoData, UIntPtr(uint32 length), cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status)
    
    member this.Handle = handle

/// Low-level: Enumeration describing the types of a metadata attribute
[<RequireQualifiedAccess>]
type TFAttributeType =

    /// The type of the attribute is a string
    | String = 0u

    /// The type of the attribute is an int.
    | Int = 1u

    /// The type of the attribute is a float32
    | Float = 2u

    /// The type of the attribute is a bool.
    | Bool = 3u

    /// The type of the attribute is a type.
    | Type = 4u

    /// The type of the attribute is a tensor shape
    | Shape = 5u

    /// The type of the attribute is a tensor
    | Tensor = 6u

    /// The type of the attribute is a placeholder
    | Placeholder = 7u

    /// The type of the attribute is a function
    | Func = 8u

/// <summary>
/// Low-level: this describes the tensorflow type information for an attribute in the low-level attributes used by operations.
/// </summary>
/// <remarks>
/// This is a low-level operation returned by the <see cref="M:TensorFlow.Operation.GetAttributeMetadata"/>.
/// This is included for completeness, but is not generally used from C#, as you have access to the high-level
/// bindings in the <see cref="T:TensorFlow.Graph"/> type.
/// </remarks>
[<StructLayout (LayoutKind.Sequential)>]
[<Struct>]
type TFAttributeMetadata =
    //byte isList
    //val IsLbool IsList => isList != 0
    val private isList : byte
    val ListSize : int64
    val Type : TFAttributeType
    val TotalSize : int64

    member this.IsList = this.isList = 0uy

    /// <summary>
    /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFAttributeMetadata"/>.
    /// </summary>
    /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFAttributeMetadata"/>.</returns>
    override this.ToString() =
         sprintf "[TFAttributeMetadata IsList=%O ListSize=%i Type=%A TotalSize=%i]" this.isList this.ListSize this.Type this.TotalSize

/// <summary>
/// Represents the shape of a tensor, it describes how many dimensions the tensor has in a given axis
/// </summary>
/// <remarks>
/// <para>
/// The shapes can be created by calling the constructor with the number of dimensions
/// in the shape.   The null value is used to specify that the shape is unknown,
/// an empty array is used to create a scalar, and other values are used to specify
/// the number of dimensions.
/// </para>
/// <para>
/// For the Unknown case, you can use <see cref="P:TensorFlor.TFShape.Unknown"/>, for
/// scalars, you can use the <see cref="P:TensorFlor.Shape.Scalar"/> shape.
/// </para>
/// <para>
/// To create a 2-element vector, use:
/// new Shape (2)
/// </para>
/// <para>
/// To create a 2x3 matrix, use:
/// new Shape (2, 3)
/// </para>
/// <para>
/// To create a shape with an unknown number of elements, you can pass the value
/// -1.  This is typically used to indicate the shape of tensors that represent a
/// variable-sized batch of values.
/// </para>
/// <para>
/// To create a matrix with 4 columns and an unknown number of rows:
/// var batch = new Shape (-1, 4)
/// </para>
/// </remarks>
/// <summary>
/// Initializes a new instance of the <see cref="T:TensorFlow.hape"/> class.
/// </summary>
/// <param name="args">This is a params argument, so you can provide multiple values to it.  
/// A null value means that this is an unknown shape, a single value is used to create a vector,
/// two values are used to create a 2-D matrix and so on.
/// </param>
/// <remarks>
/// 
/// </remarks>
type Shape(dims:int64[] option) =
    new ([<ParamArray>] dims : int64[]) = Shape(Some(dims))
    /// <summary>
    /// Represents an unknown number of dimensions in the tensor.
    /// </summary>
    /// <value>The unknown.</value>
    static member Unknown = new Shape ([||])

    /// <summary>
    /// This shape is used to represent scalar values.
    /// </summary>
    /// <value>The scalar.</value>
    static member Scalar = new Shape ([|0L|])

    /// <summary>
    /// Gets the length of the specified dimension in the tensor
    /// </summary>
    /// <returns>The length, -1 for shapes that have an unknown dimension.</returns>
    /// <param name="dimension">Dimension.</param>
    member __.GetLength (dimension : int) = match dims with | None -> -1L | Some(dims) -> dims.[dimension]

    /// <summary>
    /// Number of dimensions represented by this shape.
    /// </summary>
    /// <value>The number dimensions, -1 if the number of dimensions is unknown, 0 if the shape represent a scalar, 1 for a vector, 2 for a matrix and so on..</value>
    member __.NumDimensions = match dims with | None -> -1 | Some(dims) -> dims.Length

    /// <summary>
    /// Gets a value indicating whether all the dimensions in the <see cref="T:TensorFlow.TFShape"/> are fully specified.
    /// </summary>
    /// <value><c>true</c> if is fully specified; otherwise, <c>false</c>.</value>
    member __.IsFullySpecified  
        with get() = match dims with | Some(dims) when dims |> Array.exists ((=) -1L) |> not -> true | _ -> false

    /// <summary>
    /// Returns the shape as an array
    /// </summary>
    /// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
    member __.ToArray () =
        match dims with
        | Some dims -> dims
        | None -> null
    
    member __.Dims = dims |> Option.orDefault [||]

    /// <summary>
    /// Returns the shape as an array
    /// </summary>
    /// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
    member __.ToIntArray () = dims |> Option.map (Array.map int) |> Option.orDefault null

    /// <summary>
    /// Gets a value indicating whether one of the dimensions <see cref="T:TensorFlow.TFShape"/> in the shape is larger than Int32.MaxValue.
    /// </summary>
    /// <value><c>true</c> if is long array; otherwise, <c>false</c>.</value>
    member __.IsLongArray with get() = dims |> Option.map (Array.exists (fun x -> x > int64 Int32.MaxValue)) |> Option.orDefault false

    /// <summary>
    /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFShape"/>.
    /// </summary>
    /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFShape"/>.</returns>
    override __.ToString () =
        match dims with 
        | Some dims -> dims |> Array.map (function | -1L -> "?" | x -> string x) |> String.concat "," |> sprintf "[%s]"
        | None -> "unknown"

    /// <summary>
    /// Gets the dimensions for the specified index.
    /// </summary>
    /// <param name="idx">Index.</param>
    member __.Index (idx:int) = dims.Value.[idx]

    /// <summary>
    /// Adds a <see cref="TensorFlow.TFShape"/> to a <see cref="TensorFlow.TFShape"/>, yielding a shape made up of the concatenation of the first and the second shapes.
    /// </summary>
    /// <param name="left">The first <see cref="TensorFlow.TFShape"/> to add.</param>
    /// <param name="right">The second <see cref="TensorFlow.TFShape"/> to add.</param>
    /// <returns>The <see cref="T:TensorFlow.TFShape"/> that is the sum of the values of <c>left</c> and <c>right</c>.</returns>
    static member (+) (left:Shape,right:Shape) =
        new Shape ([|yield! left.Dims; yield! right.Dims|])

    /// <summary>
    /// Returns the shape as a 1-dimensional tensor with each element corresponding to the specified shape dimension.
    /// </summary>
    /// <returns>The tensor.</returns>
    member this.AsTensor () = new Tensor (this.ToIntArray ())

     /// <summary>
     /// Performs an implicit conversion from <see cref="TFShape"/> to <see cref="Tensor"/>.
     /// </summary>
     /// <param name="shape">The shape.</param>
     /// <returns>The result of the conversion.</returns>
     static member op_Implicit (shape : Shape) : Tensor = shape.AsTensor ()

// Use for single dimension arrays 

[<AutoOpen>]
module TensorExtension =

    // extern void * TF_TensorData (const TF_Tensor *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern IntPtr TF_TensorData (TF_Tensor tensor)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern TF_Tensor TF_AllocateTensor (DType dataType, IntPtr zeroDim, int num_dims, size_t len)

    type Tensor with

        static member SetupTensor (dt : DType, shape : Shape, data : Array, start : int, count : int, size : int) : IntPtr =
         if box shape = null then raise (ArgumentNullException "shape")
         Tensor.SetupTensor (dt, shape.Dims, data, start, count, size)

        // Convenience, should I add T[,] and T[,,] as more convenience ones?
        /// <summary>
        /// Creates a single-dimension tensor from a byte buffer.  This is different than creating a tensor from a byte array that produces a tensor with as many elements as the byte array.
        /// </summary>
        static member CreateString (buffer : byte []) : Tensor =
          if box buffer = null then raise(ArgumentNullException ("buffer"))
          //
          // TF_STRING tensors are encoded with a table of 8-byte offsets followed by
          // TF_StringEncode-encoded bytes.
          //
          let size = TFString.TF_StringEncodedSize (UIntPtr(uint64 buffer.Length))
          let handle = TF_AllocateTensor (DType.String, IntPtr.Zero, 0, UIntPtr((uint64(size) + 8uL)))
      
          // Clear offset table
          let dst = TF_TensorData (handle)
          Marshal.WriteInt64 (dst, 0L)
          use status = new TFStatus()
          use src = fixed &buffer.[0]
          TFString.TF_StringEncode (src, UIntPtr(uint64 buffer.Length), dst.Add(8) |> NativePtr.ofNativeInt<int8>, size, status.Handle) |> ignore
          if status.Ok then
              new Tensor (handle)
          else box null :?> Tensor

/// <summary>
/// A grouping of operations with defined inputs and outputs.
/// Once created and added to graphs, functions can be invoked by creating an
/// operation whose operation type matches the function name.
/// </summary>
type TFFunction internal (handle : IntPtr) =
    inherit TFDisposable (handle)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern unit TF_FunctionToFunctionDef (IntPtr func, IntPtr buffer, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern unit TF_DeleteFunction (IntPtr handle)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_FunctionImportFunctionDef (byte* proto, IntPtr len, TF_Status status)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_FunctionName (IntPtr handle)

    /// <summary>
    /// Write out a serialized representation of the function as a FunctionDef protocol message to the provided <paramref name="outputFuncDef"/>
    /// </summary>
    /// <param name="outputFuncDef">An allocated buffer where the function will be serialized.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member __.ToFunctionDef (outputFuncDef : TFBuffer, ?status : TFStatus) =
        if box outputFuncDef = null then raise(ArgumentNullException ("outputFuncDef"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_FunctionToFunctionDef (handle, outputFuncDef.Handle, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming = status, last = false)

    override __.NativeDispose (handle : TF_Status) = TF_DeleteFunction (handle)

    /// <summary>
    /// Construct and return the function whose FunctionDef representation is
    /// serialized in <paramref name="proto"/> proto
    /// </summary>
    /// <returns>The function definition, or null on failure.</returns>
    /// <param name="proto">Array containing the serialized FunctionDef in a protocol buffer.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member __.ImportFunctionDef (proto : byte [], ?status : TFStatus) : TFFunction  =
        if box proto = null then raise(ArgumentNullException ("proto"))
        let cstatus = TFStatus.Setup (?incoming=status)
        use p = fixed &proto.[0] 
        //let res = TF_FunctionImportFunctionDef (p, (IntPtr)proto.Length, cstatus.Handle)
        let res = TF_FunctionImportFunctionDef (p, IntPtr(proto.Length), cstatus.Handle)
        if (not(cstatus.CheckMaybeRaise (?incoming=status, last= false))) then
            box null :?> TFFunction
        else
            new TFFunction (handle)

/// <summary>
/// Represents a dynamically loaded library of TensorFlow operations, use to load and consume TensorFlow operations from an external library.
/// </summary>
/// <remarks>
/// Use the static method <see cref="M:Tensorflow.TFLibrary.FromFile"/> to load a dynamic library.
/// Once that function returns
/// </remarks>
type TFLibrary private (handle : IntPtr) = 
    inherit TFDisposable(handle)

    // extern TF_Library * TF_LoadLibrary (const char *library_filename, TF_Status *status)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Library TF_LoadLibrary (string library_filename, TF_Status  status)

    // extern void TF_DeleteLibraryHandle (TF_Library *lib_handle)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteLibraryHandle (TF_Library lib_handle)

    // extern TF_Buffer TF_GetOpList (TF_Library *lib_handle)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern LLBuffer TF_GetOpList (TF_Library lib_handle)

    /// <summary>
    /// Load the library specified by and register the operations and
    /// kernels present in that library.
    /// </summary>
    /// <returns>Handle to the loaded library.</returns>
    /// <param name="libraryFile">Name of the library to load, this is a platform specific name.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    /// <remarks>
    /// The provided <paramref name="libraryFile"/> is passed to the operating system dynamic loader
    /// and it will load the library using the operating system defined search paths and rules to load this.
    /// </remarks>
    static member FromFile (libraryFile : string, ?status : TFStatus) : TFLibrary  =
        let cstatus = TFStatus.Setup (?incoming=status)
        let h = TF_LoadLibrary (libraryFile, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incoming=status) |> ignore
        new TFLibrary (h)

    /// <summary>
    /// Retrieves the ProtocolBuffer describing the available operations in
    /// the loaded TensorFlow library.
    /// </summary>
    /// <returns>The buffer contains a ProtocolBuffer encoded payload, you need a ProtocolBuffer reader to process the contents.</returns>
    member __.GetOpList () : TFBuffer = new TFBuffer ((TF_GetOpList (handle)).data)

    override __.NativeDispose (handle : IntPtr) = TF_DeleteLibraryHandle (handle)
