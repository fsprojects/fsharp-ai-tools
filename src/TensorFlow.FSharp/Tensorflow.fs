namespace TensorFlow.FSharp

open System
open System.Collections
open System.Collections.Generic
open System.Runtime.InteropServices
open TensorFlow.FSharp.Utils
open FSharp.NativeInterop

#nowarn "9" "86"

type Dim = TensorFlow.FSharp.Proto.TensorShapeProto.Dim

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

    override __.NativeDispose(handle:IntPtr) = TF_DeleteStatus (handle)

    /// <summary>
    /// Sets the status code on this TFStatus.
    /// </summary>
    /// <param name="code">Code.</param>
    /// <param name="msg">Message.</param>
    member __.SetStatusCode (code : TFCode, msg : string) = TF_SetStatus (handle, code, msg)


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

/// Represents the value of one dimension in a TFShape
type Dimension(value : int64 option) = 
    do value |> Option.iter (fun x -> if x < 0L then raise (ValueError(sprintf "Dimension %i must be >= 0" x)))
    static let unknownDimension = Dimension(Option<int64>.None)

    static let mapBoth (self:Dimension) (other:Dimension) (f:int64 -> int64->int64) = 
        match self.Value, other.Value with
        | Some(x),Some(y) -> Dimension(f x y)
        | _ -> Dimension.Unknown

    static let mapBothBool (self:Dimension) (other:Dimension) (f:int64 -> int64-> bool) = 
        match self.Value, other.Value with
        | Some(x),Some(y) -> f x y
        | _ -> false

    new (value : int option) = Dimension(value |> Option.map int64)
    new (value : int64 ) = Dimension(Some(value))
    new (value : int ) = Dimension(Some(value))

    override this.ToString() = match value with | None -> "?" | Some(x) -> string x

    member this.Value = value
    
    static member op_Implicit (dimension: Dimension) : int = 
        match dimension.Value with 
        | Some(x) -> int x 
        | None -> raise(ValueError("Unknown dimension"))

    static member Unknown = unknownDimension

    interface IFSIPrint with
        member this.ToFSIString() = sprintf "Dimension(%A)" value

    /// Returns true of `other` has the same known value as this Dimension.
    static member (=) (self : Dimension, other : Dimension) = 
        match self.Value, other.Value with
        | Some(x), Some(y) -> x = y 
        | _ -> false

    /// Returns true if `other` has a different known value from `self`.
    static member (<>) (self : Dimension, other : Dimension) = 
        match self.Value, other.Value with
        | Some(x), Some(y) -> x <> y 
        | _ -> false
        
    /// <summary>
    /// Returns true if `other` is compatible with this Dimension.
    ///
    /// Two known Dimensions are compatible if they have the same value.
    /// An unknown Dimension is compatible with all other Dimensions.
    /// </summary>
    /// <returns>True if this Dimension and `other` are compatible.</returns>
    /// <param name="other> Another Dimension. </param>
    member this.IsCompatibleWith(other : Dimension) = 
        match this.Value,other.Value with 
        | Some(x),Some(y) when x <> y -> false 
        | _ -> true

    /// <summary>Raises an exception if `other` is not compatible with this Dimension.
    /// <param name="other">Another Dimension.</param>
    /// <exception cref="TensorFlow.FSharp.ValueError">If `this` and `other` are not compatible (see IsCompatibleWith) </exception>
    /// </summary>
    member this.AssertIsCompatibleWith(other : Dimension) = 
        if not(this.IsCompatibleWith(other)) then
            raise(ValueError(sprintf "Dimensions %O and %O are not compatible" this other))

    /// <summary>Returns a Dimension that combines the information in `this` and `other`.
    ///
    /// Dimensions are combined as follows:
    ///
    /// <c>
    /// Dimension(n)   .MergeWith(Dimension(n))    = Dimension(n)
    /// Dimension(n)   .MergeWith(Dimension(None)) = Dimension(n)
    /// Dimension(None).MergeWith(Dimension(n))    = Dimension(n)
    /// Dimension(None).MergeWith(Dimension(None)) = Dimension(None)
    /// Dimension(n)   .MergeWith(Dimension(m))  # raises ValueError for n != m
    /// </c>
    /// <param name="other">Another Dimension.</param>
    ///
    /// <returns> A Dimension containing the combined information of `this` and `other`. </returns>
    /// <exception cref="TensorFlow.FSharp.ValueError">If `this` and `other` are not compatible (see IsCompatibleWith) </exception>
    /// </summary>
    member this.MergeWith(other : Dimension) = 
        this.AssertIsCompatibleWith(other)
        Dimension(this.Value |> Option.orElse other.Value)

    /// <summary>Returns the sum of `self` and `other`.
    ///
    /// Dimensions are summed as follows:
    ///
    /// <code>
    /// Dimension(m)    + Dimension(n)    = Dimension(m + n)
    /// Dimension(m)    + Dimension(None) = Dimension(None)
    /// Dimension(None) + Dimension(n)    = Dimension(None)
    /// Dimension(None) + Dimension(None) = Dimension(None)
    /// </code>
    /// <param name="other>Another Dimension.</param>
    /// <returns> Dimension whose value is the sum of `self` and `other`.</returns>
    /// </summary>
    static member (+) (self : Dimension, other : Dimension) = mapBoth self other (+)

    /// <summary>Returns the subtraction of `other` from `self`.
    ///
    /// Dimensions are subtracted as follows:
    /// <code>
    /// Dimension(m)    - Dimension(n)    = Dimension(m - n)
    /// Dimension(m)    - Dimension(None) = Dimension(None)
    /// Dimension(None) - Dimension(n)    = Dimension(None)
    /// Dimension(None) - Dimension(None) = Dimension(None)
    /// </code>
    /// <param name="other">Another Dimension.</param>
    /// <returns>A Dimension whose value is the subtraction of `other` from `self`.  </returns>
    /// </summary>
    static member (-) (self : Dimension, other : Dimension) = mapBoth self other (-)

    /// <summary>Returns the product of `self` and `other`.
    ///
    /// Dimensions are summed as follows:
    ///
    /// <code>
    /// Dimension(m)    * Dimension(n)    = Dimension(m * n)
    /// Dimension(m)    * Dimension(None) = Dimension(None)
    /// Dimension(None) * Dimension(n)    = Dimension(None)
    /// Dimension(None) * Dimension(None) = Dimension(None)
    /// </code>
    /// <param name="other">Another Dimension.</param>
    /// <returns>A Dimension whose value is the product of `self` and `other`.</returns>
    /// </summary>
    static member (*) (self : Dimension, other : Dimension) = mapBoth self other (*)
    // Not sure if I should sepcify these... keeping this here for now as a reference, I may come back to this
    static member (*) (self : int, other : Dimension) = mapBoth (Dimension(self)) other (*)
    static member (*) (self : Dimension, other : int) = mapBoth self (Dimension(other)) (*)

    /// <summary>Returns the modulo of `self` and `other`.
    ///
    /// Dimensions are summed as follows:
    ///
    /// <code>
    /// Dimension(m)    % Dimension(n)    = Dimension(m % n)
    /// Dimension(m)    % Dimension(None) = Dimension(None)
    /// Dimension(None) % Dimension(n)    = Dimension(None)
    /// Dimension(None) % Dimension(None) = Dimension(None)
    /// </code>
    /// </summary>
    /// <param name="other">Another Dimension.</param>
    /// <returns>A Dimension whose value is the modulo of `self` and `other`.</returns>
    static member (%) (self : Dimension, other : Dimension) = mapBoth self other (%)

    /// <summary>Returns the quotient of `self` and `other` rounded down.
    /// Dimensions are divided as follows:
    /// <code>
    /// Dimension(m)    / Dimension(n)    = Dimension(m / n)
    /// Dimension(m)    / Dimension(None) = Dimension(None)
    /// Dimension(None) / Dimension(n)    = Dimension(None)
    /// Dimension(None) / Dimension(None) = Dimension(None)
    /// </code>
    /// <param name="other">Another Dimension.</param>
    /// <returns>A `Dimension` whose value is the integer quotient of `self` and `other`.</returns>
    /// </summary>
    static member (/) (self : Dimension, other : Dimension) = mapBoth self other (/)

    /// <summary>Returns True if `self` is known to be less than `other`.
    /// Dimensions are compared as follows:
    /// <code>
    /// (Dimension(m)    < Dimension(n))    = (m < n)
    /// (Dimension(m)    < Dimension(None)) = None
    /// (Dimension(None) < Dimension(n))    = None
    /// (Dimension(None) < Dimension(None)) = None
    /// </code>
    /// <param name="other">Another Dimension.</param>
    /// <returns>The value of `self.value < other.value` if both are known, otherwise None</returns>
    /// </summary>
    static member (<) (self : Dimension, other : Dimension) = mapBothBool self other (<)

    /// <summary>Returns True if `self` is known to be less than or equal to `other`.
    /// Dimensions are compared as follows:
    /// <code>
    /// (Dimension(m)    <= Dimension(n))    = (m <= n)
    /// (Dimension(m)    <= Dimension(None)) = None
    /// (Dimension(None) <= Dimension(n))    = None
    /// (Dimension(None) <= Dimension(None)) = None
    /// </code>
    /// <param name="other">Another Dimension.</param>
    /// <returns>The value of `self.value <= other.value` if both are known, otherwise None</returns>
    /// </summary>
    static member (<=) (self : Dimension, other : Dimension) = mapBothBool self other (<=)

    /// <summary>Returns True if `self` is known to be greater than `other`.
    /// Dimensions are compared as follows:
    /// <code>
    /// (Dimension(m)    > Dimension(n))    = (m > n)
    /// (Dimension(m)    > Dimension(None)) = None
    /// (Dimension(None) > Dimension(n))    = None
    /// (Dimension(None) > Dimension(None)) = None
    /// </code>
    /// <param name="other">Another Dimension.</param>
    /// <returns>The value of `self.value > other.value` if both are known, otherwise None</returns>
    /// </summary>
    static member (>) (self : Dimension, other : Dimension) = mapBothBool self other (>)

    /// <summary>Returns True if `self` is known to be greater than or equal to `other`.
    /// Dimensions are compared as follows:
    /// <code>
    /// (Dimension(m)    >= Dimension(n))    = (m >= n)
    /// (Dimension(m)    >= Dimension(None)) = None
    /// (Dimension(None) >= Dimension(n))    = None
    /// (Dimension(None) >= Dimension(None)) = None
    /// </code>
    /// <param name="other">Another Dimension.</param>
    /// <returns>The value of `self.value >= other.value` if both are known, otherwise None</returns>
    /// </summary>
    static member (>=) (self : Dimension, other : Dimension) = mapBothBool self other (>=) 


    /// TODO consider using implicits in the operators
    static member op_Implicit (d : int) : Dimension = Dimension(Some(d))
    static member op_Implicit (d : int option) : Dimension = match d with | None -> Dimension.Unknown | _ -> Dimension(d)
    static member op_Implicit (d : int64) : Dimension = Dimension(Some(d))
    static member op_Implicit (d : int64 option) : Dimension = match d with | None -> Dimension.Unknown | _ -> Dimension(d)

    static member op_Explicit (d : Dimension) : int64 = match d.Value with | None -> -1L | Some(x) -> x

    // NOTE This does not seem to work??
    //static member op_Explicit (d : Dimension) : int32 = match d.Value with | None -> -1 | Some(x) -> int32 x

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
type TFShape(dims:Dimension[] option) =

    static let unknown = TFShape(Option<Dimension[]>.None)
    static let scalar = TFShape(Array.empty<int64>)
    new ([<ParamArray>] dims : int64[]) = TFShape(Some(dims |> Array.map Dimension))
    new ([<ParamArray>] dims : int[]) = TFShape(Some(dims |> Array.map Dimension))
    new (proto : TensorFlow.FSharp.Proto.TensorShapeProto) =
        if proto.UnknownRank then TFShape(Option<Dimension[]>.None) 
        else new TFShape(Some([|for x in proto.Dims -> Dimension(x.Size)|]))

    /// <summary>
    /// Represents an unknown number of dimensions in the tensor.
    /// </summary>
    /// <value>The unknown.</value>
    static member Unknown = unknown


    /// <summary>
    /// Gets the length of the specified dimension in the tensor
    /// </summary>
    /// <returns>The length, -1 for shapes that have an unknown dimension.</returns>
    /// <param name="dimension">Dimension.</param>
    member __.GetLength (dimension : int) = match dims with | None -> Dimension.Unknown | Some(dims) -> dims.[dimension]

    /// Returns the rank of this shape, or raises ValueError if unspecified
    member this.Length 
        with get() = 
            match dims with 
            | None -> raise (ValueError("Cannot take the length of Shape with unknown rank."))
            | Some(xs) -> xs.Length

    /// Returns `self.Dims` if the rank is known, otherwise raises ValueError.
    interface IEnumerable<Dimension> with
        member this.GetEnumerator() = 
            match dims with
            | None -> raise (ValueError("Cannot iterate over a shape with unknown rank."))
            | Some(dims) -> (dims |> Seq.ofArray).GetEnumerator()

    /// Returns `self.Dims` if the rank is known, otherwise raises ValueError.
    interface IEnumerable with
        member this.GetEnumerator() = (this :> IEnumerable<Dimension>).GetEnumerator() :> IEnumerator

    /// <summary>
    /// Number of dimensions represented by this shape.
    /// </summary>
    /// <value>The number dimensions, -1 if the number of dimensions is unknown, 0 if the shape represent a scalar, 1 for a vector, 2 for a matrix and so on..</value>
    member __.NumDimensions = match dims with | None -> -1 | Some(dims) -> dims.Length

    member this.TryFullyDefined
        with get() = 
            match dims with
            | Some(dims) ->
                match dims |> Array.choose (fun x -> x.Value) with 
                | xs when xs.Length = dims.Length -> Some(xs)
                | _ -> None
            | None -> None

    /// <summary>
    /// Gets a value indicating whether all the dimensions in the <see cref="T:TensorFlow.TFShape"/> are fully specified.
    /// </summary>
    /// <value><c>true</c> if is fully specified; otherwise, <c>false</c>.</value>
    member this.IsFullyDefined
        with get() = this.TryFullyDefined.IsSome

    /// <summary>Raises an exception if `this` is not fully defined in every dimension</summary>
    /// <exception>If `this` does not have a known value for every dimension</exception>
    member this.AssertIsFullyDefined() =
        if not(this.IsFullyDefined) then
            raise (ValueError(sprintf "Shape %O is not fully defined" this))

    /// <summary>
    /// Returns the shape as an array
    /// </summary>
    /// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
    member __.ToArray () =
        match dims with
        | Some dims -> dims
        | None -> null
    

    member __.Dims = dims |> Option.defaultValue [||] //|> Array.map (fun x -> match x.Value with | None -> -1L | Some(y) -> y)

    //TODO: perhaps use Value?? I'm not attached to the name
    member __.TryDims = dims

    /// <summary>
    /// Returns the shape as an array
    /// </summary>
    /// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
    member __.ToIntArray () = dims |> Option.map (fun xs -> xs |> Array.map (fun x -> int32 x)) |> Option.defaultValue null

    member __.ToLongArray() = dims |> Option.map (fun xs -> xs |> Array.map int64) |> Option.defaultValue null

    /// TODO: I'm not attached to the name, this name is derived from the Python equivalent
    member this.AsList() = 
        match this.TryDims with
        | None -> raise (ValueError("AsList() is not defined on an unknown TFShape")) // NOTE: this make less sense in F# than in Python
        | Some(dims) -> dims |> Array.map (fun x -> x.Value)

    /// Returns this shape as a `TensorShapeProto`.
    member this.AsProto() =
        match this.TryDims with
        | None -> TensorFlow.FSharp.Proto.TensorShapeProto(UnknownRank=true)
        | Some(dims) ->
            let shapeProto = TensorFlow.FSharp.Proto.TensorShapeProto()
            shapeProto.Dims.AddRange((dims |> Array.map (fun x -> match x.Value with | None -> Dim(Size = -1L) | Some(y) -> Dim(Size = y))))
            shapeProto

    /// <summary>
    /// Gets a value indicating whether one of the dimensions <see cref="T:TensorFlow.TFShape"/> in the shape is larger than Int32.MaxValue.
    /// </summary>
    /// <value><c>true</c> if is long array; otherwise, <c>false</c>.</value>
    member __.IsLongArray with get() = dims |> Option.map (Array.exists (fun x -> int64 x > int64 Int32.MaxValue)) |> Option.defaultValue false

    /// <summary>
    /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFShape"/>.
    /// </summary>
    /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFShape"/>.</returns>
    override __.ToString () =
        match dims with 
        | Some [|x|] -> sprintf "(%O,)" x 
        | Some dims -> dims |> Array.map string |> String.concat "," |> sprintf "(%s)"
        | None -> "<unknown>"


    /// <summary>
    ///  Returns the value of a dimension
    /// </summary>
    /// <param name="key"></param>
    /// <returns>A dimension</returns>
    /// <exception cref="System.ArguementException">If key is outside the range of the dimensions</exception>
    member this.Item (key:int) = 
        match dims with
        | Some(dims) -> dims.[key]
        | None -> Dimension.Unknown
    
//    """Returns the value of a dimension or a shape, depending on the key.
//
//    Args:
//      key: If `key` is an integer, returns the dimension at that index;
//        otherwise if `key` is a slice, returns a TensorShape whose
//        dimensions are those selected by the slice from `self`.
//
//    Returns:
//      A dimension if `key` is an integer, or a `TensorShape` if `key` is a
//      slice.
//
//    Raises:
//      ValueError: If `key` is a slice, and any of its elements are negative, or
//        if `self` is completely unknown and the step is set.
    /// <summary>
    ///  Returns the value of a dimension
    /// </summary>
    /// <param name="slice">Slice for dimensions</param>
    /// <returns>A dimension</returns>
    /// <exception cref="System.ArguementException">If key is outside the range of the dimensions</exception>
    /// TODO fix up above summary
    member this.GetSlice(start: int option, finish : int option) = 
        let start = start |> Option.defaultValue 0
        match finish with
        | None ->
            // NOTE(mrry) This would imply that TensorShape(None) is compatible with
            // TensorShape(None)[1:], which is obivously not true. It would be
            // possible to track the nubmer of dimensions symbolically,
            // and perhaps we should do that.
            TFShape.Unknown
        | Some(finish) -> 
            if (start < 0 || finish < 0) || start >= finish then TFShape.Unknown
            else TFShape.UnknownShape(ndims = finish - start)

    /// Returns the total number of elemetns, or none for incomplete shapes.
    member this.NumElements // TODO: should this e TryNumElements?
        with get() = 
            if this.IsFullyDefined  
            then Some(this.Dims |> Array.choose (fun x -> x.Value) |> Array.fold (*) 1L)
            else None

    ///<summary>Returns a `TensorShape` combining the information in `self` and `other`.
    ///
    ///The dimensions in `self` and `other` are merged elementwise,
    ///according to the rules defined for `Dimension.MergeWith()`.
    ///
    /// <param name="other">Another `TensorShape`.</param>
    /// <returns>A `TensorShape` containing the combined information of `this` and `other`.</returns>
    /// <exception cref="TensorFlow.FSharp.ValueError">If `this` and `other` are not compatible.</exception>
    /// </summary>
    member this.MergeWith(other : TFShape) = 
        match dims with 
        | None -> other
        | Some(dims) ->
            try
                TFShape(Some(dims |> Array.mapi (fun i dim -> dim.MergeWith(other.[i]))))
            with 
            | :? ValueError as e ->
                raise (ValueError(sprintf "Shapes %O and %O are not compatible" this other))
                

    /// <summary>Returns the concatenation of the dimension in `self` and `other`.
    ///
    ///*N.B.* If either `self` or `other` is completely unknown,
    ///concatenation will discard information about the other shape. In
    ///future, we might support concatenation that preserves this
    ///information for use with slicing.
    /// </summary>
    /// <param name="other"> Another TFShape</param>
    /// <returns> A `TensorShape` whose dimensions are the concatenation of the dimensions in `this` and `other`. </returns>
    member this.Concatentate(other : TFShape) = 
        // TODO(mrry): Handle the case where we concatenate a known shape with a 
        // completely unknow shape, so that we can use partial information
        match this.TryDims, other.TryDims with
        | None,_ | _ ,None -> TFShape.Unknown
        | Some(xs), Some(ys) -> TFShape(Some([|yield! xs; yield! ys|]))


    /// <summary> /// Raises and excpetion if `this` is not compatible with the given `rank`.
    /// </summary>
    /// <param name="rank">An integer</param>
    /// <excpetion cref="TensorFlow.FShapr.ValueError">If `this` does not represent a shape with the given `rank`</exception>
    member this.AssertHasRank(rank : int) = 
        // TODO: There is probably a cleaner way to do this...
        if this.TryDims |> Option.map (fun xs -> xs.Length) <> Some(rank) then
            raise (ValueError(sprintf "Shape %O must have rank %i" this rank))

    /// <summary>Returns a shape based on `self` with the given rank.
    ///
    /// This method promotes a completely unknown shape to one with a
    /// known rank.
    /// </summary>
    /// <param name="rank">An integer. </param>
    /// <returns>A shape that is at least as specific as `self` with the given rank. </returns
    /// <excpetion cref="TensorFlow.FShapr.ValueError">If `this` does not represent a shape with the given `rank`</exception>
    member this.WithRank(rank : int) = 
        try 
            this.MergeWith(TFShape.UnknownShape(ndims=rank))
        with
        | :? ValueError  -> raise (ValueError(sprintf "Shape %O must have rank %i" this rank))


    /// <summary>
    /// Returns a shape based on `this` with at least the given rank.
    /// </summary>
    /// <param name="rank">An integer</param>
    /// <returns>A shape that is at least as specific as `this` with at least the given rank.</returns>
    /// <excpetion cref="TensorFlow.FShapr.ValueError">If `this` does not represent a shape with at least the given `rank`</exception>
    member this.WithRankAtLeast(rank : int) : TFShape =
        match this.TryDims with
        | Some(xs) when xs.Length < rank ->
            raise (ValueError(sprintf "Shape %O must have rank at least %i" this rank))
        | _ -> ()
        this

    /// <summary>
    /// Returns a shape based on `this` with at most the given rank.
    /// </summary>
    /// <param name="rank">An integer</param>
    /// <returns>A shape that is at least as specific as `this` with at most the given rank.</returns>
    /// <excpetion cref="TensorFlow.FShapr.ValueError">If `this` does not represent a shape with at most the given `rank`</exception>
    member this.WithRankAtMost(rank : int) : TFShape =
        match this.TryDims with
        | Some(xs) when xs.Length > rank ->
            raise (ValueError(sprintf "Shape %O must have rank at most %i" this rank))
        | _ -> ()
        this

    /// <summary>Returns True iff `self` is compatible with `other`.
    ///
    /// Two possibly-partially-defined shapes are compatible if there
    /// exists a fully-defined shape that both shapes can represent. Thus,
    /// compatibility allows the shape inference code to reason about
    /// partially-defined shapes. For example:
    ///
    /// * TFShape(None) is compatible with all shapes.
    ///
    /// * TFShape([None, None]) is compatible with all two-dimensional
    ///   shapes, such as TFShape([32, 784]), and also TFShape(None). It is
    ///   not compatible with, for example, TFShape([None]) or
    ///   TFShape([None, None, None]).
    ///
    /// * TFShape([32, None]) is compatible with all two-dimensional shapes
    ///   with size 32 in the 0th dimension, and also TFShape([None, None])
    ///   and TFShape(None). It is not compatible with, for example,
    ///   TFShape([32]), TFShape([32, None, 1]) or TFShape([64, None]).
    ///
    /// * TFShape([32, 784]) is compatible with itself, and also
    ///   TFShape([32, None]), TFShape([None, 784]), TFShape([None,
    ///   None]) and TFShape(None). It is not compatible with, for example,
    ///   TFShape([32, 1, 784]) or TFShape([None]).
    ///
    /// The compatibility relation is reflexive and symmetric, but not
    /// transitive. For example, TFShape([32, 784]) is compatible with
    /// TFShape(None), and TFShape(None) is compatible with
    /// TFShape([4, 4]), but TFShape([32, 784]) is not compatible with
    /// TFShape([4, 4]).
    /// </summary>
    /// <param name="other"> Another TFShape.</param>
    /// <returns>True iff `this` is compatible with `other`.</returns>
    member this.IsCompatibleWith(other : TFShape) = 
        match this.TryDims, other.TryDims with
        | Some(xs),Some(ys) -> 
           if xs.Length <> ys.Length then false
           else (xs,ys) ||> Array.zip |> Array.map (fun (x,y) -> not(x.IsCompatibleWith(y))) |> Array.exists id
        | _ -> true

    /// <summary> Raises exception if `self` and `other` do not represent the same shape.
    ///
    /// This method can be used to assert that there exists a shape that both
    /// `self` and `other` represent.
    /// </summary>
    /// <param name="other">Another TFShape</param>
    /// <excpetion cref="TensorFlow.FShapr.ValueError">If `this` and `other` do not represent the same shape.</exception>
    member this.AssertIsCompatibleWith(other : TFShape) = 
        if not(this.IsCompatibleWith(other)) then
            raise (ValueError(sprintf "Shapes %O and %O are incompatible" this other))
        
    /// <summary>Returns the most specific TensorShape compatible with `self` and `other`.
    ///
    /// * TensorShape([None, 1]) is the most specific TensorShape compatible with
    ///   both TensorShape([2, 1]) and TensorShape([5, 1]). Note that
    ///   TensorShape(None) is also compatible with above mentioned TensorShapes.
    ///
    /// * TensorShape([1, 2, 3]) is the most specific TensorShape compatible with
    ///   both TensorShape([1, 2, 3]) and TensorShape([1, 2, 3]). There are more
    ///   less specific TensorShapes compatible with above mentioned TensorShapes,
    ///   e.g. TensorShape([1, 2, None]), TensorShape(None).
    /// </summary>
    /// <param name="other"> Another `TFShape`</param>
    /// <returns> A `TensorShape` which is the most specific compatible shape of `this` and `other`</returns>
    member this.MostSpecificCompatibleShape(other : TFShape) =
        match this.TryDims, other.TryDims with
        | Some(xs), Some(ys) when xs.Length = ys.Length -> 
            TFShape(Some([| for i in [0 .. this.NumDimensions - 1] -> if xs.[i] = ys.[i] then xs.[i] else Dimension.Unknown |]))
        | _ -> TFShape.Unknown

    /// <summary>Returns an unknown tensorShape, optionally with a known rank.</summary>
    /// <param name="ndims"> (Optional) If specified, the number of dimensions in the shape.</param>
    /// <returns>An unknown TensorShape.</returns>
    static member UnknownShape(?ndims : int) = 
        match ndims with
        | None -> TFShape.Unknown
        | Some(ndims) -> TFShape(Some(Array.init ndims (fun _ -> Dimension(Option<int64>.None))))



    /// <summary>
    /// Returns a shape representing a scalar
    /// </summary>
    static member Scalar = scalar 

    /// <summary>Returns a shape representing a vector</summary>
    /// <param name="length">The length of the vector, which mayu be None if unknown.</param>
    /// <returns>A TFShape representign a vector of the given length.</returns>
    static member Vector(?length : int64) =
        TFShape(Some([|Dimension(length)|]))

    /// <summary>Returns  a shape representing a matrix. </summary>
    /// <param name="rows">The number of rows in the matrix, which may be None if unknown.</param>
    /// <param name="cols">The number of columns in the matrix, which may be None if unknown.</param>
    static member Matrix(?rows : int64, ?cols : int64) =
        TFShape(Some([|Dimension(rows); Dimension(cols)|]))

    /// Returns True if `self` is equivalent to `other`.
    static member (=) (self : TFShape, other : TFShape) = self.TryDims = other.TryDims

    /// Returns True if `self` is knwon to be different form `other`.
    static member (<>) (self : TFShape, other : TFShape) = 
        match self.TryDims, other.TryDims with
        | None,_ | _,None ->  raise (ValueError("The inequality of unknown TFShapes is undefiend."))
        | _ -> self.TryDims <> other.TryDims
         

    /// <summary>
    /// Adds a <see cref="TensorFlow.FSharp.TFShape"/> to a <see cref="TensorFlow.FSharp.TFShape"/>, yielding a shape made up of the concatenation of the first and the second shapes.
    /// </summary>
    /// <param name="left">The first <see cref="TensorFlow.FSharp.TFShape"/> to add.</param>
    /// <param name="right">The second <see cref="TensorFlow.FSharp.TFShape"/> to add.</param>
    /// <returns>The <see cref="T:TensorFlow.TFShape"/> that is the sum of the values of <c>left</c> and <c>right</c>.</returns>
    static member (+) (left:TFShape,right:TFShape) = left.Concatentate(right)

    /// <summary>
    /// Returns the shape as a 1-dimensional tensor with each element corresponding to the specified shape dimension.
    /// </summary>
    /// <returns>The tensor.</returns>
    member this.AsTensor () = new TFTensor (this.ToIntArray ())

     /// <summary>
     /// Performs an implicit conversion from <see cref="TFShape"/> to <see cref="TFTensor"/>.
     /// </summary>
     /// <param name="shape">The shape.</param>
     /// <returns>The result of the conversion.</returns>
     static member op_Implicit (shape : TFShape) : TFTensor = shape.AsTensor ()


// Use for single dimension arrays 

[<AutoOpen>]
module TensorExtension =

    // extern void * TF_TensorData (const TF_Tensor *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern IntPtr TF_TensorData (TF_Tensor tensor)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern TF_Tensor TF_AllocateTensor (TFDataType dataType, IntPtr zeroDim, int num_dims, size_t len)

    type TFTensor with

        static member SetupTensor (dt : TFDataType, shape : TFShape, data : Array, start : int, count : int, size : int) : IntPtr =
            if box shape = null then raise (ArgumentNullException "shape")
            match shape.TryFullyDefined with
            | Some(dims) -> TFTensor.SetupTensor (dt, dims, data, start, count, size)
            | None -> failwith "Shape %s is not fully specified"

        // Convenience, should I add T[,] and T[,,] as more convenience ones?
        /// <summary>
        /// Creates a single-dimension tensor from a byte buffer.  This is different than creating a tensor from a byte array that produces a tensor with as many elements as the byte array.
        /// </summary>
        static member CreateString (buffer : byte []) : TFTensor =
          if box buffer = null then raise(ArgumentNullException ("buffer"))
          //
          // TF_STRING tensors are encoded with a table of 8-byte offsets followed by
          // TF_StringEncode-encoded bytes.
          //
          let size = TFString.TF_StringEncodedSize (UIntPtr(uint64 buffer.Length))
          let handle = TF_AllocateTensor (TFDataType.String, IntPtr.Zero, 0, UIntPtr((uint64(size) + 8uL)))
      
          // Clear offset table
          let dst = TF_TensorData (handle)
          Marshal.WriteInt64 (dst, 0L)
          use status = new TFStatus()
          use src = fixed &buffer.[0]
          TFString.TF_StringEncode (src, UIntPtr(uint64 buffer.Length), dst.Add(8) |> NativePtr.ofNativeInt<int8>, size, status.Handle) |> ignore
          if status.Ok then
              new TFTensor (handle)
          else box null :?> TFTensor

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
/// Use the static method <see cref="M:TensorFlow.TFLibrary.FromFile"/> to load a dynamic library.
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
