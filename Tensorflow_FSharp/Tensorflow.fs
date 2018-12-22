// Port of TensorFlow.cs by Miguel de Icaza. 
// Once ported this file will likely be reorganized.

// TODO Make sure the attributes are float32 bit and not float64 bit


namespace Tensorflow

open System
open System.Runtime.InteropServices
open System.Text
open System.Globalization
open System.Linq
open DType
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


#nowarn "9"

module NativeBinding = 
    let [<Literal>] TensorFlowLibrary = "libtensorflow"
    let [<Literal>] TensorFlowLibraryGPU = "libtensorflowgpu"

[<AutoOpen>]
module Util = 
    type IntPtr with
        member this.GetStr() = Marshal.PtrToStringAnsi(this)

/// <summary>
/// Contains TensorFlow fundamental methods and utility functions.
/// </summary>
module TFCore =     
    let UseCPU = true

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern IntPtr TF_Version();

    do
        if sizeof<IntPtr> = 4 then
            Console.Error.WriteLine (
                "The TensorFlow native libraries were compiled in 64 bit mode, you must run in 64 bit mode\n" +
                "With Mono, do that with mono --arch=64 executable.exe, if using an IDE like MonoDevelop,\n" +
                "Xamarin Studio or Visual Studio for Mac, Build/Compiler settings, make sure that " +
                "\"Platform Target\" has x64 selected.");
            raise(Exception())
    /// <summary>
    /// Returns the version of the TensorFlow runtime in use.
    /// </summary>
    /// <value>The version.</value>
    let Version() = TF_Version().GetStr()

    // extern size_t TF_DataTypeSize (TF_DataType dt);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern IntPtr TF_DataTypeSize (DType dt);

    /// <summary>
    /// Gets the size in bytes of the specified TensorFlow data type.
    /// </summary>
    /// <returns>The data type size.</returns>
    /// <param name="dt">Dt.</param>
    let GetDataTypeSize (dt:DType) = int64 (TF_DataTypeSize (dt))


    // extern TF_Buffer * TF_GetAllOpList ();
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern IntPtr TF_GetAllOpList ();

    /// <summary>
    /// Retrieves the ProtocolBuffer describing all of the available operations in
    /// the TensorFlow library in current use.
    /// </summary>
    /// <returns>The buffer contains a ProtocolBuffer encoded payload, you need a ProtocolBuffer reader to process the contents.</returns>
    //let GetAllOpList () : TFBuffer = return new TFBuffer (TF_GetAllOpList ());

    let Init () = failwith "todo"

/// <summary>
/// Base class for many TensorFlow data types that provides a common idiom to dispose and
/// release resources associated with the native data types.   Generally, you do not need to use this.
/// </summary>
/// <remarks>
/// <para>
/// This implements the Dispose pattern in a reusable form for TensorFlow types.
/// </para>
/// <para>
/// Subclasses invoke the constructor with the handle that this will wrap, and must
/// override the NativeDispose method (internal) to release the associated resource.
/// </para>
/// </remarks>
[<AbstractClass>]
type TFDisposable (handle:IntPtr) = 
    let mutable handle = handle
    static do TFCore.Init ()

    new () = new TFDisposable(IntPtr.Zero)

    // Must be implemented in subclasses to dispose the unmanaged object, it does
    // not need to take care of zeroing out the handle, that is done by the Dispose
    // method inherited from TFDisposable
    abstract NativeDispose : IntPtr -> unit

    abstract Dispose : bool -> unit

    /// <summary>
    /// Dispose the specified object
    /// </summary>
    /// <param name="disposing">If set to <c>true</c> it means that this method was called from Dispose, otherwise from the finalizer.</param>
    default this.Dispose (disposing:bool) = 
        if disposing then
            if handle <> IntPtr.Zero then this.NativeDispose (handle)
            handle <- IntPtr.Zero
            
    /// <summary>
    /// Releases all resource used by the <see cref="T:TensorFlow.TFDisposable"/> object.
    /// </summary>
    /// <remarks>Call Dispose when you are finished using the <see cref="T:TensorFlow.TFDisposable"/>. The
    /// Dispose method leaves the <see cref="T:TensorFlow.TFDisposable"/> in an unusable state. After
    /// calling Dispose, you must release all references to the <see cref="T:TensorFlow.TFDisposable"/> so
    /// the garbage collector can reclaim the memory that the <see cref="T:TensorFlow.TFDisposable"/> was occupying.</remarks>
    member this.Dispose () = this.Dispose(true); GC.SuppressFinalize (this)
    override this.Finalize () = this.Dispose(false)

    interface IDisposable with
        member this.Dispose () = this.Dispose ()
    
    static member internal ObjectDisposedException () =
        raise (ObjectDisposedException ("The object was disposed"))
    
    member internal this.Handle with set(x) = handle <- x and get() = handle


/// <summary>
/// ase class for many TensorFlow data types that provides a common idiom to dispose and
/// release resources associated with the native data types and whose unmanaged resource
/// disposing can be called from a background thread (the finalizer).   Users do not 
/// need to deal with this class.
/// </summary>
/// <remarks>
/// Some object deletion APIs in TensorFlow can be invoked from a background thread, 
/// so the release methods are suitable to be invoked from the Finalizer thread, in
/// those scenarios, subclass from this class rather than the TFDisposable class.
/// </remarks>
[<AbstractClass>]
type TFDisposableThreadSafe(handle:IntPtr) =
    inherit TFDisposable(handle)

    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.TFDisposableThreadSafe"/> class.
    /// </summary>
    new() = new TFDisposableThreadSafe(IntPtr.Zero)

    /// <summary>
    /// Dispose the object, unlike the default implementat in TFDisposable, 
    /// this will release the unmanaged resources from a background thread.
    /// </summary>
    /// <param name="disposing">If set to <c>true</c> disposing.</param>
    override this.Dispose (disposing:bool) =
        if handle <> IntPtr.Zero then
            this.NativeDispose (handle);
            this.Handle <- IntPtr.Zero;


/// <summary>
/// This attribute can be applied to callback functions that will be invoked
/// from unmanaged code to managed code.
/// </summary>
/// <remarks>
/// <code>
/// [TensorFlow.MonoPInvokeCallback (typeof (BufferReleaseFunc))]
/// internal static void MyFreeFunc (IntPtr data, IntPtr length){..}
/// </code>
/// </remarks>
[<Sealed>]
type MonoPInvokeCallbackAttribute(t:Type) =  
    inherit Attribute()

[<StructLayout (LayoutKind.Sequential)>]
[<Struct>]
type internal LLBuffer =
    val mutable data : IntPtr
    val mutable length : size_t
    val mutable data_deallocator : IntPtr


/// <summary>
/// Signature of the method that is invoked to release the data.  
/// </summary>
/// <remarks>
/// Methods of this signature are invoked with the data pointer and the
/// lenght pointer when then TFBuffer no longer needs to hold on to the
/// data.  If you are using this on platforms with static compilation
/// like iOS, you need to annotate your callback with the MonoPInvokeCallbackAttribute,
/// like this:
/// 
/// <code>
/// [TensorFlow.MonoPInvokeCallback (typeof (BufferReleaseFunc))]
/// internal static void MyFreeFunc (IntPtr data, IntPtr length){..}
/// </code>
/// </remarks>
type BufferReleaseFunc = delegate of IntPtr * IntPtr -> unit
    
/// <summary>
/// Holds a block of data, suitable to pass, or retrieve from TensorFlow.
/// </summary>
/// <remarks>
/// <para>
/// Use the TFBuffer to blobs of data into TensorFlow, or to retrieve blocks
/// of data out of TensorFlow.
/// </para>
/// <para>
/// There are two constructors to wrap existing data, one to wrap blocks that are 
/// pointed to by an IntPtr and one that takes a byte array that we want to wrap.
/// </para>
/// <para>
/// The empty constructor can be used to create a new TFBuffer that can be populated
/// by the TensorFlow library and returned to user code.
/// </para>
/// <para>
/// Typically, the data consists of a serialized protocol buffer, but other data
/// may also be held in a buffer.
/// </para>
/// </remarks>
// TODO: the string ctor
// TODO: perhaps we should have an implicit byte [] conversion that just calls ToArray?
type TFBuffer private (handle:IntPtr) =
    inherit TFDisposable(handle) 
    // extern TF_Buffer * TF_NewBufferFromString (const void *proto, size_t proto_len);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern LLBuffer* TF_NewBufferFromString (IntPtr proto, IntPtr proto_len)

    // extern TF_Buffer * TF_NewBuffer ();
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern LLBuffer* TF_NewBuffer ()

    // extern void TF_DeleteBuffer (TF_Buffer *);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteBuffer (LLBuffer* buffer);

    // extern TF_Buffer TF_GetBuffer (TF_Buffer *buffer);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern LLBuffer TF_GetBuffer (LLBuffer* buffer);

    static let FreeBlockDelegate = TFBuffer.FreeBlock;

    static let FreeBufferFunc = Marshal.GetFunctionPointerForDelegate<BufferReleaseFunc> (BufferReleaseFunc(fun x y -> FreeBlockDelegate(x,y)));

    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> class.
    /// </summary>
    new () = new TFBuffer((TF_NewBuffer ()) |> NativePtr.toNativeInt)

    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by wrapping the unmanaged resource pointed by the buffer.
    /// </summary>
    /// <param name="buffer">Pointer to the data that will be wrapped.</param>
    /// <param name="size">The size of the buffer to wrap.</param>
    /// <param name="release">Optional, if not null, this method will be invoked to release the block.</param>
    /// <remarks>
    /// This constructor wraps the buffer as a the data to be held by the <see cref="T:TensorFlow.TFBuffer"/>,
    /// if the release parameter is null, then you must ensure that the data is not released before the TFBuffer
    /// is no longer in use.   If the value is not null, the provided method will be invoked to release
    /// the data when the TFBuffer is disposed, or the contents of the buffer replaced.
    /// </remarks>
    new (buffer : IntPtr, size : int64, release : BufferReleaseFunc) = 
        let handle = TF_NewBuffer ()
        let mutable buf = NativePtr.get handle 0  //handle |> NativePtr.ofNativeInt<LLBuffer>
        buf.data <- buffer;
        buf.length <- UIntPtr(uint64(size))
        if release = null then
            buf.data_deallocator <- IntPtr.Zero;
        else
            buf.data_deallocator <- Marshal.GetFunctionPointerForDelegate (release);
        new TFBuffer (handle |> NativePtr.toNativeInt)

    [<MonoPInvokeCallback (typeof<BufferReleaseFunc>)>]
    static member internal FreeBlock (data:IntPtr, length:IntPtr) =
        Marshal.FreeHGlobal (data);

    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by making a copy of the provided byte array.
    /// </summary>
    /// <param name="buffer">Buffer of data that will be wrapped.</param>
    /// <remarks>
    /// This constructor makes a copy of the data into an unmanaged buffer, 
    /// so the byte array is not pinned.
    /// </remarks>
    new (buffer : byte []) = new TFBuffer(buffer, 0, buffer.Length) 

    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by making a copy of the provided byte array.
    /// </summary>
    /// <param name="buffer">Buffer of data that will be wrapped.</param>
    /// <param name="start">Starting offset into the buffer to wrap.</param>
    /// <param name="count">Number of bytes from the buffer to keep.</param>
    /// <remarks>
    /// This constructor makes a copy of the data into an unmanaged buffer, 
    /// so the byte array is not pinned.
    /// </remarks>
    new (buffer : byte[], start : int, count : int) =
        if start < 0 || start >= buffer.Length then
            raise(ArgumentException ("start"))
        if count < 0 || count > buffer.Length - start then
            raise(ArgumentException ("count"))
        let handle = TF_NewBuffer ()
        let mutable buf = NativePtr.get handle 0  
        buf.data <- Marshal.AllocHGlobal (count)
        Marshal.Copy (buffer, start, buf.data, count)
        buf.length <- UIntPtr(uint64(count))
        buf.data_deallocator <- FreeBufferFunc
        new TFBuffer(handle |> NativePtr.toNativeInt)


    override this.NativeDispose (handle:IntPtr) =
        TF_DeleteBuffer (handle |> NativePtr.ofNativeInt);

    /// <summary>
    /// Returns a byte array representing the data wrapped by this buffer.
    /// </summary>
    /// <returns>The array.</returns>
    member this.ToArray() =
        if (handle = IntPtr.Zero) then
            null;
        else
            let lb = handle |> NativePtr.ofNativeInt<LLBuffer> |> NativePtr.read;
            let result = Array.zeroCreate<byte> (int lb.length)
            Marshal.Copy (lb.data, result, 0, (int lb.length))
            result;
    member internal this.LLBuffer : nativeptr<LLBuffer> =  handle |> NativePtr.ofNativeInt


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
    // extern size_t TF_StringEncode (const char *src, size_t src_len, char *dst, size_t dst_len, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern size_t TF_StringEncode (byte* src, size_t src_len, sbyte* dst, size_t dst_len, TF_Status status);

    // extern size_t TF_StringDecode (const char *src, size_t src_len, const char **dst, size_t *dst_len, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern size_t TF_StringDecode (sbyte* src, size_t src_len, IntPtr* dst, size_t* dst_len, TF_Status status);

    // extern size_t TF_StringEncodedSize (size_t len);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern size_t TF_StringEncodedSize (size_t len);

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
    // extern TF_Status * TF_NewStatus ();
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Status TF_NewStatus ();

    // extern void TF_DeleteStatus (TF_Status *);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteStatus (TF_Status status);

    // extern void TF_SetStatus (TF_Status *s, TF_Code code, const char *msg);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetStatus (TF_Status s, TFCode code, string msg);

    // extern TF_Code TF_GetCode (const TF_Status *s);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TFCode TF_GetCode (TF_Status s);

    // extern const char * TF_Message (const TF_Status *s);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_Message (TF_Status s);

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
            TF_GetCode (handle);


    /// <summary>
    /// Gets a human-readable status message.
    /// </summary>
    /// <value>The status message.</value>
    member this.StatusMessage = (TF_Message (handle)).GetStr ();

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
    member this.Ok = this.StatusCode = TFCode.Ok;

    /// <summary>
    /// Gets a value indicating whether this <see cref="T:TensorFlow.TFStatus"/> state has been set to an error.
    /// </summary>
    /// <value><c>true</c> if error; otherwise, <c>false</c>.</value>
    member this.Error = this.StatusCode <> TFCode.Ok;

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
    member internal this.CheckMaybeRaise (?incomingStatus : TFStatus, ?last : bool) = 
        let last = defaultArg last true
        match incomingStatus with
        | None ->
            if handle = IntPtr.Zero then
                Console.WriteLine ("oops")
            if this.StatusCode <> TFCode.Ok  then
                let e = new TFException (this.StatusMessage)
                if last then
                    this.Dispose ()
                raise e
            if last then
                this.Dispose ();
            true;
        | Some(_) ->
            this.StatusCode = TFCode.Ok;

    static member internal Setup (?incoming : TFStatus) = 
        match incoming with | None -> new TFStatus() | Some(x) -> x


/// <summary>
/// The session options object holds configuration options that you want to use during your session, like the TensorFlow target or the configuration.
/// </summary>
type TFSessionOptions(handle) =
    inherit TFDisposable(handle)
    // extern TF_SessionOptions * TF_NewSessionOptions ();
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_SessionOptions TF_NewSessionOptions ();

    // extern void TF_DeleteSessionOptions (TF_SessionOptions *);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteSessionOptions (TF_SessionOptions options);

    // extern void TF_SetTarget (TF_SessionOptions *options, const char *target);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetTarget (TF_SessionOptions options, string target);

    // extern void TF_SetConfig (TF_SessionOptions *options, const void *proto, size_t proto_len, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetConfig (TF_SessionOptions options, IntPtr proto, size_t proto_len, TF_Status status);

    /// <summary>
    /// Initializes a new instance of the <see cref="T:TensorFlow.TFSessionOptions"/> class.
    /// </summary>
    new() = new TFSessionOptions(TF_SessionOptions ())

    override this.NativeDispose (handle : IntPtr) = TF_DeleteSessionOptions (handle)


    /// <summary>
    /// Sets the target in options.
    /// </summary>
    /// <param name="target">target can be empty, a single entry, or a comma separated list of entries.
    /// Each entry is in one of the following formats: "local", ip:port, host:port.</param>
    member this.SetTarget (target : string) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("TFSessionOptions"))
        else TF_SetTarget (handle, target);


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
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("TFSessionOptions"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_SetConfig (handle, protoData, UIntPtr(uint32 length), cstatus.Handle);
        cstatus.CheckMaybeRaise (?incomingStatus=status);

/// Low-level: Enumeration describing the types of a metadata attribute
type TFAttributeType =
    /// The type of the attribute is a string
    | String = 0u
    /// The type of the attribute is an int.
    | Int = 1u
    /// The type of the attribute is a float
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
/// This is a low-level operation returned by the <see cref="M:TensorFlow.TFOperation.GetAttributeMetadata"/>.
/// This is included for completeness, but is not generally used from C#, as you have access to the high-level
/// bindings in the <see cref="T:TensorFlow.TFGraph"/> type.
/// </remarks>
[<StructLayout (LayoutKind.Sequential)>]
[<Struct>]
type TFAttributeMetadata =
    //byte isList;
    //val IsLbool IsList => isList != 0;
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
/// scalars, you can use the <see cref="P:TensorFlor.TFShape.Scalar"/> shape.
/// </para>
/// <para>
/// To create a 2-element vector, use:
/// new TFShape (2)
/// </para>
/// <para>
/// To create a 2x3 matrix, use:
/// new TFShape (2, 3)
/// </para>
/// <para>
/// To create a shape with an unknown number of elements, you can pass the value
/// -1.  This is typically used to indicate the shape of tensors that represent a
/// variable-sized batch of values.
/// </para>
/// <para>
/// To create a matrix with 4 columns and an unknown number of rows:
/// var batch = new TFShape (-1, 4)
/// </para>
/// </remarks>
/// <summary>
/// Initializes a new instance of the <see cref="T:TensorFlow.TFShape"/> class.
/// </summary>
/// <param name="args">This is a params argument, so you can provide multiple values to it.  
/// A null value means that this is an unknown shape, a single value is used to create a vector,
/// two values are used to create a 2-D matrix and so on.
/// </param>
/// <remarks>
/// 
/// </remarks>
type TFShape(dims:int64[] option) =
    new ([<ParamArray>] dims : int64[]) = TFShape(Some(dims))
    /// <summary>
    /// Represents an unknown number of dimensions in the tensor.
    /// </summary>
    /// <value>The unknown.</value>
    member this.Unknown = new TFShape ([||])

    /// <summary>
    /// This shape is used to represent scalar values.
    /// </summary>
    /// <value>The scalar.</value>
    member this.Scalar = new TFShape ([|0L|])

    /// <summary>
    /// Gets the length of the specified dimension in the tensor
    /// </summary>
    /// <returns>The length, -1 for shapes that have an unknown dimension.</returns>
    /// <param name="dimension">Dimension.</param>
    member this.GetLength (dimension : int) = match dims with | None -> -1L | Some(dims) -> dims.[dimension]

    /// <summary>
    /// Number of dimensions represented by this shape.
    /// </summary>
    /// <value>The number dimensions, -1 if the number of dimensions is unknown, 0 if the shape represent a scalar, 1 for a vector, 2 for a matrix and so on..</value>
    member this.NumDimensions = match dims with | None -> -1 | Some(dims) -> dims.Length

    /// <summary>
    /// Gets a value indicating whether all the dimensions in the <see cref="T:TensorFlow.TFShape"/> are fully specified.
    /// </summary>
    /// <value><c>true</c> if is fully specified; otherwise, <c>false</c>.</value>
    member this.IsFullySpecified  
        with get() = match dims with | Some(dims) when dims |> Array.exists ((=) -1L) |> not -> true | _ -> false

    /// <summary>
    /// Returns the shape as an array
    /// </summary>
    /// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
    member this.ToArray () =
        match dims with
        | Some(dims) -> dims
        | None -> null
    
    member this.Dims = dims

    /// <summary>
    /// Returns the shape as an array
    /// </summary>
    /// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
    member this.ToIntArray () = dims |> Option.map (Array.map int) |> Option.orDefault null

    /// <summary>
    /// Gets a value indicating whether one of the dimensions <see cref="T:TensorFlow.TFShape"/> in the shape is larger than Int32.MaxValue.
    /// </summary>
    /// <value><c>true</c> if is long array; otherwise, <c>false</c>.</value>
    member this.IsLongArray with get() = dims |> Option.map (Array.exists (fun x -> x > int64 Int32.MaxValue)) |> Option.orDefault false

    /// <summary>
    /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFShape"/>.
    /// </summary>
    /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFShape"/>.</returns>
    override this.ToString () =
        match dims with 
        | Some(dims) -> dims |> Array.map (function | -1L -> "?" | x -> string x) |> String.concat "," |> sprintf "[%s]"
        | None -> "unknown";

    /// <summary>
    /// Gets the dimensions for the specified index.
    /// </summary>
    /// <param name="idx">Index.</param>
    member this.Index (idx:int) = dims.Value.[idx]

    /// <summary>
    /// Adds a <see cref="TensorFlow.TFShape"/> to a <see cref="TensorFlow.TFShape"/>, yielding a shape made up of the concatenation of the first and the second shapes.
    /// </summary>
    /// <param name="left">The first <see cref="TensorFlow.TFShape"/> to add.</param>
    /// <param name="right">The second <see cref="TensorFlow.TFShape"/> to add.</param>
    /// <returns>The <see cref="T:TensorFlow.TFShape"/> that is the sum of the values of <c>left</c> and <c>right</c>.</returns>
    static member (+) (left:TFShape,right:TFShape) =
        TFShape ([|yield! left.Dims |> Option.collect; yield! right.Dims |> Option.collect |])

// TODO this should be done when Tensors are done
//     /// <summary>
//     /// Returns the shape as a 1-dimensional tensor with each element corresponding to the specified shape dimension.
//     /// </summary>
//     /// <returns>The tensor.</returns>
//     public TFTensor AsTensor ()
//     {
//         return new TFTensor (ToIntArray ());
//     }


// TODO this should be done when Tensors are done
//     /// <summary>
//     /// Performs an implicit conversion from <see cref="TFShape"/> to <see cref="TFTensor"/>.
//     /// </summary>
//     /// <param name="shape">The shape.</param>
//     /// <returns>The result of the conversion.</returns>
//     public static implicit operator TFTensor (TFShape shape)
//     {
//         return shape.AsTensor ();
//     }
// }


type TFGraph() =
    let x = 10

/// <summary>
/// Represents a computation node in the graph.  Tensorflow operations are attached to a <see cref="T:Tensorflow.TFGraph"/>.
/// </summary>
/// <remarks>
/// TFOperations are usually created by  invoking one of the methods in 
/// <see cref="T:Tensorflow.TFGraph"/>, but they can also be constructed
/// manually using the low-level <see cref="T:Tensorflow.TFOperationDesc"/> API.
/// </remarks>
and TFOperation(graph : TFGraph, handle : IntPtr)  =

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

    // extern void TF_OperationGetAttrFloat (TF_Operation *oper, const char *attr_name, float *value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_OperationGetAttrFloat (TF_Operation oper, string attr_name, float32* value, TF_Status status);

    // extern void TF_OperationGetAttrFloatList (TF_Operation *oper, const char *attr_name, float *values, int max_values, TF_Status *status);
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

    // Pointer to the graph, to keep it from collecting if there are TFOperations alive.
    member this.graph = graph

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
        cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
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
            Array.create n (fun i -> new TFOperation(graph,arr.[i]))

    member this.Device = (TF_OperationDevice (handle)).GetStr ();

    member this.GetAttributeMetadata (attrName : string, ?status : TFStatus) : TFAttributeMetadata =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
        let cstatus = TFStatus.Setup (?incoming=status);
        let x = TF_OperationGetAttrMetadata (handle, attrName, cstatus.Handle);
        cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise(?incomingStatus = status) |> ignore
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
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
        value

    member this.GetAttrFloat(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrname"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus);
        checkAttrType(attrName, metadata, TFAttributeType.Float, false);
        let mutable value = 0.0f;
        TF_OperationGetAttrFloat(handle, attrName, &&value, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
        value

    member this.GetAttrBool(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Bool, false)
        let mutable value : byte = 0uy
        TF_OperationGetAttrBool(handle, attrName, &&value, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
        values |> Array.map (Convert.ToBoolean)

    member this.GetAttrType(attrName : string, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException()
        if box attrName = null then raise(ArgumentNullException("attrName"))
        let cstatus = TFStatus.Setup(?incoming=status)
        let metadata = this.GetAttributeMetadata(attrName, cstatus)
        checkAttrType(attrName, metadata, TFAttributeType.Type, false)
        let mutable value = DType.Unknown
        TF_OperationGetAttrType(handle, attrName, &&value, cstatus.Handle);
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
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
        cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
        let returnValues = Array.zeroCreate<TFShape> (int metadata.ListSize)
        let mutable offset = 0;
        for i = 0 to int metadata.ListSize - 1 do
            let xs = Array.zeroCreate<int64> numDims.[i]
            for j = 0 to numDims.[i] - 1 do
                xs.[j] <- storage.[offset + j]
            returnValues.[i] <- new TFShape(xs)
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
       cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
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
       cstatus.CheckMaybeRaise(?incomingStatus=status) |> ignore
       rs

    // member this.GetAttrTensor(attrName : string, ?status : TFStatus) =
    //     if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
    //     let cstatus = TFStatus.Setup (?incoming=status)
    //     let metadata = this.GetAttributeMetadata(attrName, cstatus)
    //     checkAttrType(attrName, metadata, TFAttributeType.Tensor, false)
    //     let tensor = IntPtr.Zero : TF_Tensor
    //     TF_OperationGetAttrTensor(handle, attrName, &&tensor, cstatus.Handle)
    //     cstatus.CheckMaybeRaise (?incomingStatus=status)
    //     new TFTensor(tensor)

    //     // WARN: untested
    //     public TFTensor[] GetAttrTensorList(string attrName, TFStatus status = null)
    //     {
    //         if (handle == IntPtr.Zero)
    //             TFDisposable.ObjectDisposedException ();
    //         var cstatus = TFStatus.Setup (status);
    //         var metadata = GetAttributeMetadata(attrName, cstatus);
    //         if (metadata.TotalSize < 0)
    //             throw new Exception("Metadata Error");
    //         checkAttrType(attrName, metadata, TFAttributeType.Tensor, true);
    //         TFTensor[] returnValue = new TFTensor[metadata.ListSize];
    //         unsafe
    //         {
    //             TF_Tensor[] tensorPointers = new TF_Tensor[metadata.ListSize];
    //             fixed (TF_Tensor* tensorPointersF = &tensorPointers[0])
    //                 TF_OperationGetAttrTensorList(handle, attrName, tensorPointersF, (int) metadata.ListSize, cstatus.Handle);
    //             for (int i = 0; i < metadata.ListSize; i++)
    //             {
    //                 returnValue[i] = new TFTensor(tensorPointers[i]);
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
    /// Encodes the TFOperation as a protocol buffer payload
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
                else (this :> IComparable<TFOperation>).CompareTo(x :?> TFOperation);

    interface IComparable<TFOperation>
        with 
            member this.CompareTo(x : TFOperation) =
                if box x = null then -1
                else this.Name.CompareTo(x.Name);

    // /// <summary>
    // /// Returns the handle to the idx-th output of the operation.
    // /// </summary>
    // /// <param name="idx">Index of the output in the operation.</param>
    // member this.Item(idx:int) : TFOutput = TFOutput (this, idx);

