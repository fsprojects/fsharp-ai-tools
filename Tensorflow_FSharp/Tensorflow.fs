// Port of TensorFlow.cs by Miguel de Icaza. 
// Once ported this file will likely be reorganized.

// TODO Make sure the attributes are float32 bit and not float64 bit


namespace Tensorflow

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
type TFBuffer internal (handle:IntPtr) =
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
    
    member this.Handle = handle

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
    static member Unknown = new TFShape ([||])

    /// <summary>
    /// This shape is used to represent scalar values.
    /// </summary>
    /// <value>The scalar.</value>
    static member Scalar = new TFShape ([|0L|])

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


/// <summary>
/// A grouping of operations with defined inputs and outputs.
/// Once created and added to graphs, functions can be invoked by creating an
/// operation whose operation type matches the function name.
/// </summary>
type TFFunction internal (handle : IntPtr) =
    inherit TFDisposable (handle)

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern unit TF_FunctionToFunctionDef (IntPtr func, IntPtr buffer, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern unit TF_DeleteFunction (IntPtr handle);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_FunctionImportFunctionDef (byte* proto, IntPtr len, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_FunctionName (IntPtr handle);

    /// <summary>
    /// Write out a serialized representation of the function as a FunctionDef protocol message to the provided <paramref name="outputFuncDef"/>
    /// </summary>
    /// <param name="outputFuncDef">An allocated buffer where the function will be serialized.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.ToFunctionDef (outputFuncDef : TFBuffer, ?status : TFStatus) =
        if box outputFuncDef = null then raise(ArgumentNullException ("outputFuncDef"))
        let cstatus = TFStatus.Setup (?incoming=status);
        TF_FunctionToFunctionDef (handle, outputFuncDef.Handle, cstatus.Handle);
        cstatus.CheckMaybeRaise (?incomingStatus = status, last = false);

    override this.NativeDispose (handle : TF_Status) = TF_DeleteFunction (handle);


    /// <summary>
    /// Construct and return the function whose FunctionDef representation is
    /// serialized in <paramref name="proto"/> proto
    /// </summary>
    /// <returns>The function definition, or null on failure.</returns>
    /// <param name="proto">Array containing the serialized FunctionDef in a protocol buffer.</param>
    /// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
    member this.ImportFunctionDef (proto : byte [], ?status : TFStatus) : TFFunction  =
        if box proto = null then raise(ArgumentNullException ("proto"))
        let cstatus = TFStatus.Setup (?incoming=status)
        use p = fixed &proto.[0] 
        //let res = TF_FunctionImportFunctionDef (p, (IntPtr)proto.Length, cstatus.Handle);
        // TODO Double check that this function takes length as IntPtr
        let res = TF_FunctionImportFunctionDef (p, IntPtr(proto.Length), cstatus.Handle);
        if (not(cstatus.CheckMaybeRaise (?incomingStatus=status, last= false))) then
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

    // extern TF_Library * TF_LoadLibrary (const char *library_filename, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Library TF_LoadLibrary (string library_filename, TF_Status  status);

    // extern void TF_DeleteLibraryHandle (TF_Library *lib_handle);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteLibraryHandle (TF_Library lib_handle);

    // extern TF_Buffer TF_GetOpList (TF_Library *lib_handle);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern LLBuffer TF_GetOpList (TF_Library lib_handle);

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
        let h = TF_LoadLibrary (libraryFile, cstatus.Handle);
        cstatus.CheckMaybeRaise (?incomingStatus=status) |> ignore
        new TFLibrary (h);

    /// <summary>
    /// Retrieves the ProtocolBuffer describing the available operations in
    /// the loaded TensorFlow library.
    /// </summary>
    /// <returns>The buffer contains a ProtocolBuffer encoded payload, you need a ProtocolBuffer reader to process the contents.</returns>
    member this.GetOpList () : TFBuffer = new TFBuffer ((TF_GetOpList (handle)).data)


    override this.NativeDispose (handle : IntPtr) = TF_DeleteLibraryHandle (handle);

/// <summary>
/// Low-level TensorFlow operation builder
/// </summary>
/// <remarks>
/// <para>This is the low-level API that is used to create operations by manually specificying all
/// the parameters of an operation (inputs, outputs, attribute descriptions) that can then
/// be attached into a graph.
/// </para>
/// <para>
/// Generally, you will instead be using the methods surfaced in <see cref="T:TensorFlow.TFGraph"/> 
/// that surfaces a C# high-level API that has already been bound to the built-in TensorFlow
/// nodes.
/// </para>
/// <para>
/// You create instances bound to a graph, add inputs, attributes and so on, and when you are done
/// you can call the <see cref="FinishOperation"/> method that will turn this TFOperationDesc 
/// into a <see cref="T:TensorFlow.TFOperation"/>.
/// </para>
/// </remarks>
type TFOperationDesc private (graph : Graph, opType : string, name : string, handle : IntPtr) =
    inherit TFDisposable(handle)
    let mutable handle = handle

    // extern TF_OperationDescription * TF_NewOperation (TF_Graph *graph, const char *op_type, const char *oper_name);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_OperationDescription TF_NewOperation (TF_Graph graph, string opType, string oper_name);

    // extern void TF_AddInputList (TF_OperationDescription *desc, const TF_Output *inputs, int num_inputs);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddInputList (TF_OperationDescription desc, TF_Output [] inputs, int num_inputs);

    // extern void TF_SetDevice (TF_OperationDescription *desc, const char *device);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetDevice (TF_OperationDescription desc, string device);

    // extern void TF_AddInput (TF_OperationDescription *desc, TF_Output input);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddInput (TF_OperationDescription desc, TF_Output input);

    // extern void TF_AddControlInput (TF_OperationDescription *desc, TF_Operation *input);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_AddControlInput (TF_OperationDescription desc, TF_Operation input);

    // extern void TF_SetAttrString (TF_OperationDescription *desc, const char *attr_name, const void *value, size_t length);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrString (TF_OperationDescription desc, string attr_name, IntPtr value, size_t length);

    // extern void TF_SetAttrStringList (TF_OperationDescription *desc, const char *attr_name, const void *const *values, const size_t *lengths, int num_values);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrStringList (TF_OperationDescription desc, string attr_name, IntPtr [] values, UIntPtr [] lengths, int num_values);

    // extern void TF_ColocateWith (TF_OperationDescription *desc, TF_Operation *op);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_ColocateWith (TF_OperationDescription desc, TF_Operation op);

    // extern void TF_SetAttrInt (TF_OperationDescription *desc, const char *attr_name, int64_t value);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrInt (TF_OperationDescription desc, string attr_name, int64 value);
    // extern void TF_SetAttrIntList (TF_OperationDescription *desc, const char *attr_name, const int64_t *values, int num_values);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrIntList (TF_OperationDescription desc, string attr_name, int64 [] values, int num_values);
    // extern void TF_SetAttrFloat (TF_OperationDescription *desc, const char *attr_name, float value);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrFloat (TF_OperationDescription desc, string attr_name, float value);
    // extern void TF_SetAttrFloatList (TF_OperationDescription *desc, const char *attr_name, const float *values, int num_values);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrFloatList (TF_OperationDescription desc, string attr_name, float [] values, int num_values);

    // extern void TF_SetAttrBool (TF_OperationDescription *desc, const char *attr_name, unsigned char value);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrBool (TF_OperationDescription desc, string attr_name, byte value);

    // extern void TF_SetAttrBoolList (TF_OperationDescription *desc, const char *attr_name, const unsigned char *values, int num_values);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrBoolList (TF_OperationDescription desc, string attr_name, bool [] values, int num_values);

    // extern void TF_SetAttrType (TF_OperationDescription *desc, const char *attr_name, TF_DataType value);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrType (TF_OperationDescription desc, string attr_name, TFDataType value);

    // extern void TF_SetAttrTypeList (TF_OperationDescription *desc, const char *attr_name, const TF_DataType *values, int num_values);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTypeList (TF_OperationDescription desc, string attr_name, TFDataType [] values, int num_values);

    // extern void TF_SetAttrShape (TF_OperationDescription *desc, const char *attr_name, const int64_t *dims, int num_dims);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, int64 [] dims, int num_dims);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, IntPtr dims, int num_dims);

    // extern void TF_SetAttrShapeList (TF_OperationDescription *desc, const char *attr_name, const int64_t *const *dims, const int *num_dims, int num_shapes);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrShapeList (TF_OperationDescription desc, string attr_name, IntPtr[] dims, int[] num_dims, int num_shapes);

    // extern void TF_SetAttrTensorShapeProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTensorShapeProto (TF_OperationDescription desc, string attr_name, IntPtr proto, size_t proto_len, TF_Status status);

    // extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription *desc, const char *attr_name, const void *const *protos, const size_t *proto_lens, int num_shapes, TF_Status *status);
    //[DllImport (NativeBinding.TensorFlowLibrary)]
    //static extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription desc, string attr_name, void** protos, size_t* proto_lens, int num_shapes, TF_Status status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription desc, string attr_name, IntPtr[] protos, size_t* proto_lens, int num_shapes, TF_Status status);

    // extern void TF_SetAttrTensor (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *value, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTensor (TF_OperationDescription desc, string attr_name, TF_Tensor value, TF_Status status);

    // extern void TF_SetAttrTensorList (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *const *values, int num_values, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrTensorList (TF_OperationDescription desc, string attr_name, IntPtr [] values, int num_values, TF_Status status);
    // extern void TF_SetAttrValueProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrValueProto (TF_OperationDescription desc, string attr_name, IntPtr proto, size_t proto_len, TF_Status status);

    // extern TF_Operation * TF_FinishOperation (TF_OperationDescription *desc, TF_Status *status);
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Operation TF_FinishOperation (TF_OperationDescription desc, TF_Status status);

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_SetAttrFuncName (TF_OperationDescription desc, string attr_name, string value, IntPtr len);

    new (graph : Graph, opType : string, name: string) =
        handle = TF_NewOperation (graph.handle, opType, operName);
        TFOperationDesc(graph, opType, name,handle)

    override this.NativeDispose (handle : IntPtr) =
        // If you reach this, you never called FinishOperation
        printf "TFOperationDescription(%s,%s was never turned into an TFOperation" opType name

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
    member this.AddInput (input : Output) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        TF_AddInput (handle, input);
        this;


        /// <summary>
        /// Adds a series of inputs to the operation.
        /// </summary>
        /// <param name="inputs">Inputs, this is a params array for your convenience.</param>
    member this.AddInputs([<ParamAray>] inputs : Output []) =
            if (handle = IntPtr.Zero) then raise (ObjectDisposedException ("handle"))
            if not (box inputs = null || inputs.Length = 0) then
                TF_AddInputList (handle, inputs, inputs.Length)
            this;

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
    member this.AddControlInput (control : Operation) =
        if (handle = IntPtr.Zero) then raise (ObjectDisposedException ("handle"))
        if (box control = null) then raise (ArgumentNullException ("input"))
        TF_AddControlInput (handle, control.handle)
        this

    member this.ColocateWith (op : Operation) = 
        if (handle = IntPtr.Zero) then (raise (ObjectDisposedException ("handle")))
        if (box op = null) then raise (ArgumentNullException ("op"))
        TF_ColocateWith (handle, op.handle);
        this;

    member this.SetAttr (attrName : string, value : string) =
        if (handle = IntPtr.Zero) then raise (ObjectDisposedException ("handle"))
        if (box attrName = null) then raise (ArgumentNullException ("attrName"))
        let bytes = Encoding.UTF8.GetBytes (value)
        let buf = Marshal.AllocHGlobal (bytes.Length + 1);
        Marshal.Copy (bytes, 0, buf, bytes.Length)
        TF_SetAttrString (handle, attrName, buf, UIntPtr(bytes.Length))
        this


    member this.SetAttr (attrName : string, values : string []) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box attrName = null then raise (ArgumentNullException ("attrName"))
        if box values = null then raise (ArgumentNullException ("values"))
        let n = values.Length;
        let unmanaged,lengths = 
            Array.init n (fun i ->
                let bytes = Encoding.UTF8.GetBytes (values [i]);
                let buf = Marshal.AllocHGlobal (bytes.Length + 1);
                let bc = bytes.Length;
                Marshal.Copy (bytes, 0, buf, bc);
                (buf,(size_t)bc;)) |> Array.unzip
        TF_SetAttrStringList (handle, attrName, unmanaged, lengths, n);
        this


    member this.SetAttr (attrName : string, value : int64) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("attrName"))
        TF_SetAttrInt (handle, attrName, value);
        this;

    member this.SetAttr (attrName : string, values : int64[]) = 
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("attrName"))
        if box values = null then raise(ArgumentNullException ("values"))
        TF_SetAttrIntList (handle, attrName, values, values.Length);
        this;

    member this.SetAttr (attrName : string, value : float) =
        if handle = IntPtr.Zero then raise( ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        TF_SetAttrFloat (handle, attrName, value);
        this;

    member this.SetAttr (attrName : string, values : float[]) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("attrname"))
        if box values = null then raise(ArgumentNullException ("values"))
        TF_SetAttrFloatList (handle, attrName, values, values.Length);
        this


    member this.SetAttr (attrName : string, value : bool) =
        if handle = IntPtr.Zero then raise( ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        TF_SetAttrBool (handle, attrName, if value then 1uy else 0uy);
        this;


    member this.SetAttr (attrName : string, values : bool[]) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("handle"))
        if box values = null then raise(ArgumentNullException ("values"))
        TF_SetAttrBoolList (handle, attrName, values, values.Length);
        this

    member this.SetAttr (attrName : string, value : DataType) =
        if handle = IntPtr.Zero then raise( ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        TF_SetAttrType (handle, attrName, value)
        this;


    member this.SetAttr (attrName : string, [<ParamAray>] values : DataType[]) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("handle"))
        if box values = null then raise(ArgumentNullException ("values"))
        TF_SetAttrTypeList(handle, attrName, values, values.Length);
        this


    member this.SetAttr (attrName : string, shape : Shape) =
        if handle = IntPtr.Zero then raise( ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        if (box shape = null || box shape.dims = null) then
            TF_SetAttrType (handle, attrName, null, -1)
        else
            TF_SetAttrType (handle, attrName, shape.dims, shape.dims.Length)
        this

    // TODO (matt): this originally had the name this.SetAttrShape
    member this.SetAttr (attrName : string, shapeList : Shape []) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if attrName = null then raise(ArgumentNullException ("attrName"))
        if box shapeList = null then raise(ArgumentNullException ("shapeList"))
        let num_shapes = shapeList.Length
        let num_dims = Array.zeroCreate<int> shapeList.Length 
        let dims = Array.init num_shapes (fun i -> 
            num_dims.[i] <- shapeList.[i].NumDimensions
            let array = Marshal.AllocHGlobal(sizeof(int64) * shapeList.[i].dims.Length)
            Marshal.Copy(shapeList.[i].dims, 0, array, shapeList.[i].dims.Length)
            array)
        TF_SetAttrShapeList (handle, attrName, dims, num_dims, num_shapes);
        this

    member this.SetAttrTensorShapeProto (attrName : string, proto : IntPtr, protoLen : size_t, ?status : TFStatus) = 
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_SetAttrTensorShapeProto (handle, attrName, proto, protoLen, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incomingStatus=status)
        this


    // WARN: untested
    // TODO: consider why ShapeProt is IntPtr and ShapeProtoList is TFBuffer[]
    member this.SetAttrShapeProtoList (attrName : string, protos : TFBuffer[], ?status : TFStatus) =
        if handle = IntPtr.Zero then raise(ObjectDisposedException ("handle"))
        if box attrName = null then raise(ArgumentNullException ("attrName"))
        if box protos = null then raise(ArgumentNullException ("protos"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let lengths = Array.zeroCreate<size_t> protos.Length
        let unmanaged = Array.init protos.Length (fun i -> protos.[i].Handle)
        use protoLengths = fixed &lengths.[0]
        TF_SetAttrTensorShapeProtoList(handle, attrName, unmanaged, protoLengths, unmanaged.Length, cstatus.handle);
        // prevent finalization of managed TFBuffer
        GC.KeepAlive(protos);
        cstatus.CheckMaybeRaise (?incomingStatus=status)
        this

    member this.SetAttr (attrName : string, tensor : TFTensor, ?status : TFStatus) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box attrName = null then raise (ArgumentNullException ("attrName"))
        if box tensor = null then raise (ArgumentNullException ("tensor"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_SetAttrTensor (handle, attrName, tensor.handle, cstatus.handle);
        cstatus.CheckMaybeRaise (?incomingStatus=status);
        this;


    member this.SetAttr (attrName : string, tensors : TFTensor [], ?status : TFStatus) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        if box attrName = null then raise (ArgumentNullException ("attrName"))
        if box tensor = null then raise (ArgumentNullException ("tensors"))
        let cstatus = TFStatus.Setup (?incoming=status)
        let unmanaged = Array.init tensor.Length (fun i -> tensor.[i].Handle)
        TF_SetAttrTensorList (handle, attrName, unmanaged, unmanaged.Length, cstatus.handle)
        // prevent finalization of managed TFTensors
        GC.KeepAlive(tensor)
        cstatus.CheckMaybeRaise (?incomingStatus=status)
        this

    // WARN: untested
    member this.SetAttr(attrName : string, proto : TFProto, ?status : TFStatus) =
        if handle = IntPtr.Zero then TFDisposable.ObjectDisposedException ()
        if box attrName = null then raise(ArgumentNullException ("attrName"))
        if box proto = null then raise(ArgumentNullException ("proto"))
        let cstatus = TFStatus.Setup (?incoming=status)
        TF_SetAttrValueProto(handle, attrName, proto.LLBuffer.data, proto.LLBuffer.length, cstatus.Handle)
        cstatus.CheckMaybeRaise (?incomingStatus=status)
        this

    /// <summary>
    /// Turns the operation description into an actual operation in the graph.
    /// </summary>
    /// <returns>The operation on success, or null on error.</returns>
    /// <param name="status">Optional status, on failure the operation is not added to the graph.  If you pass null (the default), this operation throws on error conditions.</param>
    member this.FinishOperation (?status : TFStatus) =
        if handle = IntPtr.Zero then raise (ObjectDisposedException ("handle"))
        let cstatus = TFStatus.Setup (?incoming=status);
        let h = TF_FinishOperation (handle, cstatus.handle);
        cstatus.CheckMaybeRaise (?incomingStatus=status);
        handle <- IntPtr.Zero;
        GC.SuppressFinalize (this);
        match status with 
        | None | _ when status.Error -> null :?> TFOpeartion
        | _ -> new TFOperation (graph, h)

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