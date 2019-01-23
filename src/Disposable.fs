namespace TensorFlow

open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop

#nowarn "9"

module NativeBinding = 
    let [<Literal>] TensorFlowLibrary = "libtensorflow"
    let [<Literal>] TensorFlowLibraryGPU = "libtensorflowgpu"

[<AutoOpen>]
module Util = 
    type IntPtr with
        member this.GetStr() = Marshal.PtrToStringAnsi(this)

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
type LLBuffer =
    val mutable data : IntPtr
    val mutable length : UIntPtr
    val mutable data_deallocator : IntPtr

/// <summary>
/// Contains TensorFlow fundamental methods and utility functions.
/// </summary>
module TFCore =     
    let UseCPU = true

    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern IntPtr TF_Version()

    /// <summary>
    /// Returns the version of the TensorFlow runtime in use.
    /// </summary>
    /// <value>The version.</value>
    let Version() = TF_Version().GetStr()

    // extern size_t TF_DataTypeSize (TF_DataType dt)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern IntPtr TF_DataTypeSize (DType dt)

    /// <summary>
    /// Gets the size in bytes of the specified TensorFlow data type.
    /// </summary>
    /// <returns>The data type size.</returns>
    /// <param name="dt">Dt.</param>
    let GetDataTypeSize (dt:DType) = int64 (TF_DataTypeSize (dt))

    // extern TF_Buffer * TF_GetAllOpList ()
    //[<DllImport (NativeBinding.TensorFlowLibrary)>]
    //extern IntPtr TF_GetAllOpList ()

    /// <summary>
    /// Retrieves the ProtocolBuffer describing all of the available operations in
    /// the TensorFlow library in current use.
    /// </summary>
    /// <returns>The buffer contains a ProtocolBuffer encoded payload, you need a ProtocolBuffer reader to process the contents.</returns>
    //let GetAllOpList () : TFBuffer = new TFBuffer (TF_GetAllOpList ())

    let Init () = 
        if sizeof<IntPtr> = 4 then
            Console.Error.WriteLine (
                "The TensorFlow native libraries were compiled in 64 bit mode, you must run in 64 bit mode\n" +
                "With Mono, do that with mono --arch=64 executable.exe, if using an IDE like MonoDevelop,\n" +
                "Xamarin Studio or Visual Studio for Mac, Build/Compiler settings, make sure that " +
                "\"Platform Target\" has x64 selected.")
            raise(Exception("Requries 64 bit"))

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
    // extern TF_Buffer * TF_NewBufferFromString (const void *proto, size_t proto_len)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern LLBuffer* TF_NewBufferFromString (IntPtr proto, IntPtr proto_len)

    // extern TF_Buffer * TF_NewBuffer ()
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern LLBuffer* TF_NewBuffer ()

    // extern void TF_DeleteBuffer (TF_Buffer *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteBuffer (LLBuffer* buffer)

    // extern TF_Buffer TF_GetBuffer (TF_Buffer *buffer)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern LLBuffer TF_GetBuffer (LLBuffer* buffer)

    static let FreeBlockDelegate = TFBuffer.FreeBlock

    static let FreeBufferFunc = Marshal.GetFunctionPointerForDelegate<BufferReleaseFunc> (BufferReleaseFunc(fun x y -> FreeBlockDelegate(x,y)))

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
        buf.data <- buffer
        buf.length <- UIntPtr(uint64(size))
        if release = null then
            buf.data_deallocator <- IntPtr.Zero
        else
            buf.data_deallocator <- Marshal.GetFunctionPointerForDelegate (release)
        new TFBuffer (handle |> NativePtr.toNativeInt)

    [<MonoPInvokeCallback (typeof<BufferReleaseFunc>)>]
    static member internal FreeBlock (data:IntPtr, length:IntPtr) =
        Marshal.FreeHGlobal (data)

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
        TF_DeleteBuffer (handle |> NativePtr.ofNativeInt)

    /// <summary>
    /// Returns a byte array representing the data wrapped by this buffer.
    /// </summary>
    /// <returns>The array.</returns>
    member this.ToArray() =
        if (handle = IntPtr.Zero) then
            null
        else
            let lb = handle |> NativePtr.ofNativeInt<LLBuffer> |> NativePtr.read
            let result = Array.zeroCreate<byte> (int lb.length)
            Marshal.Copy (lb.data, result, 0, (int lb.length))
            result

    member internal this.LLBuffer : nativeptr<LLBuffer> =  handle |> NativePtr.ofNativeInt


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
and
    [<AbstractClass>]
    TFDisposable (handle:IntPtr) = 
    let mutable handle = handle
    //static do TFCore.Init ()

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
    member this.Dispose () = 
        this.Dispose(true)
        GC.SuppressFinalize (this)

    override this.Finalize () = this.Dispose(false)

    interface IDisposable with
        member this.Dispose () = this.Dispose ()
    
    static member internal ObjectDisposedException () =
        raise (ObjectDisposedException ("The object was disposed"))
    
    member (*internal*) this.Handle with set(x) = handle <- x and get() = handle


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
            this.NativeDispose (handle)
            this.Handle <- IntPtr.Zero

