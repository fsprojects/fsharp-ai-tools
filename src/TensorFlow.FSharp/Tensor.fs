namespace TensorFlow.FSharp

open System
open System.Collections
open System.Collections.Generic
open System.Linq
open System.Numerics
open System.Runtime.InteropServices
open System.Text
type size_t = System.UIntPtr
open FSharp.NativeInterop
open Util

#nowarn "9"

/// <summary>
/// Signature that methods must conform to to be used to release memory that was passed to a manually allocated TFTensor
/// </summary>
type Deallocator = delegate of IntPtr * IntPtr * IntPtr -> unit

module TensorNative =
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    extern TF_Tensor TF_NewTensor (uint32 dataType, IntPtr zeroDims, int num_dims, IntPtr data, size_t len, Deallocator deallocator, IntPtr deallocator_arg)

/// <summary>
/// TFTensor holds a multi-dimensional array of elements of a single data type.
/// </summary>
/// <remarks>
/// <para>
/// You can create tensors with the various constructors in this class, or using
/// the implicit conversions from various data types into a TFTensor, including
/// the creation of tensors from simple constants (returning a tensor that reprensets
/// a scalar, that is, it is a 0D tensor), arrays (returning a tensor of a single
/// dimension, 1D) or arbitrary multidimensional arrays.
///</para>
/// <para>
///   Given a tensor, you can retrieve the number of dimensions in it via the
///   NumDims property, or you can retrieve the shape of a tensor, that is how many
///   elements on each dimension the tensor has, by fetching the Shape property.
/// </para>
/// <para>
/// The implicit conversions for basic types produce tensors of one dimesion with
/// a single element, while the implicit conversion from an array, expects a multi-dimensional
/// array that is converted into a tensor of the right dimensions.
/// </para>
/// <para>
/// The special "String" tensor data type that you will find in TensorFlow documentation
/// really represents a byte array.   You can create string tensors by using the <see cref="M:TensorFlow.TFTensor.CreateString"/> 
/// method that takes a byte array buffer as input.
/// </para>
/// <example>
/// <code>
///   TFTensor scalar = 1           // Creates a 0D tensor, for the integer value 1
///   int d = scalar.NumDims        // d will be equal to zero, as it is a 0D tensor
///   long [] shape = scalar.Shape   // returns an empty array, as it is a 0D tensor
///   
///   TFTensor list = new [] {1,2,3} // Creates a 1D tensor, or vector, for the values 1, 2, 3
///   d = list.NumDims              // d will be one
///   shape = list.Shape            // shape will be an array with a single value 3, representing that the dimension 0 has 3 elements
/// 
///                                  // Creates a 3D tensor, 
///   TFTensor cube = new [,,] { {{1,2,3},{4,5,6}}}
///   d = cube.NumDims               // d will be 3
///   shape = list.Shape             // shape will be [1,2,3] which is the shape of the above 3D array
/// </code>
/// </example>
/// </remarks>
type TFTensor internal (handle: IntPtr) =
    inherit TFDisposableThreadSafe(handle)

    // extern TF_Tensor * TF_NewTensor (TF_DataType, const int64_t *dims, int num_dims, void *data, size_t len, void (* deallocator)(void *, size_t, void *), void *deallocator_arg)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Tensor TF_NewTensor (uint32 dataType, int64 [] dims, int num_dims, IntPtr data, size_t len, Deallocator deallocator, IntPtr deallocator_arg)

    // extern TF_Tensor * TF_AllocateTensor (TF_DataType, const int64_t *dims, int num_dims, size_t len)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TF_Tensor TF_AllocateTensor (uint32 dataType, int64 [] dims, int num_dims, size_t len)

    // [<DllImport (NativeBinding.TensorFlowLibrary)>]
    // static extern TF_Tensor TF_AllocateTensor (uint32 dataType, IntPtr zeroDim, int num_dims, size_t len);

    // extern void TF_DeleteTensor (TF_Tensor *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern void TF_DeleteTensor (TF_Tensor tensor)

    // extern TF_DataType TF_TensorType (const TF_Tensor *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern TFDataType TF_TensorType (TF_Tensor tensor)

    // extern int TF_NumDims (const TF_Tensor *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int TF_NumDims (TF_Tensor tensor)

    // extern int64_t TF_Dim (const TF_Tensor *tensor, int dim_index)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern int64 TF_Dim (TF_Tensor tensor, int dim_index)

    // extern size_t TF_TensorByteSize (const TF_Tensor *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern size_t TF_TensorByteSize (TF_Tensor tensor)

    // extern void * TF_TensorData (const TF_Tensor *)
    [<DllImport (NativeBinding.TensorFlowLibrary)>]
    static extern IntPtr TF_TensorData (TF_Tensor tensor)

    static let FreeTensorDataDelegate: Deallocator = Deallocator(TFTensor.FreeTensorData)
    static let FreeTensorHandleDelegate: Deallocator  = Deallocator(TFTensor.FreeTensorHandle)

    static member SizeFromDimensionAndType (t: Type, dims: int64[]) =
        let tc = Type.GetTypeCode (t)
        let dt, size = 
            match tc with 
            | TypeCode.Boolean -> TFDataType.Bool, 1
            | TypeCode.SByte -> TFDataType.Int8, 1
            | TypeCode.Byte -> TFDataType.UInt8, 1
            | TypeCode.Int16 -> TFDataType.Int16, 2
            | TypeCode.UInt16 -> TFDataType.UInt16, 2
            | TypeCode.Int32 -> TFDataType.Int32, 4
            | TypeCode.Int64 -> TFDataType.Int64, 8
            | TypeCode.Single -> TFDataType.Float32, 4
            | TypeCode.Double -> TFDataType.Float64, 8
            // Check types that are not handled by the typecode
            | _ when t.IsAssignableFrom (typeof<Complex>) -> TFDataType.Complex128, 16
            | _ -> raise(ArgumentException(sprintf "The data type %O is not supported" tc))

        let totalSize = dims |> Array.fold (*) (int64 size)
        dt, totalSize

    static member CreateFromMultidimensionalArrays (array: Array) =
        let t = array.GetType().GetElementType ()
        let dims = [|for i = 0 to array.Rank - 1 do yield int64(array.GetLength (i)) |]
        let dt, totalSize = TFTensor.SizeFromDimensionAndType(t, dims)
        TFTensor.SetupMulti (dt, dims, array, totalSize)

    [<MonoPInvokeCallback (typeof<Deallocator>)>]
    static member FreeTensorData (data : IntPtr) (len : IntPtr) (closure : IntPtr) = 
        Marshal.FreeHGlobal (data)

    [<MonoPInvokeCallback (typeof<Deallocator>)>]
    static member FreeTensorHandle (data : IntPtr) (len : IntPtr) (closure : IntPtr) =
        let gch = GCHandle.FromIntPtr (closure)
        gch.Free ()

    // General purpose constructor, specifies data type and gets pointer to buffer
    // Is the default good, one where we let the user provide their own deallocator, or should we make a copy in that case?
    /// <summary>
    /// Low-level tensor constructor that creates a tensor from a buffer pointed to by an IntPtr.
    /// </summary>
    /// <param name="dataType">Specifies the data type held by the tensor, as well as how to interpret the provided data.</param>
    /// <param name="dims">Describes the tensor shape, an array that indicates .</param>
    /// <param name="data">Pointer to the raw data that will be used to initialize the tensor.</param>
    /// <param name="dataSize">The size of the data being passed in.</param>
    /// <param name="deallocator">Deallocator method, it is invoked when the tensor is destroyed to release the data pointed to by <paramref name="data"/>.   On platforms like iOS (or other static compilation platforms), yiou must annotate the method specified in the deallocator with a <see cref="T:TensorFlow.MonoPInvokeCallbackAttribute"/>.</param>
    /// <param name="deallocatorData">An optional argument of data that is passed to the deallocator method when the tensor is destroyed, you can use this to pass context information.</param>
    new (dataType: TFDataType, dims: int64 [], data: IntPtr, dataSize: size_t, deallocator: Deallocator, deallocatorData: IntPtr) =
        if box dims = null then raise (ArgumentNullException ("dims"))
        let handle = TF_NewTensor (uint32 dataType, dims, dims.Length, data, dataSize, deallocator, deallocatorData)
        new TFTensor (handle)


    override this.NativeDispose (handle: IntPtr) = TF_DeleteTensor (handle)

    /// <summary>
    /// Low-level: Creates an empty tensor of the specified type and shape, with the specified number of elements
    /// </summary>
    /// <param name="dataType">Data type.</param>
    /// <param name="dims">TFTensor shape.</param>
    /// <param name="size">Size in bytes of the tensor, this will be the actual memory allocated.</param>
    /// <remarks>
    /// It is the responsibility of the caller to ensure that the size is correct given the data type size
    /// and the tensor dimension specified in dims.
    /// </remarks>
    new (dataType: TFDataType, dims: int64 [], size: int64) =
      if dims = null then raise (ArgumentNullException ("dims"))
      let handle = TF_AllocateTensor (uint32 dataType, dims, dims.Length, unativeint size)
      new TFTensor (handle)


    /// <summary>
    /// Returns the data type for the tensor.
    /// </summary>
    /// <value>The type of the tensor.</value>
    member this.TFDataType: TFDataType = TF_TensorType (handle) 

    /// <summary>
    /// Returns the number of dimensions in the tensor.
    /// </summary>
    /// <remarks>
    /// For single-dimension tensors the return is 1, 2 dimensions is 2 and so on.
    /// </remarks>
    member this.NumDims: int= TF_NumDims (handle)

    /// <summary>
    /// Returns the number of elements on a specific dimension in the tensor.
    /// </summary>
    /// <returns>The tensor dimension.</returns>
    /// <param name="dimIndex">Dimension that you are querying.</param>
    /// <remarks>
    /// If you have a tensor of 3 elements by 5, represented by [3 5],
    /// the GetTensorDimension(0) will return 3, the GetTensorDimension(1)
    /// will return 5.
    /// </remarks>
    member this.GetTensorDimension (dimIndex: int) = TF_Dim (handle, dimIndex)

    member this.TensorByteSize: size_t = TF_TensorByteSize (handle)

    /// <summary>
    /// Returns a pointer to the raw data in the tensor.
    /// </summary>
    /// <remarks>
    /// The contents of the Data must be interpreted according to the type of the
    /// data as described by the DataType property.   The amount of data
    /// is given by the the TensorByteSize property.
    /// </remarks>
    member this.Data: IntPtr =  TF_TensorData (handle)

    /// <summary>
    /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
    /// </summary>
    /// <remarks>
    ///     An array of size 0 is used for constants, an array of size 1 is used
    ///     for single-dimension arrays, where the dimension is the value of the
    ///     first element.   And so on.
    /// </remarks>
    member this.Shape = [|for i = 0 to TF_NumDims (handle) - 1 do yield TF_Dim (handle, i)|]

    static member internal FetchSimpleObj (dt : TFDataType, data : obj) : obj =
        match dt with 
        | TFDataType.Float32 -> Convert.ToSingle (data) |> box
        | TFDataType.Float64 -> Convert.ToDouble (data) |> box
        | TFDataType.Int32   -> Convert.ToInt32 (data) |> box
        | TFDataType.UInt8   -> Convert.ToByte (data) |> box
        | TFDataType.Int16   -> Convert.ToInt16 (data) |> box
        | TFDataType.Int8    -> Convert.ToSByte (data) |> box 
        | TFDataType.String  -> raise(NotImplementedException())
        | TFDataType.Int64   -> Convert.ToInt64 (data) |> box
        | TFDataType.Bool    -> Convert.ToBoolean (data) |> box
        | TFDataType.UInt16  -> Convert.ToUInt16 (data) |> box
        | TFDataType.Complex128 -> data :?> Complex |> box
        | _ -> box null

    static member FetchSimple (dt : TFDataType, data : IntPtr ) : obj =
        match dt with 
        | TFDataType.Float32 -> data |> NativePtr.nativeIntRead<float32> |> box
        | TFDataType.Float64 -> data |> NativePtr.nativeIntRead<double> |> box
        | TFDataType.Int32   -> data |> NativePtr.nativeIntRead<int32> |> box
        | TFDataType.UInt8   -> data |> NativePtr.nativeIntRead<uint8> |> box
        | TFDataType.Int16   -> data |> NativePtr.nativeIntRead<int16>|> box
        | TFDataType.String  -> raise(NotImplementedException())
        | TFDataType.Int64   -> data |> NativePtr.nativeIntRead<int64> |> box
        | TFDataType.Bool    -> data |> NativePtr.nativeIntRead<bool> |> box
        | TFDataType.UInt16  -> data |> NativePtr.nativeIntRead<uint16> |> box
        | TFDataType.Complex128 -> data |> NativePtr.nativeIntRead<Complex>  |> box
        | _ -> box null

    //used to create multidementional arrays / tensor with a constant value
    static member Set (target: Array, dt: TFDataType, shape: int64 [], idx: int [], level: int, value: obj) =
        if level < shape.Length - 1 then
            idx.[level] <- 0
            while (idx.[level] < int shape.[level]) do
                TFTensor.Set (target, dt, shape, idx, level + 1, value)
                idx.[level] <- idx.[level] + 1
        else
            idx.[level] <- 0
            while (idx.[level] < int shape.[level]) do
                match dt with 
                | TFDataType.Float32
                | TFDataType.Float64
                | TFDataType.Int32
                | TFDataType.UInt32
                | TFDataType.Int16
                | TFDataType.Int8
                | TFDataType.Int64
                | TFDataType.Bool
                | TFDataType.Complex128 -> target.SetValue (value, idx)
                | TFDataType.String -> 
                    raise(NotImplementedException ("String decoding not implemented for tensor vecotrs yet"))
                | _ -> raise(NotImplementedException ())
                idx.[level] <- idx.[level] + 1

    static member Copy(src: IntPtr,  target: voidptr, size: int) =
        Buffer.MemoryCopy (src |> NativePtr.intPtrToVoidPtr, target, uint64 size, uint64 size)

    static member Copy(target: Array, dt: TFDataType, shape: int64 [], idx: int [], level: int, data: byref<IntPtr>) =
        if level < shape.Length - 1 then
            idx.[level] <- 0
            while (idx.[level] < int shape.[level]) do
                TFTensor.Copy (target, dt, shape, idx, level + 1, &data)
                idx.[level] <- idx.[level] + 1
        else
            for i = 0 to int shape.[level] - 1 do
                idx.[level] <- i
                let offset = dt.ByteSize
                match dt with 
                | TFDataType.Float32 ->
                    target.SetValue (data |> NativePtr.nativeIntRead<float32>, idx)
                | TFDataType.Float64 ->
                    target.SetValue (data |> NativePtr.nativeIntRead<double>, idx)
                | TFDataType.Int32 ->
                    target.SetValue (data |> NativePtr.nativeIntRead<int32>, idx)
                | TFDataType.UInt32 ->
                    target.SetValue (data |> NativePtr.nativeIntRead<uint32>, idx)
                | TFDataType.Int16 ->
                    target.SetValue (data |> NativePtr.nativeIntRead<int16>, idx)
                | TFDataType.Int8 ->
                    target.SetValue (data |> NativePtr.nativeIntRead<int8>, idx)
                | TFDataType.Int64 ->
                    target.SetValue (data |> NativePtr.nativeIntRead<int64>, idx)
                | TFDataType.Bool ->
                    target.SetValue (data |> NativePtr.nativeIntRead<bool>, idx)
                | TFDataType.Complex128 -> 
                    target.SetValue (data |> NativePtr.nativeIntRead<Complex>, idx)
                | TFDataType.String -> 
                    raise(NotImplementedException ("String decoding not implemented for tensor vecotrs yet"))
                | _ ->
                    raise(NotImplementedException ())
                data <- data.Add(offset)

    static member FetchFlatArray (target: Array, dt: TFDataType, data: IntPtr) =
        let len = target.Length
        let size = dt.ByteSize
        match dt, target with 
        | TFDataType.Int8, (:? array<int8> as asbyte) -> 
            use p = fixed &asbyte.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.Bool, (:? array<bool> as abool) ->
            use p = fixed &abool.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.UInt16, (:? array<uint16> as ushort) ->
            use p = fixed &ushort.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.Complex128, (:? array<Complex> as acomplex) ->
            use p = fixed &acomplex.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.Float32, (:? array<float32> as afloat) ->
            use p = fixed &afloat.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.Float64, (:? array<double> as adouble) ->
            use p = fixed &adouble.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.Int32, (:? array<int32> as aint) ->
            use p = fixed &aint.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.UInt8, (:? array<uint8> as abyte) ->
            use p = fixed &abyte.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.Int16, (:? array<int16> as ashort) ->
            use p = fixed &ashort.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.Int64, (:? array<int64> as along) ->
            use p = fixed &along.[0]
            TFTensor.Copy (data, (p |> NativePtr.toVoidPtr), len * size)
        | TFDataType.String ,_-> 
           // need to return an array of TFStrings []
           raise (NotImplementedException ())
        | _ -> raise (NotImplementedException ())
        


    static member FetchJaggedArray (t: Type, dt: TFDataType, data: IntPtr ref, shape: int64 [], ?level: int ) =
        let level = defaultArg level 0
        let size = dt.ByteSize
        // If we are at the last node
        if level = shape.Length - 1 then
            let target = Array.CreateInstance (t, int(shape.[level]))
            for  l in  0L .. shape.[level] - 1L do
                match dt with
                | TFDataType.Float32 ->  target.SetValue((!data) |> NativePtr.nativeIntRead<float32>,l)
                | TFDataType.Float64 ->  target.SetValue((!data) |> NativePtr.nativeIntRead<double>,l)
                | TFDataType.Int32 ->    target.SetValue((!data) |> NativePtr.nativeIntRead<int32>,l)
                | TFDataType.UInt8 ->    target.SetValue((!data) |> NativePtr.nativeIntRead<uint8>,l)
                | TFDataType.Int16 ->    target.SetValue((!data) |> NativePtr.nativeIntRead<int16>,l)
                | TFDataType.Int8 ->     target.SetValue((!data) |> NativePtr.nativeIntRead<int8>,l)
                | TFDataType.Int64 ->        target.SetValue((!data) |> NativePtr.nativeIntRead<int64>,l)
                | TFDataType.Bool ->         target.SetValue((!data) |> NativePtr.nativeIntRead<bool>,l)
                | TFDataType.Complex128 ->   target.SetValue((!data) |> NativePtr.nativeIntRead<Complex>,l)
                | TFDataType.String -> raise(NotImplementedException ("String decoding not implemented for tensor vecotrs yet"))
                | _ -> raise(NotImplementedException ())
                data:= (!data).Add(size)
            target
        else
            let mutable target = None
            let top = shape.[level]
            if top < int64 Int32.MaxValue then
                for i = 0 to int(top) - 1 do
                    let childArray = TFTensor.FetchJaggedArray (t, dt, data, shape, level + 1)
                    if target.IsNone then target <- Some(Array.CreateInstance (childArray.GetType (), shape.[level]))
                    target.Value.SetValue (childArray, i)
            else
                for  l in 0L .. shape.[level] - 1L do
                    let childArray = TFTensor.FetchJaggedArray (t, dt, data, shape, level + 1)
                    if target.IsNone then target <- Some(Array.CreateInstance (childArray.GetType (), shape.[level]))
                    target.Value.SetValue (childArray, l)
            target.Value


    static member FetchMultiDimensionalArray (target : Array, dt : TFDataType, data : IntPtr, shape : int64 []) =
        let idx = Array.zeroCreate<int> shape.Length
        for i = 0 to shape.Length - 1 do
            if (shape.[i] > int64(Int32.MaxValue)) then raise(ArgumentOutOfRangeException ("Shape can not be longer than 32 bits"))
        TFTensor.Copy (target, dt, shape, idx, 0, ref data)

    /// <summary>
    /// Returns the value of the TFTensor as a C# type if possible, or null if the data type can not be represented in C#
    /// </summary>
    /// <param name="jagged">
    /// The default is set to false, which returns .NET multi-dimensional arrays for multi-dimensional
    /// tensors.    This is useful to feed the data back as a TFTensor created from an array.   Set to
    /// true if you want to get arrays pointing to arrays, which are slightly more convenient to work
    /// with from C#
    /// </param>
    /// <remarks>
    /// Jagged arrays create various intermediate arrays, while multi-dimensional arrays are more
    /// efficient memory-wise.
    /// </remarks>
    /// <returns>The value encodes the contents of the tensor, and could include simple values, arrays and multi-dimensional values.</returns>
    member this.GetValue(?jagged: bool) : obj =
        if this.NumDims = 0 then
            TFTensor.FetchSimple (this.TFDataType, this.Data)
        else
            match this.TFDataType.ToType with
            | null -> null
            | t ->
                let jagged = defaultArg jagged false
                if this.NumDims = 1 then
                    let result = Array.CreateInstance (t, this.Shape.[0])
                    TFTensor.FetchFlatArray (result, this.TFDataType, this.Data)
                    box result
                else
                    if jagged then
                        box (TFTensor.FetchJaggedArray (t, this.TFDataType, ref this.Data, this.Shape))
                    else
                        let result = Array.CreateInstance (t, this.Shape)
                        TFTensor.FetchMultiDimensionalArray (result, this.TFDataType, this.Data, this.Shape)
                        box result

    /// <summary>
    /// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFTensor"/>.
    /// </summary>
    /// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFTensor"/>.</returns>
    override this.ToString () : string =
        let n = this.NumDims
        if n = 0 then this.GetValue().ToString ()
        else
            let sb = new StringBuilder ("shape [")
            for i = 0 to n - 1 do
              sb.Append (TF_Dim (handle, i)) |> ignore
              if i + 1 < n then
                sb.Append("x") |> ignore
            sb.Append ("]") |> ignore
            sb.ToString ()
  
    static member GetLength (array: Array, ?deep: bool, ?max: bool) : int[] =
      let deep = defaultArg deep true
      let max  = defaultArg max false
      // This function gets the length of all dimensions in a multidimensional, jagged, or mixed array.
      // https://github.com/accord-net/framework/blob/b4990721a61f03602d04c12b148215c7eca1b7ac/Sources/Accord.Math/Matrix/Matrix.Construction.cs#L1118
      // Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
      if array.Rank = 0 then [||]
      elif deep && TFTensor.IsJagged(array) then
          if array.Length = 0 then [||]
          else
            let rest =
                if not(max) then TFTensor.GetLength (array.GetValue (0) :?> Array, deep)
                else
                  // find the max
                  let mutable rest = TFTensor.GetLength (array.GetValue (0) :?> Array, deep)
                  for i = 1 to array.Length - 1 do
                        let r = TFTensor.GetLength (array.GetValue(i) :?> Array, deep)
                        for j = 0 to r.Length - 1 do
                          if r.[j] > rest.[j] then
                            rest.[j] <- r.[j]
                  rest
            [| yield array.Length; yield! rest |] // not sure about this
      else Array.init array.Rank (fun i -> array.GetUpperBound (i) + 1)

    static member deepFlatten (array: Array) : Array =
      // This function converts multidimensional, jagged, or mixed arrays into a single unidimensional array (i.e. flattens the mixed array).
      // https://github.com/accord-net/framework/blob/f78181b82eb6ee6cc7fd10d2a7a55334982c40df/Sources/Accord.Math/Matrix/Matrix.Common.cs#L1625
      // Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
      let totalLength = TFTensor.GetTotalLength (array, deep = true)
      let elementType = TFTensor.GetInnerMostType (array)
      let result = Array.CreateInstance (elementType, totalLength)
      let mutable k = 0
      for v in TFTensor.enumerateJagged (array) do
        result.SetValue (v, k)
        k <- k + 1
      result
  
    static member enumerateJagged (array: Array) : IEnumerable =
          // This function can enumerate all elements in a multidimensional ,jagged, or mixed array.
          // From https://github.com/accord-net/framework/blob/b4990721a61f03602d04c12b148215c7eca1b7ac/Sources/Accord.Math/Matrix/Jagged.Construction.cs#L1202
          // Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
          let arrays = new Stack<Array> ()
          let counters = new Stack<int> ()
  
          arrays.Push (array)
          counters.Push (0)
  
          let mutable depth = 1
          let mutable a = array
          let mutable i = 0
  
          [|
              while arrays.Count > 0 do
                if (i >= a.Length) then
                  a <- arrays.Pop ()
                  i <- counters.Pop () + 1
                  depth <- depth - 1
                else
                  let e = a.GetValue (i)
                  match e |> Option.ofType<Array> with
                  | None -> 
                    yield e 
                    i <- i + 1
                  | Some next ->
                    arrays.Push (a)
                    counters.Push (i)
                    a <- next
                    i <- 0
                    depth <- depth + 1
          |]:> IEnumerable
  
    static member GetTotalLength (array: Array, ?deep: bool, ?rectangular: bool) : int =
        // From https://github.com/accord-net/framework/blob/b4990721a61f03602d04c12b148215c7eca1b7ac/Sources/Accord.Math/Matrix/Matrix.Construction.cs#L1087
        // Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
        let deep = defaultArg deep true
        let rectangular = defaultArg rectangular true
        if deep && TFTensor.IsJagged(array.GetType ()) then
          if rectangular then
            let rest = TFTensor.GetTotalLength (array.GetValue(0) :?> Array, deep)
            array.Length * rest
          else 
            let mutable sum = 0
            for i = 0 to array.Length - 1 do
              sum <- sum + TFTensor.GetTotalLength (array.GetValue(i) :?> Array, deep)
            sum
        else array.Length
  
    static member IsJagged (array: Array) = 
        // From https://github.com/accord-net/framework/blob/f78181b82eb6ee6cc7fd10d2a7a55334982c40df/Sources/Accord.Math/Matrix/Matrix.Construction.cs#L1204
        // Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
        if (array.Length = 0) then array.Rank = 1
        else array.Rank = 1 && (array.GetValue (0) |> isAssignableTo<Array>)
  
    static member IsJagged (_type: Type) =
       // From https://github.com/accord-net/framework/blob/eb371fbc540a41c1a711b6ab1ebd49889316e7f7/Sources/Accord.Math/Matrix/Matrix.Common.cs#L84
       // Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
        _type.IsArray && _type.GetElementType().IsArray
  
    static member GetInnerMostType (array: Array) : Type =
       // From https://github.com/accord-net/framework/blob/eb371fbc540a41c1a711b6ab1ebd49889316e7f7/Sources/Accord.Math/Matrix/Matrix.Common.cs#L95
       // Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
         let mutable _type = array.GetType ()
         while _type.IsArray do
           _type <- _type.GetElementType ()
         _type

// Factory methods to create tensors from a constant

     /// <summary>
     /// Converts an integer into a 1-dimensional, 1-valued tensor.
     /// </summary>
     /// <returns>The tensor representing the integer value.</returns>
     /// <param name="value">Value to initialize the tensor with.</param>
     static member op_Implicit(value: int) = new TFTensor (value)

     /// <summary>
     /// Converts a boolean into a 1-dimensional, 1-valued tensor.
     /// </summary>
     /// <returns>The tensor representing the integer value.</returns>
     /// <param name="value">Value to initialize the tensor with.</param>
     static member op_Implicit(value: bool) = new TFTensor (value)

     /// <summary>
     /// Converts a long into a 1-dimensional, 1-valued tensor.
     /// </summary>
     /// <returns>The tensor representing the long value.</returns>
     /// <param name="value">Value to initialize the tensor with.</param>
     static member op_Implicit(value: int64) = new TFTensor (value)

     /// <summary>
     /// Converts a double into a 1-dimensional, 1-valued tensor.
     /// </summary>
     /// <returns>The tensor representing the double value.</returns>
     /// <param name="value">Value to initialize the tensor with.</param>
     static member op_Implicit(value: double) = new TFTensor (value)

     /// <summary>
     /// Converts a float32 into a 1-dimensional, 1-valued tensor.
     /// </summary>
     /// <returns>The tensor representing the float32 value.</returns>
     /// <param name="value">Value to initialize the tensor with.</param>
     static member op_Implicit(value: float32) = new TFTensor (value)

     /// <summary>
     /// Converts a Complex number into a 1-dimensional, 1-valued tensor.
     /// </summary>
     /// <returns>The tensor representing the complex value.</returns>
     /// <param name="value">Value to initialize the tensor with.</param>
     static member op_Implicit(value: Complex) = new TFTensor (value)

     /// <summary>
     /// Converts a byte into a 1-dimensional, 1-valued tensor.
     /// </summary>
     /// <returns>The tensor representing the byte value.</returns>
     /// <param name="value">Value to initialize the tensor with.</param>
     static member op_Implicit(value: byte) = new TFTensor (value)

    /// <summary>
    /// Creates a 1 dimensional tensor from an array of booleans.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: bool []) = new TFTensor(TFTensor.SetupTensor (TFDataType.Bool, data, size = 1))

    /// <summary>
    /// Creates a 1 dimensional tensor from an array of sbytes.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: int8[]) = new TFTensor(TFTensor.SetupTensor (TFDataType.Int8, data, size = 1))
    /// <summary>
    /// Creates a 1 dimensional tensor from an array of bytes.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: byte []) = new TFTensor(TFTensor.SetupTensor (TFDataType.UInt8, data, size = 1))
    /// <summary>
    /// Creates a 1 dimensional tensor from an array of shorts.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: int16[]) = new TFTensor(TFTensor.SetupTensor (TFDataType.Int16, data, size = 2))
    /// <summary>
    /// Creates a 1 dimensional tensor from an array of ushorts
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: uint16[]) = new TFTensor(TFTensor.SetupTensor (TFDataType.UInt16, data, size = 2))
    /// <summary>
    /// Creates a 1 dimensional tensor from an array of ints.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: int []) = new TFTensor(TFTensor.SetupTensor (TFDataType.Int32, data, size = 4))
    /// <summary>
    /// Creates a 1 dimensional tensor from an array of floats.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: float32 []) = new TFTensor(TFTensor.SetupTensor (TFDataType.Float32, data, size = 4))
    /// <summary>
    /// Creates a 1 dimensional tensor from an array of doubles.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: double []) = new TFTensor(TFTensor.SetupTensor (TFDataType.Float64, data, size = 8))
    /// <summary>
    /// Creates a 1 dimensional tensor from an array of longs.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: int64 [] ) = new TFTensor(TFTensor.SetupTensor (TFDataType.Int64, data, size = 8))
    /// <summary>
    /// Creates a 1 dimensional tensor from an array of complex numbers.
    /// </summary>
    /// <param name="data">Data.</param>
    new (data: Complex []) = new TFTensor(TFTensor.SetupTensor (TFDataType.Complex128, data, size = 16))
//
//
// TODO: Other overloads we could add: String, Complex (float32), Bool, QInt8, QUInt8, QInt32, Bfloat16,
// QInt16, QUint16, Half, Resource
// TODO: not clear that this is very useful (the dims versions), perhaps to reduce the surface of
// construcors these rarer blobs should be "FromSpec" or something like that
    /// <summary>
    /// Creates a new tensor from a portion of an array of sbytes
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: int8[], start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.Int8, shape, data, start, count, size = 1))
    
    /// <summary>
    /// Creates a new tensor from a portion of an array of bytes
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: byte[] , start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.UInt8, shape, data, start, count, size = 1))
    
    /// <summary>
    /// Creates a new tensor from a portion of an array of shorts
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: int16[], start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.Int16, shape, data, start, count, size = 2))
    
    /// <summary>
    /// Creates a new tensor from a portion of an array of ushorts
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: uint16[], start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.UInt16, shape, data, start, count, size = 2))
  
    /// <summary>
    /// Creates a new tensor from a portion of an array of ints
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: int32[], start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.Int32, shape, data, start, count, size = 4))
    
    /// <summary>
    /// Creates a new tensor from a portion of an array of floats
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: float32[], start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.Float32, shape, data, start, count, size = 4))
    
    /// <summary>
    /// Creates a new tensor from a portion of an array of doubles
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: double[], start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.Float64, shape, data, start, count, size = 8))
    
    /// <summary>
    /// Creates a new tensor from a portion of an array of longs
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: int64[], start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.Int64, shape, data, start, count, size = 8))
    
    /// <summary>
    /// Creates a new tensor from a portion of an array of Complex numbers
    /// </summary>
    /// <param name="shape">Represents the tensor shape.</param>
    /// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
    /// <param name="start">The offset into the provided data array where the data resides.</param>
    /// <param name="count">The number of bytes to copy from count into the tensor.</param>
    /// <remarks>
    /// Use the FromBuffer method to create a tensor that has the specified dimensions
    /// and is initialized with data from the data array.   The data is copied starting
    /// at the start offset, for count bytes and is laid out into the tensor following the
    /// specified dimensions.
    /// </remarks>
    static member FromBuffer (shape, data: Complex[], start: int, count: int) =
      new TFTensor (TFTensor.SetupTensor (TFDataType.Complex128, shape, data, start, count, size = 16))
  
    /// <summary>
    /// Creates a constant tensor from an array, the shape reflects the shape of the C# array and the underlying type reflects the C# type.
    /// </summary>
    new (array: Array) =
      if array = null then raise (new ArgumentNullException ("array"))
      // Ensure that, if we have arrays of arrays, we can handle them accordingly:
      if TFTensor.IsJagged(array.GetType ()) then
        let elementType = TFTensor.GetInnerMostType (array)
        let length: int[] = TFTensor.GetLength (array)
        let multidimensional = Array.CreateInstance (elementType, length)
        let flatten: Array = TFTensor.deepFlatten (array)
        Buffer.BlockCopy (flatten, 0, multidimensional, 0, flatten.Length * Marshal.SizeOf (elementType))
        new TFTensor(TFTensor.CreateFromMultidimensionalArrays (multidimensional))
      else
        new TFTensor(TFTensor.CreateFromMultidimensionalArrays (array))
  
    static member TFCreate(x: 'a) : IntPtr =
        // TODO does this create a memory leak?
        let v = valueToIntPtr x
        TensorNative.TF_NewTensor (uint32 <| TFDataType.FromType(typeof<'a>), IntPtr.Zero, 0,  v,  UIntPtr(uint64 sizeof<'a>),  FreeTensorDataDelegate,  IntPtr.Zero)
  
    // Creates a constant tensor with a single dimension from an integer value.
    new (value: int) = new TFTensor(TFTensor.TFCreate(value))
//        let intPtr = Marshal.AllocHGlobal (sizeof<int>)
//        NativePtr.nativeIntWrite intPtr value
//        new TFTensor(TensorNative.TF_NewTensor (uint32 <| TFDataType.FromType(typeof<int>), IntPtr.Zero, 0,  intPtr,  UIntPtr(uint64 sizeof<'a>),  FreeTensorDataDelegate,  IntPtr.Zero))
  
  
    /// Creates a constant tensor with a single dimension from a boolean value.
    new (value: bool) = new TFTensor(TFTensor.TFCreate(value))
  
    /// Creates a constant tensor with a single dimension from an sbyte value.
    new (value: int8) = new TFTensor(TFTensor.TFCreate(value))
  
    /// Creates a constant tensor with a single dimension from a short value.
    new (value: int16) = new TFTensor(TFTensor.TFCreate(value))
  
    /// Creates a constant tensor with a single dimension from an ushort value.
    new (value: uint16) = new TFTensor(TFTensor.TFCreate(value))
  
    /// Creates a constant tensor with a single dimension from an byte value.
    new (value: byte) = new TFTensor(TFTensor.TFCreate(value))
  
    /// Creates a constant tensor with a single dimension from a Complex value.
    new (value: Complex) = new TFTensor(TFTensor.TFCreate(value))
  
    /// Creates a constant tensor with a single dimension from a float32 value.
    new (value: float32) = new TFTensor(TFTensor.TFCreate(value))
  
    /// Creates a constant tensor with a single dimension from a double value.
    new (value: double) = new TFTensor(TFTensor.TFCreate(value))
  
    /// Creates a constant tensor with a single dimension from a long value.
    new (value: int64) = new TFTensor(TFTensor.TFCreate(value))
  
    // Convenience function to factor out the setup of a new tensor from an array
    static member SetupTensor (dt: TFDataType, dims: int64 [], data: Array, size: int) : IntPtr = 
      TFTensor.SetupTensor (dt, dims, data, start = 0, count = data.Length, size = size)
  
    // Convenience function to factor out the setup of a new tensor from an array
    static member SetupTensor (dt: TFDataType, data: Array, size: int) : IntPtr =
      let dims = Array.init data.Rank (fun i -> int64( data.GetLength (i)))
      TFTensor.SetupTensor (dt, dims, data, start = 0, count = data.Length, size = size)
  
    // Use for single dimension arrays 
    static member SetupTensor (dt: TFDataType, dims: int64[], data: Array, start: int, count: int, size: int) : IntPtr  =
      if start < 0 || start > data.Length - count then raise(ArgumentException ("start + count > Array size"))
      let dataHandle = GCHandle.Alloc (data, GCHandleType.Pinned)
      if box dims = null then
          TensorNative.TF_NewTensor (uint32 dt, IntPtr.Zero, 0, dataHandle.AddrOfPinnedObject().Add(start * size), UIntPtr(uint64 (count * size)), FreeTensorHandleDelegate, GCHandle.ToIntPtr (dataHandle))
      else 
          TF_NewTensor (uint32 dt, dims, dims.Length, dataHandle.AddrOfPinnedObject().Add(start * size), UIntPtr(uint64 (count * size)), FreeTensorHandleDelegate, GCHandle.ToIntPtr (dataHandle))
  
    // Use for multiple dimension arrays 
    static member SetupMulti (dt: TFDataType, dims: int64 [], data: Array, bytes: int64) : IntPtr =
      let dataHandle = GCHandle.Alloc (data, GCHandleType.Pinned)
      if box dims = null then 
          TensorNative.TF_NewTensor (uint32 dt, IntPtr.Zero, 0, dataHandle.AddrOfPinnedObject (), UIntPtr(uint64 bytes), FreeTensorHandleDelegate, GCHandle.ToIntPtr (dataHandle))
      else
          TF_NewTensor (uint32 dt, dims, dims.Length, dataHandle.AddrOfPinnedObject (), UIntPtr(uint64 bytes), FreeTensorHandleDelegate, GCHandle.ToIntPtr (dataHandle))
  
    /// <summary>
    /// Converts a C# array into a tensor.
    /// </summary>
    /// <returns>The tensor containing the data.</returns>
    /// <param name="array">single dimension, or multi-dimensional array.</param>
    /// <remarks>
    /// This implicit conversion can convert single or multidimensional arrays of
    /// booleans, sbytes, byte, shorts, ushorts, ints, longs, doubles, floats and
    /// complex numbers into a tensor with the same dimensional shape as the provided
    /// array.
    /// </remarks>
    static member op_Implicit (array: Array) = new TFTensor (array)
  
  
  