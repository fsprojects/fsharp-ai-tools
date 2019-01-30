module TensorFlow.HDF5ReaderWriter
// Originally from https://github.com/DeepMLNet/DeepNet/blob/master/Tensor/Tensor/HDF5.fs
// Apache 2.0 licence

// NOTE: invalidOpf may be useful to get rid of the need of the backpipe and sprintf
// NOTE: perhaps re-introduce record support

// WARN: There appears to be issues running on Ubuntu 18.04
//       It is recommended to avoid H5 where possible

open System
open System.IO
open System.Runtime.InteropServices
open FSharp.Reflection
open HDF.PInvoke 
 
/// HDF5 support functions
module private HDF5Support =

    /// Raises an HDF5 error with an error message generated form the 
    /// current HDF5 error stack.
    let raiseErr () =
        let mutable fullMsg = "HDF5 error:\n"
        let walkFn (pos: uint32) (err: H5E.error_t) =
            let majorMsg = Text.StringBuilder()
            let minorMsg = Text.StringBuilder()
            let mutable msgType = H5E.type_t.MAJOR
            H5E.get_msg (err.maj_num, ref msgType, majorMsg, nativeint 1024) |> ignore
            H5E.get_msg (err.min_num, ref msgType, minorMsg, nativeint 1024) |> ignore
            let line =
                sprintf "%s in %s(%d): %s / %s / %s\n" 
                        err.func_name err.file_name err.line 
                        err.desc (majorMsg.ToString()) (minorMsg.ToString())
            fullMsg <- fullMsg + line
            0
        H5E.walk (H5E.DEFAULT, H5E.direction_t.H5E_WALK_UPWARD, 
                  H5E.walk_t (fun pos err _ -> walkFn pos err), nativeint 0) |> ignore
        invalidOp <| sprintf "%s" fullMsg

    /// Checks if HDF5 return value indicates success and if not,
    /// raises an error.
    let inline check retVal =
        match box retVal with
        | :? int as rv ->            
            if rv < 0 then raiseErr()
            else retVal
        | :? int64 as rv ->
            if rv < 0L then raiseErr()
            else retVal        
        | rv -> 
            failwithf "Internal error: unknown HDF5 return type: %A" (rv.GetType())

    do
        H5.``open`` () |> check |> ignore

    let hdfTypeTable = 
        [ typeof<bool>,     H5T.NATIVE_UINT8
          typeof<int8>,     H5T.NATIVE_INT8
          typeof<int16>,    H5T.NATIVE_INT16
          typeof<int32>,    H5T.NATIVE_INT32
          typeof<int64>,    H5T.NATIVE_INT64
          typeof<uint8>,    H5T.NATIVE_UINT8
          typeof<uint16>,   H5T.NATIVE_UINT16
          typeof<uint32>,   H5T.NATIVE_UINT32
          typeof<uint64>,   H5T.NATIVE_UINT64
          typeof<single>,   H5T.NATIVE_FLOAT
          typeof<double>,   H5T.NATIVE_DOUBLE
          typeof<string>,   H5T.C_S1 ]

    let hdfTypeInst t =     
        match hdfTypeTable |> List.tryPick (fun (nt, ht) -> 
            if nt=t then Some ht else None) with
        | Some ht -> ht
        | None -> invalidOp <| sprintf "Unsupported type for HDF5: %A" t

    let hdfType<'T> =  
        hdfTypeInst typeof<'T>

    let netType t =     
        match hdfTypeTable |> List.tryPick (fun (nt, ht) -> 
            if H5T.equal(ht, t) > 0 then Some nt else None) with
        | Some nt -> nt
        | None -> invalidOp <| sprintf "Unsupported HDF5 type: %A" t

    let hdfShape (shape: int64 list) =
        shape |> List.map uint64 |> List.toArray

    let intShape (shape: uint64 array) =
        shape |> Array.toList |> List.map int64

open HDF5Support


type private HDF5Mode =
    /// read HDF5 file
    | HDF5Read
    /// (over-)write HDF5 file
    | HDF5Overwrite


/// <summary>An object representing an HDF5 file.</summary>
/// <remarks>
/// <p>HDF5 is an open, cross-plattform, industry-standard file format for data exchange. 
/// More information is available at <see href="https://www.hdfgroup.org"/>.</p>
/// <p>This object represents an HDF5 file. Use <see cref="OpenRead"/> and <see cref="OpenWrite"/> to open an HDF5 file 
/// for reading or writing.</p>
/// <p>This object does not aim to expose all functions provided by the HDF5 standard. Instead, it focuses on providing
/// a simple interface for reading and writing arrays as well as attributes.</p>
/// </remarks>
type HDF5 private (path: string, mode: HDF5Mode) = 
        
    let mutable disposed = false

    let fileAccessProps = H5P.create(H5P.FILE_ACCESS) |> check
    do  
        H5P.set_libver_bounds(fileAccessProps, H5F.libver_t.LATEST, H5F.libver_t.LATEST) 
        |> check |> ignore

    let fileHnd =
        match mode with
        | HDF5Read -> 
            if not (File.Exists path) then 
                raise (FileNotFoundException (sprintf "HDF5 file not found: %s" path, path))
            H5F.``open`` (path, H5F.ACC_RDONLY, plist=fileAccessProps)
        | HDF5Overwrite -> H5F.create (path, H5F.ACC_TRUNC, access_plist=fileAccessProps)
        |> check

    let checkShape data shape =
        let nElems = List.fold (*) 1L shape
        if int64 (Array.length data) < nElems then
            invalidArg "data" <| sprintf "Shape %A does not match number of elements in data array." shape

        if List.exists ((>) 0L) shape then
            invalidArg "shape" <| sprintf "Shape %A has negative elements." shape

    let checkDisposed () =
        if disposed then raise (ObjectDisposedException("HDF5", "HDF5 file was previously disposed"))

    /// <summary>Closes the HDF5 file.</summary>
    member this.Dispose () = 
        if not disposed then             
            if fileHnd >= 0L then
                H5F.close fileHnd |> check |> ignore
            if fileAccessProps >= 0L then
                H5P.close fileAccessProps |> check |> ignore
            disposed <- true

    interface IDisposable with
        /// <summary>Closes the HDF5 file.</summary>
        member this.Dispose () = this.Dispose ()

    /// <summary>Closes the HDF5 file.</summary>
    override this.Finalize () =
        this.Dispose ()
                    
    /// <summary>Opens the specified HDF5 file for reading.</summary>
    /// <param name="path">The path to the HDF5 file.</param>
    /// <returns>An HDF5 object representing the opened file.</returns>
    static member OpenRead  path = new HDF5 (path, HDF5Read)

    /// <summary>Opens the specified HDF5 file for writing.</summary>
    /// <param name="path">The path to the HDF5 file.</param>
    /// <returns>An HDF5 object representing the opened file.</returns>
    /// <remarks>If the file already exists, it will be overwritten.</remarks>
    static member OpenWrite path = new HDF5 (path, HDF5Overwrite)

    /// Splits a HDF5 path string into a list.
    static member private SplitPath (path: string) =
        path.Split('/') 
        |> List.ofArray
        |> List.filter (fun d -> String.length d > 0)

    /// Combines a list of groups into a HDF5 path string.
    static member private CombinePath (dirs: string list) =
        dirs
        |> List.filter (fun d -> String.length d > 0)
        |> String.concat "/" 
            
    /// <summary>Checks whether an object (array or group) with the given name exists.</summary>
    /// <param name="name">HDF5 path to check.</param>
    /// <returns><c>true</c> if the path exists; <c>false</c> otherwise.</returns>
    member this.Exists (name: string) =
        checkDisposed ()
        let rec exists prefix dirs =
            match dirs with
            | [] -> true
            | dir::dirs ->
                let nextPrefix = prefix @ [dir]
                if H5L.exists (fileHnd, HDF5.CombinePath nextPrefix) |> check <= 0 then false
                else
                    exists nextPrefix dirs
        exists [] (HDF5.SplitPath name) 

    /// <summary>Creates the given group path.</summary>
    /// <param name="path">HDF5 group path to create.</param>
    /// <remarks>All necessary parent groups are created automatically.
    /// If the group with the given path already exists, nothing happens.</remarks>
    member this.CreateGroups (path: string) =
        checkDisposed ()
        let rec create prefix dirs =
            match dirs with
            | [] -> ()
            | dir::dirs ->
                let nextPrefix = prefix @ [dir]
                let nextPrefixPath = HDF5.CombinePath nextPrefix
                if not (this.Exists nextPrefixPath) then
                    let groupHnd = H5G.create(fileHnd, nextPrefixPath) |> check 
                    H5G.close groupHnd |> check |> ignore
                create nextPrefix dirs
        if path.Length > 0 then
            create [] (HDF5.SplitPath path)                

    /// Create all necessary parent groups for the given path.
    member private this.CreateParentGroups (path: string) =
        checkDisposed ()
        match HDF5.SplitPath path with
        | [] -> ()
        | [_] -> ()
        | pl ->
            pl.[0 .. pl.Length-2]
            |> HDF5.CombinePath
            |> this.CreateGroups

    /// <summary>Write data array to HDF5 file.</summary>
    /// <typeparam name="'T">Type of the data.</typeparam>
    /// <param name="name">HDF5 path to write to.</param>
    /// <param name="data">Data array to write.</param>
    /// <param name="shape">Array shape to use.</param>
    /// <remarks>All HDF5 groups are automatically created as necessary.</remarks>
    /// <seealso cref="Read``1"/><seealso cref="HostTensor.write"/>
    member this.Write (name: string, data: 'T array, shape: int64 list) =
        checkDisposed ()
        if mode <> HDF5Overwrite then
            invalidOp <| sprintf "HDF5 file %s is opened for reading." path
        checkShape data shape
        this.CreateParentGroups name
        if this.Exists name then
            invalidOp <| sprintf "HDF5 dataset %s already exists in file %s." name path
            
        let typeHnd = H5T.copy hdfType<'T> |> check
        let shapeHnd = H5S.create_simple (List.length shape, hdfShape shape, hdfShape shape) |> check
        let dataHnd = H5D.create (fileHnd, name, typeHnd, shapeHnd) |> check

        let gcHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
        H5D.write (dataHnd, typeHnd, int64 H5S.ALL, int64 H5S.ALL, H5P.DEFAULT, gcHnd.AddrOfPinnedObject()) |> check |> ignore
        gcHnd.Free ()

        H5D.close dataHnd |> check |> ignore
        H5S.close shapeHnd |> check |> ignore
        H5T.close typeHnd |> check |> ignore

    /// <summary>Read data array from HDF5 file.</summary>
    /// <typeparam name="'T">Type of the data.</typeparam>
    /// <param name="name">HDF5 path to read from.</param>
    /// <returns>A tuple of <c>(data, shape)</c> where <c>data</c> is the read data array and <c>shape</c> is the 
    /// corresponding shape.</returns>
    /// <remarks>The type <c>'T</c> must match the data type stored in the HDF5 file, otherwise an exception is raised.
    /// </remarks>
    /// <seealso cref="Write``1"/><seealso cref="GetDataType"/><seealso cref="HostTensor.read``1"/>
    member this.Read<'T> (name: string) = 
        checkDisposed ()
        if not (this.Exists name) then
            invalidOp <| sprintf "HDF5 dataset %s does not exist in file %s." name path
        let dataHnd = H5D.``open`` (fileHnd, name) |> check
        let typeHnd = H5D.get_type dataHnd |> check
        let shapeHnd = H5D.get_space (dataHnd) |> check

        if H5T.equal (hdfType<'T>, typeHnd) = 0 then
            invalidOp <| sprintf "HDF5 dataset %s has other type than %A." name typeof<'T>

        if H5S.is_simple (shapeHnd) = 0 then
            invalidOp <| sprintf "HDF5 dataset %s is not simple." name
        let nDims = H5S.get_simple_extent_ndims (shapeHnd) |> check
        let shape : uint64 array = Array.zeroCreate nDims
        let maxShape : uint64 array = Array.zeroCreate nDims
        H5S.get_simple_extent_dims(shapeHnd, shape, maxShape) |> check |> ignore
        let nElems = Array.fold (*) 1UL shape |> int

        let data : 'T array = Array.zeroCreate nElems
        let gcHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
        H5D.read (dataHnd, typeHnd, int64 H5S.ALL, int64 H5S.ALL, H5P.DEFAULT, gcHnd.AddrOfPinnedObject()) |> check |> ignore
        gcHnd.Free ()

        H5D.close dataHnd |> check |> ignore
        H5S.close shapeHnd |> check |> ignore
        H5T.close typeHnd |> check |> ignore

        data, shape |> intShape

    /// <summary>Get data type of array in HDF5 file.</summary>
    /// <param name="name">HDF5 path to read from.</param>
    /// <returns>Data type.</returns>    
    member this.GetDataType (name: string) =
        checkDisposed ()
        if not (this.Exists name) then
            invalidOp <| sprintf "HDF5 dataset %s does not exist in file %s." name path
        let dataHnd = H5D.``open`` (fileHnd, name) |> check
        let typeHnd = H5D.get_type dataHnd |> check
        let netType = netType typeHnd
        H5D.close dataHnd |> check |> ignore
        H5T.close typeHnd |> check |> ignore
        netType

    /// <summary>Set attribute value on an HDF5 object.</summary>
    /// <typeparam name="'T">Type of the attribute value.</typeparam>
    /// <param name="name">HDF5 path to operate on.</param>
    /// <param name="atrName">Name of the attribute.</param>
    /// <param name="value">Value to set attribute to.</param>    
    /// <seealso cref="GetAttribute``1"/><seealso cref="SetRecord``1"/>    
    member this.SetAttribute (name: string, atrName: string, value: 'T) =
        checkDisposed ()
        if not (this.Exists name) then
            invalidOp <| sprintf "HDF5 object %s does not exist in file %s." name path

        let valType = value.GetType()
        let hdfValType, data, dataLength =
            if valType = typeof<string> then
                let bytes = System.Text.Encoding.UTF8.GetBytes (box value :?> string) |> box :?> Array
                let strType = H5T.copy(H5T.C_S1) |> check
                H5T.set_size(strType, nativeint bytes.LongLength) |> check |> ignore
                strType, bytes, 1UL
            elif valType.IsArray then 
                let ary = box value :?> Array
                hdfTypeInst (valType.GetElementType()), ary, uint64 ary.LongLength
            else 
                let ary = Array.CreateInstance(valType, 1)
                ary.SetValue (value, 0)
                hdfTypeInst valType, ary, 1UL

        let typeHnd = H5T.copy hdfValType |> check
        let shapeHnd = H5S.create_simple (1, [|dataLength|], [|dataLength|]) |> check

        if H5A.exists_by_name (fileHnd, name, atrName) > 0 then
            H5A.delete_by_name (fileHnd, name, atrName) |> check |> ignore
        let atrHnd = H5A.create_by_name (fileHnd, name, atrName, typeHnd, shapeHnd) |> check

        let gcHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
        H5A.write (atrHnd, typeHnd, gcHnd.AddrOfPinnedObject()) |> check |> ignore
        gcHnd.Free ()

        H5A.close atrHnd |> check |> ignore
        H5S.close shapeHnd |> check |> ignore
        H5T.close typeHnd |> check |> ignore

    /// <summary>Get attribute value on an HDF5 object.</summary>
    /// <typeparam name="'T">Type of the attribute value.</typeparam>
    /// <param name="name">HDF5 path to operate on.</param>
    /// <param name="atrName">Name of the attribute.</param>
    /// <returns>Value of the attribute.</returns>    
    /// <seealso cref="SetAttribute``1"/><seealso cref="GetRecord``1"/>
    member this.GetAttribute (name: string, atrName: string) : 'T =
        checkDisposed ()
        if not (this.Exists name) then
            invalidOp <| sprintf "HDF5 object %s does not exist in file %s." name path
        if not (H5A.exists_by_name (fileHnd, name, atrName) > 0) then
            invalidOp <| sprintf "HDF5 attribute %s does not exist on object %s in file %s." atrName name path

        let elementType =
            if typeof<'T> = typeof<string> then typeof<byte>
            elif typeof<'T>.IsArray then typeof<'T>.GetElementType()
            else typeof<'T>

        let atrHnd = H5A.open_by_name (fileHnd, name, atrName) |> check
        let typeHnd = H5A.get_type atrHnd |> check
        let shapeHnd = H5A.get_space (atrHnd) |> check

        if typeof<'T> = typeof<string> then
            if H5T.get_class (typeHnd) <> H5T.class_t.STRING then
                invalidOp <| sprintf "HDF5 attribute %s on object %s is not a string." atrName name
        elif H5T.equal (hdfTypeInst elementType, typeHnd) = 0 then
            invalidOp <| sprintf "HDF5 attribute %s on object %s has other type than %A." atrName name elementType

        if H5S.is_simple (shapeHnd) = 0 then
            invalidOp <| sprintf "HDF5 attribute %s on object %s is not simple." atrName name
        let nDims = H5S.get_simple_extent_ndims (shapeHnd) |> check
        if nDims <> 1 then
            invalidOp <| sprintf "HDF5 attribute %s on object %s is not of rank 1." atrName name
        let shape : uint64 array = Array.zeroCreate nDims
        let maxShape : uint64 array = Array.zeroCreate nDims
        H5S.get_simple_extent_dims(shapeHnd, shape, maxShape) |> check |> ignore
        let nElems = 
            if typeof<'T> = typeof<string> then H5T.get_size(typeHnd) |> int
            else shape.[0] |> int
        if typeof<'T> <> typeof<string> && nElems <> 1 && not typeof<'T>.IsArray then
            invalidOp <| sprintf "HDF5 attribute %s on object %s has %d elements, but a scalar is expected."
                      atrName name nElems

        let data = Array.CreateInstance(elementType, nElems)
        let gcHnd = GCHandle.Alloc(data, GCHandleType.Pinned)
        H5A.read (atrHnd, typeHnd, gcHnd.AddrOfPinnedObject()) |> check |> ignore
        gcHnd.Free()

        H5A.close atrHnd |> check |> ignore
        H5S.close shapeHnd |> check |> ignore
        H5T.close typeHnd |> check |> ignore

        if typeof<'T> = typeof<string> then
            System.Text.Encoding.UTF8.GetString (data :?> byte[]) |> box :?> 'T
        elif typeof<'T>.IsArray then
            box data :?> 'T
        else
            data.GetValue(0) :?> 'T

    // /// <summary>Set attribute values on an HDF5 object using the provided record.</summary>
    // /// <typeparam name="'R">Type of the F# record. It must contain only field of primitive data types.</typeparam>
    // /// <param name="name">HDF5 path to operate on.</param>
    // /// <param name="record">Record containing the attribute values.</param>    
    // /// <remarks>
    // /// <p>The record must consists only of fields of primitive data types (int, float, string, etc.).</p>
    // /// <p>Each record field is stored as an HDF5 attribute using the same name.</p>
    // /// </remarks>
    // /// <seealso cref="GetRecord``1"/><seealso cref="SetAttribute``1"/>
    // member this.SetRecord (name: string, record: 'R) =
    //     if not (FSharpType.IsRecord typeof<'R>) then
    //         invalidArg "record" "Must specify a value of record type."
    //     for fi, value in Array.zip (FSharpType.GetRecordFields typeof<'R>) 
    //                                 (FSharpValue.GetRecordFields record) do
    //         callGenericInst<HDF5, unit> this "SetAttribute" [fi.PropertyType]
    //             (name, fi.Name, value)

    // /// <summary>Get attribute values on an HDF5 object and returns them as a record.</summary>
    // /// <typeparam name="'R">Type of the F# record. It must contain only field of primitive data types.</typeparam>
    // /// <param name="name">HDF5 path to operate on.</param>
    // /// <returns>Record containing the attribute values.</returns>    
    // /// <remarks>
    // /// <p>The record must consists only of fields of primitive data types (int, float, string, etc.).</p>
    // /// <p>Each record field is read from an HDF5 attribute using the same name.</p>
    // /// </remarks>
    // /// <seealso cref="SetRecord``1"/><seealso cref="GetAttribute``1"/>
    // member this.GetRecord (name: string) : 'R =
    //     if not (FSharpType.IsRecord typeof<'R>) then
    //         invalidArg "return" "Must use a record as return type."
    //     let values =
    //         FSharpType.GetRecordFields typeof<'R>
    //         |> Array.map (fun fi ->
    //             callGenericInst<HDF5, obj> this "GetAttribute" [fi.PropertyType]
    //                 (name, fi.Name)
    //         )
    //     FSharpValue.MakeRecord (typeof<'R>, values) :?> 'R