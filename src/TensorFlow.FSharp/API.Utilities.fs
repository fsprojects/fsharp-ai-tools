module TensorFlow.FSharp.API.Utilities
open TensorFlow.FSharp

/// Files https://github.com/eaplatanios/tensorflow_scala/tree/a1a704fe79c1415df6eda9a8f1651c3b1984223a/modules/api/src/main/scala/org/platanios/tensorflow/api/utilities

/// DONE:
/// Coding.scala
/// Collections.scala
/// Proto.scala
/// Reservoir.scala
/// CRC32.scala // Will need some testing... may have some issues

/// Don't Do: (I believe this is handled elsewhere)
/// Diposer.scala
/// NativeHandleWrapper.scala
/// package.scala

open System
open System.IO
open System.Collections.Concurrent

/// Revisit this if we have any issues with it...
type CRC32() = 

    static let maskDelta = 0xa282ead8u

    static let crcTable : Lazy<uint32[]> =
        lazy 
            [|
                for n in 0..255 do
                    let mutable c = uint32 n
                    for k in 0..7 do
                        if ((c &&& 1u) <> 0u) then c <- uint32 (0xedb88320L ^^^ ((int64 c) >>> 1)) else c  <- c >>> 1
                    yield c
           |]         

    static let getCrc(buf:byte[],start:int,finish:int, inital : uint32 option) =
        let crcTable = crcTable.Force()
        let inital = defaultArg inital 0u
        let mutable c = inital ^^^ 0xffffffffu 
        for n in start..finish-1 do 
            c <- crcTable.[int(c ^^^ uint32 buf.[n] &&& 0xffu)] ^^^ (c >>> 8)
        c ^^^ 0xffffffffu 
    
    /// Returns the CRC32C of the provided string.
    static member Value(value : string) : uint32 = CRC32.Value(System.Text.UTF8Encoding.UTF8.GetBytes(value))

    /// Returns the CRC32C of `a` concatentated with `value` where `initCrc` is the CRC32C of some string `a`. this method
    /// is often used to maintain the CRC32C of a stream of data.
    static member Extend(initCrc : uint32, value : string, ?size : int) : uint32 = 
        CRC32.Extend(initCrc,System.Text.UTF8Encoding.UTF8.GetBytes(value), ?size=size)
        
    /// Returns the CRC32C of `data[0 .. size - 1)`.
    static member Value(bytes : byte[], ?size : int) = 
        let n = match size with | None -> bytes.Length | Some(n) -> n
        getCrc(bytes,0,n,None)

    /// Returns the CRC32C of `a` concatenated with `data[0, size - 1]` where `initCrc` is the CRC32C of some string `a`.
    static member Extend(initCrc : uint32, bytes : byte[], ?size : int) : uint32 =
        let n = match size with | None -> bytes.Length | Some(n) -> n
        getCrc(bytes,0,n,Some(initCrc))

    /// Returns a masked representation of `crc`
    static member Mask(crc : uint32) : uint32 = 
        // Rorate the right 15 bits and add a constant
        let l = crc &&& 0xffffffffu
        (((l >>> 15) ||| (l <<< 17)) + maskDelta)

    /// Returns the CRC whos masked representation is `maskedCrc`
    static member Unmask(maskedCrc : uint32) : uint32 =
        let l = maskedCrc &&& 0xffffffffu
        let rot = l - maskDelta
        ((rot >>> 17) ||| (rot <<< 15))


/// NOTE: depending on how often we have to do bigEndian we may skip out of this type all together
type Coding() =
    static member EncodeFixedInt16(value : int16, ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.GetBytes(value)
        else BitConverter.GetBytes(value) |> Array.rev
    
    static member EncodeFixedInt32(value : int32, ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.GetBytes(value)
        else BitConverter.GetBytes(value) |> Array.rev
        
    static member EncodeFixedInt64(value : int64, ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.GetBytes(value)
        else BitConverter.GetBytes(value) |> Array.rev

    static member DecodeFixedInt16(value : byte[], ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.ToInt16(value,0)
        else BitConverter.ToInt16(value |> Array.rev,0)
    
    static member DecodeFixedInt32(value : byte[], ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.ToInt32(value,0)
        else BitConverter.ToInt32(value |> Array.rev,0)
        
    static member DecodeFixedInt64(value : byte[], ?littleEndian : bool) = 
        let littleEndian = defaultArg littleEndian true
        if littleEndian 
        then BitConverter.ToInt64(value,0)
        else BitConverter.ToInt64(value |> Array.rev,0)

    static member EncodeStrings(values : String[]) : byte[] = 
        // NOTE: It must be assumed that the number of strings is encoded elsewhere
        let buffers = values |> Array.map System.Text.UTF8Encoding.UTF8.GetBytes
        [|
            for buffer in buffers do yield! BitConverter.GetBytes(buffer.Length)
            for buffer in buffers do yield! buffer
        |]
        
    static member VarIntLength(value  : int) : int = 
        // We are treating `value` as an unsigned integer.
        let B = 128L
        let mutable unsigned = (int64 value) &&& 0xffffffffL
        let mutable length = 1
        while unsigned >= B do
            unsigned <- unsigned >>> 7
            length <- length + 1
        length
    
    static member EncodeVarInt32(value : int) : byte [] =
        // We are treating `value` as an unsigned integer.
        let B = 128L
        let mutable unsigned = (int64 value) &&& 0xffffffffL
        let mutable position = 0
        [|
            while unsigned >= B do
                yield byte ((unsigned &&& (B - 1L)) ||| B)
                unsigned <- unsigned >>> 7
                position <- position + 1
        |]

    static member DecodeVarInt32(bytes : byte[]) : (int*int) = 
        // TODO double check this there may be errors introduced by casting
        let B = 128L
        let mutable b = bytes.[0]
        let mutable value = int b
        let mutable position = 1
        let mutable shift = 7
        while (b &&& byte B) <> 0uy && shift <= 28 && position < bytes.Length do
            // More bytes are present.
            b <- bytes.[position]
            value <- value ||| ((int (b &&& (byte B - 1uy))) <<< shift)
            position <- position + 1
            shift <- shift + 7
        if (b &&& 128uy) <> 0uy then (-1,-1)
        else (value,position)

// NOTE: ProtoBuf does not seem to have a common interface so an unrestricted generic is used
//       Normally would consider trying to use Google.Protobuf.IMessage is used in place of GeneratedMessageV3 

type Proto() =
    /// <summary>
    /// Writes `message` to the specified file.
    /// </summary>
    /// <param name="directory">Directoryh in which to write the file.</param>
    /// <param name="filename">Name of the file.</param>
    /// <param name="message">ProtoBuf message to write.</param>
    /// <param name="asText">Boolean value indicating wheter to serialize the ProtoBuf message in the human-friendly text format, or in the more efficent binary format.</param>
    /// <returns>Path of the written file.</returns>
    static member Write(directory : string, filename : string, message : 'T, ?asText : bool) : string =
        let asText = defaultArg asText false
        if not(Directory.Exists(directory)) && not(directory.StartsWith("gs:")) then
            Directory.CreateDirectory(directory) |> ignore
        let filePath = Path.Combine(directory,filename)
        if asText then File.WriteAllText(filePath, message.ToString())
        else 
            use file = File.OpenWrite(filename)
            ProtoBuf.Serializer.Serialize(file,message)
        filePath

/// <summary> Container for items coming from a stream, that implements reservoir sampling so that its size never exceeds
/// `maxSize`.
/// </summary>
/// <param  name="maxSize">Maximum size of this bucket.</param>
/// <param  name="random">Random number generator to use while sampling.</param>
/// <param  name="alwaysKeepLast"> Boolean flag indicating whether to always store the last seen item. If set to `true` and the
///                        last seen item was not sampled to be stored, then it replaces the last item in this bucket.</param>
type ReservoirBucket<'T>(maxSize : int, random : Random, alwaysKeepLast : bool) = 
    let mutable items = Array.empty<'T>
    let mutable numItemsSeen = 0
    /// <summary>Adds an item to this reservoir bucket, replacing an old item, if necessary.
    ///
    /// If `alwaysKeepLast` is `true`, then the new item is guaranteed to be added to the bucket, and to be the last
    /// element in the bucket. If the bucket has reached capacity, then an old item will be replaced. With probability
    /// `maxSize / numItemsSeen` a random item in the bucket will be popped out and the new item will be appended to the
    /// end. With probability `1 - maxSize / numItemsSeen`, the last item in the bucket will be replaced.
    ///
    /// If `alwaysKeepLast` is `false`, then with probability `1 - maxSize / numItemsSeen` the new item may not be added
    /// to the reservoir at all.
    ///
    /// Since the `O(n)` replacements occur with `O(1/numItemsSeen)` likelihood, the amortized runtime cost is `O(1)`.
    /// </summary>
    /// <param name="item">Item to add.</param>
    /// <param name="transformFn">A function used to transform the item before addition, if the item will be kept in the reservoir. </param>
    member this.Add(item : 'T, ?transformFn : 'T -> 'T) = 
        let item = match transformFn with | Some(f) -> f(item) | _ -> item
        if items.Length < maxSize || maxSize = 0 then
            items <- [|yield! items; yield item|]
        else
            let r = random.Next(numItemsSeen)
            if r < maxSize then
                items <- items |> Array.removeByIndex r
                items <- [|yield! items ; yield item|]
            elif alwaysKeepLast then
                items.[items.Length - 1] <- item
        numItemsSeen <- numItemsSeen + 1
        
    /// <summary>Filters the items in this reservoir using the provided filtering function.
    ///
    /// When filtering items from the reservoir bucket, we must update the internal state variable `numItemsSeen`, which
    /// is used for determining the rate of replacement in reservoir sampling. Ideally, `numItemsSeen` would contain the
    /// exact number of items that have ever been seen by the `add` function of this reservoir, and that satisfy the
    /// provided filtering function. However, the reservoir bucket does not have access to all of the items it has seen --
    /// it only has access to the subset of items that have survived sampling (i.e., `_items`). Therefore, we estimate
    /// `numItemsSeen` by scaling its original value by the same ratio as the ratio of items that were not filtered out
    /// and that are currently stored in this reservoir bucket.
    /// </summary>
    /// <param name="filterFn"> Filtering function that returns `true` for the items to be kept in the reservoir. </param>
    /// <returns>Number of items filtered from this reservoir bucket.</returns>
    member this.Filter(filterFn : 'T -> bool) = 
        let sizeBefore = items.Length
        items <- items |> Array.filter filterFn
        let numFiltered = sizeBefore - items.Length
        // Estimate a correction for the nubmer of items seen.
        let proportionRemaining = if sizeBefore > 0 then float items.Length / float sizeBefore else 0.0
        numItemsSeen <- int(Math.Round(proportionRemaining * float numItemsSeen))
        numFiltered

/// <summary>
/// /** A key-value store using deterministic reservoir sampling.
///
/// Items are added with an associated key. Items may be retrieved by the corresponding key, and a list of keys can also
/// be retrieved. If `maxSize` is not zero, then it dictates the maximum number of items that will be stored for each
/// key. Once there are more items for a given key, they are replaced via reservoir sampling, such that each item has an
/// equal probability of being included in the sample.
///
/// Deterministic means that for any given seed and bucket size, the sequence of values that are kept for any given key
/// will always be the same, and that this is independent of any insertions for other keys. That is:
///
/// {{{
///   val reservoirA = ReservoirKVStore(10)
///   val reservoirB = ReservoirKVStore(10)
///   (0 until 100).foreach(i => reservoirA.add("key1", i))
///   (0 until 100).foreach(i => reservoirA.add("key2", i))
///   (0 until 100).foreach(i => {
///     reservoirB.add("key1", i)
///     reservoirB.add("key2", i)
///   })
/// }}}
///
/// After executing this code, `reservoirA` and `reservoirB` will be in identical states.
///
/// For more information on reservoir sampling, refer to [this page](https://en.wikipedia.org/wiki/Reservoir_sampling).
///
/// Note that, adding items has amortized `O(1)` runtime cost.
/// </summary>
/// <param name="maxSize"></param>
/// <param name="seed"></param>
/// <param name="alwaysKeepLast"></param>
type Reservoir<'K,'V>(maxSize : int, ?seed : int32, ?alwaysKeepLast : bool) =
    let seed = defaultArg seed 0
    let alwaysKeepLast = defaultArg alwaysKeepLast true
    do if maxSize < 0 then invalidArg "maxSize" (sprintf "maxSize (= %i) Must be a non-negative integer." maxSize)
    // NOTE: Will try to use ConcurrentDictionary as a base and avoid the need for a lock, though may need to reverse this decision
    let buckets = ConcurrentDictionary<'K,ReservoirBucket<'V>>()

    /// Returns all the keys in the reservoir.
    member this.Keys = buckets.Keys

    /// REturns all the items stored for the provided key and throws an exception if the key does not exist
    member this.Item(key : 'K) = buckets.Item(key)

    ///  <summary>Adds a new item to the reservoir with the provided key.
    /// 
    ///  If the corresponding reservoir has not yet reached full size, then the new item is guaranteed to be added. If the
    ///  reservoir is full, then the behavior of this method depends on the value of `alwaysKeepLast`.
    /// 
    ///  If `alwaysKeepLast` is set to `true`, then the new item is guaranteed to be added to the reservoir, and either the
    ///  previous last item will be replaced, or (with low probability) an older item will be replaced.
    /// 
    ///  If `alwaysKeepLast` is set to `false`, then the new item may replace an old item with low probability.
    /// 
    ///  If `transformFn` is provided, then it will be applied to transform the provided item (lazily, if and only if the
    ///  item is going to be included in the reservoir).
    ///  </summary>
    ///  <param name="key">Key for the item to add.</param>
    ///  <param name="item">Item to add.</param>
    ///  <param name="transformFn">Transform function for the item to add.</transform>
    member this.Add(key : 'K, item : 'V, ?transformFn : 'V -> 'V) =
        let transformFn = defaultArg transformFn id
        buckets.GetOrAdd(key, ReservoirBucket(maxSize, new Random(seed), alwaysKeepLast)).Add(item, transformFn)

    /// <summary>Filters the items in this reservoir using the provided filtering function.
    ///
    /// When filtering items from each reservoir bucket, we must update the internal state variable `numItemsSeen`, which
    /// is used for determining the rate of replacement in reservoir sampling. Ideally, `numItemsSeen` would contain the
    /// exact number of items that have ever been seen by the `add` function of this reservoir, and that satisfy the
    /// provided filtering function. However, the reservoir bucket does not have access to all of the items it has seen --
    /// it only has access to the subset of items that have survived sampling (i.e., `_items`). Therefore, we estimate
    /// `numItemsSeen` by scaling its original value by the same ratio as the ratio of items that were not filtered out
    /// and that are currently stored in this reservoir bucket.
    /// </summary>
    /// <param  name="filterFn">Filtering function that returns `true` for the items to be kept in the reservoir.</param>
    /// <param  nake="key">Optional key for which to filter the values. If `None` (the default), then the values for all
    ///                  keys in the reservoir are filtered.</param>
    /// <return>Number of items filtered from this reservoir.</return>
    member this.Filter(filterFn : 'V -> bool, ?key : 'K) : int =
        match key with
        | Some(key) ->
            match buckets.TryGet(key) with 
            | Some(bucket) -> bucket.Filter(filterFn)
            | None -> 0
        | None -> 
            [|for bucket in buckets.Values  -> bucket.Filter(filterFn)|] |> Array.sum
