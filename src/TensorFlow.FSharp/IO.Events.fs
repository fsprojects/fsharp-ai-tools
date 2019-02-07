namespace TensorFlow.FSharp.IO.Events
open TensorFlow.FSharp.API.Utilities
open TensorFlow.FSharp.Proto
open System
open System.IO

// From
//https://github.com/eaplatanios/tensorflow_scala/tree/master/modules/api/src/main/scala/org/platanios/tensorflow/api/io/events

// EventPluginUtilities.scala
// EventRecord.scala
// EventType.scala

//TODO
// EventAccumulator.scala
// EventFileReader.scala
// EventFileWriter.scala
// EventMultiplexer.scala
// SummaryFileWriter.scala
// SummaryFileWriterCache.scala

type EventType =
| ScalarEventType
| ImageEventType
| AudioEventType
| HistogramEventType
| CompressedHistogramEventType
| TensorEventType
| GraphEventType
| MetaGraphEventType
| RunMetadataEventType


type EventRecord<'T> =
    { 
        walltime : float32
        step : int64
        value : 'T
    }

type ByteString = byte[]

type ImageValue = 
    {   encodedImage : ByteString; 
        width : int; 
        height : int; 
        colorSpace : int } // Could also use a ByteString from proto??

type AudioValue = 
    {   encodedAudo : ByteString; 
        contentType : string; 
        sampleRate : float32; 
        numChannels : int64; 
        lengthFrames : int64 }

type HistogramValue =
    {   min : double
        max : double
        num : double
        sum : double
        sumSquares : double
        bucketLimits : double[]
        }

type ScalarEventRecord = EventRecord<float32>
type ImageEventRecord = EventRecord<ImageValue>
type AudioEventRecord = EventRecord<AudioValue>
type HistogramEventRecord = EventRecord<HistogramValue>
type CompressedHistogramEventRecord = EventRecord<HistogramValue[]>
type TensorEventRecord = EventRecord<TensorFlow.FSharp.Proto.TensorProto>

[<RequireQualifiedAccess>]
type Event =
| ScalarEventRecord of EventRecord<float32>
| ImageEventRecord of EventRecord<ImageValue>
| AudioEventRecord of EventRecord<AudioValue>
| HistogramEventRecord of EventRecord<HistogramValue>
| CompressedHistogramEventRecord of EventRecord<HistogramValue[]>
| TensorEventRecord of EventRecord<TensorFlow.FSharp.Proto.TensorProto>

module EventPluginUtilities =
    open System.IO

    let private PLUGINS_DIR = "plugins"

    /// Returns the plugin directory for hte provided plugin name.
    let pluginDir(logDir : string, pluginName : string) : string =
        // TODO Double check agains java Path resolve behavior
        Path.Combine(logDir, PLUGINS_DIR, pluginName)

    /// Returns a sequence with paths to all the registered assets for the provided plugin name, in `logDir`.
    ///
    /// If a plugins directory does not exist in `logDir`, then this method returns an empty list. This maintains
    /// compatibility with old log directories that contain no plugin sub-directories.
    let listPluginAssets(logDir : string, pluginName : string) : string[] =
        let pluginsDir = Path.Combine(logDir, pluginName)
        if not(Directory.Exists(pluginsDir)) then [||]
        else Directory.GetDirectories(pluginsDir)

    /// Returns a sequence with all the plugin directories that have contained regiesterd assets, in `logDir`.
    /// If a plugins directory does not exit in `logDir`, the this method returns an empty list. This maintains
    /// compatibility with old log directories that contain no plugin sub-directores.
    let listPluginDirs(logDir : string) : string[] = listPluginAssets(logDir, PLUGINS_DIR)

    /// Retrieves a particular plugin asset from `logDir` and returns it as a string
    let retrievePluginAsset(logDir : string, pluginName : string, assetName : string ) : string =
        let assetPath = Path.Combine(pluginDir(logDir, pluginName), assetName)
        if File.Exists(assetPath) then
            File.ReadAllText(assetPath)
        else
            failwithf "Asset path '%s' not found" assetPath

            
/// <summary>Event file reader.
///
/// An event file reader is used to create iterators over the events stored in the file at the provided path (i.e.,
/// `filePath`).
///
/// Note that this reader ignores any corrupted records at the end of the file. That is to allow for "live tracking" of
/// summary files while they're being written to.
/// </summary>
/// <param name="filePath">Path to the file being read.</param>
/// <param  name="compressionType">Compression type used for the file.</param>
type EventFileReader internal ( filePath : string, ?compressionType : CompressionType) 
    // TODO inherit TFDisposable(x : Handle) = 
    // extends Clodable with Loader[Event]

    member this.Load() : Seq<Event> = 
        // Catches the enxt event stored in the file
        seq {
            try

            with
            // TODO narrow to OutOfRangeException and DataLossException or the equivilent 
            | :? Exception
        
        }
       NativeReader.RecordReaderWrapperReadnext()
        //Event.parseFrom
    let x =1 0

/// Accumulates event values collected from the provided path.
///
/// The [[EventAccumulator]] is intended to provide a convenient interface for loading event data written during a
/// TensorFlow run (or otherwise). TensorFlow writes out event ProtoBuf objects, which have a timestamp and step number
/// associated with them, and often also contain a [[Summary]]. Summaries can store different kinds of data like a
/// scalar value, an image, audio, or a histogram. Each summary also has a tag associated with it, which we use to
/// organize logically related data. The [[EventAccumulator]] supports retrieving the event and summary data by their
/// tags.
///
/// Calling `tags` returns a map from event types to the associated tags for those types, that were found in the loaded
/// event files. Then, various functional endpoints (e.g., `scalars(tag)`) allow for the retrieval of all data
/// associated with each tag.
///
/// The `reload()` method synchronously loads all of the data written so far.
///
/// @param  path                    Path to a directory containing TensorFlow events files, or a single TensorFlow
///                                 events file. The accumulator will load events from this path.
/// @param  sizeGuidance            Information on how much data the event accumulator should store in memory. The
///                                 default size guidance tries not to store too much so as to avoid consuming all of
///                                 the client's memory. The `sizeGuidance` should be a map from event types to integers
///                                 representing the number of items to keep in memory, per tag for items of that event
///                                 type. If the size is `0`, then all events are stored. Images, audio, and histograms
///                                 tend to be very large and thus storing all of them is not recommended.
/// @param  histogramCompressionBps Information on how the event accumulator should compress histogram data for the
///                                 [[CompressedHistogramEventType]] event type.
/// @param  purgeOrphanedData       Boolean value indicating whether to discard any events that were "orphaned" by a
///                                 TensorFlow restart.
///
type EventAccumulator(
                      path : string, 
                      ?sizeGuidance : Map<EventType,int>,
                      ?histogramCompressionBps : int[],
                      ?purgeOrphanedData : bool) as this = 

    // TODO maybe something other than seq here
    let eventLoader : unit -> seq<Event> = EventAccumulator.EventLoaderFromPath(path)
    static let defaultSizeGuidance (e:EventType) = 
        match e with
        | ScalarEventType -> 110000
        | ImageEventType -> 4
        | AudioEventType -> 4
        | HistogramEventType -> 1
        | CompressedHistogramEventType -> 500
        | TensorEventType -> 10
        | GraphEventType -> 1
        | MetaGraphEventType -> 1
        | RunMetadataEventType -> 1

    /// Default histogram compression BPS to use. The Normal CDF for standard deviations:
    /// (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf) naturally gives bands around the median of width 1 std dev, 2 std dev,
    /// 3 std dev, and then the long tail. 
    static let defaultHistogramCompressionBps = 
        [|0; 668; 1587; 3085; 5000; 6915; 8413; 9332; 10000|]

    let sizeGuidance = defaultArg sizeGuidance Map.empty
    let histogramComrpessionBps = defaultArg histogramCompressionBps defaultHistogramCompressionBps
    let purgeOrphanedData = defaultArg purgeOrphanedData 

    let actualSizeGuidance (e : EventType) = sizeGuidance.TryFind(e) |> Option.defaultWith (fun _ -> defaultSizeGuidance(e))
        
    let scalarReservoir = Reservoir<string,ScalarEventRecord>(actualSizeGuidance(ScalarEventType))
    let imageReservoir = Reservoir<string,ImageEventRecord>(actualSizeGuidance(ImageEventType))
    let audioReservoir = Reservoir<string,AudioEventRecord>(actualSizeGuidance(AudioEventType))
    let histogramReservoir = Reservoir<string,HistogramEventRecord>(actualSizeGuidance(HistogramEventType))
    let compressedHistogramReservoir = Reservoir<string,CompressedHistogramEventRecord>(actualSizeGuidance(CompressedHistogramEventType))
    let tensorReservoir = Reservoir<string,TensorEventRecord>(actualSizeGuidance(TensorEventType))

    let mutable graphDef : ByteString option = None
    let mutable graphFromMetaGraph : bool = false
    let mutable metaGraphDef : ByteString option = None
    let mutable taggedRunMetadata : Map<string,ByteString> = Map.empty
    let mutable summaryMetadata : Map<string,SummaryMetadata> = Map.empty

    /// Keep a mapping from plugin name to a map from tag to plugin data content obtained from the summary metadata for
    /// that plugin (this is not the entire summary metadata proto - only the content for that plugin). The summary writer
    /// only keeps the content on the first event encountered per tag, and so we must store that first instance of content
    /// for each tag.
    let mutable pluginTagContent = Map.empty<string,Map<string,string>>

    /// Loads all events added since the last call to `reload()` and returns this event accumulator. If `reload()` was
    /// never called before, then it loads all events in the path.
    let reload() = 
        lock this (fun _ ->
            //eventLoader().foreach(processEvent)
            failwith "todo"
        )
    //private[this] val _pluginTagContent: mutable.Map[String, mutable.Map[String, String]] = mutable.Map.empty

    member this.FirstEventTimeStampe 
        with get() : float =
            lock this (fun _ ->
                failwith "todo"
            )

    static member EventLoaderFromPath(path : string) : unit -> seq<Event> = 
        if File.Exists(path) && Path.GetFileName.Contains("tfevents") then
            fun () -> EventFileReader(path).Load()
        else 
            fun () -> failwith "todo" //TODO    DirectoryLoader(path, EventFileReader(_), fun p -> Path.GetFileName(p).Contains("tfevents")).Load()
//    //let eventLoader : unit -> seq<Event> = EventAccumulator.eventLoaderFromPth(path)
//
