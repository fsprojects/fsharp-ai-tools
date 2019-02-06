module TensorFlow.FSharp.IO.SummaryFileWriter

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

type Event =
| ScalarEventRecord of EventRecord<float32>
| ImageEventRecord of EventRecord<ImageValue>
| AudoeEventRecord of EventRecord<AudioValue>
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

            
/// Event file reader.
///
/// An event file reader is used to create iterators over the events stored in the file at the provided path (i.e.,
/// `filePath`).
///
/// Note that this reader ignores any corrupted records at the end of the file. That is to allow for "live tracking" of
/// summary files while they're being written to.
///
/// @param  filePath        Path to the file being read.
/// @param  compressionType Compression type used for the file.
/// TODO pick up from here
//type EventFileReader internal (
//    filePath : string
//    ?compressionType : CompressionType
//
//    ) inherit TFDisposable(x : Handle)

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
                      ?sizeGuidance : EventType -> int,
                      ?histogramCompressionBps : int[],
                      ?purgeOrphanedData : bool) = 

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

    let sizeGuidance = defaultArg sizeGuidance defaultSizeGuidance
    let histogramComrpessionBps = defaultArg histogramCompressionBps defaultHistogramCompressionBps
    let purgeOrphanedData = defaultArg purgeOrphanedData 

    let eventLoader : unit -> seq<Event> = EventAccumulator.eventLoaderFromPth(path)

