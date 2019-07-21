module FSharp.AI.Tests.Data.MNist

open NumSharp
open System
open System.IO
open System.IO.Compression
open System.Linq
open System.Net
open System.Threading
open System.Threading.Tasks
open Tensorflow

type Compress() = 
    static member ExtractGZip(gzipFileName : string, targetDir : string) =
        //let dataBuffer = Array.zeroCreate<byte> 4096
        //use fs = new FileStream(gzipFileName, FileMode.Open, FileAccess.Read)
        //use gzipStream = new Ionic.Zlib.GZipStream(fs, Ionic.Zlib.CompressionMode.Decompress)
        let fnOut = Path.Combine(targetDir, Path.GetFileNameWithoutExtension(gzipFileName))
        File.WriteAllBytes(fnOut,Ionic.Zlib.GZipStream.UncompressBuffer(File.ReadAllBytes(gzipFileName)))
        //let fsOut = File.Create(fnOut)
        //StreamUtils.Copy(gzipStream, fsOut, dataBuffer)
        
    static member UnZip(gzArchiveName : string, destFolder : string) =
        let flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin"
        if not(File.Exists(Path.Combine(destFolder, flag))) then
            printfn "Extracting."
            let task = Task.Run(fun () -> ZipFile.ExtractToDirectory(gzArchiveName, destFolder))
            while (not task.IsCompleted) do
                Thread.Sleep(200)
                Console.Write(".")
            File.Create(Path.Combine(destFolder, flag)) |> ignore
            printfn ""
            printfn "Extracting is complete"

type Web() =
    /// TODO make this better 
    static member Download(url : string, destDir : string, ?destFileName : string) = 
        let destFileName = defaultArg destFileName (url.Split(Path.DirectorySeparatorChar).Last())
        Directory.CreateDirectory(destDir) |> ignore
        let relativeFilePath = Path.Combine(destDir, destFileName)
        if File.Exists(relativeFilePath) then
            printfn "%s already exists." destFileName
            false
        else 
            use wc = new WebClient()
            printfn "Downloading %s" relativeFilePath
            let download = Task.Run(fun () -> wc.DownloadFile(url, relativeFilePath))
            while (not download.IsCompleted) do
                Thread.Sleep(1000)
                Console.Write(".")
            printfn ""
            printfn "Downloaded %s" relativeFilePath
            true
            

type DataSet(images : NDArray, labels : NDArray, dtype : TF_DataType, reshape : bool) = 
    let _num_examples = images.shape.[0]
    let images = images.reshape(images.shape.[0], images.shape.[1] * images.shape.[2])
    let mutable _images = np.multiply(images.astype(dtype.as_numpy_datatype()), NDArray.op_Implicit(1.0f / 255.f))
    let mutable _labels = labels.astype(dtype.as_numpy_datatype())
    let mutable _epochs_completed = 0
    let mutable _index_in_epoch = 0

    member this.Images = _images
    member this.Labels = _labels
    member this.EpochsCompleted = _epochs_completed
    member this.IndexInEpoch = _index_in_epoch
    member this.NumExamples = _num_examples

    member this.next_batch(batch_size : int, ?fake_data : bool, ?shuffle : bool) = 
        let fake_data = defaultArg fake_data false
        let shuffle = defaultArg shuffle true
        let start = _index_in_epoch
        let applyShuffle() = 
            let perm0 = np.arange(_num_examples)
            np.random.shuffle(perm0)
            _images <- _images.[perm0]
            _labels <- _labels.[perm0]

        if _epochs_completed = 0 && start = 0 && shuffle then applyShuffle()

        // Go to the next epoch
        if start + batch_size > _num_examples then
            // Finished epoch
            _epochs_completed <- _epochs_completed + 1

            // Get the rest examples in this epoch
            let rest_num_examples = _num_examples - start
            if shuffle then applyShuffle()

            let mutable start = 0
            _index_in_epoch <- batch_size - rest_num_examples
            let _end = _index_in_epoch
            (_images.[np.arange(start, _end)], _images.[np.arange(start, _end)])
        else
            _index_in_epoch <- _index_in_epoch + batch_size
            let _end = _index_in_epoch
            (_images.[np.arange(start,_end)],_labels.[np.arange(start,_end)])


type Datasets = {
    train : DataSet
    validation : DataSet
    test : DataSet
}


// Re-doing the dataset program
[<AutoOpen>]
module Dataset = 

    let DEFAULT_SOURCE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    let TRAIN_IMAGES = "train-images-idx3-ubyte.gz";
    let TRAIN_LABELS = "train-labels-idx1-ubyte.gz";
    let TEST_IMAGES = "t10k-images-idx3-ubyte.gz";
    let TEST_LABELS = "t10k-labels-idx1-ubyte.gz";
    let private _read32(bytestream : FileStream) =  
        let buffer = Array.zeroCreate<byte> (sizeof<uint32>)
        let count = bytestream.Read(buffer, 0, 4)
        np.frombuffer(buffer, ">u4").Data<uint32>(0) //MM) Is this really necessary?
    let dense_to_one_hot(labels_dense : NDArray, num_classes : int) =
        let num_labels = labels_dense.shape.[0]
        let index_offset = np.arange(num_labels) * NDArray.op_Implicit(num_labels)
        let labels_one_hot = np.zeros(num_labels, num_classes)
        for row in 0 .. num_labels - 1 do
            let col = int(labels_dense.Data<byte>(row))
            labels_one_hot.SetData(1.0, row,col)
        labels_one_hot

    type MNistDataset() =
        static member extract_images(file : string, ?limit : int) : NDArray =
            use bytestream = new FileStream(file, FileMode.Open)
            let magic = _read32(bytestream)
            if magic <> 2051u then raise <| ValueError(sprintf "Invalid magic number %i in MNIST image file %s" magic file)
            let num_images = int(_read32(bytestream))
            let rows = int(_read32(bytestream))
            let cols = int(_read32(bytestream))
            let buf = Array.zeroCreate<byte> (rows * cols * num_images)
            bytestream.Read(buf, 0, buf.Length) |> ignore
            let data = np.frombuffer(buf, np.uint8)
            data.reshape(num_images, rows, cols, 1)
            
        static member extract_labels(file : string, ?one_hot : bool, ?num_classes : int, ?limit : int) : NDArray =
            let one_hot = defaultArg one_hot false
            let num_classes = defaultArg num_classes 10
            use bytestream = new FileStream(file, FileMode.Open)
            let magic = _read32(bytestream)
            if magic <> 2049u then raise <| ValueError(sprintf "Invalid magic number %i in MNIST label file %s" magic file)
            let num_images = int(_read32(bytestream))
            let buf = Array.zeroCreate<byte> (num_images)
            bytestream.Read(buf, 0, buf.Length) |> ignore
            let labels = np.frombuffer(buf, np.uint8)
            if one_hot then dense_to_one_hot(labels,num_classes)
            else labels

        static member read_data_sets(train_dir : string, 
                                     ?one_hot : bool, 
                                     ?dtype : TF_DataType,
                                     ?reshape : bool,
                                     ?validation_size : int,
                                     ?train_size : int,
                                     ?test_size : int,
                                     ?source_url) =
            let one_hot = defaultArg one_hot false
            let dtype   = defaultArg dtype TF_DataType.TF_FLOAT
            let reshape = defaultArg reshape true
            let validation_size = defaultArg validation_size 5000
            let source_url = defaultArg source_url DEFAULT_SOURCE_URL
            train_size |> Option.iter (fun train_size -> 
                if validation_size >= train_size then raise <| ArgumentException("Validation set should be smaller than training set"))
            Web.Download(source_url + TRAIN_IMAGES, train_dir, TRAIN_IMAGES) |> ignore
            Compress.ExtractGZip(Path.Combine(train_dir, TRAIN_IMAGES), train_dir)
            let train_images = MNistDataset.extract_images(Path.Combine(train_dir, TRAIN_IMAGES.Split('.').[0]), ?limit = train_size)

            Web.Download(source_url + TRAIN_LABELS, train_dir, TRAIN_LABELS) |> ignore
            Compress.ExtractGZip(Path.Combine(train_dir, TRAIN_LABELS), train_dir)
            let train_labels = MNistDataset.extract_labels(Path.Combine(train_dir, TRAIN_LABELS.Split('.').[0]), one_hot = one_hot, ?limit = train_size)

            Web.Download(source_url + TEST_IMAGES, train_dir, TEST_IMAGES) |> ignore
            Compress.ExtractGZip(Path.Combine(train_dir, TEST_IMAGES), train_dir)
            let test_images = MNistDataset.extract_images(Path.Combine(train_dir, TEST_IMAGES.Split('.').[0]), ?limit = test_size)

            Web.Download(source_url + TEST_LABELS, train_dir, TEST_LABELS) |> ignore
            Compress.ExtractGZip(Path.Combine(train_dir, TEST_LABELS), train_dir)
            let test_labels = MNistDataset.extract_labels(Path.Combine(train_dir, TEST_LABELS.Split('.').[0]), one_hot = one_hot, ?limit = test_size)

            let _end = train_images.shape.[0]
            let validation_images = train_images.[np.arange(validation_size)]
            let validation_labels = train_labels.[np.arange(validation_size)]
            let train_images = train_images.[np.arange(validation_size)]
            let train_labels = train_labels.[np.arange(validation_size)]

            let train = new DataSet(train_images, train_labels, dtype, reshape)
            let validation = new DataSet(validation_images, validation_labels, dtype, reshape)
            let test = new DataSet(test_images, test_labels, dtype, reshape)
            {train = train; validation = validation; test = test}
