#I __SOURCE_DIRECTORY__
#r "netstandard"
<<<<<<< Updated upstream
#I "../tests/bin/Debug/net472/"
#r "FSAI.Tools.dll"
=======
#I "../tests/bin/Debug/netcoreapp2.0/"
#r "FSharp.AI.dll"
>>>>>>> Stashed changes
#r "Tensorflow.NET.dll"
#r "NumSharp.Core.dll"
#r "FSAI.Tools.Tests.dll"
#r "System.IO.Compression.dll"
#r "System.Memory"
#r "Markeli.Half"
#nowarn "760" "49"

open Tensorflow
open FSAI.Tools
open FSAI.Tools.Tests.TF
open FSAI.Tools.Tests
open System.IO
open NumSharp

#if FS47
open Tensorflow.Binding
#else
let tf = Tensorflow.Binding.tf
#endif

let input_data = tf.placeholder(TF_DataType.TF_STRING,name="input")
let input_img = ResNet50Classifier.binaryJPGToImage(input_data)
let weights = Utils.fetchClassifierWeights()
let output = ResNet50Classifier.model(input_img,weights)

let labels = File.ReadAllLines(Path.Combine(Utils.basePath,"pretrained","imagenet1000.txt"))

let sess = Session()

let classifyImage(path:string) = 
    let label = sess.run(output,FeedItem(input_data,NDArray(File.ReadAllBytes(path)))).Data<int>().[0]
    printfn "%i: %s" label labels.[label]

for i in 0..5 do
    Path.Combine(Utils.basePath,"images",sprintf "example_%i.jpeg" i) |> classifyImage

