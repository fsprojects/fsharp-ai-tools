#if INTERACTIVE
#I __SOURCE_DIRECTORY__
#I "../../tests/bin/Debug/netcoreapp2.0/"
#r "Tensorflow.NET.dll"
#r "NumSharp.Core.dll"
#r "System.IO.Compression.dll"
#r "System.Memory"
#r "Markeli.Half"
#endif
#if NOTEBOOK
#r "nuget: TODO"
#endif

#nowarn "760" "49"

open Tensorflow
open FSAI.Tools
open FSAI.Tools.Tests.TF
open FSAI.Tools.Tests
open System.IO
open NumSharp

let tf = Tensorflow.Binding.tf

let input_data = tf.placeholder(TF_DataType.TF_FLOAT,name="input")
let weights = Utils.fetchClassifierWeights()
let output = ResNet50Classifier.model(input_data,weights)

let labels = File.ReadAllLines(Path.Combine(Utils.basePath,"pretrained","imagenet1000.txt"))

let sess = Session()

let classifyImage(path:string) = 
    let image = ResNet50Classifier.binaryJPGToImage(path)
    let label = sess.run(output,FeedItem(input_data,image)).Data<int>().[0]
    printfn "%i: %s" label labels.[label]

for i in 0..5 do
    Path.Combine(Utils.basePath,"images",sprintf "example_%i.jpeg" i) |> classifyImage

#if COMPILED

let v = sprintf "running test in %s at %A" __SOURCE_FILE__ System.DateTime.Now
open NUnit.Framework
[<Test>]
let ``run test`` () = 
    v |> ignore
#endif


