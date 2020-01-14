#if INTERACTIVE
#I __SOURCE_DIRECTORY__
#I "../../tests/bin/Debug/netcoreapp3.1/"
#r "NumSharp.Core.dll"
#r "TensorFlow.Net.dll"
#r "FSAI.Tools.dll"
#r "FSAI.Tools.Tests.dll"
#nowarn "25"
#endif

#if NOTEBOOK
#r "nuget: TODO"
#endif

open System
open System.IO
open Tensorflow
open NumSharp
open FSAI.Tools.Tests

let tf = Tensorflow.Binding.tf

let sess = new Session()
let weights = Utils.fetchStyleWeights("rain")
let input_data = tf.placeholder(TF_DataType.TF_FLOAT)
let output = TF.VGGStyleTransfer.model(input_data,weights)
let input_path = Path.Combine(Utils.basePath,"images","chicago.jpg")
let input_img = TF.VGGStyleTransfer.binaryJPGToImage(input_path)
let styled_img = sess.run(output,FeedItem(input_data,input_img)) 

#if COMPILED
let v = sprintf "running test in %s at %A" __SOURCE_FILE__ System.DateTime.Now
open NUnit.Framework
[<Test>]
let ``run test`` () = 
    v |> ignore
#endif
