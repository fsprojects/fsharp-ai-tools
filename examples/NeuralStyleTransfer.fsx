#I __SOURCE_DIRECTORY__
#r "netstandard"
#I "../tests/bin/Debug/net461/"
#r "TensorFlow.FSharp.dll"
#r "TensorFlow.Net.dll"
#r "NumSharp.Core.dll"
#r "FSharp.AI.Tests.dll"
#r "System.Memory.dll"
#nowarn "25"

open System
open System.IO
open Tensorflow
open Tensorflow.Operations
open NumSharp
open FSharp.AI.Tests
open TensorFlow.FSharp.ImageWriter

let sess = new Session()
let weights = Utils.fetchStyleWeights("rain")
let input_data = tf.placeholder(TF_DataType.TF_STRING)
let input_img = TF.VGGStyleTransfer.binaryJPGToImage(input_data)
let output = TF.VGGStyleTransfer.model(input_img,weights)
let img_tf = NDArray(File.ReadAllBytes(Path.Combine(Utils.basePath,"images","chicago.jpg")))
let styled_img = sess.run(output,FeedItem(input_data,img_tf)) 
File.WriteAllBytes(Path.Combine(Utils.basePath,"images","chicago_rain.png"),styled_img |> ndarrayToPNG_NHWC)


// TODO: Find out why String output does not appear to return the enture amount of jpeg data
//let input = tf.placeholder(TF_DataType.TF_UINT8)
//let output_string = sess.run(gen_ops.encode_base64(gen_ops.encode_jpeg(tf.squeeze(input))),FeedItem(input,styled_img))
