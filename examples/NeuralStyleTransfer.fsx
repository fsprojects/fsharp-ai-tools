#I __SOURCE_DIRECTORY__
#r "netstandard"
#I "../tests/bin/Debug/net461/"
#r "TensorFlow.FSharp.dll"
#r "TensorFlow.Net.dll"
#r "NumSharp.Core.dll"
#r "FSharp.AI.Tests.dll"
#r "System.Memory.dll"
#nowarn "25"

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

let img_tf = new NDArray(File.ReadAllBytes(Path.Combine(Utils.basePath,"images","chicago.jpg")))

let res = sess.run(output,FeedItem(input_data,img_tf))

let getImg(m:NDArray) : byte[] = 
    let [|_;H;W;C|] = m.shape
    let xs = m.Data<uint8>()
    Array3D.init H W C (fun h w c ->  xs.[h * (W * C) + w * C + c])
    |> arrayToPNG_HWC

let data = getImg(res)

File.WriteAllBytes(Path.Combine(Utils.basePath,"images","chicargo_rain"),data)
