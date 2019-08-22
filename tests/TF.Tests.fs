module FSharp.AI.Tests.TF.Tests

open NUnit.Framework

open System
open System.IO
open Tensorflow
open Tensorflow.Operations
open NumSharp
open FSharp.AI.Tests

let tf = Tensorflow.Binding.tf

[<Test>]
let ``resnet 50 classifier`` () = 
    let input_data = tf.placeholder (TF_DataType.TF_STRING, name = "input")
    let input_img = ResNet50Classifier.binaryJPGToImage input_data
    let weights = Utils.fetchClassifierWeights ()
    let output = ResNet50Classifier.model (input_img, weights)
    let labels = File.ReadAllLines (Path.Combine (Utils.basePath, "pretrained", "imagenet1000.txt"))
    use sess = new Session ()
    let predicted_labels = [|for i in 0..5 -> sess.run(output, FeedItem (input_data, NDArray (File.ReadAllBytes (Path.Combine (Utils.basePath, "images", sprintf "example_%i.jpeg" i))))).Data<int>().[0]|]
    let actual_labels = [|360;319;610;84;696;272|]
    if predicted_labels <> actual_labels then failwithf "Predicted labels of %A did not match actual labels of %A" predicted_labels actual_labels

[<Test>]
let ``vgg style transfer`` () = 
    use sess = new Session ()
    let weights = Utils.fetchStyleWeights ("rain")
    let input_data = tf.placeholder (TF_DataType.TF_STRING)
    let input_img = TF.VGGStyleTransfer.binaryJPGToImage (input_data)
    let output = TF.VGGStyleTransfer.model (input_img, weights)
    let img_tf = NDArray (File.ReadAllBytes (Path.Combine (Utils.basePath, "images", "chicago.jpg")))
    let styled_img = sess.run (output, FeedItem (input_data, img_tf)) 
    let baseline = 
        let data = File.ReadAllBytes ( (Path.Combine (Utils.basePath, "images", "chicago_rain.jpg")))
        let input = tf.placeholder (TF_DataType.TF_STRING) 
        sess.run (gen_ops.decode_jpeg (input, channels = Nullable (3)), FeedItem (input, NDArray (data)))
    let diffImg (img1: NDArray, img2: NDArray) = 
        let input1 = tf.placeholder (TF_DataType.TF_UINT8)
        let input2 = tf.placeholder (TF_DataType.TF_UINT8)
        sess.run(tf.reduce_mean (tf.abs (tf.cast (tf.sub (input1, input2), tf.float32))), [|FeedItem (input1, img1); FeedItem (input2, img2)|]).Data<float32>().[0]
    if diffImg (styled_img, baseline) > 200.f then failwith "Styled image differs from pre-styled baseline image."
