module FSAI.Tools.Tests.TF.Tests

open NUnit.Framework

open System
open System.IO
open Tensorflow
open Tensorflow.Operations
open NumSharp
open FSAI.Tools.Tests

#if FS47
open Tensorflow.Binding
#else
let tf = Tensorflow.Binding.tf
#endif

[<Test>]
let ``resnet 50 classifier`` () = 
    let input_data = tf.placeholder (TF_DataType.TF_FLOAT, name = "input")
    let weights = Utils.fetchClassifierWeights ()
    let output = ResNet50Classifier.model (input_data, weights)
    let labels = File.ReadAllLines (Path.Combine (Utils.basePath, "pretrained", "imagenet1000.txt"))
    use sess = new Session ()
    let predicted_labels =
        [| for i in 0..5 -> 
              let file_name = Path.Combine (Utils.basePath, "images", sprintf "example_%i.jpeg" i)
              let image = ResNet50Classifier.binaryJPGToImage(file_name)
              sess.run(output, FeedItem (input_data, image)).Data<int>().[0]|]
    let actual_labels = [|360;319;610;84;696;272|]
    if predicted_labels <> actual_labels then failwithf "Predicted labels of %A did not match actual labels of %A" predicted_labels actual_labels

[<Test>]
let ``vgg style transfer`` () = 
    use sess = new Session ()
    let weights = Utils.fetchStyleWeights ("rain")
    let input_data = tf.placeholder (TF_DataType.TF_FLOAT)
    let output = TF.VGGStyleTransfer.model (input_data, weights)
    let input_file = Path.Combine (Utils.basePath, "images", "chicago.jpg")
    let input_img = TF.VGGStyleTransfer.binaryJPGToImage (input_file)
    let styled_img = sess.run (output, FeedItem (input_data, input_img)) 
    let baseline = 
        let file = Path.Combine (Utils.basePath, "images", "chicago_rain.jpg")
        TF.VGGStyleTransfer.decodeJPG (file)
    let diffImg (img1: NDArray, img2: NDArray) = 
        let input1 = tf.placeholder (TF_DataType.TF_UINT8)
        let input2 = tf.placeholder (TF_DataType.TF_UINT8)
        sess.run(tf.reduce_mean (tf.abs (tf.cast (tf.sub (input1, input2), tf.float32))), [|FeedItem (input1, img1); FeedItem (input2, img2)|]).Data<float32>().[0]
    if diffImg (styled_img, baseline) > 200.f then failwith "Styled image differs from pre-styled baseline image."
