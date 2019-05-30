module Program

#nowarn "760" "49"

open Tensorflow
open Tensorflow.Operations
open TensorFlow.FSharp
open System.IO
open System
open System.Collections.Generic
open TensorFlow.FSharp.NPYReaderWriter

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

let test_dir = Path.Combine(__SOURCE_DIRECTORY__, "..", "tests")

let pretrained_dir = Path.Combine(test_dir,"pretrained")
let weights_path = Path.Combine(pretrained_dir, "resnet_classifier_1000.npz")
let labels_path = Path.Combine(pretrained_dir,"imagenet1000.txt")
let example_dir = Path.Combine(test_dir,"examples")
let label_map = File.ReadAllLines(labels_path)
let sess = new Session()
let graph = sess.graph


let xs2 = readFromNPZ((File.ReadAllBytes(weights_path)))

let weights_map = 
    readFromNPZ(File.ReadAllBytes(weights_path))
    |> Map.map (fun k (metadata,arr) ->
        // TODO: make initialization of tensor to be the right shape w/o doing a graph.Reshape here
        // This way the graph functions defined here will be defined as part of the model and not feeding into the model
        tf.reshape(tf.constant(arr), tf.constant(metadata.shape)))

//weights_map.Count
//xs2.Count
//let weights_map2 = xs |> Map.map (fun k (metadata,arr) -> 1)

for KeyValue(x,y) in weights_map do 
  printfn "%s" x

/// This is from TensorflowSharp (Examples/ExampleCommon/ImageUtil.cs)
/// It's intended for inception but used here for resnet as an example
/// of this type of functionality 
let construtGraphToNormalizeImage(destinationDataType : TF_DataType) =
    let W = 224
    let H = 224
    let Mean = 117.0f
    let Scale = 1.0f
    let input = tf.placeholder(TF_DataType.TF_STRING)
    let loaded_img = tf.cast(gen_ops.decode_jpeg(contents=input,channels=Nullable(3)),TF_DataType.TF_FLOAT)
    let expanded_img = gen_ops.expand_dims(input=loaded_img, dim = tf.constant(0))
    let resized_img = gen_ops.resize_bilinear(expanded_img,tf.constant([|W;H|]))
    let final_img = gen_ops.div(gen_ops.sub(resized_img, tf.constant([|Mean|])), tf.constant(([|Scale|])))
    (input,tf.cast(final_img,destinationDataType))

let img_input,img_output = construtGraphToNormalizeImage(TF_DataType.TF_FLOAT)

let input_placeholder = tf.placeholder(TF_DataType.TF_FLOAT, shape=TensorShape(-1,-1,-1,3), name="new_input")
let output = ResNet50.model(input_placeholder,weights_map)



let getWeights(name:string) = 
  printfn "key name %s" name
  printfn "key name %s" name
  printfn "key name %s" name
  printfn "key name %s" name
  weights_map.[name + ".npy"]

let finalWeights = getWeights("fc1000/fc1000_W:0")

let classifyFile(path:string) =
    let createTensorFromImageFile(file:string,destinationDataType:TF_DataType) =
        let tensor = Tensor(File.ReadAllBytes(file)) // This was CreateString, if this works then delete this note
        //sess.run(inputs = [|img_input|], inputValues = [|tensor|], outputs = [|img_output|]).[0]
        sess.run(img_output,FeedItem(img_input,tensor))
    let example = createTensorFromImageFile(path, TF_DataType.TF_FLOAT)
    let index = gen_ops.arg_max(output,tf.constant(1))
    let res = sess.run(index, FeedItem(input_placeholder,example)) //inputs = [|input_placeholder|], inputValues = [|example|], outputs = [|index|])
    printfn "%A" res // TODO remove this debug line
    //label_map.[(res.[0] :?> int64[]) |> Array.item 0 |> int]
    label_map.[res.GetInt32(0)]


[<EntryPoint>]
let main argv = 
    printfn "%A" argv
    printfn "example_0.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_0.jpeg")))
    printfn "example_1.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_1.jpeg")))
    printfn "example_2.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_2.jpeg")))
    0 // return an integer exit code
