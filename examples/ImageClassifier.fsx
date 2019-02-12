#I __SOURCE_DIRECTORY__
#r "netstandard"
#I "../tests/bin/Debug/net461/"
#r "TensorFlow.FSharp.dll"
#r "TensorFlow.FSharp.Tests.dll"
#r "TensorFlow.FSharp.Proto.dll"
#nowarn "760" "49"

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
let sess = new TFSession()
let graph = sess.Graph

let weights_map = 
    readFromNPZ((File.ReadAllBytes(weights_path)))
    |> Map.map (fun k (metadata,arr) ->
        // TODO: make initialization of tensor to be the right shape w/o doing a graph.Reshape here
        // This way the graph functions defined here will be defined as part of the model and not feeding into the model
        graph.Reshape(graph.Const(new TFTensor(arr)), graph.Const(TFShape(metadata.shape |> Array.map int64).AsTensor()))) 


/// This is from TensorflowSharp (Examples/ExampleCommon/ImageUtil.cs)
/// It's intended for inception but used here for resnet as an example
/// of this type of functionality 
let construtGraphToNormalizeImage(destinationDataType:TFDataType) =
    let W = 224
    let H = 224
    let Mean = 117.0f
    let Scale = 1.0f
    let input = graph.Placeholder(TFDataType.String)
    let loaded_img = graph.Cast(graph.DecodeJpeg(contents=input,channels=3L),TFDataType.Float32)
    let expanded_img = graph.ExpandDims(input=loaded_img, dim = graph.Const(TFTensor(0)))
    let resized_img = graph.ResizeBilinear(expanded_img,graph.Const(TFTensor([|W;H|])))
    let final_img = graph.Div(graph.Sub(resized_img, graph.Const(TFTensor([|Mean|]))), graph.Const(TFTensor([|Scale|])))
    (input,graph.Cast(final_img,destinationDataType))

let img_input,img_output = construtGraphToNormalizeImage(TFDataType.Float32)

let input_placeholder = graph.Placeholder(TFDataType.Float32, shape=TFShape(-1L,-1L,-1L,3L), name="new_input")
let output = ResNet50.model(graph,input_placeholder,weights_map)

let classifyFile(path:string) =
    let createTensorFromImageFile(file:string,destinationDataType:TFDataType) =
        let tensor = TFTensor.CreateString(File.ReadAllBytes(file))
        sess.Run(inputs = [|img_input|], inputValues = [|tensor|], outputs = [|img_output|]).[0]
    let example = createTensorFromImageFile(path, TFDataType.Float32)
    let index = graph.ArgMax(output,graph.Const(TFTensor(1)))
    let res = sess.Run(inputs = [|input_placeholder|], inputValues = [|example|], outputs = [|index|])
    label_map.[res.[0].GetValue() :?> int64[] |> Array.item 0 |> int]

printfn "example_0.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_0.jpeg")))
printfn "example_1.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_1.jpeg")))
printfn "example_2.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_2.jpeg")))
