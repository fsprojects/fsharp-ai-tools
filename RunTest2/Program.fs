module Program
// There are some problems with mixing .net backends which resulted in some weird errors

//#I __SOURCE_DIRECTORY__
//#r "netstandard"
//#I "../tests/bin/Debug/net461/"
//#r "TensorFlow.FSharp.dll"
//#r "Tensorflow.NET.dll"
//#r "NumSharp.Core.dll"
//#r "TensorFlow.FSharp.Tests.dll"
////#r "Ionic.ZLib.Core.dll"
//#r "System.IO.Compression.dll"
//#nowarn "760" "49"
//#load "../tests/NPYReaderWriter.fsx"

module Classifier = 
    open Tensorflow
    open Tensorflow.Operations
    open TensorFlow.FSharp
    open System.IO
    open System
    open System.Collections.Generic
    open TensorFlow.FSharp.NPYReaderWriter
    open NumSharp

    let run() =
      if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

      let test_dir = Path.Combine(__SOURCE_DIRECTORY__, "..", "tests")

      let pretrained_dir = Path.Combine(test_dir,"pretrained")
      let weights_path = Path.Combine(pretrained_dir, "resnet_classifier_1000.npz")
      let labels_path = Path.Combine(pretrained_dir,"imagenet1000.txt")
      let example_dir = Path.Combine(test_dir,"examples")
      let label_map = File.ReadAllLines(labels_path)
      let sess = new Session()
      let weights_map = 
          readFromNPZ(File.ReadAllBytes(weights_path))
          |> Map.map (fun k (metadata,arr) ->
              // TODO: make initialization of tensor to be the right shape w/o doing a graph.Reshape here
              // This way the graph functions defined here will be defined as part of the model and not feeding into the model
              tf.reshape(tf.constant(arr), tf.constant(metadata.shape)))

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

      let classifyFile(path:string) =
          let createTensorFromImageFile(file:string,destinationDataType:TF_DataType) =
              let tensor = new NDArray(File.ReadAllBytes(file)) // This was CreateString, if this works then delete this note
              //sess.run(inputs = [|img_input|], inputValues = [|tensor|], outputs = [|img_output|]).[0]
              sess.run(img_output,FeedItem(img_input,tensor))
          let example = createTensorFromImageFile(path, TF_DataType.TF_FLOAT)
          let index = gen_ops.arg_max(output,tf.constant(1),output_type=Nullable(TF_DataType.TF_INT32))
          let res = sess.run(index, FeedItem(input_placeholder,example)) //inputs = [|input_placeholder|], inputValues = [|example|], outputs = [|index|])
          //label_map.[(res.[0] :?> int64[]) |> Array.item 0 |> int]
          label_map.[res.GetInt32(0)]

      printfn "example_0.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_0.jpeg")))
      printfn "example_1.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_1.jpeg")))
      printfn "example_2.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_2.jpeg")))

module FFStyle = 
    open System
    open System.IO
    open NumSharp
    open Argu
    open Tensorflow
    open Tensorflow.Operations
    open TensorFlow.FSharp
    open TensorFlow.FSharp.NPYReaderWriter
    open TensorFlow.FSharp.ImageWriter
    //    type Argument =
    //        | [<AltCommandLine([|"-s"|])>] Style of string
    //        with
    //            interface IArgParserTemplate with
    //                member this.Usage =
    //                    match this with
    //                    |  Style _ -> "Specify a style of painting to use."

    let run(style:string) =
      if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)
      //let style = ArgumentParser<Argument>().Parse(fsi.CommandLineArgs.[1..]).GetResult(<@ Argument.Style @>, defaultValue = "rain")
      let test_dir = Path.Combine(__SOURCE_DIRECTORY__, "..", "tests")
      let pretrained_dir = Path.Combine(test_dir,"pretrained")
      let example_dir = Path.Combine(test_dir,"examples")
      let sess = new Session()
      let weights_path = Path.Combine(pretrained_dir, sprintf "fast_style_weights_%s.npz" style)
      let weights = 
          readFromNPZ((File.ReadAllBytes(weights_path)))
          |> Map.toArray 
          |> Array.map (fun (k,(metadata, arr)) -> 
              k.Substring(0, k.Length-4), gen_ops.reshape(tf.constant(arr), tf.constant(metadata.shape))) 
          |> Map.ofArray

      let input_string = tf.placeholder(TF_DataType.TF_STRING)
      let output_img = 
          let mean_pixel = tf.constant([|123.68f; 116.778f; 103.939f|])
          let img = 
              let decoded = tf.cast(gen_ops.decode_jpeg(contents=input_string, channels=Nullable(3)), TF_DataType.TF_FLOAT)
              let preprocessed = tf.sub(decoded,mean_pixel)
              let expanded = gen_ops.expand_dims(input=preprocessed, dim = tf.constant(0))
              //let resized = graph.ResizeBicubic(expanded,graph.Const(new TFTensor([|256;256|])),align_corners=Nullable(true))
              expanded
          let output = FFStyleVGG.model(img, weights)
          let (+) x (y:float32) = tf.add(x,tf.constant(y))
          let (*) x (y:float32) = tf.multiply(x,tf.constant(y))
          let tanh x = tf.tanh(x)
          let clip_by_value(low:float32,hight:float32) x = gen_ops.clip_by_value(x,tf.constant(low), tf.constant(hight))
          let to_pixel_value x = (tanh(x) * 150.f) + (255.f/2.f)
          output
          |> to_pixel_value
          |> clip_by_value(0.f,255.f)
      let img_tf = new NDArray(File.ReadAllBytes(Path.Combine(example_dir,"chicago.jpg")))
      let img_styled = sess.run(output_img,FeedItem(input_string,img_tf))
      let res = img_styled.Array :?> single[]
      let shape = img_styled.shape
      let _,H,W,C = shape.[0], shape.[1], shape.[2], shape.[3] 
      let xss : single[,,] = Array3D.init H W C (fun h w c ->  res.[h * (W * C) + w * C + c])
      File.WriteAllBytes(Path.Combine(__SOURCE_DIRECTORY__, sprintf "chicago_in_%s_style.bmp" style), ImageWriter.arrayToPNG_HWC(xss))

[<EntryPoint>]
let main argv =
    //FFStyle.run("rain")
    //FFStyle.run("starry_night")
    //FFStyle.run("wave")
    //Classifier.run()
    NeuralStyleTransferDSL.run()
    0 