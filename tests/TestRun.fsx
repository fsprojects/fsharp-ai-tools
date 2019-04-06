#r "netstandard"
#r @"C:\EE\Git\Tensorflow_FSharp\packages\Google.Protobuf.3.7.0\lib\net45\Google.Protobuf.dll"
#r @"C:\EE\Git\TensorFlow.NET\src\TensorFlowNET.Core\bin\Debug\netstandard2.0\NumSharp.Core.dll"
#r @"C:\EE\Git\TensorFlow.NET\src\TensorFlowNET.Core\bin\Debug\netstandard2.0\TensorFlow.NET.dll"


open Tensorflow

type ops = Operations.gen_ops



let a = tf.constant(4.0f)
let b = tf.constant(5.0f)
let c = tf.add(a,b)

let sess = tf.Session()
let o = sess.run(c)

let input_img = tf.constant(1.0f,TF_DataType.TF_FLOAT, [|1;10;10;1|])

let filters = tf.random_normal([|3;3;1;1|])

//let conv = tf2.nn.conv2d(input_img, filters, [|1;1;1;1|], padding="SAME", dilations = [|1;1;1;1|], data_format = "NHWC", use_cudnn_on_gpu = false, name = "Conv2D2")

let conv = ops.conv2d(input_img, filters, [|1;2;2;1|], padding="SAME", dilations = [|1;1;1;1|])

//sess.run(tf.nn.relu(tf.constant(-4.0f)))
let foo = sess.run(conv) 

printfn "%A" foo