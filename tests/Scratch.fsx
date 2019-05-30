open System.Windows.Forms

#I __SOURCE_DIRECTORY__
#r "netstandard"
#I @"..\tests\bin\Debug\net461"
#r "TensorFlow.FSharp.dll"
#r "TensorFlow.Net.dll"
#r "NumSharp.Core.dll"

#nowarn "49"

//open TensorFlow.FSharp
open System.Text
open Tensorflow
open Tensorflow.Operations
open System.ComponentModel
open NumSharp

let a = tf.constant(4.0f)
let b = tf.constant(5.0f)
let c = tf.add(a,b)
let sess = tf.Session()
let o = sess.run(c)



let foo = tf.get_variable("foo",TensorShape(1,2,3), TF_DataType.TF_FLOAT)
let input_img = tf.constant(1.0f,TF_DataType.TF_FLOAT, [|1;10;10;1|])
let filters = tf.random_normal([|3;3;1;1|])
let input_sizes = tf.constant([|1;20;20;1|], TF_DataType.TF_INT64)

//let conv2res = tf2.nn.conv2d_tranpose(input_sizes, filters, input_img, strides = [|1;2;2;1|], padding="SAME", dilations = [|1;1;1;1|], data_format = "NHWC", use_cudnn_on_gpu = false, name = "Conv2D22")
let convres = gen_ops.conv2d(input_img, filters, [|1;2;2;1|], padding="SAME", dilations = [|1;1;1;1|])


sess.run(gen_ops.relu(tf.constant(-4.0f)))
//sess.run(tf2.global_variables_initializer().output)
let convres2 = sess.run(convres)


//[|1|].GetType().FullName


module Array3D = 
    /// Not properly tested, probably slow, not to be used beyond a temporary work around
    let flatten(xsss:'a[,,]) = 
      [| for x in 0..(xsss.GetLength(0)-1) do 
          for y in 0..(xsss.GetLength(1)-1) do 
            for z in 0..(xsss.GetLength(2)-1) -> xsss.[x,y,z]|]

let value = Array3D.zeroCreate<single> 712 474 3
value |> Array3D.flatten

let nd = new NDArray(value |> Array3D.flatten, NumSharp.Shape(value.GetLength(0),value.GetLength(1),value.GetLength(2)))
nd.Array
nd.Data<single>()
nd.shape
(*
let convres2 = sess.run(conv2res)
convres2.shape
convres2.shape 

let x = { features = tf.constant(4.0f)} : tf.ReluArgs 

tf._op_def_lib.Force()._apply_op_helper("Relu", args = x)

[|for propertyDescriptor in TypeDescriptor.GetProperties(x) -> propertyDescriptor.GetValue(x) |]
*)
