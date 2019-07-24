#I __SOURCE_DIRECTORY__
#r "netstandard"
#I "../tests/bin/Debug/net472/"
#r "TensorFlow.FSharp.dll"
#r "TensorFlow.Net.dll"
#r "NumSharp.Core.dll"
#r "FSharp.AI.Tests.dll"
#r "ICSharpCode.SharpZipLib.dll"
#r "System.IO.Compression.dll"
#r "System.IO.Compression.FileSystem.dll"
#r "System.IO.Compression.ZipFile.dll"
#r "System.Runtime.Extensions.dll"

// TODO needs better random initializers that draw from a normal
// TODO tf.get_variable is not implemented
// TODO get_gradient_function Conv2D is not implemented
// TODO implement dropout to improve accuracy

open System
open Tensorflow
open Tensorflow.Operations
open NumSharp
open FSharp.AI.Tests.Data

let mnist = MNist.Dataset.MNistDataset.read_data_sets("mnist",one_hot = true, validation_size = 5000)

let xtr = tf.placeholder(tf.float32, TensorShape(-1, 784))
let ytr = tf.placeholder(tf.float32, TensorShape(-1, 10))

let getRandom(shape:int[]) = 
    np.random.randn(shape).astype(typeof<single>)

let basicModel(x) : Tensor = 
    let W = tf.Variable(getRandom([|784;128|]), name = "weight")
    let b = tf.Variable(getRandom([|128|]), name = "bias",dtype = TF_DataType.TF_FLOAT)
    let x =  gen_ops.relu(tf.matmul(x, (W._AsTensor())) + (b._AsTensor()))
    let W1 = tf.Variable(getRandom([|128;48|]), name = "bias1")
    let b1 = tf.Variable(getRandom([|48|]), name = "weight1",dtype = TF_DataType.TF_FLOAT)
    let x =  gen_ops.relu(tf.matmul(x, (W1._AsTensor())) + (b1._AsTensor()))
    let W2 = tf.Variable(getRandom([|48;10|]), name = "weight2")
    let b2 = tf.Variable(getRandom([|10|]), name = "bias2",dtype = TF_DataType.TF_FLOAT)
    tf.sigmoid(tf.matmul(x, (W2._AsTensor())) + (b2._AsTensor()))

// NHWC
let cnnModel(xtr) : Tensor= 
    let x = gen_ops.reshape(xtr,tf.constant([|-1;28;28;1|]))
    let c1f = tf.Variable(getRandom([|5;5;1;32|]),name = "c1f")
    let x = gen_ops.relu(gen_ops.conv2d(x,c1f._AsTensor(),[|1;2;2;1|],"SAME",data_format="NHWC"))
    let c2f = tf.Variable(getRandom([|5;5;32;64|]),name = "c2f")
    let x = gen_ops.relu(gen_ops.conv2d(x,c2f._AsTensor(),[|1;2;2;1|],"SAME",data_format="NHWC"))
    let x = tf.reshape(x,[|-1;7*7*64|])
    let W = tf.Variable(getRandom([|7*7*64;1024|]), name = "weight1")
    let b = tf.Variable(getRandom([|1024|]), name = "bias1")
    let x = gen_ops.relu(tf.matmul(x, (W._AsTensor())) + (b._AsTensor()))
    let W = tf.Variable(getRandom([|1024;10|]), name = "weight2")
    let b = tf.Variable(getRandom([|10|]), name = "bias2")
    gen_ops.relu(tf.matmul(x, (W._AsTensor())) + (b._AsTensor()))


let batches = 1000
let display_step = 100

let toItems(xs : ('a*'b) seq) = [|for (x,y) in xs -> FeedItem(x,y)|]

let output = basicModel(xtr)
//let output = cnnModel(xtr)

let sess = tf.Session()

let (loss,_) = gen_ops.softmax_cross_entropy_with_logits(output,ytr).ToTuple()
let optimizer = tf.train.AdamOptimizer(0.001f).minimize(loss)
let init = tf.global_variables_initializer()
sess.run(init)

let train(res,loss,optimizer,batches,display_step) =
    for epoch in 0 .. batches do
        let (x,y) = mnist.train.next_batch(64)
        // for some super weird reason y sometimes reutrns [|4;784|]
        if y.shape = [|64;10|] then
            sess.run(optimizer,[(xtr,x);(ytr,y)] |> toItems) |> ignore
            if ((epoch + 1)) % display_step = 0 then
                let getAccuracy(xs : NDArray,ys : NDArray) = 
                    let equal = (np.argmax(xs,1).Data<int32>(), np.argmax(ys,1).Data<int32>()) ||> Seq.zip 
                                |> Seq.sumBy (function | (x,y) when x = y -> 1 | _ -> 0)
                    (float equal / float xs.shape.[0])
                let ts = sess.run(res,FeedItem(xtr,mnist.test.Images))
                let accuracy = getAccuracy(mnist.test.Labels, ts)
                let c = sess.run(loss,[(xtr,x);(ytr,y)] |> toItems)
                printfn "Batch: %i cost=%f accuracy =%f " (epoch + 1) (c.Data<float32>().[0]) accuracy

for _ in 0..10 do
    train(output,loss,optimizer,batches,display_step)

