#I __SOURCE_DIRECTORY__
#r "netstandard"
#I "../tests/bin/Debug/net472/"
#r "FSharp.AI.dll"
#r "TensorFlow.Net.dll"
#r "NumSharp.Core.dll"
#r "Argu.dll"
#r "FSharp.AI.Tests.dll"

open System
open System.IO
open Tensorflow
open NumSharp
open Tensorflow.Operations

// Linear Regression

// We can set a fixed inti value in order to debug

let X = tf.placeholder(TF_DataType.TF_FLOAT)
let Y = tf.placeholder(TF_DataType.TF_FLOAT)

let W = tf.Variable(-0.06f, name = "weight")
let b = tf.Variable(-0.73f, name = "bias")

// Construct a linear model
let pred = tf.add(tf.multiply(X,W),b)


let train_X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f, 7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);
let train_Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f, 2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
let test_X = np.array(6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f);
let test_Y = np.array(1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f);

let n_samples = train_X.shape.[0];

let training_epochs = 1000
let display_step = 40
let rng = np.random;

// Mean squared error
let cost = tf.reduce_sum(tf.square(pred - Y)) / (2.0f * float32 n_samples)

// Gradient descent
// Note, minimiuze() knows to modify W and b because Variable objects are trainable=True by default
let optimizer = tf.train.GradientDescentOptimizer(0.1f).minimize(cost)

// Initialize the variables (i.e. assign their default value)
let init = tf.global_variables_initializer()

// Start training
let sess = tf.Session()
sess.run(init) |> ignore

let toItems(xs : ('a*'b) seq) = [|for (x,y) in xs -> FeedItem(x,y)|]

for epoch in 0 .. training_epochs do

  // Mini-batches of 1
  for x,y in (train_X.Data<float32>(),train_Y.Data<float32>()) ||> Array.zip do
    sess.run(optimizer,[(X,x);(Y,y)] |> toItems) |> ignore
  
  // Display logs per epoch step
  if ((epoch + 1)) % display_step = 0 then
    let c = sess.run(cost,[(X,train_X);(Y,train_Y)] |> toItems)
    printfn "Epoch: %i cost=%f" (epoch + 1) (c.Data<float32>().[0])
  
printfn "Optimization Finished!"
/// NOTE: clearely we need to support multiple out
//let c = sess.run([|cost;W._AsTensor();b._AsTensor()|],[(X,train_X);(Y,train_Y)] |> toItems)
let training_cost = sess.run(cost,[(X,train_X);(Y,train_Y)] |> toItems).Data<float32>().[0]
let W_res = sess.run(W._AsTensor(),[(X,train_X);(Y,train_Y)] |> toItems).Data<float32>().[0]
let b_res = sess.run(b._AsTensor(),[(X,train_X);(Y,train_Y)] |> toItems).Data<float32>().[0]
printfn "Training cost=%f W=%f b=%f"  training_cost W_res b_res

let testing_cost = sess.run(tf.reduce_sum(tf.square(pred - Y)) / (2.0f * float32 test_X.shape.[0]),[(X,test_X);(Y,test_Y)] |> toItems).Data<float32>().[0]
printfn "Testing cost=%f"  testing_cost 
printfn "Absolute mean square loss difference: %f" (Math.Abs(training_cost - testing_cost))

