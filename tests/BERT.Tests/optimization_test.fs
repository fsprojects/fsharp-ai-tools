module OptimizationTests

open System
open Tensorflow
open NUnit.Framework

let shouldEqual (msg: string) (v1: 'T) (v2: 'T) = 
    if v1 <> v2 then 
        Assert.Fail(sprintf "fail %s: expected %A, got %A" msg v1 v2)

let tf = Tensorflow.Binding.tf

[<Test>]
let ``test adam``() =
    use sess = tf.Session()
    let w = tf.get_variable("w", 
                            shape=TensorShape(3), 
                            initializer = tf.constant_initializer([|0.1f;-0.2f;-0.1f|]))
    let x = tf.constant([|0.4f; 0.2f; -0.5f|])
    let loss = tf.reduce_mean(tf.square(w - x))
    let tvars = tf.trainable_variables() |> Array.map (fun x -> x._variable)
    let grads = tf.gradients(loss, tvars)
    let global_stop = tf.get_or_create_global_step() 
    let optimizer = Optimization.AdamWeightDecayOptimizer(learning_rate=tf.constant(0.2f))
    let train_op = optimizer.apply_gradients2((grads,tvars) ||> Array.zip, global_stop._variable, "train_op")

    let init_op = tf.group([|tf.global_variables_initializer(); (* tf.local_variables_initializer()*)|])
    sess.run(init_op)
    for _i in 0 .. 100 do
        sess.run(train_op)
    let w_np = sess.run(w)
    let res = w_np.[0].Data<float32>().ToArray()
    let expected = [|0.4f; 0.2f; -0.5f|]
    if (res, expected) ||> Array.zip |> Array.exists (fun (x,y) -> Math.Abs(x-y) > 1e-2f ) 
    then Assert.Fail(sprintf "fail test adam: expected %A, got %A" expected res)


