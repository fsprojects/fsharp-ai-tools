// Apache 2.0 from https://github.com/google-research/bert/blob/master/optimization.py
/// Functions and classes related to optimization (weight updates). 
module Optimization
open Tensorflow
open System
open System.Text.RegularExpressions

let tf = Tensorflow.Binding.tf
type gen_ops = Tensorflow.Operations.gen_ops

/// A basic Adam optimizer that includes "correct" L2 weight decay.
type AdamWeightDecayOptimizer(learning_rate: Tensor, 
        ?weight_decay_rate: float32,
        ?beta_1: float32,
        ?beta_2: float32,
        ?epsilon: float32,
        ?exclude_from_weight_decay: string[],
        ?name: string) =

    inherit Optimizer(learning_rate, false, name = defaultArg name "AdamWeightDecayOptimizer")
    let beta_1 = defaultArg beta_1 0.9f
    let beta_2 = defaultArg beta_2 0.999f
    let epsilon = defaultArg epsilon 0.0000001f
    let weight_decay_rate = defaultArg weight_decay_rate 0.0f
    let exclude_from_weight_decay = defaultArg exclude_from_weight_decay [||]
    let name_regex = Regex("^(.*):\\d+$")

    let get_variable_name(param_name: string) : string = 
        let m = name_regex.Match(param_name) 
        if m.Groups.Count > 1 then m.Groups.[1].Value else param_name

    /// Whether to use L2 weight decay for `param_name`.
    let do_use_weight_decay(param_name: string) = 
        if weight_decay_rate = 0.0f then false
        else
            exclude_from_weight_decay 
            |> Array.exists param_name.Contains 
            |> not

    member _.Beta1 = beta_1
    member _.Beta2 = beta_2
    member _.Epsion = epsilon
    member _.ExcludeFromWeightDecay = exclude_from_weight_decay

    // TODO This should be an override, for some reason this is not wokring
    //member _.apply_gradients(grads_and_vars: (Tensor*RefVariable)[], ?global_step: RefVariable, ?name: string): Operation = 
    member _.apply_gradients2(grads_and_vars: (Tensor*Tensor)[], ?global_step: Tensor, ?name: string): Operation = 
        [|
            for (grad, param) in grads_and_vars do
                // if grad is None or param is None:
                //  continue
                let param_name = get_variable_name(param.name)

                let m = tf.get_variable(name=param_name + "/adam_m",
                                        shape=TensorShape(param.shape),
                                        dtype=tf.float32,
                                        trainable=Nullable(false),
                                        initializer=tf.zeros_initializer)

                let v = tf.get_variable(name=param_name + "/adam_v",
                                        shape=TensorShape(param.shape),
                                        dtype=tf.float32,
                                        trainable=Nullable(false),
                                        initializer=tf.zeros_initializer)

                // Standard Adam update.
                let next_m = tf.multiply(beta_1, m) + tf.multiply(1.0f - beta_1, grad)
                let next_v = tf.multiply(beta_2, v) + tf.multiply(1.0f - beta_2, tf.square(grad))

                let update = next_m / (tf.sqrt(next_v) + epsilon)

                // Just adding the square of the weights to the loss function is *not*
                // the correct way of using L2 regularization/weight decay with Adam,
                // since that will interact with the m and v parameters in strange ways.
                // 
                // Instead we want ot decay the weights in a manner that doesn't interact
                // with the m/v parameters. This is equivalent to adding the square
                // of the weights to the loss with plain (non-momentum) SGD.
                let update = if do_use_weight_decay(param_name) then update + param * weight_decay_rate  else update
                
                let update_with_lr = learning_rate * update

                let next_param = param - update_with_lr

                yield gen_ops.assign(param,next_param)
                yield tf.assign(m,next_m)
                yield tf.assign(v,next_v)
        |] 
        // TODO fix optional name when C# optional interop is fixed
        |> fun assignments -> tf.group(assignments,name = (defaultArg name "fix_me"))


/// Creates an optimizer training op.
let create_optimizer(loss: Tensor, init_lr: float32, num_train_steps: int, num_warmup_steps: int option) = 
    let global_step = tf.get_or_create_global_step()
    //let learning_rate = tf.constant(value = init_lr, shape=[||], dtype = tf.float32)
    let learning_rate = init_lr

    // Implements linear decay of the learning rate
    let learning_rate = 
        tf.train.polynomial_decay(learning_rate, 
                                  global_step, 
                                  float32 num_train_steps,
                                  end_learning_rate = 0.0f,
                                  power = 1.0f,
                                  cycle = false)

    // Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    // learning rate will be `global_step/num_warmup_steps * init_lr`.
    let learning_rate = 
         match num_warmup_steps with
         | None -> learning_rate
         | Some num_warmup_steps ->
            let global_steps_init = tf.cast(global_step._AsTensor(), tf.int32)
            let warmup_steps_init = tf.constant(num_warmup_steps, dtype=tf.int32)

            let global_steps_float = tf.cast(global_steps_init, tf.float32)
            let warmup_steps_float = tf.cast(warmup_steps_init, tf.float32)

            let warmup_percent_done = global_steps_float / warmup_steps_float
            let warmup_learning_rate = init_lr * warmup_percent_done

            let is_warmup = tf.cast(tf.less(global_steps_init , warmup_steps_init), tf.float32)
            let learning_rate = ((1.0f - is_warmup)  * learning_rate + is_warmup * warmup_learning_rate)
            learning_rate

    // It is recommended that you use this optimizer for fine tuning, since this
    // is how the model was trained (note that the Adam m/v variables are NOT
    // loaded from init_checkpoint.)

    let optimizer = 
        AdamWeightDecayOptimizer(learning_rate=learning_rate,
            weight_decay_rate=0.01f,
            beta_1=0.9f,
            beta_2=0.999f,
            epsilon=0.0000001f,
            exclude_from_weight_decay=[|"LayerNorm"; "layer_norm"; "bias"|])

    let tvars = tf.trainable_variables() |> Array.map (fun x -> x._variable)
    let grads = tf.gradients(loss, tvars)

    // This is how the model was pre-trained.
    let (grads, _) = tf.clip_by_global_norm(grads, clip_norm = tf.constant(1.0f))
    let train_op = optimizer.apply_gradients2((grads, tvars) ||> Array.zip, global_step = global_step._AsTensor())

    // Normally the global step update is done inside of `apply_gradients`.
    // However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    // a different optimizer, you should probably take this line out.
    let new_global_step = global_step + 1
    let train_op = tf.group [|train_op :> ITensorOrOperation; tf.assign(global_step,new_global_step) :> ITensorOrOperation|]
    train_op

