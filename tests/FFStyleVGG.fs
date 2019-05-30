module FFStyleVGG

open Tensorflow
open Tensorflow.Operations
type ops = gen_ops

// TODO: replace fixed weights with variables
// NOTE: This model architecture is tailored for fast feed forward style transfer and not as genearully useful as something like ResNet
// TODO recover from checkpoint

let model(input_img : Tensor, weights : Map<string,Tensor>) =
    // TODO: Create the following using Variables and use a checkpoint loader to load the values
    //       This will require a checkpoint saver/loader to be built

    let conv_init_vars(input:Tensor, out_channels:int, filter_size:int,is_transpose:bool,name:string) =
        //let weights_shape = 
        //    let in_channels = graph.GetShape(input).[3]
        //    if not is_transpose then
        //        [|filter_size; filter_size; in_channels; out_channels|]
        //    else
        //        [|filter_size; filter_size; out_channels; in_channels|]
        //let truncatedNormal = graph.TruncatedNormal(graph.Const(TFShape(weights_shape).AsTensor()),TFDataType.Float, seed=System.Nullable(1L))
        weights.[name + "/weights"]
        //graph.Variable(graph.Mul(truncatedNormal,graph.Const(new TFTensor(0.1f))),operName="weights").Read

    let instance_norm(input:Tensor, train:bool, name:string) =
        use scope = tf.name_scope("instance_norm")
        let struct (mu, sigma_sq) = nn_impl.moments(input, [|1;2|], keep_dims=true) 

        let shift = weights.[name + "/shift"]
        let scale = weights.[name + "/scale"]
        //let var_shape = TFShape(graph.GetShape(input).[3])
        //let shift = graph.Variable(graph.Zeros(var_shape),operName="shift").Read
        //let scale = graph.Variable(graph.Ones(var_shape),operName="scale").Read
        let epsilon = tf.constant(0.001f)
        // Note: The following would benefit from operator overloads
        let normalized = gen_ops.div(gen_ops.sub(input,mu), gen_ops.pow(gen_ops.add(sigma_sq,epsilon),tf.constant(0.5f)))
        math_ops.add(tf.multiply(scale,normalized),shift, name=scope._name)

    let conv_layer(num_filters:int, filter_size:int, strides:int, is_relu:bool, name:string) (input:Tensor) = 
        let weights_init = conv_init_vars(input, num_filters, filter_size,false,name)
        
        let x = instance_norm(ops.conv2d(input, weights_init, [|1;strides;strides;1|], padding="SAME"),true, name)
        if is_relu then ops.relu(x) else x

    let residual_block(filter_size:int, name:string) (input:Tensor) = 
        let tmp = input |> conv_layer(128, filter_size, 1, true, name + "_c1")
        math_ops.add(input, tmp |> conv_layer(128, filter_size, 1, false, name + "_c2"))

    let conv_transpose_layer(num_filters:int, filter_size:int, strides:int, name:string) (input:Tensor) =
        let weights_init = conv_init_vars(input, num_filters, filter_size,true,name)
        let strides_array = [|1;strides;strides;1|]
        let out_shape = 
            //ops.shape
            let x = if strides = 1 then array_ops.shape(input) else tf.multiply(array_ops.shape(input), tf.constant(strides_array))
            ops.concat_v2([|ops.slice(x,tf.constant([|0|]),tf.constant([|3|]));tf.constant([|num_filters|])|],axis=tf.constant(0))
        let rev(x) = ops.reverse(x,tf.constant([|true|]))
        //let out_shape = graph.Print(out_shape,[|rev(graph.Shape(weights_init));rev(out_shape); rev(graph.Shape(input))|],"out_shape") 
        ops.relu(instance_norm(ops.conv2d_backprop_input(input_sizes = out_shape, filter = weights_init, out_backprop = input, strides = strides_array, padding = "SAME", data_format = "NHWC"), true, name))

    input_img
    |> conv_layer(32,9,1,true,"conv1")
    |> conv_layer(64,3,2,true,"conv2")
    |> conv_layer(128,3,2,true,"conv3")
    |> residual_block(3,"resid1")
    |> residual_block(3,"resid2")
    |> residual_block(3,"resid3")
    |> residual_block(3,"resid4")
    |> residual_block(3,"resid5")
    |> conv_transpose_layer(64,3,2,"conv_t1")
    |> conv_transpose_layer(32,3,2,"conv_t2")
    |> conv_layer(3,9,1,false,"conv_t3")


