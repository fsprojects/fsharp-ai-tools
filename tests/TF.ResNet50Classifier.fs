module FSAI.Tools.Tests.TF.ResNet50Classifier

open Tensorflow
open Tensorflow.Operations
open System

// TODO replace fixed weights with variables
// TODO add the ability to use variable name scoping
#if FS47
open Tensorflow.Binding
#else
let tf = Tensorflow.Binding.tf
#endif

let model (input :  Tensor, weights: Map<string, Tensor>) =
    use ns = tf.name_scope ("Resnet50")
    let relu x = gen_ops.relu (x)
    let add y x = gen_ops.add (x, y)
    let matMul y x = gen_ops.mat_mul (x, y) 
    let dense (W, b) x = x |> matMul W |> add b
    let softmax x = gen_ops.softmax (x)
    let maxPool (ksizes: int*int*int*int, strides: int*int*int*int, padding: string, dataFormat: string) x = 
        let f (a, b, c, d) = [|a;b;c;d|] 
        gen_ops.max_pool (x, f ksizes, f strides, padding=padding, data_format=dataFormat)

    let getWeights (name: string) = weights.[name + ".npy"]

    let get_conv_tensor (conv_name: string) = getWeights (sprintf "%s/%s_W:0" conv_name conv_name)

    let batch_norm (bn_name: string) bnx = 
        use _scope = tf.name_scope ("batchnorm")
        let getT nm = getWeights (sprintf "%s/%s_%s:0" bn_name bn_name nm)
        let moving_variance = getT "running_std"
        let gamma = getT "gamma" // AKA scale
        let moving_mean = getT "running_mean"
        let beta = getT "beta"
        let fbn, _, _, _, _ = 
            gen_ops.fused_batch_norm(bnx, gamma, beta,
                mean=moving_mean, 
                variance=moving_variance,
                epsilon=Nullable (0.00001f), 
                is_training=Nullable (false),
                data_format="NHWC").ToTuple()
        fbn                         

    let res_block (stage: int, 
                   block: char, 
                   is_strided: bool, 
                   conv_shortcut: bool)
                  (input_tensor: Tensor) =
        use _scope = tf.name_scope "resblock" // TODO add input_tensor
        let conv_name_base = sprintf "res%i%c_branch" stage block
        let bn_name_base = sprintf "bn%i%c_branch" stage block

        let conv (postfix, is_strided: bool) cx =
            use _scope = tf.name_scope ("conv")
            let conv_name = sprintf "res%i%c_branch" stage block
            let strides = if is_strided then [|1;2;2;1|] else [|1;1;1;1|]
            gen_ops.conv2d (cx, 
                get_conv_tensor (conv_name_base + postfix), 
                strides, 
                padding = "SAME", 
                data_format = "NHWC", 
                dilations = [|1;1;1;1|], 
                name = conv_name + postfix)

        let right = 
            input_tensor
            |> conv ("2a", is_strided)
            |> batch_norm (bn_name_base + "2a")
            |> relu
            |> conv ("2b", false)
            |> batch_norm (bn_name_base + "2b")
            |> relu
            |> conv ("2c", false)
            |> batch_norm (bn_name_base + "2c")

        let left = 
            if conv_shortcut then 
                input_tensor |> conv ("1", is_strided) |> batch_norm (bn_name_base + "1")
            else input_tensor

        (right, left) ||> add |> relu

    let paddings = gen_ops.reshape (tf.constant [|0;0;3;3;3;3;0;0|], tf.constant [|4;2|])
    let padded_input = gen_ops.pad (input, paddings, "CONSTANT")

    let build_stage (stage: int, blocks: string) (x: Tensor) =
        blocks.ToCharArray () 
        |> Array.fold (fun x c -> res_block (stage, c, c='a' && stage<>2, c='a') (x)) x

    let reduceMean (axis: int[]) (x: Tensor) = tf.reduce_mean (x, axis = axis)
    //let matMul x y = graph.MatMul (x, y)
    let finalWeights = getWeights ("fc1000/fc1000_W:0")
    let finalBias = getWeights ("fc1000/fc1000_b:0")

    let initial_conv x = 
        gen_ops.conv2d (x, 
            get_conv_tensor "conv1", 
            [|1;2;2;1|], 
            padding = "VALID", 
            data_format = "NHWC", 
            name = "conv1")

    let output = 
        padded_input
        |> initial_conv
        |> batch_norm ("bn_conv1") 
        |> relu
        |> maxPool ( (1, 3, 3, 1), (1, 2, 2, 1), "SAME", "NHWC")
        |> build_stage (2, "abc")
        |> build_stage (3, "abcd")
        |> build_stage (4, "abcdef")
        |> build_stage (5, "abc")
        |> reduceMean ([|1;2|])
        |> dense (finalWeights, finalBias)
        |> softmax

    gen_ops.arg_max (output, tf.constant 1, output_type = Nullable (TF_DataType.TF_INT32))

/// Load and normalize JPEG Binary
let binaryJPGToImage (inputfile: string) =
    /// This is from TensorflowSharp (Examples/ExampleCommon/ImageUtil.cs)
    /// It's intended for inception but used here for resnet as an example of this type of functionality 
    let W, H, Mean, Scale = 224, 224, 117.0f, 1.0f
    let read_img = tf.read_file (inputfile)
    let loaded_img = tf.cast (gen_ops.decode_jpeg (contents = read_img, channels = Nullable 3), TF_DataType.TF_FLOAT)
    let expanded_img = gen_ops.expand_dims (input = loaded_img, dim = tf.constant 0)
    let resized_img = gen_ops.resize_bilinear (expanded_img, tf.constant [|W;H|])
    let final_img = gen_ops.div (gen_ops.sub (resized_img, tf.constant [|Mean|]), tf.constant [|Scale|])
    let output = tf.cast (final_img, TF_DataType.TF_FLOAT)
    let sess = new Session()
    let result = sess.run(output)
    result
