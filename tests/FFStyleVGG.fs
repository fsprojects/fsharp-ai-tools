module FFStyleVGG
open TensorFlow.FSharp

// TODO: replace fixed weights with variables
// TODO: add the ability to use variable name scoping
// NOTE: This model architecture is tailored for fast feed forward style transfer and not as genearully useful as something like ResNet

let model(graph:TFGraph, input_img : TFOutput, weights : Map<string,TFOutput>) =
    // TODO: Create the following using Variables and use a checkpoint loader to load the values
    //       This will require a checkpoint saver/loader to be built

    let conv_init_vars(input:TFOutput, out_channels:int64, filter_size:int64,is_transpose:bool,name:string) =
        //let weights_shape = 
        //    let in_channels = graph.GetShape(input).[3]
        //    if not is_transpose then
        //        [|filter_size; filter_size; in_channels; out_channels|]
        //    else
        //        [|filter_size; filter_size; out_channels; in_channels|]
        //let truncatedNormal = graph.TruncatedNormal(graph.Const(TFShape(weights_shape).AsTensor()),TFDataType.Float, seed=System.Nullable(1L))
        weights.[name + "/weights"]
        //graph.Variable(graph.Mul(truncatedNormal,graph.Const(new TFTensor(0.1f))),operName="weights").Read

    let instance_norm(input:TFOutput, train:bool, name:string) =
        use scope = graph.NameScope(name + "/instance_norm")
        let mu, sigma_sq = graph.Moments(input, graph.Const(TFShape([|1L;2L|]).AsTensor()), keep_dims=true)
        let shift = weights.[name + "/shift"]
        let scale = weights.[name + "/scale"]
        //let var_shape = TFShape(graph.GetShape(input).[3])
        //let shift = graph.Variable(graph.Zeros(var_shape),operName="shift").Read
        //let scale = graph.Variable(graph.Ones(var_shape),operName="scale").Read
        let epsilon = graph.Const(new TFTensor(0.001f))
        // Note: The following would benefit from operator overloads
        let normalized = graph.Div(graph.Sub(input,mu),graph.Pow(graph.Add(sigma_sq,epsilon),graph.Const(new TFTensor(0.5f))))
        graph.Add(graph.Mul(scale,normalized),shift,name=scope.Scope)

    let conv_layer(num_filters:int64, filter_size:int64, strides:int64, is_relu:bool, name:string) (input:TFOutput) = 
        let weights_init = conv_init_vars(input, num_filters, filter_size,false,name)
        let x = instance_norm(graph.Conv2D(input, weights_init, [|1L;strides;strides;1L|], padding="SAME"),true, name)
        if is_relu then graph.Relu(x) else x

    let residual_block(filter_size:int64, name:string) (input:TFOutput) = 
        let tmp = input |> conv_layer(128L, filter_size, 1L, true, name + "_c1")
        graph.Add(input, tmp |> conv_layer(128L, filter_size, 1L, false, name + "_c2"))

    let conv_transpose_layer(num_filters:int64, filter_size:int64, strides:int64, name:string) (input:TFOutput) =
        let weights_init = conv_init_vars(input, num_filters, filter_size,true,name)
        let strides_array = [|1L;strides;strides;1L|]
        let g = graph
        let out_shape = 
            let x = if strides = 1L then graph.Shape(input) else graph.Mul(graph.Shape(input), graph.Cast(graph.Const(new TFTensor(strides_array)),TFDataType.Int32))
            g.ConcatV2([|g.Slice(x,g.Const(new TFTensor([|0|])),g.Const(new TFTensor([|3|])));g.Const(new TFTensor([|int32 num_filters|]))|],axis=g.Const(new TFTensor(0)))
        let rev(x) = g.Reverse(x,g.Const(new TFTensor([|true|])))
        //let out_shape = graph.Print(out_shape,[|rev(graph.Shape(weights_init));rev(out_shape); rev(graph.Shape(input))|],"out_shape") 
        graph.Relu(instance_norm(graph.Conv2DBackpropInput( input_sizes = out_shape, filter = weights_init, out_backprop = input, strides = strides_array, padding = "SAME", data_format = "NHWC"), true, name))

    input_img
    |> conv_layer(32L,9L,1L,true,"conv1")
    |> conv_layer(64L,3L,2L,true,"conv2")
    |> conv_layer(128L,3L,2L,true,"conv3")
    |> residual_block(3L,"resid1")
    |> residual_block(3L,"resid2")
    |> residual_block(3L,"resid3")
    |> residual_block(3L,"resid4")
    |> residual_block(3L,"resid5")
    |> conv_transpose_layer(64L,3L,2L,"conv_t1")
    |> conv_transpose_layer(32L,3L,2L,"conv_t2")
    |> conv_layer(3L,9L,1L,false,"conv_t3")
