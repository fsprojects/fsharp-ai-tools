module ResNet50
open TensorFlow.FSharp
open System

// TODO replace fixed weights with variables
// TODO add the ability to use variable name scoping

let model(graph : TFGraph, input : TFOutput, weights:Map<string,TFOutput>) =
    use ns = graph.NameScope("Resnet50")
    let relu x = graph.Relu(x)
    let add y x = graph.Add(x,y)
    let matMul y x = graph.MatMul(x,y) 
    let dense(W,b) x = x |> matMul W |> add b
    let softmax x = graph.Softmax(x)
    let maxPool(ksizes:int*int*int*int,strides:int*int*int*int,padding:string,dataFormat:string) x = 
        let f(a,b,c,d) = [|a;b;c;d|] |> Array.map int64
        graph.MaxPool(x,f(ksizes),f(strides),padding=padding,data_format=dataFormat)

    let getWeights(name:string) = weights.[name + ".npy"]

    let get_conv_tensor(conv_name:string) = getWeights(sprintf "%s/%s_W:0" conv_name conv_name)

    let batch_norm(bn_name:string) bnx = 
        use ns = graph.NameScope("batchnorm")
        let getT(nm) = getWeights(sprintf "%s/%s_%s:0" bn_name bn_name nm)
        let moving_variance = getT("running_std")
        let gamma           = getT("gamma") // AKA scale
        let moving_mean     = getT("running_mean")
        let beta            = getT("beta")
        let (fbn,_,_,_,_) = graph.FusedBatchNorm(bnx,gamma,beta,mean=moving_mean,
                             variance=moving_variance, epsilon=0.00001f,
                             is_training=false, data_format="NHWC")
        fbn                         

    let res_block(stage:int, 
                  block:char, 
                  is_strided:bool, 
                  conv_shortcut:bool)
                  (input_tensor:TFOutput) =
        use scope = graph.NameScope("resblock", input_tensor)
        let conv_name_base = sprintf "res%i%c_branch" stage block
        let bn_name_base = sprintf "bn%i%c_branch" stage block
        let conv(postfix,is_strided:bool) cx =
            use ns = graph.NameScope("conv")
            let conv_name = sprintf "res%i%c_branch" stage block
            let strides = if is_strided then [|1L;2L;2L;1L|] else [|1L;1L;1L;1L|]
            graph.Conv2D(cx,
                         get_conv_tensor(conv_name_base + postfix),
                         strides,
                         padding="SAME",
                         data_format="NHWC",
                         dilations=[|1L;1L;1L;1L|],
                         name=conv_name + postfix)
        let right = 
            input_tensor
            |> conv("2a",is_strided)
            |> batch_norm(bn_name_base + "2a")
            |> relu
            |> conv("2b",false)
            |> batch_norm(bn_name_base + "2b")
            |> relu
            |> conv("2c",false)
            |> batch_norm(bn_name_base + "2c")
        let left = 
            if conv_shortcut then 
                input_tensor |> conv("1",is_strided) |> batch_norm(bn_name_base + "1")
            else input_tensor
        (right,left) ||> add |> relu

    /// TODO make this simpler with helper functions
    let paddings = graph.Reshape(graph.Const(new TFTensor([|0;0;3;3;3;3;0;0|])), graph.Const(TFShape(4L,2L).AsTensor()))
    let padded_input = graph.Pad(input,paddings, "CONSTANT")

    let build_stage(stage:int,blocks:string) (x:TFOutput) =
        blocks.ToCharArray() 
        |> Array.fold (fun x c -> res_block(stage,c,c='a' && stage<>2,c='a')(x)) x

    let toAxis (xs:int[]) : TFOutput = 
        graph.Const(new TFTensor(xs),TFDataType.Int32)

    let reduceMean(axis:int list) (x:TFOutput) = graph.ReduceMean(x,axis = (axis  |> Array.ofList |> toAxis))
    //let matMul x y = graph.MatMul(x,y)
    let finalWeights = getWeights("fc1000/fc1000_W:0")
    let finalBias = getWeights("fc1000/fc1000_b:0")
    let initial_conv x = 
        graph.Conv2D(x, 
                     get_conv_tensor("conv1"),
                     [|1L;2L;2L;1L|],
                     padding="VALID",
                     data_format="NHWC",
                     name="conv1")

    let output = 
        padded_input
        |> initial_conv
        |> batch_norm("bn_conv1") 
        |> relu
        |> maxPool((1,3,3,1),(1,2,2,1),"SAME","NHWC")
        |> build_stage(2,"abc")
        |> build_stage(3,"abcd")
        |> build_stage(4,"abcdef")
        |> build_stage(5,"abc")
        |> reduceMean([1;2])
        |> dense(finalWeights,finalBias)
        |> softmax

    output

