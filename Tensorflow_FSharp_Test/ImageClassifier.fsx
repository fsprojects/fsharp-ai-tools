(* TODO Port from TensorFlowSharp to Tensorflow_FSharp
#r "netstandard"
#r "lib/TensorFlowSharp.dll"
#load "shared/NPYReaderWriter.fsx"
#nowarn "760"
open TensorFlow
open System
open System.IO
open System.Collections.Generic
open NPYReaderWriter

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

let pretrained_dir = Path.Combine(__SOURCE_DIRECTORY__,"pretrained")
let weights_path = Path.Combine(pretrained_dir, "resnet_classifier_1000.npz")
let labels_path = Path.Combine(pretrained_dir,"imagenet1000.txt")
let example_dir = Path.Combine(__SOURCE_DIRECTORY__,"examples")
let label_map = File.ReadAllLines(labels_path)

let sess = new TFSession()
// NOTE: Graph.ToString() returns the whole protobuf as txt to console
// NOTE: fsi does not type check in Ionide. This can be ignored.
fsi.AddPrinter(fun (x:TFGraph) -> sprintf "TFGraph %i" (int64 x.Handle))

let graph = sess.Graph
let relu x = graph.Relu(x)
let add y x = graph.Add(x,y)
let matMul y x = graph.MatMul(x,y) 
let dense(W,b) x = x |> matMul W |> add b
let softmax x = graph.Softmax(x)
let maxPool(ksizes:int*int*int*int,strides:int*int*int*int,padding:string,dataFormat:string) x = 
    let f(a,b,c,d) = [|a;b;c;d|] |> Array.map int64
    graph.MaxPool(x,f(ksizes),f(strides),padding=padding,data_format=dataFormat)


let buildResnet(graph:TFGraph,weights_path:string) =
    // NOTE: This behaviour should be built into graph
    // NOTE: This is needed as by default the graph will use the same block for each invocation
    let withScope =
        let nameCount = Dictionary<string,int>()
        fun (name:string) -> 
            let namePrime = 
                if nameCount.ContainsKey(name) then 
                    nameCount.[name] <- nameCount.[name] + 1
                    sprintf "%s_%i" name nameCount.[name] 
                else nameCount.Add((name,1)); name
            graph.WithScope(namePrime)


    let weights_map = 
        readFromNPZ((File.ReadAllBytes(weights_path)))
        |> Map.map (fun k (metadata,arr) ->
            graph.Reshape(graph.Const(new TFTensor(arr)), graph.Const(TFShape(metadata.shape |> Array.map int64).AsTensor()))) 

    let getWeights(name:string) =
        weights_map.[name + ".npy"]
        //let data, shape = h5.Read<float32>(name)
        //graph.Reshape(graph.Const(new TFTensor(data)), graph.Const(TFShape(shape |> Array.ofList).AsTensor()))

    let get_conv_tensor(conv_name:string) = getWeights(sprintf "%s/%s_W:0" conv_name conv_name)

    let batch_norm(bn_name:string) bnx = 
        use ns = withScope("batchnorm")
        let getT(nm) = getWeights(sprintf "%s/%s_%s:0" bn_name bn_name nm)
        let moving_variance = getT("running_std")
        let gamma           = getT("gamma") // AKA scale
        let moving_mean     = getT("running_mean")
        let beta            = getT("beta")
        let (fbn,_,_,_,_) = graph.FusedBatchNorm(bnx,gamma,beta,mean=moving_mean,
                             variance=moving_variance, epsilon=Nullable(0.00001f),
                             is_training=Nullable(false), data_format="NHWC").ToTuple()
        fbn                         

    let res_block(stage:int, 
                  block:char, 
                  is_strided:bool, 
                  conv_shortcut:bool)
                  input_tensor:TFOutput =
        use scope = withScope("resblock")
        let conv_name_base = sprintf "res%i%c_branch" stage block
        let bn_name_base = sprintf "bn%i%c_branch" stage block
        let conv(postfix,is_strided:bool) cx =
            use ns = withScope("conv")
            let conv_name = sprintf "res%i%c_branch" stage block
            let strides = if is_strided then [|1L;2L;2L;1L|] else [|1L;1L;1L;1L|]
            graph.Conv2D(cx,
                         get_conv_tensor(conv_name_base + postfix),
                         strides,
                         padding="SAME",
                         data_format="NHWC",
                         dilations=[|1L;1L;1L;1L|],
                         operName=conv_name + postfix)
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
        
    let input_placeholder = 
        graph.Placeholder(TFDataType.Float, 
                          shape=TFShape(-1L,-1L,-1L,3L), 
                          operName="new_input")

    /// TODO make this simpler with helper functions
    let paddings = graph.Reshape(graph.Const(new TFTensor([|0;0;3;3;3;3;0;0|])), graph.Const(TFShape(4L,2L).AsTensor()))
    let padded_input = graph.Pad(input_placeholder,paddings, "CONSTANT")

    let build_stage(stage:int,blocks:string) (x:TFOutput) =
        blocks.ToCharArray() 
        |> Array.fold (fun x c -> res_block(stage,c,c='a' && stage<>2,c='a')(x)) x

    let toAxis (xs:int[]) : Nullable<TFOutput> = 
        Nullable(graph.Const(new TFTensor(xs),TFDataType.Int32))
    let reduceMean(axis:int list) (x:TFOutput) = graph.ReduceMean(x,axis  |> Array.ofList |> toAxis)
    //let matMul x y = graph.MatMul(x,y)
    let finalWeights = getWeights("fc1000/fc1000_W:0")
    let finalBias = getWeights("fc1000/fc1000_b:0")
    let initial_conv x = 
        graph.Conv2D(x, 
                     get_conv_tensor("conv1"),
                     [|1L;2L;2L;1L|],
                     padding="VALID",
                     data_format="NHWC",
                     operName="conv1")

    let output = 
        padded_input
        |> initial_conv
        |> batch_norm("bn_conv1") 
        |> relu
        |> maxPool((1,3,3,1),(1,2,2,1),"SAME","NCHW")
        |> build_stage(2,"abc")
        |> build_stage(3,"abcd")
        |> build_stage(4,"abcdef")
        |> build_stage(5,"abc")
        |> reduceMean([1;2])
        |> dense(finalWeights,finalBias)
        |> softmax

    (input_placeholder,output)


/// This is from TensorflowSharp (Examples/ExampleCommon/ImageUtil.cs)
/// It's intended for inception but used here for resnet as an example
/// of this type of functionality 
let construtGraphToNormalizeImage(destinationDataType:TFDataType) =
    let W = 224
    let H = 224
    let Mean = 117.0f
    let Scale = 1.0f
    let input = graph.Placeholder(TFDataType.String)
    let loaded_img = graph.Cast(graph.DecodeJpeg(contents=input,channels=Nullable(3L)),TFDataType.Float)
    let expanded_img = graph.ExpandDims(input=loaded_img, dim = graph.Const(TFTensor(0)))
    let resized_img = graph.ResizeBilinear(expanded_img,graph.Const(TFTensor([|W;H|])))
    let final_img = graph.Div(graph.Sub(resized_img, graph.Const(TFTensor([|Mean|]))), graph.Const(TFTensor([|Scale|])))
    (input,graph.Cast(final_img,destinationDataType))

let img_input,img_output = construtGraphToNormalizeImage(TFDataType.Float)

let (input,output) = buildResnet(graph,weights_path)

let classifyFile(path:string) =
    let createTensorFromImageFile(file:string,destinationDataType:TFDataType) =
        let tensor = TFTensor.CreateString(File.ReadAllBytes(file))
        sess.Run(runOptions = null, inputs = [|img_input|], inputValues = [|tensor|], outputs = [|img_output|]).[0]
    let example = createTensorFromImageFile(path, TFDataType.Float)
    let index = graph.ArgMax(output,graph.Const(TFTensor(1)))
    let res = sess.Run(runOptions = null, inputs = [|input|], inputValues = [|example|], outputs = [|index|])
    label_map.[res.[0].GetValue() :?> int64[] |> Array.item 0 |> int]

printfn "example_0.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_0.jpeg")))
printfn "example_1.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_1.jpeg")))
printfn "example_2.jpeg is %s " (classifyFile(Path.Combine(example_dir,"example_2.jpeg")))
*)
