namespace Tests
open Microsoft.ML.OnnxRuntime.Tensors
open NUnit.Framework
open Onnx
open FSharp.ML.Onnx.Protobuf
open System
open System.IO
open Microsoft.ML.OnnxRuntime
open Microsoft.FSharp.Quotations
open FSharp.ML.Onnx.Utils
open FSharp.ML.Onnx.Utils.Expr
open FSharp.ML.Onnx.Expr
open FSharp.ML.Onnx.Extensions

type on = FSharp.ML.Onnx.API.SnakeCase.Onnx
type DV<'a> = DisposableValue<'a>

module MiniGraphs = 
    let input1 = ArrayTensorExtensions.ToTensor(Array2D.create 1 32 2.f) :> Tensor<float32>
    let input2 = ArrayTensorExtensions.ToTensor(Array2D.create 32 1 3.f) :> Tensor<float32>

    let input1Int = ArrayTensorExtensions.ToTensor(Array2D.create 1 32 2L) :> Tensor<int64>
    let input2Int = ArrayTensorExtensions.ToTensor(Array2D.create 32 1 3L) :> Tensor<int64>

    let input4D1 = ArrayTensorExtensions.ToTensor(Array4D.create 3 3 1 3 2.f) :> Tensor<float32>
    let input4D2 = ArrayTensorExtensions.ToTensor(Array4D.create 3 3 3 1 1.f) :> Tensor<float32>


    [<Test>]
    let ``add float``() = 
        let res1 = on.add(input1,input2)
        let res2 = on.add(input1Int,input2Int)
        if res1.Dimensions.ToArray() <> [|32;32|] then failwith "Incorrect dimmesions"
        if res1 |> Seq.exists (fun x -> x <> 5.f) then failwith "An incorrect value"
        if res2.Dimensions.ToArray() <> [|32;32|] then failwith "Incorrect dimmesions"
        if res2 |> Seq.exists (fun x -> x <> 5L) then failwith "An incorrect value"

    [<Test>]
    let relu() = 
        let xx = Array2D.create 2 2 0.f
        xx.[0,0] <- -1.0f
        xx.[1,1] <- 1.0f
        let res = on.relu(ArrayTensorExtensions.ToTensor(xx) :> Tensor<float32>)
        Assert.AreEqual(0.0, float res.[0,0], 0.001)
        Assert.AreEqual(1.0, float res.[1,1], 0.001)

    [<Test>]
    let convolution() = 
        let img = ArrayTensorExtensions.ToTensor(Array4D.create 1 1 32 32 1.f) :> Tensor<float32>
        let kernel = ArrayTensorExtensions.ToTensor(Array4D.create 8 1 5 5 1.f) :> Tensor<float32>
        let convRes = on.conv(img, kernel, auto_pad= "SAME_UPPER")
        Assert.AreEqual(9.0, float convRes.[0,0,0,0], 0.001)
        Assert.AreEqual(25.0, float convRes.[0,0,5,5], 0.001)

    [<Test>]
    let ``matmul broadcast``() = 
        let res1 = on.mat_mul(input1,input2)
        Assert.AreEqual([|1;1|], res1.shape)
        Assert.AreEqual(192.0f, res1.[0])
        let res2 = on.mat_mul(input2,input1)
        Assert.AreEqual([|32;32|], res2.shape)
        Assert.AreEqual(6.0f, res2.[0])

    [<Test>]
    let ``matmul batch``() = 
        let res1 = on.mat_mul(input4D1,input4D2)
        Assert.AreEqual([|3;3;1;1|], res1.Dimensions.ToArray())
        Assert.AreEqual(6.0f, res1.[0])

    [<Test>]
    let ``eager api``() =
        let input1 = ArrayTensorExtensions.ToTensor(Array2D.create 10000 40 -2.f) :> Tensor<float32>
        let input2 = ArrayTensorExtensions.ToTensor(Array2D.create 40 10000 -2.f) :> Tensor<float32>
        let res = on.mat_mul(input2,on.abs(input1))
        Assert.AreEqual([|40;40|], res.shape)
        Assert.AreEqual(-40000., float res.[0,0], 0.00001)
        

module FullModel = 

    let shouldEqual (msg: string) (v1: 'T) (v2: 'T) = 
        if v1 <> v2 then 
            Assert.Fail(sprintf "fail %s: expected %A, got %A" msg v1 v2)

    let mnistDir = Path.Combine(__SOURCE_DIRECTORY__,"..","..","data","mnist")

    let test_data = 
            let f(path: string) = 
                TensorProto.Parser.ParseFrom(File.ReadAllBytes(path))
            [| for i in [0;1;2] ->
                    Path.Combine(mnistDir,sprintf "test_data_set_0") 
                    |> fun dir -> (f(Path.Combine(dir,"input_0.pb")),f(Path.Combine(dir,"output_0.pb")))|]

    let testModel(model : byte[]) = 
        use sess = new InferenceSession(model)
        for (index,(input,output)) in test_data |> Array.indexed do
            use values2 = sess.Run([|NamedOnnxValue.CreateFromTensor("Input3",Tensor.FromTensorProtoFloat32(input))|])
            let diff = 
                (values2 |> Seq.toArray |> Array.head |> fun v -> v.AsTensor<float32>() |> Seq.toArray, Tensor.FromTensorProtoFloat32(output) |> Seq.toArray)
                ||> Array.zip
                |> Array.sumBy (fun (x,y) -> System.Math.Abs(x-y))
            if diff > 0.1f then failwithf "Unexpected result in example %i with a difference of %f" index diff

    let testModel2(f: Tensor<float32> -> DV<Tensor<float32>>) = 
        let test_data = 
            let f(path: string) = 
                TensorProto.Parser.ParseFrom(File.ReadAllBytes(path))
            [| for i in [0;1;2] ->
                    Path.Combine(mnistDir,sprintf "test_data_set_%i" i) 
                    |> fun dir -> (f(Path.Combine(dir,"input_0.pb")),f(Path.Combine(dir,"output_0.pb")))|]
        for (index,(input,output)) in test_data |> Array.indexed do
            use values2 = f(Tensor.FromTensorProtoFloat32(input)) 
            let ys = values2.Value |> Seq.toArray
            let diff = 
                (ys, Tensor.FromTensorProtoFloat32(output) |> Seq.toArray)
                ||> Array.zip
                |> Array.sumBy (fun (x,y) -> System.Math.Abs(x-y))
            if diff > 0.1f then failwithf "Unexpected result in example %i with a difference of %f" index diff
            printfn "%f %A" diff ys


    [<Test>]
    let ``prebuilt model``() = 
        let model = File.ReadAllBytes(Path.Combine(mnistDir, "model.onnx")) 
        model |> testModel

    [<AutoOpen>]
    module Node = 

        let unaryOp op (attrs: AttributeProto[]) (name: string, input: string, output: string)  =
            simple op (name, [|input|],[|output|],attrs)

        let binaryOp op (attrs: AttributeProto[]) (name: string, left: string, right: string, output: string)  =
            simple op (name, [|left;right|],[|output|],attrs)

        let reshape = binaryOp "Reshape" [||]
        let add = binaryOp "Add" [||]

        let cnn(name: string, 
                input: string, 
                kernel: string, 
                output: string, 
                kernel_shape: int64[], 
                strides: int64[], 
                auto_pad: string , 
                group: int64,
                dilations: int64[]) = 

            let attrs = 
                [|
                    Attr.ints("kernel_shape", kernel_shape)// [|5L;5L|]
                    Attr.ints("strides", strides) // [|1L;1L|]
                    Attr.string("auto_pad", auto_pad) //"SAME_UPPER"
                    Attr.int("group",group) // 1L
                    Attr.ints("dilations",dilations) //[|1L;1L|]
                |] |> Array.choose id

            let np = simple "Conv" (name, [|input;kernel|],[|output|],attrs)
            np

        let pool opType
                   (name: string, 
                    input: string, 
                    output: string, 
                    kernel_shape: int64[], 
                    strides: int64[], 
                    pads: int64[], 
                    auto_pad : string) = 

            let attrs = 
                [|
                    Attr.ints("kernel_shape",kernel_shape)
                    Attr.ints("strides",strides)
                    Attr.ints("pads",pads)
                    Attr.string("auto_pad",auto_pad)
                |] |> Array.choose id
            let np = simple opType (name, [|input|],[|output|],attrs)
            np

        let maxPool = pool "MaxPool"
        let relu(name: string, input: string, output: string) = unaryOp "Relu" [||] (name, input,output)
        let matmul = binaryOp "MatMul"  [||]

    /// This is a full MNist example that exactly matches the pre-trained model
    [<Test>]
    let ``manual model``() =
        let nodes = 
            [|
                reshape ("Times212_reshape1","Parameter193", "Parameter193_reshape1_shape","Parameter193_reshape1")
                cnn("Convolution28","Input3","Parameter5","Convolution28_Output_0",[|5L;5L|],[|1L;1L|],"SAME_UPPER",1L,[|1L;1L|])
                add ("Plus30", "Convolution28_Output_0", "Parameter6","Plus30_Output_0")
                relu("ReLU32","Plus30_Output_0","ReLU32_Output_0")
                maxPool("Pooling66","ReLU32_Output_0", "Pooling66_Output_0", [|2L;2L|],[|2L;2L|],[|0L;0L;0L;0L|],"NOTSET")
                cnn("Convolution110","Pooling66_Output_0","Parameter87","Convolution110_Output_0",[|5L;5L|],[|1L;1L|],"SAME_UPPER",1L,[|1L;1L|])
                add ("Plus112", "Convolution110_Output_0", "Parameter88" ,"Plus112_Output_0")
                relu("ReLU114", "Plus112_Output_0", "ReLU114_Output_0")
                maxPool("Pooling160","ReLU114_Output_0", "Pooling160_Output_0", [|3L;3L|],[|3L;3L|],[|0L;0L;0L;0L|],"NOTSET")
                reshape("Times212_reshape0","Pooling160_Output_0", "Pooling160_Output_0_reshape0_shape","Pooling160_Output_0_reshape0")
                matmul("Times212", "Pooling160_Output_0_reshape0", "Parameter193_reshape1", "Times212_Output_0")
                add("Plus214", "Times212_Output_0", "Parameter194" , "Plus214_Output_0")
            |]

        let tensorProtos = 
            [|
                "Parameter193", DataType.FLOAT32, [|16L; 4L; 4L; 10L|]
                "Parameter87", DataType.FLOAT32, [|16L; 8L; 5L; 5L|]
                "Parameter5", DataType.FLOAT32, [|8L; 1L; 5L; 5L|]
                "Parameter6", DataType.FLOAT32, [|8L; 1L; 1L|]
                "Parameter88", DataType.FLOAT32, [|16L; 1L; 1L|]
                "Pooling160_Output_0_reshape0_shape", DataType.INT64, [|2L|]
                "Parameter193_reshape1_shape", DataType.INT64, [|2L|]
                "Parameter194", DataType.FLOAT32, [|1L; 10L|]
            |] |> Array.map (fun (name, dt,dims) -> 
                let tp = TensorProto(DataType = int dt, Name = name)
                tp.Dims.AddRange(dims)
                let path = Path.Combine(mnistDir, name)
                let data = File.ReadAllBytes(path)
                match dt with
                | DataType.FLOAT32 -> 
                    tp.FloatData.AddRange(data |> bytesToFloats)
                | DataType.INT64 -> 
                    tp.Int64Data.AddRange(data |> bytesToInts)
                | _ -> failwith "err"
                tp)

        let inputs = 
            [| 
                "Input3", DataType.FLOAT32, [|1L;1L;28L;28L|]
                "Parameter5", DataType.FLOAT32, [|8L;1L;5L;5L|]
                "Parameter6", DataType.FLOAT32, [|8L;1L;1L|]
                "Parameter87", DataType.FLOAT32, [|16L;8L;5L;5L|]
                "Parameter88", DataType.FLOAT32, [|16L;1L;1L|]
                "Pooling160_Output_0_reshape0_shape", DataType.INT64, [|2L|]
                "Parameter193",DataType.FLOAT32,[|16L;4L;4L;10L|]
                "Parameter193_reshape1_shape", DataType.INT64,[|2L|]
                "Parameter194", DataType.FLOAT32,[|1L;10L|]
            |]
            |> Array.map (fun (name,dt,shape) -> ValueInfoProto(DocString = "", Name = name, Type = TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 dt, Shape = makeShape shape))))

        let outputs =
            [|"Plus214_Output_0", DataType.FLOAT32,[|1L;10L|]|]
            |> Array.map (fun (name,dt,shape) -> ValueInfoProto(DocString = "", Name = name, Type = TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 dt, Shape = makeShape shape))))

        let valueInfo = 
            [|
                "Parameter193_reshape1", DataType.FLOAT32, [|256L;10L|]
                "Convolution28_Output_0", DataType.FLOAT32, [|1L;8L;28L;28L|]
                "Plus30_Output_0", DataType.FLOAT32, [|1L;8L;28L;28L|]
                "ReLU32_Output_0", DataType.FLOAT32, [|1L;8L;28L;28L|]
                "Pooling66_Output_0", DataType.FLOAT32, [|1L;8L;14L;14L|]
                "Convolution110_Output_0", DataType.FLOAT32, [|1L;16L;14L;14L|]
                "Plus112_Output_0", DataType.FLOAT32, [|1L;16L;14L;14L|]
                "ReLU114_Output_0", DataType.FLOAT32, [|1L;16L;14L;14L|]
                "Pooling160_Output_0", DataType.FLOAT32, [|1L;16L;4L;4L|]
                "Pooling160_Output_0_reshape0", DataType.FLOAT32, [|1L; 256L|]
                "Times212_Output_0", DataType.FLOAT32, [|1L;10L|]
            |]
            |> Array.map (fun (name,dt,shape) -> ValueInfoProto(DocString = "", Name = name, Type = TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 dt, Shape = makeShape shape))))

        let mp = 
            let graph = GraphProto(Name = "CNTKGraph")
            graph.Input.AddRange(inputs)
            graph.Output.AddRange(outputs)
            graph.ValueInfo.AddRange(valueInfo)
            graph.Node.AddRange(nodes)
            graph.Initializer.AddRange(tensorProtos)
            let mp = 
                ModelProto(DocString = "",
                    Domain = "ai.cntk",
                    IrVersion = 3L,
                    ModelVersion = 1L,
                    ProducerName = "CNTK",
                    ProducerVersion = "2.5.1",
                    Graph = graph)
            mp.OpsetImport.Add(OperatorSetIdProto(Version = 8L))
            mp

        let mpData = writeModelToStream(mp)

        mpData |> testModel


    /// NOTE: This is roughly 14x slower with the API overhead
    [<Test>]
    let ``eager mnist``() = 
        let getTensorF(name,shape) =
            let dts = File.ReadAllBytes(Path.Combine(mnistDir, name)) |> bytesToFloats
            on.reshape(ArrayTensorExtensions.ToTensor(dts) ,ArrayTensorExtensions.ToTensor(shape))

        let p193 = getTensorF("Parameter193", [|16L; 4L; 4L; 10L|])
        let p87  = getTensorF("Parameter87",  [|16L; 8L; 5L; 5L|])
        let p5   = getTensorF("Parameter5",  [|8L; 1L; 5L; 5L|])
        let p6   = getTensorF("Parameter6", [|8L; 1L; 1L|])
        let p88  = getTensorF("Parameter88", [|16L; 1L; 1L|])
        let p194 = getTensorF("Parameter194", [|1L; 10L|]) 

        let mnist (x:Tensor<float32>) = 
            let f (x:Tensor<float32>) (p1:Tensor<float32>) (p2:Tensor<float32>) k = 
                on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
            on.add(on.mat_mul(on.reshape((f (f x p5 p6 2L) p87 p88 3L),[|1;256|]),on.reshape(p193,[|256;10|])),p194)

        let test_data = 
            let f (x: TensorProto) = x.RawData.ToByteArray() |> bytesToFloats 
            test_data |> Array.map (fun (x,y) -> (f x).ToTensor().Reshape([|1;1;28;28|]), f y)

        for (index,(x,y1)) in Array.indexed(test_data) do
            let y2 = mnist x
            let diff = (y2.ToArray(),y1) ||> Array.zip |> Array.sumBy (fun (x,y) -> System.Math.Abs(x-y))
            if diff > 0.1f then failwithf "Unexpected result in example %i with a difference of %f" index diff

    type ong = FSharp.ML.Onnx.API.Graph.OnnxGraph

    type MNISTGraph() = 

        let bytesToFloats(buffer : byte[]) = 
            let xs= Array.zeroCreate<float32> (buffer.Length / 4)
            System.Buffer.BlockCopy(buffer, 0, xs, 0, buffer.Length)
            xs

        let getTensorF(name,shape) =
            let dts = File.ReadAllBytes(Path.Combine(mnistDir, name)) |> bytesToFloats
            on.reshape(ArrayTensorExtensions.ToTensor(dts) ,ArrayTensorExtensions.ToTensor(shape))

        let p193 = getTensorF("Parameter193", [|16L; 4L; 4L; 10L|])
        let p87  = getTensorF("Parameter87",  [|16L; 8L; 5L; 5L|])
        let p5   = getTensorF("Parameter5",  [|8L; 1L; 5L; 5L|])
        let p6   = getTensorF("Parameter6", [|8L; 1L; 1L|])
        let p88  = getTensorF("Parameter88", [|16L; 1L; 1L|])
        let p194 = getTensorF("Parameter194", [|1L; 10L|]) 

        [<ReflectedDefinition>]
        member this.Rec(graph:Graph, x:ValueInfo,p1,p2,k) = 
           ong.MaxPool(graph,ong.Relu(graph,ong.Add(graph,ong.Conv(graph,x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst

        [<ReflectedDefinition>]
        member this.Forward(graph: Graph, x: ValueInfo) = 
            let constant (x:Tensor<float32>) = Constants.constant(graph,x)
            let x = this.Rec(graph, x, constant p5,constant p6,2L)
            let x = this.Rec (graph, x, constant p87,constant p88,3L)
            ong.Add(graph, ong.MatMul(graph, ong.Reshape(graph, x,Constants.constant(graph,[|1L;256L|].ToTensor())),ong.Reshape(graph,constant p193,Constants.constant(graph,[|256L;10L|].ToTensor()))),constant p194)

        [<ReflectedDefinition>]
        member this.Rec(x:Tensor<float32>,p1,p2,k) = 
           on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst

        [<ReflectedDefinition>]
        member this.Forward(x: Tensor<float32>) = 
            let layer (p1,p2,k) (x:Tensor<float32>) = 
                on.max_pool(on.relu(on.add(on.conv(x,p1,auto_pad = "SAME_UPPER"),p2)),kernel_shape = [|k;k|], strides = [|k;k|]) |> fst
            on.add(on.mat_mul(on.reshape(x |> layer(p5,p6,2L) |> layer (p87, p88, 3L),[|1;256|]),on.reshape(p193,[|256;10|])),p194)

    [<Test>]
    let ``full mnist``() = 

        let mnistG = MNISTGraph()
        let makeValueInfoProto(valueInfo: ValueInfo) = 
            ValueInfoProto(Name = valueInfo.name, Type = 
                TypeProto(TensorType = TypeProto.Types.Tensor(ElemType = int32 valueInfo.dt)))

        let input = {name = "Input3";dt=DataType.FLOAT32}
        let graph = Graph.Default()
        let output = mnistG.Forward(graph,input)

        let gp = GraphProto(Name = "G")
        gp.Input.Add(makeValueInfoProto(input))
        gp.Output.Add(makeValueInfoProto(output))
        gp.Node.Add(graph.ops)
        testModel(writeModelToStream(gp |> graphToModel))

    [<Test>]
    let ``converted graph``() = 
        let mnistG = MNISTGraph()
        use graphFunction : DV<Tensor<float32> -> DV<Tensor<float32>>> = toOnnxGraph(<@ mnistG.Forward @>)
        testModel2(graphFunction.Value)
        testModel2(fun x -> new DV<Tensor<float32>>(mnistG.Forward(x),(fun () -> ())))
        
    type RecA = {a:Tensor<float32>;b:Tensor<float32>}
    type RecB = {a:Tensor<float32>;b:RecA; c:Tensor<float32>*Tensor<float32>}

    [<Test>]
    let ``complex input and output``() = 
        let tupleFunction = <@ fun (x:Tensor<float32>,y:Tensor<float32>) -> (on.add(x,y),on.sub(x,y),on.log(x)) @>
        let recFunction = <@ fun (x:RecA,y:Tensor<float32>) -> {a=x.a;b=x.b},{a = on.add(x.a,x.b); b = x; c = (x.a,x.b)} @>
        use ff : DV<RecA*Tensor<float32> -> DV<RecA*RecB>> = toOnnxGraph(recFunction)
        let p1 = [|0.1f|].ToTensor() :> Tensor<float32>
        let x = ({a=p1;b=p1},p1)
        use y = ff.Value(x)
        let r1 = (fst y.Value).a.ToArray() 
        let diff = (p1.ToArray(),r1) ||> Array.zip |> Array.sumBy (fun (x,y) -> System.Math.Abs(x-y))
        if diff > 0.1f then failwith "Error running function"

module ONNXExample = 
    [<Test>]
    let ``squeezenet example``() = 
        let loadTensorFromFile(filename: string) = 
            File.ReadAllLines(filename).[1..]
            |> Array.collect (fun line -> line.Split([|',';'[';']'|], StringSplitOptions.RemoveEmptyEntries))
            |> Array.map Single.Parse

        let dir = Path.Combine(__SOURCE_DIRECTORY__ ,"..", "..", "data", "squeezenet")
        let modelPath = Path.Combine(dir,"squeezenet.onnx")

        // Optional : Create session options and set the graph optimization level for the session
        let options = new SessionOptions()
        options.GraphOptimizationLevel <- GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        use session = new InferenceSession(modelPath, options)
        let inputMeta = session.InputMetadata
        let inputData = loadTensorFromFile(Path.Combine(dir,"bench.in"))
        let container = 
            [|
                for name in inputMeta.Keys do
                    let tensor = new DenseTensor<float32>(Memory.op_Implicit(inputData),ReadOnlySpan.op_Implicit(inputMeta.[name].Dimensions)) 
                    yield NamedOnnxValue.CreateFromTensor<float32>(name, tensor)
            |]
        use results = session.Run(container)

        // TODO verify output
        ()
//        for r in results do
//            printfn "Output for %s" r.Name
//            printfn "%s" (r.AsTensor<float32>().GetArrayString())

module ExpressionFunctions =                                         
    open FSharp.Quotations.Evaluator
    type O(a: string) = 
        [<ReflectedDefinition>] static member A = "A"
        [<ReflectedDefinition>] static member AA with get() = "A" and set(x) = printf "%s" x
        [<ReflectedDefinition>] static member B () = "B"
        [<ReflectedDefinition>] static member BA () () = "BA"
        [<ReflectedDefinition>] static member C (c1:string) = c1 + "C"
        [<ReflectedDefinition>] static member D (d1:string, d2:int) = d1 + string d2 + "D"
        [<ReflectedDefinition>] static member E (e1:string) (e2:int) = e1 + string e2 + "E"
        [<ReflectedDefinition>] static member F( f1:string) (f2:int,f3:bool) = f1 + string f2 + string f3 + "F"
        [<ReflectedDefinition>] static member G (g1:string) = fun (g2:int) -> g1 + string g2 + "G"
        [<ReflectedDefinition>] static member H () (h1:string) (h2:string,h3:int) (h4:string,h5:int,h6:bool) = h1 + h2 + string h3 + h4 + string h5 + string h6 + "H"
        [<ReflectedDefinition>] member x.A0 = "A" + a
        [<ReflectedDefinition>] member x.B0 () = "B" + a
        [<ReflectedDefinition>] member x.C0 (c1:string) = c1 + "C" + a
        [<ReflectedDefinition>] member x.D0 (d1:string, d2:int) = d1 + string d2 + "D" + a
        [<ReflectedDefinition>] member x.E0 (e1:string) (e2:int) = e1 + string e2 + "E" + a
        [<ReflectedDefinition>] member x.F0 (f1:string) (f2:int,f3:bool) = f1 + string f2 + string f3 + "F" + a
        [<ReflectedDefinition>] member x.G0 (g1:string) = fun (g2:int) -> g1 + string g2 + "G" + a
        [<ReflectedDefinition>] member x.H0 () (h1:string) (h2:string,h3:int) (h4:string,h5:int,h6:bool) = h1 + h2 + string h3 + h4 + string h5 + string h6 + "H" + a
        [<ReflectedDefinition>] static member Combo() =
                                            O.A + O.B() + O.BA () () +
                                            O.C "c1" + O.D("d1",2) + O.E "e1" 2 + O.F "f1" (2,true) +
                                            O.G "g1" 2 + O.H () "h1" ("h2",3) ("h4",5,true)

        [<ReflectedDefinition>] member x.Combo0() = 
                                            x.A0 + x.B0 () + 
                                            x.C0 "c1" + x.D0("d1",2) + x.E0 "e1" 2 + 
                                            x.F0 "f1" (2,true) + x.G0 "g1" 2 + x.H0 () "h1" ("h2",3) ("h4",5,true) + 
                                            O.Combo()

    [<Test>]
    let ``static method quotation and simplification``() = 
        let qt = <@ O.Combo() @>
        let minQuote = 
            qt 
            |> Expr.unfoldWhileChanged ExprTransforms.expandWithReflectedDefinition |> Seq.last
            |> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last
            |> Expr.Cast<string>
        Assert.AreEqual(minQuote.Evaluate(), qt.Evaluate(), "Static Method Quotation expanded and reduced")
        Assert.AreEqual(minQuote |> Expr.getCallNames, set ["ToString"; "op_Addition"], "Expanded expression should only have ToString and op_Addition calls")

    [<Test>]
    let ``object method quotation and simplification``() = 
        let qt = <@ O("A").Combo0() @>
        let minQuote = 
            qt 
            |> Expr.unfoldWhileChanged ExprTransforms.expandWithReflectedDefinition |> Seq.last
            |> Expr.unfoldWhileChanged (ExprTransforms.reduceApplicationsAndLambdas true) |> Seq.last
            |> Expr.Cast<string>
        Assert.AreEqual(minQuote.Evaluate(), qt.Evaluate(), "Objct Method Quotation expanded and reduced")
        Assert.AreEqual(minQuote |> Expr.getCallNames, set ["ToString"; "op_Addition"], "Expanded expression should only have ToString and op_Addition calls")

