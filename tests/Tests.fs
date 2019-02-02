module Tests
open NUnit.Framework
open TensorFlow.FSharp
open TensorFlow.FSharp.Operations
open System.Runtime.CompilerServices
open System
open System.Runtime.InteropServices

//open TensorFlow.FSharp.RandomOps
//open RandomOps

[<Test>]
let ``smoke test`` () = ()

/// TensorFlow has some quirky naming rules that we need to mathc
[<Test>]
let ``naming check`` () =
    // with tf.new_scope("bar") as scope"
    //   y = tf.add(1,2)                    # y.name = bar/Add:0
    //   z = tf.add(1,2,name=scope)         # z.name = bar:0
    //   print(scope)                       # bar/
    let graph = new TFGraph()
    let a = graph.Const(new TFTensor(1))
    let b = graph.Const(new TFTensor(1))
    do
        use ns = graph.NameScope("bar")
        let x = graph.Add(a,b)
        let y = graph.Add(a,b)
        let z = graph.Add(a,b,name= string ns)
        Assert.AreEqual(ns.ToString(),"bar/") 
        Assert.AreEqual(graph.CurrentNameScope,"bar") 
        Assert.AreEqual(x.Name,"bar/Add:0") 
        Assert.AreEqual(y.Name,"bar/Add_1:0") 
        Assert.AreEqual(z.Name,"bar:0")
        try
            graph.Add(a,b,name= string ns) |> ignore
            Assert.Fail("Tensorflow failed to raise an error upon repeated use of a namespace for an op name")
        with
        | :? TFException -> ()
    do
        use ns = graph.NameScope("bar")
        Assert.AreEqual(ns.ToString(), "bar_1/") // ensure namescopes are unique
        let name1 = graph.MakeName(None,"X",false) |> ignore
        let name2 = graph.MakeName(None,"X",false) |> ignore
        Assert.AreEqual(name1,name2)


/// From SampleTest form TensorFlowSharp
[<Test>]
let ``Basic Constant Ops`` () =
    // Test the manual GetRunner, 
    use s = new TFSession()
    let g = s.Graph
    let a = g.Const(new TFTensor(2))
    let b = g.Const(new TFTensor(3))

    // Add two constants
    let results = s.GetRunner().Run (g.Add (a, b))
    Assert.AreEqual(results.GetValue (),5)

    // Test Zeros
    let shape = TFShape(4,4)
    let o = g.Ones(shape)
    let r = g.RandomNormal(shape)
    let z = g.Zeros(shape)
    let m = g.Mul(o,r)
    let res1 = s.GetRunner().Run(m)
    let res2 = s.GetRunner().Run(g.Mul(g.Mul(o,r),z))

    // Test Constants
    let co = g.Constant(1.0f, shape, TFDataType.Float32)
    let cz = g.Constant(0.0f, shape, TFDataType.Float32)
    let cr = g.RandomNormal(shape)
    let cm = g.Mul(co,cr)
    let cres1 = s.GetRunner().Run(cm)
    let cres2 = s.GetRunner().Run(g.Mul(g.Mul(co, cr), cz))

    let so = g.Ones(TFShape(4, 3), TFDataType.Float32)
    let sr = g.RandomNormal(TFShape(3, 5))
    let sz = g.Zeros(TFShape(5, 6))
    let sm = g.MatMul(so, sr)
    let sres1 = s.GetRunner().Run(sm)
    let sres2 = s.GetRunner().Run(g.MatMul(g.MatMul(so, sr), sz))

    // Multiply two constants
    let results = s.GetRunner().Run (g.Mul (a, b))
    Assert.AreEqual(results.GetValue (), 6)

[<Test>]    
let ``Basic Variables`` () = 
    use s = new TFSession()
    let g = s.Graph

    // We use "shorts" here, so notice the casting to short to get the 
    // tensor with the right data type
    let var_a = g.Placeholder (TFDataType.Int16)
    let var_b = g.Placeholder (TFDataType.Int16)

    let add = g.Add(var_a, var_b)
    let mul = g.Mul(var_a, var_b)

    let addV = 
        s.GetRunner()
            .AddInput(var_a, new TFTensor(3s))
            .AddInput(var_b, new TFTensor(2s))
            .Run(add).GetValue() 
    Assert.AreEqual(addV,5s)
     
    let mulV = 
        s.GetRunner()
            .AddInput(var_a, new TFTensor(3s))
            .AddInput(var_b, new TFTensor(2s))
            .Run(mul).GetValue()
    Assert.AreEqual(mulV,6s)

[<Test>]
let ``Test Variable`` =
    use s = new TFSession()
    let g = s.Graph
    let initValue = g.Const(new TFTensor(1.5))
    let increment = g.Const(new TFTensor(0.5))
    let tfv = g.Variable(initValue)
    // Not 100% on the following
    let update = g.AssignVariableOp(tfv.VariableOp, g.Add(tfv.Read, increment))
    for i = 0 to 4 do
        let result = s.GetRunner().Fetch(tfv.Read).AddTarget(update).Run()
        Assert.AreEqual(result, 1.5 + (float i * 0.5)) // there should not be floating point errors but ther may be

[<Test>]
let ``Basic Multidimensional Array`` () =
    use g = new TFGraph()
    use s = new TFSession(g)
    let var_a = g.Placeholder(TFDataType.Int32)
    let mul = g.Mul(var_a, g.Const(new TFTensor(2)))
    let a = new TFTensor([|[|[|0;1|];[|2;3|]|];[|[|4;5|];[|6;7|]|]|]) 
    let actual = s.GetRunner().AddInput(var_a,a).Fetch(mul).Run().[0].GetValue() :?> int[,,]
    let expected = (new TFTensor([|[|[|0;2|];[|4;6|]|];[|[|8;10|];[|12;14|]|]|])).GetValue() :?> int[,,]
    Assert.AreEqual(expected,actual)
    

[<Test>]
let ``Basic Matrix`` () =
    use g = new TFGraph()
    use s = new TFSession(g)

    // 1x2 matrix
    let matrix1 = g.Const(new TFTensor([|[|3.;3.|]|]))
    // 2x1 matrix
    let matrix2 = g.Const(new TFTensor([|[|2.|];[|2.|]|]))

    let product = g.MatMul(matrix1, matrix2)
    Assert.AreEqual(12,(s.GetRunner().Run(product).GetValue() :?> float[,]).[0,0])


// Low Level Tests

let placeholder (graph : TFGraph, s : TFStatus) : TFOperation =
    let desc = new TFOperationDesc(graph, "Placeholder", "feed")
    desc.SetAttr("dtype", TFDataType.Int32) |> ignore
    desc.FinishOperation()

let scalarConst (v : TFTensor, graph : TFGraph, status : TFStatus, name : string option) =
    let desc = new TFOperationDesc(graph, "Const", name |> Option.defaultValue "scalar")
    desc.SetAttr("value", v, status) |> ignore
    desc.SetAttr("dtype", TFDataType.Int32) |> ignore
    desc.FinishOperation()

let add (left : TFOperation, right :TFOperation, graph : TFGraph, status : TFStatus) =
    let op = new TFOperationDesc(graph, "AddN", "add")
    op.AddInputs(new TFOutput(left,0), new TFOutput (right, 0)) |> ignore
    op.FinishOperation()

type Assert with
    static member TFStatus(status : TFStatus, [<CallerMemberName>] ?caller : string, ?message : string) = 
        if status.StatusCode <> TFCode.Ok then 
            Assert.Fail(sprintf "%s: %s %s " (defaultArg caller "") status.StatusMessage (defaultArg message ""))

[<Test>]
let ``Test Import Graph Def`` () =
    let status = new TFStatus()
    let mutable graphDef = Option<TFBuffer>.None
    // Create graph with two nodes, "x", and "3"
    do
        use graph = new TFGraph() 
        Assert.TFStatus(status)
        placeholder(graph, status) |> ignore
        Assert.IsNotNull(graph.["feed"])
        scalarConst(new TFTensor(3), graph, status, None) |> ignore
        Assert.IsNotNull(graph.["scalar"])

        // Export to GraphDef
        let buff = new TFBuffer()
        graph.ToGraphDef(buff, status) |> ignore
        Assert.TFStatus(status)
        graphDef <- Some(buff)

    // Import it again, with a prefix, in a fresh group
    do
        use graph = new TFGraph()
        use options = new TFImportGraphDefOptions()
        options.SetPrefix("imported")
        graph.Import(graphDef.Value, options, status) 
        Assert.TFStatus(status)
        let scalar = graph.["imported/scalar"]
        let feed = graph.["imported/feed"]
        Assert.IsNotNull(scalar)
        Assert.IsNotNull(feed)

        // Can add nodes to the imported graph without trouble
        add(feed,scalar, graph, status) |> ignore
        Assert.TFStatus(status)

[<Test>]
let ``Test Session`` () =
    let status = new TFStatus()
    use graph = new TFGraph()
    let feed = placeholder(graph,status)
    let two = scalarConst(new TFTensor(2), graph, status,None)
    let add = add(feed,two, graph,status)
    Assert.TFStatus(status)

    // Create a session for this graph
    use session = new TFSession(graph, status=status) 
    Assert.TFStatus(status)
    //Run the graph
    let inputs = [|new TFOutput(feed,0)|]
    let inputValues = [|new TFTensor(3)|]
    let addOutput = new TFOutput(add,0)
    let outputs = [|addOutput|]

    let results = session.Run(inputs,inputValues,outputs,status=status)
    Assert.TFStatus(status)
    let res = results.[0]
    Assert.AreEqual(res.TFDataType, TFDataType.Int32)
    Assert.AreEqual(res.NumDims, 0)
    Assert.AreEqual(res.TensorByteSize, UIntPtr(uint32 4)) // Double check this
    Assert.AreEqual(Marshal.ReadInt32(res.Data), 5)


[<Test>]
let ``Test Operation Outut List Size`` () =
    use graph = new TFGraph()
    let c1 = graph.Const(new TFTensor(1L), "c1") 
    let c1 = graph.Const(new TFTensor([|1;2|]), "c1") // Source material does duplicate name here
    let c2 = graph.Const(new TFTensor([|[|1;2|];[|3;4|]|]))
    let outputs = graph.ShapeN([|c1;c2|])
    let op = outputs.[0].Op
    Assert.AreEqual(op.OutputListLength("output"),2)
    Assert.AreEqual(op.NumOutputs,2)

[<Test>]
let ``Test Output Shape`` () =
    use graph = new TFGraph()
    let c1 = graph.Const(new TFTensor(0L),"c1")
    let s1 = graph.GetShape(c1)
    let c2 = graph.Const(new TFTensor([|1L;2L;3L|]), "c2")
    let s2 = graph.GetShape(c2)
    let c3 = graph.Const(new TFTensor([|[|1L;2L;3L|]; [|4L;5L;6L|]|]), "c3")
    let s3 = graph.GetShape(c3)
    ()

//type WhileTester() = 
//    let status = new TFStatus()
//    let graph = new TFGraph()
//    let mutable session = Option<TFSession>.None
//    let mutable runner = Option<TFRunner>.None
//    let mutable inputs : TFOutput[] = [||]
//    let mutable outputs : TFOutput[] = [||]
//
//    member this.Init(ninputs : int, constructor : WhileConstructor) =
//        inputs <- Array.init ninputs (fun i -> graph.Placeholder(TFDataType.Int32, name = sprintf "p%i" i))
//        Assert.TFStatus(status)
//        let outputs = graph.While(inputs, constructor, status)
//        Assert.TFStatus(status)
//    
//    member this.Run([<ParamArray>] inputValues : int[]) =
//        Assert.AreEqual(inputValues.Length, inputs.Length)
//        session <- Some(new TFSession(graph))
//        runner <- Some(session.Value.GetRunner())
//        for i = 0 to inputs.Length - 1 do
//            runner.Value.AddInput (inputs.[i], new TFTensor(inputValues.[i])) |> ignore
//        runner.Value.Fetch(outputs) |> ignore
//        runner.Value.Run()
//
//    interface IDisposable with
//        member this.Dispose() = 
//            status.Dispose()
//            graph.Dispose()

//  TODO pull in WhileTester and figure out how to handle the Dispatcher
//[<Test>]
//let ``While Test`` () =
//    use j = new WhileTester()
//    // Create loop: while (input1 < input2) input1 += input2 + 1
//    let whileFunction(conditionGraph, condInputs : TFOutput[], [<Out>] condOutput : TFOutput, bodyGraph : TFGraph, bodyInputs : TFOutput [], [<Out>] ?name : string) =
//        Assert.AreNotEqual(bodyGraph.Handle,IntPtr.Zero)
//        Assert.AreNotEqual(conditionGraph.Handle,IntPtr.Zero)
//        let status = new TFStatus()
//        let lessThan = conditionGraph.Less(condInputs.[0], condInputs.[1])
//        Assert.TFStatus(status)
//        contOutput <- new TFOutput(lessThan.Operation, 0)
//        let add1 = bodyGraph.Add(bodyInputs.[0], bodyInputs.[1])
//        let one = bodyGraph.Const(new TFTensor(1))
//        let add2 = bodyGraph.Add(add1,one)
//        bodyOutputs.[0] <- new TFOutput(add2,new TFTensor(0))
//        bodyOutputs.[1] <- bodyInputs.[1]
//        name <- "Simple1"
//    let res = j.Run (-9,2)
//
//    Assert.AreEqual(res.[0].GetValue() :?> int, 3)
//    Assert.AreEqual(res.[1].GetValue() :?> int, 2)


// TODO / NOTE: The following test fails due to the Op type of name AttributeTestListshape does not exist. It's probably not supposed to exist
//[<Test>]
//let ``Attribute Test`` () = 
//    use graph = new TFGraph()
//    use status = new TFStatus()
//    let mutable counter = 0
//    let init (op : string) =
//        let opname = 
//            if op.StartsWith("list(") 
//            then "AttributeTestList" + op.Substring(5, op.Length - 6)
//            else "AttributeTest" + op
//        let desc = new TFOperationDesc(graph, opname, "name" + string counter)
//        counter <- counter + 1
//        desc
//    let shape1 = new TFShape(1, 3)
//    let shape2 = new TFShape(2, 4, 6)
//    let desc = init("list(shape)")
//    desc.SetAttr("v", [|shape1;shape2|]) |> ignore
//    let op = desc.FinishOperation()
//    let expectMeta(op : TFOperation, name : string, expectedListSize : int, expectedType : TFAttributeType, expectedTotalSize : int) =
//        let meta = op.GetAttributeMetadata(name)
//        Assert.AreEqual(meta.IsList, expectedListSize >= 0 )
//        Assert.AreEqual(meta.ListSize,expectedListSize)
//        Assert.AreEqual(meta.Type, expectedType)
//        Assert.AreEqual(expectedTotalSize, meta.TotalSize)
//    expectMeta(op, "v", 2, TFAttributeType.Shape, 5)

[<Test>]
let ``Add Control Input`` () =
    let status = new TFStatus()
    use g = new TFGraph()
    use s = new TFSession(g)
    let yes = new TFTensor(true)
    let no = new TFTensor(false)
    let placeholder = g.Placeholder( TFDataType.Bool, name = "boolean")
    let check = 
        (new TFOperationDesc(g, "Assert", "assert"))
            .AddInput(placeholder) 
            .AddInputs(placeholder) // Source code repeats the adding of input, this 
            .FinishOperation()
    let noop = 
        (new TFOperationDesc(g, "NoOp", "noop"))
            .AddControlInput(check)
            .FinishOperation()
    s.GetRunner()
       .AddInput(placeholder, yes) 
       .AddTarget(noop)             
        // No problems when the Assert check succeeds
       .Run() |> ignore

    // Excpetion thrown by the execution of the Assert node
    try 
        s.GetRunner()
            .AddInput(placeholder,no)
            .AddTarget(noop)
            .Run() |> ignore
        Assert.Fail("This sould have thrown an exception")
    with  _ -> ()
        

[<Test>]
let ``Test Parameters With Indexes`` () =
    let status = new TFStatus()
    use g = new TFGraph()
    use s = new TFSession(g)
    let split = 
        (new TFOperationDesc(g, "Split", "Split"))
            .AddInput(scalarConst(new TFTensor(0),g,status,None).[0])
            .AddInput(scalarConst(new TFTensor([|1;2;3;4|]),g,status,Some("array")).[0])
            .SetAttr("num_split",2L)
            .FinishOperation()

    let add = 
        (new TFOperationDesc(g, "Add", "Add"))
            .AddInput(split.[0])
            .AddInput(split.[1])
            .FinishOperation().[0]
    
    // fetch using colo sepearted names
    let fetched = s.GetRunner().Fetch("Split:1").Run().[0]
    let vals = fetched.GetValue() :?> int[]
    if (vals.[0] <> 3 || vals.[1] <> 4) then
        Assert.Fail("Expected the values 3 and 4")
    
    // Add inputs using colo seperated names
    let t = new TFTensor([|4;3;2;1|])
    let ret = (s.GetRunner().AddInput("Split:0",t).AddInput("Split:1",t).Fetch("Add").Run()).GetValue(0) :?> TFTensor
    let value = ret.GetValue() :?> int[]
    if value.[0] <> 8 || value.[1] <> 6 || value.[2] <> 4 || value.[3] <> 2 then
        Assert.Fail("Expected 8, 6, 4, 2")
    

        
