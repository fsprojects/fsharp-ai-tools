// Build the Debug 'FSAI.Tools.Tests' before using this

#I __SOURCE_DIRECTORY__
#r "netstandard"
#r "../tests/bin/Debug/net472/FSAI.Tools.Proto.dll"
#r "bin/Debug/net472/FSAI.Tools.dll"
#nowarn "49"

//open Argu
open System
open System.IO
open FSAI.Tools

if not System.Environment.Is64BitProcess then System.Environment.Exit(-1)

module Play1 = 
    let t1 = new TFTensor( [| 6 |])
    let shape1 = TFShape [| 1L |] 
    let shape2 = TFShape [| 1L |] 

    // TODO: TFShape should support basic things like equality
    (shape1 = shape2)
    (shape1.Equals(shape2))

    //tf.add(1, 2)
    //tf.add([1, 2], [3, 4])
    //tf.square(5)
    //tf.reduce_sum([1, 2, 3])
    //tf.encode_base64("hello world")



module PlayWithAddingConstants = 
    let xv = 3.0
    let t1 = new TFTensor( [| xv |])
    let graph = new TFGraph()
    let x = graph.Const (t1)

    let y1 = graph.Square (x, "Square1") // x^2
    let y2 = graph.Square (y1, "Square2") // x^4
    let y3 = graph.Square (y2, "Square3") // x^8
    
    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], [| y3 |])
    let v = res1.[0].GetValue() :?> double[] |> Array.item 0
    let answer = (xv ** 8.0)
    if v <> answer then failwith "fail"
    
module PlayWithAddingAddN = 
    let xv = 3.0
    let t1 = new TFTensor( [| xv |])
    let graph = new TFGraph()
    let x = graph.Const (t1)

    let y1 = graph.AddN ([| x; x; x |]) 
    
    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], [| y1 |])
    let v = res1.[0].GetValue() :?> double[] |> Array.item 0
    if v <> 9.0 then failwith "fail"

module PlayWithStack = 
    let xv = 3.0
    let graph = new TFGraph()
    let sample_0 = new TFTensor (1.0)
    let sample_1 = new TFTensor ([| 1.0 |])
    let sample_1_by_1 = new TFTensor (array2D [ [ 1.0 ] ])
    let sample_2_by_1 = new TFTensor (array2D [ [ 1.0 ]; [2.0] ])
    let sample_2_by_3 = new TFTensor (array2D [ [ 1.0; 1.0; 1.0 ]; [2.0; 1.0; 1.0] ])
    sample_0.Shape // [| |]
    sample_1.Shape // [| 1 |]
    sample_1_by_1.Shape // [| 1; 1 |]
    sample_2_by_1.Shape // [| 2 rows, 1 column |]
    sample_2_by_3.Shape // [| 2 rows, 1 column |]

    //let x = graph.Concat (graph.Const(new TFTensor(0)), [| graph.Const sample_0; graph.Const sample_0 |])
    let x1 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_0 |]) // --> shape [ 2 ]
    let x2 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_1 |], axis=0)
    let x3 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_1 |], axis=1)
    let x4 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_2_by_1 |], axis=0)
    let x5 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_2_by_1 |], axis=1)
    let x6 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_2_by_1 |], axis=2)
    let x7 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_2_by_3 |], axis=0)
    let x8 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_2_by_3 |], axis=1)
    let x9 = graph.Stack ([| for i in 1 .. 4 -> graph.Const sample_2_by_3 |], axis=2)
    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], [| x1; x2; x3; x4; x5; x6; x7; x8; x9 |])
//  [|shape [4]; shape [4x1]; shape [1x4]; shape [4x2x1]; shape [2x4x1]; shape [2x1x4]; shape [4x2x3]; shape [2x4x3]; shape [2x3x4]|]  
    let v = res1.[0].Shape


module PlayWithAddingConcat = 
    let xv = 3.0
    let t1 = new TFTensor( [| xv |])
    let graph = new TFGraph()
    let x = graph.Const (t1)

    let y1 = graph.Concat(graph.Const(new TFTensor(0)), [| x; x; x |])
    
    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], [| y1 |])
    let v = res1.[0].GetValue() :?> double[] 
    if v <> [| 3.0; 3.0; 3.0 |] then failwith "fail"
    
module PlayWithAddingConcat2 = 
    let xv1 = [| 3.0; 4.0; 5.0 |]
    let xv2 = [| 3.1; 4.1; 5.1 |]
    let t1 = new TFTensor( xv1 )
    let t2 = new TFTensor( xv2 )
    let graph = new TFGraph()
    let x1 = graph.Const (t1)
    let x2 = graph.Const (t2)

    let y1 = graph.Concat(graph.Const(new TFTensor(0)), [| x1; x2 |])
    let y2 = graph.Stack([| x1; x2 |])
    
    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], [| y1; y2 |])
    let v1 = res1.[0].GetValue() :?> double[] 
    let v2 = res1.[1].GetValue() :?> double[,] 
    res1.[1].Shape
    if v2.[0,*] <> xv1 then failwith "fail"
    if v2.[1,*] <> xv2 then failwith "fail"
    if v1 <> [| 3.0; 3.0; 3.0 |] then failwith "fail"
    
    
module PlayWithGradients1D = 
    let xv = 3.0
    let t1 = new TFTensor( [| xv |])
    let graph = new TFGraph()
    let x = graph.Const (t1)

    let y1 = graph.Square (x, "Square1") // x^2
    let y2 = graph.Square (y1, "Square2") // x^4
    let y3 = graph.Square (y2, "Square3") // x^8
    let g = graph.AddGradients ([| y1; y3 |], [| x |]) // d(x^2)/dx + d(x^8)/dx | x = 3.0  --> 2x + 8x^7 --> 
    
    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], g)
    let v = res1.[0].GetValue() :?> double[] |> Array.item 0
    let answer = 2.0 * xv + 8.0 * (xv ** 7.0)
    if v <> answer then failwith "fail"
    
module PlayWithGradients_2R_to_R = 
    let xv1 = 3.0
    let t1 = new TFTensor( [| xv1 |])
    t1.Shape
    let xv2 = 4.0
    let t2 = new TFTensor( [| xv2 |])
    let graph = new TFGraph()
    let x1 = graph.Const (t1)
    let x2 = graph.Const (t2)

    let y = graph.Add(graph.Square(graph.Square x1), graph.Square x2) // y = x1^4 + x2^2 
    // first element differentiates y by first variable, second element differentiates y by second variable
    let g = graph.AddGradients ([| y |], [| x1; x2 |]) // [dy/dx1; dy/dx2] = [4*x1^3; 2*x2]
    
    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], g)
    let v1 = res1.[0].GetValue() :?> double[] |> Array.item 0
    let v2 = res1.[1].GetValue() :?> double[] |> Array.item 0
    let answer1 = 4.0 * xv1 ** 3.0 
    let answer2 = 2.0 * xv2
    if v1 <> answer1 then failwith "fail"
    if v2 <> answer2 then failwith "fail"
    
module PlayWithGradients_2R_to_R_grad_slice = 
    let xv1 = 3.0
    let xv2 = 4.0
    let graph = new TFGraph()
    let x = graph.Const (new TFTensor( [| xv1; xv2 |]))

    let y = graph.Slice(x, graph.Const(new TFTensor( [| 0 |])),graph.Const(new TFTensor( [| 1 |]))) // y = xv1
    let y2 = graph.Squeeze(y, [| 0L |])

    let g = graph.AddGradients ([| y2 |], [| x |]) 
    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y2; yield! g |])
    let v = results.[0].GetValue() :?> double
    let tangent = results.[1].GetValue() :?> double[]
    if v <> 3.0 then failwith "fail"
    
    let t1 = tangent.[0]
    let t2 = tangent.[1]
    if t1 <> 1.0 then failwith "fail"
    if t2 <> 0.0 then failwith "fail"

    
module PlayWithGradients_2R_to_R_grad = 
    let xv1 = 3.0
    let xv2 = 4.0
    let graph = new TFGraph()
    let x = graph.Const (new TFTensor( [| xv1; xv2 |]))

    let y = graph.ReduceSum(x) // y = x1 + x2  
    // partial derivative by first variable, then by second variable
    let g = graph.AddGradients ([| y |], [| x |]) // [dy/dx1; dy/dx2] 

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y; yield! g |])
    let v = results.[0].GetValue() :?> double
    let tangent = results.[1].GetValue() :?> double[]
    if v <> 3.0 + 4.0 then failwith "fail"
    
    let t1 = tangent.[0]
    let t2 = tangent.[1]
    if t1 <> 1.0 then failwith "fail"
    if t2 <> 1.0 then failwith "fail"

module PlayWithGradients_2R_to_R_grad_mean = 
    let xv1 = 3.0
    let xv2 = 4.0
    let graph = new TFGraph()
    let x = graph.Const (new TFTensor( [| xv1; xv2 |]))

    let y = graph.ReduceMean(x) // y = mean (x1,x2)  
    // partial derivative by first variable, then by second variable
    let g = graph.AddGradients ([| y |], [| x |]) // [dy/dx1; dy/dx2] 
    g.Length

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y; yield! g |])
    let v = results.[0].GetValue() :?> double
    let tangent = results.[1].GetValue() :?> double[]
    if v <> 3.0 + 4.0 then failwith "fail"
    
    let t1 = tangent.[0]
    let t2 = tangent.[1]
    if t1 <> 0.5 then failwith "fail"
    if t2 <> 0.5 then failwith "fail"

module PlayWithGradients_2R_to_R_grad_prod = 
    let xv1 = 3.0
    let xv2 = 4.0
    let graph = new TFGraph()
    let x = graph.Const (new TFTensor( [| xv1; xv2 |]))

    let y = graph.ReduceProd(x) // y = x1 * x2  
    // partial derivative by first variable, then by second variable
    let g = graph.AddGradients ([| y |], [| x |]) // [dy/dx1; dy/dx2] 
    g.Length

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y; yield! g |])
    let v = results.[0].GetValue() :?> double
    let tangent = results.[1].GetValue() :?> double[]
    if v <> 3.0 * 4.0 then failwith "fail"
    
    let t1 = tangent.[0]
    let t2 = tangent.[1]
    if t1 <> 4.0 then failwith "fail"
    if t2 <> 3.0 then failwith "fail"

    
module PlayWithGradients_2R_to_R_grad_with_initial_tangents = 
    let xv1 = 3.0
    let xv2 = 4.0
    let graph = new TFGraph()
    let x = graph.Const (new TFTensor( [| xv1; xv2 |]))
    let dyv = 2.0 
    let dy = graph.Const (new TFTensor( [| dyv |]))

    // y = x1 * x2 + x2  * x1
    let y = graph.Sum(graph.Mul(graph.ReverseV2(x,graph.Const (new TFTensor( [| 0 |]))),x), graph.Const  (new TFTensor( [| 0 |]))) 
    let g = graph.AddGradients ([| y |], [| x |], dy=[| dy |])  
    g.Length

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y; yield! g |])
    let v = results.[0].GetValue() :?> double
    let tangent = results.[1].GetValue() :?> double[]
    if v <> 3.0 * 4.0 + 4.0 * 3.0 then failwith "fail"
    
    let t1 = tangent.[0]
    let t2 = tangent.[1]
    if t1 <> 2.0 * xv2 * dyv then failwith "fail"
    if t2 <> 2.0 * xv1 * dyv then failwith "fail"

    
module PlayWithGradients_2R2R_to_R_grad = 
    let xv11 = 3.0
    let xv21 = 4.0
    let xv12 = 5.0
    let xv22 = 6.0
    let graph = new TFGraph()
    let x = graph.Const (new TFTensor( array2D [| [| xv11; xv21 |]; [| xv12; xv22 |] |] ))

    let y = graph.Sum(graph.Mul(x,x), graph.Const  (new TFTensor( [| 0;1 |]))) // y = x1 + x2  
    // partial derivative by first variable, then by second variable
    let g = graph.AddGradients ([| y |], [| x |]) // [dy/dx1; dy/dx2] 
    g.Length

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y; yield! g |])
    let v = results.[0].GetValue() :?> double
    let tangent = results.[1].GetValue() :?> double[,]
    if v <> xv11 * xv11 + xv12 * xv12 + xv21 * xv21 + xv22 * xv22 then failwith "fail"
    
    let t11 = tangent.[0,0]
    let t21 = tangent.[1,0]
    let t12 = tangent.[0,1]
    let t22 = tangent.[1,1]
    if t11 <> 2.0 * xv11  then failwith "fail"
    if t12 <> 2.0 * xv21  then failwith "fail"
    if t21 <> 2.0 * xv12  then failwith "fail"
    if t22 <> 2.0 * xv22  then failwith "fail"

module PlayWithGradients_2_by_R_to_R_pointwise = 
    let xv1 = 3.0
    let xv2 = 4.0
    let graph = new TFGraph()
    let x = graph.Const (new TFTensor( [| xv1; xv2 |]))

    let y = graph.Add(graph.Square(graph.Square x), graph.Square x) // y1 = x1^4 + x1^2; y2 = x2^4 + x2^2;  
    // first element differentiates by first variable, second element by second variable
    let g = graph.AddGradients ([| y |], [| x |]) // [dy1/dx; dy2/dx] 
    g.Length

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y; yield! g |])
    let primals = results.[0].GetValue() :?> double[]
    let tangent = results.[1].GetValue() :?> double[]
    let v1 = primals.[0]
    let v2 = primals.[1]
    if v1 <> xv1 ** 4.0 + xv1 ** 2.0 then failwith "fail"
    if v2 <> xv2 ** 4.0 + xv2 ** 2.0 then failwith "fail"
    
    let t1 = tangent.[0]
    let t2 = tangent.[1]
    if t1 <> 4.0 * xv1 ** 3.0 + 2.0 * xv1 then failwith "fail"
    if t2 <> 4.0 * xv2 ** 3.0 + 2.0 * xv2 then failwith "fail"

module PlayWithGradients_R_to_2R = 
    let xv1 = 3.0
    let graph = new TFGraph()
    let x = graph.Const (new TFTensor( [| xv1 |]))

    // y1 = 5.0 * x + x
    // y2 = 6.0 * x + x
    let y = graph.Add(graph.Mul(graph.Const (new TFTensor( [| 5.0; 6.0 |])), x), x)
    // differentiate results and add
    let g = graph.AddGradients ([| y |], [| x |]) // [dy1/dx + dy2/dx] 

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y; yield! g |])
    let primal1 = results.[0].GetValue() :?> double[] |> Array.item 0
    let primal2 = results.[0].GetValue() :?> double[] |> Array.item 1
    let dy1_dx_plus_dy2_dx = results.[1].GetValue() :?> double[] |> Array.item 0
    if primal1 <> 5.0 * xv1 + xv1 then failwith "fail"
    if primal2 <> 6.0 * xv1 + xv1 then failwith "fail"
    if dy1_dx_plus_dy2_dx <> 5.0 + 1.0 + 6.0 + 1.0 then failwith "fail"

module PlayWithGradients_2R_to_2R = 
    let xv1 = 3.0
    let xv2 = 4.0
    let graph = new TFGraph()
    let x1 = graph.Const (new TFTensor( [| xv1 |]))
    let x2 = graph.Const (new TFTensor( [| xv2 |]))

    let y1 = graph.Add(graph.Mul(graph.Square x1, x2), x2) // y1 = x1^2 * x2 + x2
    let y2 = graph.Add(graph.Mul(graph.Square x2, x1), x1) // y2 = x2^2 * x1 + x1
    // first element differentiates by first variable, second element by second variable
    let g1 = graph.AddGradients ([| y1 |], [| x1; x2 |]) // [dy1/dx1; dy1/dx2] 
    let g2 = graph.AddGradients ([| y2 |], [| x1; x2 |]) // [dy2/dx1; dy2/dx2] 

    let session = new TFSession(graph)
    let results = session.Run( [| |], [| |], [| yield y1; yield y2; yield! g1; yield! g2 |])
    let primal1 = results.[0].GetValue() :?> double[] |> Array.item 0
    let primal2 = results.[1].GetValue() :?> double[] |> Array.item 0
    let dy1_dx1 = results.[2].GetValue() :?> double[] |> Array.item 0
    let dy1_dx2 = results.[3].GetValue() :?> double[] |> Array.item 0
    let dy2_dx1 = results.[4].GetValue() :?> double[] |> Array.item 0
    let dy2_dx2 = results.[5].GetValue() :?> double[] |> Array.item 0
    if primal1 <> xv1 ** 2.0 * xv2 + xv2 then failwith "fail"
    if primal2 <> xv2 ** 2.0 * xv1 + xv1 then failwith "fail"
    if dy1_dx1 <> 2.0 * xv1 * xv2 then failwith "fail"
    if dy1_dx2 <> xv1 ** 2.0 + 1.0 then failwith "fail"
    if dy2_dx1 <> xv2 ** 2.0 + 1.0 then failwith "fail"
    if dy2_dx2 <> 2.0 * xv2 * xv1 then failwith "fail"

    //graph.ReduceMean
    graph.Mean

module PlayWithUsingVariables = 
    let t1 = new TFTensor( [| 6 |])
    let graph = new TFGraph()
    let out1 = graph.Const(new TFTensor( [| 6 |]))
    let out2 = graph.Const(new TFTensor( [| 1 |]))
    let var2 : TFVariable = graph.Variable(out2, trainable = true, name= "var2")
    let out = graph.Add(out1, var2.Read)

    let session = new TFSession(graph)
    let res1 = session.Run( [|  |], [| |], [| out; var2.Read |])
    res1.[0].GetValue()
    res1.[1].GetValue()

    //graph.Const("a", "b")
    //graph.Variable( .MakeName("a", "b")

    //TFTensor(TFDataType.Double, [| 2;2|], )
    new TFTensor(array2D [| [| 0.0; 1.0 |]; [| 0.0; 1.0 |] |])
    (new TFTensor(array2D [| [| 0.0; 1.0 |]; [| 0.0; 1.0 |] ; [| 0.0; 1.0 |] |])).GetTensorDimension(1)
    TFOutput
    //graph.Inpu

