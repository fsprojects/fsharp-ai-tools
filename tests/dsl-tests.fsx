// Build the Debug 'TensorFlow.FSharp.Tests' before using this

#I __SOURCE_DIRECTORY__
#r "netstandard"

#r "bin/Debug/net472/TensorFlow.FSharp.dll"
#nowarn "49"

//open Argu
open System
open System.IO
open TensorFlow.FSharp
open TensorFlow.FSharp.DSL

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
    
module PlayWithGradients2D = 
    let xv1 = 3.0
    let t1 = new TFTensor( [| xv1 |])
    let xv2 = 4.0
    let t2 = new TFTensor( [| xv2 |])
    let graph = new TFGraph()
    let x1 = graph.Const (t1)
    let x2 = graph.Const (t2)

    let y = graph.Add(graph.Square(graph.Square x1), graph.Square x2) // y = x1^4 + x2^2 
    let g = graph.AddGradients ([| y |], [| x1; x2 |]) // [dy/dx1; dy/dx2] = [4*x1^3; 2*x2]
    
    let session = new TFSession(graph)
    let res1 = session.Run( [| |], [| |], g)
    let v1 = res1.[0].GetValue() :?> double[] |> Array.item 0
    let v2 = res1.[1].GetValue() :?> double[] |> Array.item 0
    let answer1 = 4.0 * xv1 ** 3.0 
    let answer2 = 2.0 * xv2
    if v1 <> answer1 then failwith "fail"
    if v2 <> answer2 then failwith "fail"
    
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

let apply x f = f x

module PlayWithTF = 
    tf { return v 1.0 }
    |> DT.Run
   
    tf { return v 1.0 + v 4.0 }
    |> DT.Run

    let f x = tf { return x * x + v 4.0 * x }
    let df = f |> DT.Diff

    df (v 3.0)
    |> DT.Run
    |> fun x -> x.GetValue() :?> double
    |> (=) (2.0 * 3.0 + 4.0)

    tf { return vec [1.0; 2.0] + v 4.0 }
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return vec [1.0; 2.0] + v 4.0 }
    //|> DT.Diff
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { use _ = DT.WithScope("foo")
         return vec [1.0; 2.0] + v 4.0 }
   // |> DT.Diff
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] }
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return matrix [ [ 1.0; 2.0 ]; [ 3.0; 4.0 ] ] + v 4.0 }
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> DT.Run
    |> fun x -> x.GetValue()

    let var v nm = DT.Variable (v, name=nm)
    tf { return var (vec [ 1.0 ]) "x" + v 4.0 }
    |> fun dt -> DT.Run(dt, ["x", (vec [2.0] :> _)] )
    |> fun x -> x.GetValue()

    tf { return var (vec [ 1.0 ]) "x" * var (vec [ 1.0 ]) "x" + v 4.0 }
   // |> DT.Diff (var "x")
    |> DT.Run
    |> fun x -> x.GetValue()

    tf { return DT.Variable (vec [ 1.0 ], name="hey") + v 4.0 }
    |> fun dt -> DT.Run(dt, ["hey", upcast (vec [2.0])])
    |> fun x -> x.GetValue()
       // Gives 6.0

    let input = matrix4 [ for i in 0 .. 9 -> [ for j in 1 .. 40 -> [ for k in 1 .. 40 -> [ for m in 0 .. 2 -> double (i+j+k+m) ]]]]
    let name = "a"
    let instance_norm (input, name) =
        tf { use _ = TF.WithScope(name + "/instance_norm")
             let mu, sigma_sq = DT.Moments (input, axes=[0;1])
             let shift = DT.Variable (v 0.0, name + "/shift")
             let scale = DT.Variable (v 1.0, name + "/scale")
             let epsilon = v 0.001
             let normalized = (input - mu) / sqrt (sigma_sq + epsilon)
             return scale * normalized + shift }

    let friendly4D (d : 'T[,,,]) =
        [| for i in 0..Array4D.length1 d - 1 -> [| for j in 0..Array4D.length2 d - 1 -> [| for k in 0..Array4D.length3 d - 1 -> [| for m in 0..Array4D.length4 d - 1 -> d.[i,j,k,m]  |]|]|]|]
        |> array2D |> Array2D.map array2D

    instance_norm (input, name) |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D
    (fun input -> instance_norm (input, name)) |> DT.Diff |> apply input |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

    let out_channels = 128
    let filter_size = 7
    let conv_init_vars (out_channels:int, filter_size:int, is_transpose: bool, name) =
        let weights_shape = 
            if is_transpose then
                Shape [| Dim filter_size; Dim filter_size; Dim out_channels; Dim.Inferred |]
            else
                Shape [| Dim filter_size; Dim filter_size; Dim.Inferred; Dim out_channels |]
        tf { let truncatedNormal = DT.TruncatedNormal(weights_shape)
             return DT.Variable (truncatedNormal * v 0.1, name + "/weights") }

    let is_relu = 1
    let stride = 1
    let conv_layer (input, out_channels, filter_size, stride, is_relu, name) = 
        tf { let filters = conv_init_vars (out_channels, filter_size, false, name)
             let x = DT.Conv2D (input, filters, stride=stride)
             let x = instance_norm (x, name)
             if is_relu then 
                 return DT.Relu x 
             else 
                 return x }

    conv_layer (input, out_channels, filter_size, 1, true, "layer")  |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D
    (fun input -> conv_layer (input, out_channels, filter_size, 1, true, "layer")) |> DT.Diff |> apply input |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

    let residual_block (input, filter_size, name) = 
        tf { let tmp = conv_layer(input, 128, filter_size, 1, true, name + "_c1")
             return input + conv_layer(tmp, 128, filter_size, 1, false, name + "_c2") }

    let conv2D_transpose (input, filter, stride) = 
        tf { return DT.Conv2DBackpropInput(filter, input, stride, padding = "SAME") }
  
    let conv_transpose_layer (input: DT<double>, num_filters, filter_size, stride, name) =
        tf { let filters = conv_init_vars (num_filters, filter_size, true, name)
             return DT.Relu (instance_norm (conv2D_transpose (input, filters, stride), name))
           }

    let to_pixel_value (input: DT<double>) = 
        tf { return tanh input * v 150.0 + (v 255.0 / v 2.0) }

    // The style-transfer tf

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         return x }
    |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         return x }
    |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         return x }
    |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         let x = residual_block (x, 3, "resid1")
         return x }
    |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         let x = residual_block (x, 3, "resid1")
         let x = residual_block (x, 3, "resid2")
         let x = residual_block (x, 3, "resid3")
         let x = residual_block (x, 3, "resid4")
         let x = residual_block (x, 3, "resid5")
         return x }
    |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D


    let t1 = 
        tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
             let x = conv_layer (x, 64, 3, 2, true, "conv2")
             let x = conv_layer (x, 128, 3, 2, true, "conv3")
             let x = residual_block (x, 3, "resid1")
             let x = residual_block (x, 3, "resid2")
             let x = residual_block (x, 3, "resid3")
             let x = residual_block (x, 3, "resid4")
             let x = residual_block (x, 3, "resid5")
             return x }

    let t2 = 
        tf { return conv_transpose_layer (t1, 64, 3, 2, "conv_t1") }
        |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

    tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
         let x = conv_layer (x, 64, 3, 2, true, "conv2")
         let x = conv_layer (x, 128, 3, 2, true, "conv3")
         let x = residual_block (x, 3, "resid1")
         let x = residual_block (x, 3, "resid2")
         let x = residual_block (x, 3, "resid3")
         let x = residual_block (x, 3, "resid4")
         let x = residual_block (x, 3, "resid5")
         let x = conv_transpose_layer (x, 64, 3, 2, "conv_t1") // TODO: check fails
         return x }
    |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

(*
(fun input -> 
        tf { let x = conv_layer (input, 32, 9, 1, true, "conv1")
             let x = conv_layer (x, 64, 3, 2, true, "conv2")
             let x = conv_layer (x, 128, 3, 2, true, "conv3")
             let x = residual_block (x, 3, "resid1")
             let x = residual_block (x, 3, "resid2")
             let x = residual_block (x, 3, "resid3")
             let x = residual_block (x, 3, "resid4")
             let x = residual_block (x, 3, "resid5")
             let x = conv_transpose_layer (x, 64, 3, 2, "conv_t1") // TODO: check fails
             return x })
    |> DT.Diff |> apply input |> DT.Run |> fun x -> x.GetValue() :?> double[,,,] |> friendly4D

//2019-01-24 17:21:17.095544: F tensorflow/core/grappler/costs/op_level_cost_estimator.cc:577] Check failed: iz == filter_shape.dim(in_channel_index).size() (64 vs. 128)
    *)


(*
let x = conv_layer (x, 3, 9, 1, false, "conv_t3")
             let x = to_pixel_value x
             let x = DT.ClipByValue (x, v 0.0, v 255.0)
             return x }
*)

 (*
    let d = array2D [| [| 1.0f; 2.0f |]; [| 3.0f; 4.0f |] |]
    let shape = shape [ d.GetLength(0); d.GetLength(1)  ]
    let graph = new TFGraph()
    let session= new TFSession(graph)
    let n1 = graph.Const(new TFTensor(d))
    let res1 = session.Run( [|  |], [| |], [| n1 |])
*)

    //graph.Const("a", "b")
    //graph.Variable( .MakeName("a", "b")

    //TFTensor(TFDataType.Double, [| 2;2|], )
    (TFDataType.Double, [| 2;2|])
    TFOutput
    //graph.Inpu

