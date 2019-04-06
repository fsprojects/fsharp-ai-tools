open System.Windows.Forms

#I __SOURCE_DIRECTORY__
#r "netstandard"

#I @"..\tests\bin\Debug\net461"

//#r "TensorFlow.FSharp.dll"
#r "TensorFlow.Net.dll"
#r "NumSharp.Core.dll"

#nowarn "49"

//open TensorFlow.FSharp
open System.Text


open Tensorflow
open System.ComponentModel


let a = tf.constant(4.0f)
let b = tf.constant(5.0f)
let c = tf.add(a,b)

let sess = tf.Session()
let o = sess.run(c)






let foo = tf.get_variable("foo",TensorShape(1,2,3), TF_DataType.TF_FLOAT)


//let graph2 = tf.get_default_graph()
//tf.get_variable("var", dtype= TF_DataType.TF_FLOAT)
//let vOps = [| |]

//(input : TFOutput, filter : TFOutput, strides : int64[], padding : string, ?use_cudnn_on_gpu : bool, ?data_format : string, ?dilations : int64[],  ?name : string) : TFOutput =

(*
module tf =

    let _op_def_lib = lazy new OpDefLibrary()

    type ReluArgs = { features : Tensor }

    type Conv2DArgs = {
        input : Tensor
        filter : Tensor
        strides : int[]
        padding : string
        use_cudnn_on_gpu : bool
        data_format : string
        dilations : int[]
    }

    type Conv2DTransposeArgs = {
        input_sizes : Tensor
        filter : Tensor
        out_backprop : Tensor
        strides : int[]
        padding : string
        use_cudnn_on_gpu : bool
        data_format : string
        dilations : int[]
    }
        
    type nn() =
        static member conv2d_tranpose(input_sizes : Tensor, filter : Tensor, out_backprop : Tensor, ?strides : int[], ?padding : string, ?use_cudnn_on_gpu : bool, ?data_format : string, ?dilations : int[], ?name : string) =
            // We're adding the defaults here because there isn't a straight forward mechanism to do this
            // This code is temporary anyway
            let args : Conv2DTransposeArgs = {
                input_sizes = input_sizes
                filter = filter
                out_backprop = out_backprop
                strides = defaultArg strides [|1;1;1;1|]
                padding = defaultArg padding "SAME"
                use_cudnn_on_gpu = defaultArg use_cudnn_on_gpu true
                data_format = defaultArg data_format "NHWC"
                dilations = defaultArg dilations [|1;1;1;1|]
            }
            let name = defaultArg name "Conv2DBackpropInput"
            _op_def_lib.Force()._apply_op_helper("Conv2DBackpropInput", name=name,args = args).output

        static member conv2d(input : Tensor, filter : Tensor, ?strides : int[], ?padding : string, ?dilations : int[],  ?data_format : string, ?use_cudnn_on_gpu : bool, ?name : string) =
            // We're adding the defaults here because there isn't a straight forward mechanism to do this
            // This code is temporary anyway
            let args : Conv2DArgs = {
                input = input
                filter = filter
                strides = defaultArg strides [|1;1;1;1|]
                padding = defaultArg padding "SAME"
                use_cudnn_on_gpu = defaultArg use_cudnn_on_gpu true
                data_format = defaultArg data_format "NHWC"
                dilations = defaultArg dilations [|1;1;1;1|]
            }
            printfn "%A" args
            let name = defaultArg name "Conv2D"
            _op_def_lib.Force()._apply_op_helper("Conv2D", name=name,args = args).output

        static member relu(input : Tensor, ?name : string) = 
            _op_def_lib.Force()._apply_op_helper("Relu", name= (name |> Option.defaultValue "Relu"), args = {features = input} ).output
*)

module tf2 =
    open System.Collections.Generic
    let private maybeSet (name : string) (dict : Dictionary<string,obj>) (x : #obj option) =
        match x with
        | Some(x) -> dict.[name] <- x
        | None -> ()

    let _op_def_lib = lazy new OpDefLibrary()
        
    type nn() =
        static member conv2d_tranpose(input_sizes : Tensor, filter : Tensor, out_backprop : Tensor, strides : int[], padding : string, ?use_cudnn_on_gpu : bool, ?data_format : string, ?dilations : int[], ?name : string) =
            let args = new Dictionary<string,obj>()
            args.["input_sizes"] <- input_sizes
            args.["filter"] <- filter
            args.["input_sizes"] <- input_sizes
            args.["out_backprop"] <- out_backprop
            args.["strides"] <- strides
            args.["padding"] <- padding
            use_cudnn_on_gpu |> maybeSet "use_cudnn_on_gpu" args
            data_format |> maybeSet "data_format" args
            dilations |> maybeSet "dilations" args
            let name = defaultArg name "Conv2DBackpropInput"
            _op_def_lib.Force()._apply_op_helper("Conv2DBackpropInput", name=name,keywords = args).output


        static member conv2d(input : Tensor, filter : Tensor, strides : int[], padding : string, ?dilations : int[],  ?data_format : string, ?use_cudnn_on_gpu : bool, ?name : string) =
            let args = new Dictionary<string,obj>()
            args.["input"] <- input
            args.["filter"] <- filter
            args.["strides"] <- strides
            args.["padding"] <- padding
            use_cudnn_on_gpu |> maybeSet "use_cudnn_on_gpu" args
            data_format |> maybeSet "data_format" args
            dilations |> maybeSet "dilations" args
            dilations |> Option.iter (fun x -> args.["dilations"] <- x)
            printfn "%A" args
            let name = defaultArg name "Conv2D"
            _op_def_lib.Force()._apply_op_helper("Conv2D", name=name,keywords = args).output

        static member relu(features : Tensor, ?name : string) = 
            let args = new Dictionary<string,obj>()
            args.["features"] <- features
            _op_def_lib.Force()._apply_op_helper("Relu", name= (name |> Option.defaultValue "Relu"), keywords = args ).output



let input_img = tf.constant(1.0f,TF_DataType.TF_FLOAT, [|1;10;10;1|])

let filters = tf.random_normal([|3;3;1;1|])

let input_sizes = tf.constant([|1;20;20;1|], TF_DataType.TF_INT64)

//let conv2res = tf2.nn.conv2d_tranpose(input_sizes, filters, input_img, strides = [|1;2;2;1|], padding="SAME", dilations = [|1;1;1;1|], data_format = "NHWC", use_cudnn_on_gpu = false, name = "Conv2D22")

let convres = tf2.nn.conv2d(input_img, filters, [|1;2;2;1|], padding="SAME", dilations = [|1;1;1;1|])

sess.run(tf2.nn.relu(tf.constant(-4.0f)))
//sess.run(tf2.global_variables_initializer().output)
let convres2 = sess.run(convres)

convres2 

(*
let convres2 = sess.run(conv2res)
convres2.shape
convres2.shape 

let x = { features = tf.constant(4.0f)} : tf.ReluArgs 

tf._op_def_lib.Force()._apply_op_helper("Relu", args = x)

[|for propertyDescriptor in TypeDescriptor.GetProperties(x) -> propertyDescriptor.GetValue(x) |]
*)
