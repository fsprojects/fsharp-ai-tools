[<AutoOpen>]
module Utils

open Tensorflow
open System
open System.Collections.Generic
open System.Reflection
open Tensorflow.Operations.Activation
type gen_ops = Tensorflow.Operations.gen_ops

let tf = Tensorflow.Binding.tf

module vs = 
    let variable_scope(name : string) =
        let vs = tf.variable_scope(name)
        vs.__init__()
        vs.__enter__()
        {new IDisposable with 
            member this.Dispose() = 
                vs.__exit__()
                vs.__del__()}

let init =
    lazy
        ops.RegisterGradientFunction("Sqrt",Func<Operation,Tensor[],Tensor[]>(fun op grads -> 
            let grad = grads.[0]
            let y = op.outputs.[0]
            [|gen_ops.sqrt_grad(y,grad)|]))

        ops.RegisterGradientFunction("Rsqrt",Func<Operation,Tensor[],Tensor[]>(fun op grads -> 
            let grad = grads.[0]
            let y = op.outputs.[0]
            [|gen_ops.rsqrt_grad(y,grad)|]))

        /// Returns the gradient for (x-y)^2.
        ops.RegisterGradientFunction("SquaredDifference",Func<Operation,Tensor[],Tensor[]>(fun op grads ->
            // TODO support skip_input_indices
            // TODO suport IndexedSlices
            let x = op.inputs.[0]
            let y = op.inputs.[1]
            let grad = grads.[0]
            let x_grad = 
                Tensorflow.Binding.tf_with(ops.control_dependencies([|grad|]), fun _ -> 
                    2.0 *  grad * (x - y))
            if x.TensorShape.is_fully_defined() && y.TensorShape.is_fully_defined() then
                if x.shape = y.shape 
                then [|x_grad; -x_grad|]
                else
                    match x.shape,y.shape with
                    | [|_;1|],_ -> [|array_ops.reshape(math_ops.reduce_sum(x_grad,1),x.shape);-x_grad|]
                    | _,[|_;1|] -> [|x_grad;-array_ops.reshape(math_ops.reduce_sum(x_grad,1),y.shape)|]
                    | [|_;_;1|],_ -> [|array_ops.reshape(math_ops.reduce_sum(x_grad,2),x.shape);-x_grad|]
                    | _,[|_;_;1|] -> [|x_grad;-array_ops.reshape(math_ops.reduce_sum(x_grad,2),y.shape)|]
                    | _ -> failwith "fix hack"
            else
                failwith "todo"
            ))

        /// Fixes bug
        ops.RegisterGradientFunction("Slice", Func<Operation,Tensor[],Tensor[]>(fun op grads -> 
            let grad = grads.[0]
            let input_vec = op.inputs.[0]
            let begin_vec = op.inputs.[1]
            let input_rank = array_ops.rank(input_vec)
            let slice_size = array_ops.shape(op.outputs.[0])
            let shape = array_ops.stack([|input_rank; tf.constant(1)|])
            let before_pad = array_ops.reshape(begin_vec, shape)
            let after_pad = array_ops.reshape(array_ops.shape(input_vec) - slice_size - begin_vec, shape)
            let paddings = array_ops.concat([|before_pad; after_pad|], 1)
            [|array_ops.pad(grad, paddings); null; null|]
        ))

        // fixes another bug
        ops.RegisterGradientFunction("GatherV2", Func<Operation,Tensor[],Tensor[]>(fun op grads -> 
            /// Converts an IndexedSlices to a Tensor without sparse->dense warnings.
            let indexedSlicesToTensorNoWarning(indexed_slices: Tensorflow.Framework.IndexedSlices) = 
                match indexed_slices.dense_shape with
                | null -> raise (ValueError(sprintf "Tensor conversion requested for IndexedSlices without dense_shape: %s"
                     indexed_slices.name))
                | _ -> math_ops.unsorted_segment_sum(indexed_slices.values,
                                                     indexed_slices.indices,
                                                     indexed_slices.dense_shape.slice(0))

            let grad = grads.[0]
            let ps = op.inputs.[0]
            ops.colocate_with(ps)
            let params_shape = array_ops.shape(ps, out_type = tf.int32)
            let indices = op.inputs.[1]
            let indices_size = array_ops.expand_dims(array_ops.size(indices), 0)
            let axis = op.inputs.[2]
            let axis_static = tensor_util.constant_value(axis)
            if int axis_static = 0 then
                let params_tail_shape = params_shape.slice(NumSharp.Slice(start = Nullable(1)));
                let values_shape = array_ops.concat([|indices_size; params_tail_shape|], 0);
                let values = array_ops.reshape(grad, values_shape);
                let indices = array_ops.reshape(indices, indices_size);
                let indexed_slices = Tensorflow.Framework.IndexedSlices(values, indices, params_shape)
                [| indexedSlicesToTensorNoWarning(indexed_slices); null; null|]
            else failwith "only supports static axis of 0 at this time"
        ))

let setup() = init.Force()

module Async =
    let mapiChunkBySize (chunkSize: int) (f: int -> 'T -> 'b) (xs: 'T[]) =
        xs 
        |> Array.mapi (fun i x -> (i,x))
        |> Array.chunkBySize chunkSize
        |> Array.map (fun xs -> async {return  [| for (i,x) in xs -> f i x|]})
        |> Async.Parallel
        |> Async.RunSynchronously
        |> Array.collect id


type Tensorflow.variable_scope with
    member scope.name = 
        let m = typeof<Tensorflow.variable_scope>.GetField("_name", BindingFlags.Instance ||| BindingFlags.NonPublic)
        m.GetValue(scope)  :?> string

let loggingf = printfn

type Tensorflow.tensorflow.train_internal with
    member _.get_or_create_global_step(): Tensorflow.RefVariable = 
        failwith "todo"

// https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/training/checkpoint_utils.py#L203-L291

// Replaces `tf.Variable` initializers so they load from a checkpoint file.
type Tensorflow.tensorflow.train_internal with
    member _.init_from_checkpoint(ckpt_dir_or_file, assignemnt_map) = 
        failwith "todo"

type Tensorflow.tensorflow with
    member _.constant(values: string[], ?shape: int[], ?name: string) =
        let name = defaultArg name "Const"
        let g = ops.get_default_graph()
        let tensor_proto = 
            let tp = TensorProto()
            tp.Dtype <- tf.string.as_datatype_enum()
            tp.StringVal.AddRange(values |> Array.map (fun x -> Google.Protobuf.ByteString.CopyFromUtf8(x)))
            tp.TensorShape <- tensor_util.as_shape(defaultArg shape [|values.Length|])
            tp
        let attrs = Dictionary([|"value",AttrValue(Tensor = tensor_proto); "dtype", AttrValue(Type = tensor_proto.Dtype)|] |> dict)
        g.create_op("Const",[||],[|tf.string|],attrs = attrs, name = name).output

    member _.all_variable_names() = 
        let graph = tf.get_default_graph()
        [| for x in graph.get_operations() do if x.op.OpType = "VariableV2" then yield x.op.name|]

    member tf.restore(path: string, ?variable_names: string[], ?mapping: string -> string, ?name: string) = 
        let name = defaultArg name "restore"
        let mapping = defaultArg mapping id
        let graph = tf.get_default_graph()
        let variable_names = variable_names |> Option.defaultWith (fun _ -> tf.all_variable_names())
        let variables = [| for x in variable_names  -> graph.get_operation_by_name(mapping(x))|]
        let dataTypes = [| for x in variables -> x.op.output.dtype.as_base_dtype()|]
        // TODO proper slices requires making an extra C shim to expose types and mapping
        let restore = Tensorflow.Operations.gen_ops.restore_v2(tf.constant(path), 
                                         tf.constant(variable_names),
                                         tf.constant(Array.create variables.Length ""),
                                         dataTypes)
        let assignOps = [|for r,v in (restore,variables) ||> Array.zip -> tf.assign(v.output,r)|]
        tf.group(assignOps,name=name)

    // Not tested yet
    member tf.save(path: string, ?variableNames: string[], ?name: string) = 
        let name = defaultArg name "save"
        let graph = tf.get_default_graph()
        let variable_names = variableNames |> Option.defaultWith (fun _ -> tf.all_variable_names())
        let variables= variable_names |> Array.map (fun x -> graph.get_operation_by_name(x).output)
        Tensorflow.Operations.gen_ops.save_v2(tf.constant(path),
                        tf.constant(variable_names),
                        tf.constant(Array.create variables.Length ""),
                        variables, 
                        name = name)

module Array = 
    let shuffleInPlace (xs: 'T[]) = 
        let rand = System.Random()
        let swap (a: _[]) x y =
            let tmp = a.[x]
            a.[x] <- a.[y]
            a.[y] <- tmp
        Array.iteri (fun i _ -> swap xs i (rand.Next(i, Array.length xs))) xs

    let shuffle (xs: 'T[]) = 
        xs |> Array.map id |> fun ys -> ys |> shuffleInPlace; ys

    let subSample (count: int) (xs: 'T[]) =
        let rand = System.Random()
        Array.init count (fun _ -> xs.[rand.Next(0,Array.length xs)])


[<AutoOpen>]
module Auto = 
    //https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/python/ops/math_ops.py#L2565-L2754
    type Tensorflow.tensorflow with
        //<summary>Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
        //  The inputs must, following any transpositions, be tensors of rank >= 2
        //  where the inner 2 dimensions specify valid matrix multiplication arguments,
        //  and any further outer dimensions match.
        //  Both matrices must be of the same type. The supported types are:
        //  `float16`, `float32`, `float64`, `int32`, `complex64`, `complex128`.
        //  Either matrix can be transposed or adjointed (conjugated and transposed) on
        //  the fly by setting one of the corresponding flag to `True`. These are `False`
        //  by default.
        //  If one or both of the matrices contain a lot of zeros, a more efficient
        //  multiplication algorithm can be used by setting the corresponding
        //  `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.
        //  This optimization is only available for plain matrices (rank-2 tensors) with
        //  datatypes `bfloat16` or `float32`.
        //  For example:
        //  ```python
        //  # 2-D tensor `a`
        //  # [[1, 2, 3],
        //  #  [4, 5, 6]]
        //  a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
        //  # 2-D tensor `b`
        //  # [[ 7,  8],
        //  #  [ 9, 10],
        //  #  [11, 12]]
        //  b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
        //  # `a` * `b`
        //  # [[ 58,  64],
        //  #  [139, 154]]
        //  c = tf.matmul(a, b)
        //  # 3-D tensor `a`
        //  # [[[ 1,  2,  3],
        //  #   [ 4,  5,  6]],
        //  #  [[ 7,  8,  9],
        //  #   [10, 11, 12]]]
        //  a = tf.constant(np.arange(1, 13, dtype=np.int32),
        //                  shape=[2, 2, 3])
        //  # 3-D tensor `b`
        //  # [[[13, 14],
        //  #   [15, 16],
        //  #   [17, 18]],
        //  #  [[19, 20],
        //  #   [21, 22],
        //  #   [23, 24]]]
        //  b = tf.constant(np.arange(13, 25, dtype=np.int32),
        //                  shape=[2, 3, 2])
        //  # `a` * `b`
        //  # [[[ 94, 100],
        //  #   [229, 244]],
        //  #  [[508, 532],
        //  #   [697, 730]]]
        //  c = tf.matmul(a, b)
        //  # Since python >= 3.5 the @ operator is supported (see PEP 465).
        //  # In TensorFlow, it simply calls the `tf.matmul()` function, so the
        //  # following lines are equivalent:
        //  d = a @ b @ [[10.], [11.]]
        //  d = tf.matmul(tf.matmul(a, b), [[10.], [11.]])
        //  ```
        // </summary>
        // <param name="a"> `Tensor` of type `float16`, `float32`, `float64`, `int32`, `complex64`,
        // `complex128` and rank > 1. </param>
        // <param name="b"> `Tensor` with same type and rank as `a`.</param>
        // <param name="transpose_a"> If `True`, `a` is transposed before multiplication.</param>
        // <param name="transpose_b"> If `True`, `b` is transposed before multiplication.</param>
        // <param name="adjoint_a"> If `True`, `a` is conjugated and transposed before
        // multiplication.</param>
        // <param name="adjoint_b"> If `True`, `b` is conjugated and transposed before multiplication. </param>
        // <param name="a_is_sparse"> If `True`, `a` is treated as a sparse matrix.</param>
        // <param name="b_is_sparse"> If `True`, `b` is treated as a sparse matrix.</param>
        // <param name="name"> Name for the operation (optional).</param>
        // <returns>
        //    A `Tensor` of the same type as `a` and `b` where each inner-most matrix is
        //    the product of the corresponding matrices in `a` and `b`, e.g. if all
        //    transpose or adjoint attributes are `False`:
        //    `output`[..., i, j] = sum_k (`a`[..., i, k] * `b`[..., k, j]),
        //    for all indices i, j.
        //    Note: This is matrix product, not element-wise product.
        // </returns>
        // <exception cdef="Tensorflow.ValueError">
        // If transpose_a and adjoint_a, or transpose_b and adjoint_b
        // are both set to True.
        // </exception>
        member _.matmul2(a: Tensor, 
                b: Tensor, 
                ?transpose_a: bool,
                ?transpose_b: bool,
                ?adjoint_a: bool,
                ?adjoint_b: bool,
                ?a_is_sparse: bool,
                ?b_is_sparse: bool,
                ?name: string) = 
            let transpose_a = defaultArg transpose_a false
            let transpose_b = defaultArg transpose_b false
            let adjoint_a = defaultArg adjoint_a false
            let adjoint_b = defaultArg adjoint_b false
            let a_is_sparse = defaultArg a_is_sparse false
            let b_is_sparse = defaultArg b_is_sparse false
            let name = defaultArg name "MatMul"
            Tensorflow.Binding.tf_with(tf.name_scope(name,"MatMul",[|a;b|]), fun (name:ops.NameScope) -> 
                if transpose_a && adjoint_a then
                    raise (ValueError("Only one of transpose_a and adjoint_a can be True."))
                if transpose_b && adjoint_b then
                    raise (ValueError("Only one of transpose_b and adjoint_b can be True."))
                if a_is_sparse || b_is_sparse then failwith "todo"
                if adjoint_a || adjoint_b then failwith "todo"
    //            let output_may_have_non_empty_batch_shape, batch_mat_mul_fn =
    //                true, tf.matmul
      //          if false then
                let output_may_have_non_empty_batch_size = (a.shape.Length > 2) && (b.shape.Length > 2)
                if (not a_is_sparse) && (not b_is_sparse) && output_may_have_non_empty_batch_size then
                    // BatchMatmul does not support transpose, so we conjugate the matrix and
                    // use adjoint instead. Conj() is a noop for real matrices.
                    let conj = id
                    //https://github.com/tensorflow/tensorflow/blob/590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b/tensorflow/python/ops/math_ops.py#L3416
                    if a.dtype.is_complex() || b.dtype.is_complex() then failwith "todo"
                    let a, adjoint_a = if transpose_a then conj(a), true else a,adjoint_a
                    let b, adjoint_b = if transpose_b then conj(b), true else b,adjoint_b
                    Tensorflow.gen_math_ops.batch_mat_mul(a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name._name)
                else
                    if a.shape.Length = 2 && b.shape.Length = 2 && not(transpose_a) && not(transpose_b) then tf.matmul(a,b)
                    else failwith "todo"
                )

        //<summary>Computes the global norm of multiple tensors.
        //Given a tuple or list of tensors `t_list`, this operation returns the
        //global norm of the elements in all tensors in `t_list`. The global norm is
        //computed as:
        //`global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`
        //Any entries in `t_list` that are of type None are ignored.</summary>
        //<param name="t_lsit"> A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.</param>
        //<param name="name"> A name for the operation (optional).</param>
        //<returns>A 0-D (scalar) `Tensor` of type `float`.</returns>
        member _.global_norm(t_list: Tensor[], ?name: string) = 
            Tensorflow.Binding.tf_with(tf.name_scope(defaultArg name "global_norm", "global_norm", values = t_list ), fun _ -> 
    //            let values = [|
    //                ops.convert_to_tensor(
    //                    t.values if isinstance(t, ops.IndexedSlices) else t,
    //                    name="t_%d" % i)
    //                if t is not None else t
    //                for i, t in enumerate(t_list)|]
                let values = t_list
                let half_squared_norms = 
                    values 
                    //|> Array.choose id
                    |> Array.map (fun v -> 
                        // TODO! colocate_with is currently a no-op
                        //use _colo = ops.colocate_with(v)
                        gen_ops.l2loss(v))

                let half_squared_norm = math_ops.reduce_sum(array_ops.stack(half_squared_norms))
                let norm = math_ops.sqrt(half_squared_norm *
                                         constant_op.constant(2.0, dtype=half_squared_norm.dtype),
                                         name="global_norm")
                norm)

    //Tensorflow.Operations.gen_nn_ops.log_softmax

        // <summary> Clips values of multiple tensors by the ratio of the sum of their norms.
        //Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
        //this operation returns a list of clipped tensors `list_clipped`
        //and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
        //if you've already computed the global norm for `t_list`, you can specify
        //the global norm with `use_norm`.
        //To perform the clipping, the values `t_list[i]` are set to:
        //    t_list[i] * clip_norm / max(global_norm, clip_norm)
        //where:
        //    global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
        //If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
        //otherwise they're all shrunk by the global ratio.
        //If `global_norm == infinity` then the entries in `t_list` are all set to `NaN`
        //to signal that an error occurred.
        //Any of the entries of `t_list` that are of type `None` are ignored.
        //This is the correct way to perform gradient clipping (for example, see
        //[Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
        //([pdf](http://arxiv.org/pdf/1211.5063.pdf))).
        //However, it is slower than `clip_by_norm()` because all the parameters must be
        //ready before the clipping operation can be performed.</summary>
        // <param name="t_list">A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.</param>
        // <param name="clip_norm"> A 0-D (scalar) `Tensor` > 0. The clipping ratio.</param>
        // <param name="use_norm"> A 0-D (scalar) `Tensor` of type `float` (optional). The global
        //      norm to use. If not provided, `global_norm()` is used to compute the norm.</param>
        // <param name="name"> A name for the operation (optional).</param>
        // <returns> list_clipped: A list of `Tensors` of the same type as `list_t`.
        //    global_norm: A 0-D (scalar) `Tensor` representing the global norm.</returns>
        member _.clip_by_global_norm(t_list: Tensor[], clip_norm: Tensor, ?use_norm: Tensor, ?name: string) = 
            let use_norm = use_norm |> Option.defaultWith (fun () -> tf.global_norm(t_list)) 
            let name = defaultArg name "clip_by_global_norm"
            Tensorflow.Binding.tf_with(tf.name_scope(name, "clip_by_global_norm", values = [|yield! t_list; yield clip_norm |]), fun _ -> 
            // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
                let scale_for_finite = 
                    clip_norm * math_ops.minimum(1.0 / use_norm,
                                                 constant_op.constant(1.0, dtype=use_norm.dtype) / clip_norm)
                let scale = 
                    array_ops.where(tf.is_finite(use_norm),
                                    scale_for_finite,
                                    // Return NaN if use_norm is not finite.
                                    constant_op.constant(System.Single.NaN, dtype=use_norm.dtype))

    //            values = [
    //                ops.convert_to_tensor(
    //                    t.values if isinstance(t, ops.IndexedSlices) else t,
    //                    name="t_%d" % i)
    //                if t is not None else t
    //                for i, t in enumerate(t_list)]
                let values = t_list

                let values_clipped = 
                    values 
                    |> Array.mapi (fun i v -> tf.identity(v * scale, name=sprintf "%s_%i" name i))
    //                for i, v in enumerate(values):
    //                  if v is None:
    //                    values_clipped.append(None)
    //                  else:
    //                    with ops.colocate_with(v):
    //                      values_clipped.append(
    //                          array_ops.identity(v * scale, name="%s_%d" % (name, i)))

    //           list_clipped = [
    //                ops.IndexedSlices(c_v, t.indices, t.dense_shape)
    //                if isinstance(t, ops.IndexedSlices)
    //                else c_v
    //                for (c_v, t) in zip(values_clipped, t_list)]

                let list_clipped = values_clipped // TODO IndexedSlices

                list_clipped, use_norm)

        member _.get_or_create_global_step() = 
            let graph = tf.get_default_graph()
            match graph.get_collection(ops.GraphKeys.GLOBAL_STEP_) with
            | :? System.Collections.Generic.List<Tensorflow.VariableV1> as xs -> xs |> Seq.toArray
            | _ -> [|tf.train.create_global_step(graph)|]
            // NOTE: On the odd chance that this is not a RefVariable then find a fix
            |> Seq.head :?> RefVariable 

module utils = 
//https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/layers/python/layers/utils.py
//    def get_variable_collections(variables_collections, name):
//      if isinstance(variables_collections, dict):
//        variable_collections = variables_collections.get(name, None)
//      else:
//        variable_collections = variables_collections
//      return variable_collections
//    let get_variable_collections(variables_collections: string[], name: string): string[] = 
//        variables_collection
//
//    let get_variable_collections(variables_collections: Map<string,string[]>, name: string): string[] = 
//        variable_collections.[name]

      ///<summary>Append an alias to the list of aliases of the tensor.</summary>
      ///<param name=tensor>A `Tensor`</param>
      ///<param name=alias>String, to add to the list of aliases of the tensor.</param>
      ///<returns> The tensor with a new alias appended to its list of aliases.</returns>
    let append_tensor_alias(tensor: Tensor, alias: string) = 
        let dropSlash(x:string) = 
            if x.[x.Length-1] = '/' 
            then if x = "/" then "" else x.[.. x.Length - 2] 
            else x

//        let alias = dropSlash(alias)
////    TODO - tensors do not have alias yet. We are ignoring this for now
//          if hasattr(tensor, 'Tliases'):
//            tensor.aliases.append(alias)
//          else:
//            tensor.aliases = [alias]
//          return tensor
        tensor

//https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/layers/python/layers/utils.py
    let collect_named_outputs(collections: string[], alias: string, outputs: Tensor) = 
        if collections.Length > 0 then 
            ops.add_to_collections(System.Collections.Generic.List<string>(collections), outputs)
        outputs
        
// see https://stackoverflow.com/questions/47608357/difference-between-get-variable-and-model-variable-function
//https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/contrib/framework/python/ops/variables.py
type variables with
    static member model_variable(name: string, 
                                 shape: TensorShape, 
                                 dtype: TF_DataType, 
                                 initializer: IInitializer,
                                 collections: string[],
                                 trainable: bool) = 

        let collections = System.Collections.Generic.List<string>(collections)
        collections.Add(ops.GraphKeys.GLOBAL_VARIABLES_)
        collections.Add(ops.GraphKeys.MODEL_VARIABLES_)

        tf.get_variable(name,
            shape = shape,
            dtype = dtype,
            initializer = initializer,
            collections = collections,
            trainable = Nullable(trainable))


type Layers () = 
    static member dense(input: Tensor, 
                    units: int, 
                    ?activation: IActivation, 
                    ?use_bias: bool,
                    ?kernel_initializer: IInitializer,
                    ?bias_initializer: IInitializer,
                    ?trainable: Nullable<bool>,
                    ?reuse: Nullable<bool>,
                    ?name: string) =

        let dtype = input.dtype
        let name = defaultArg name String.Empty
        let reuse = defaultArg reuse (Nullable<bool>())
        let trainable = defaultArg trainable (Nullable<bool>())
        let use_bias = defaultArg use_bias true
        let bias_initializer = defaultArg bias_initializer tf.zeros_initializer
        Tensorflow.Binding.tf_with(tf.variable_scope(name,"dense",reuse=reuse), fun _vs ->
            match input.shape with
            | [|_;n|] when n > 0 ->
                let kernel = tf.get_variable("kernel",TensorShape(n,units),dtype=dtype,?initializer=kernel_initializer,trainable=trainable)
                let x = 
                    if use_bias 
                    then
                        let bias = tf.get_variable("bias",TensorShape(units),dtype=dtype,initializer=bias_initializer,trainable=trainable)
                        gen_ops.bias_add(tf.matmul2(input,kernel._AsTensor()),bias._AsTensor())
                    else
                        tf.matmul2(input,kernel._AsTensor())
                let x = match activation with None -> x | Some(f) -> f.Activate(x)
                //tf.identity(x,name=Tensorflow.ops.NameScope.op_Implicit(ns))
                x
            | _ ->
                raise (ValueError(sprintf "Input shape of %A is not suitable for a dense network " input.shape))
        )

    // https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/ops/nn_impl.py#L1382-L1442
    ///<summary>Batch normalization.
    ///Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
    ///`scale` \\(\gamma\\) to it, as well as an `offset` \\(\beta\\):
    ///\\(\frac{\gamma(x-\mu)}{\sigma}+\beta\\)
    ///`mean`, `variance`, `offset` and `scale` are all expected to be of one of two
    ///shapes:
    ///* In all generality, they can have the same number of dimensions as the
    ///  input `x`, with identical sizes as `x` for the dimensions that are not
    ///  normalized over (the 'depth' dimension(s)), and dimension 1 for the
    ///  others which are being normalized over.
    ///  `mean` and `variance` in this case would typically be the outputs of
    ///  `tf.nn.moments(..., keep_dims=True)` during training, or running averages
    ///  thereof during inference.
    ///* In the common case where the 'depth' dimension is the last dimension in
    ///  the input tensor `x`, they may be one dimensional tensors of the same
    ///  size as the 'depth' dimension.
    ///  This is the case for example for the common `[batch, depth]` layout of
    ///  fully-connected layers, and `[batch, height, width, depth]` for
    ///  convolutions.
    ///  `mean` and `variance` in this case would typically be the outputs of
    ///  `tf.nn.moments(..., keep_dims=False)` during training, or running averages
    ///  thereof during inference.
    ///See Source: [Batch Normalization: Accelerating Deep Network Training by
    ///Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy]
    ///(http://arxiv.org/abs/1502.03167).
    ///</summary>
    /// <param name="x">Input `Tensor` of arbitrary dimensionality.</param>
    /// <param name="mean"> A mean `Tensor`.</param>
    /// <param name="variance"> A variance `Tensor`.</param>
    /// <param name="offset"> An offset `Tensor`, often denoted \\(\beta\\) in equations, or
    ///  None. If present, will be added to the normalized tensor.
    ///     scale: A scale `Tensor`, often denoted \\(\gamma\\) in equations, or
    ///      `None`. If present, the scale is applied to the normalized tensor.</param>
    /// <param name="variance_epsilon"> A small float number to avoid dividing by 0.</param>
    /// <param name="name"> A name for this operation (optional).</param>
    /// <returns> the normalized, scaled, offset tensor.</returns>
    static member batch_normalization(x: Tensor,
                    mean: Tensor,
                    variance: Tensor,
                    ?offset: Tensor,
                    ?scale: Tensor,
                    ?epsilon: float,
                    ?name: string) =

        let epsilon = defaultArg epsilon 1.0e-6
        let inputs: Tensor[] = [| Some(x); Some(mean); Some(variance); scale; offset|] |> Array.choose id
        Tensorflow.Binding.tf_with(ops.name_scope("batchnorm","batchnorm",inputs), fun ns ->
            let inv = math_ops.rsqrt(variance + epsilon)
            let inv2 = match scale with | Some(scale) -> inv * scale | _ -> inv
            x * math_ops.cast(inv2,x.dtype) + 
                math_ops.cast((match offset with | Some(offset) ->(offset - mean * inv2) | None ->  -mean * inv2), x.dtype)
            )

    // https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/layers/python/layers/layers.py#L2204
    /// <summary>
    ///Adds a Layer Normalization layer.
    ///  Based on the paper:
    ///    "Layer Normalization"
    ///    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    ///    https://arxiv.org/abs/1607.06450.
    ///  Can be used as a normalizer function for conv2d and fully_connected.
    ///  Given a tensor `inputs` of rank `R`, moments are calculated and normalization
    ///  is performed over axes `begin_norm_axis ... R - 1`.  Scaling and centering,
    ///  if requested, is performed over axes `begin_params_axis .. R - 1`.
    ///  By default, `begin_norm_axis = 1` and `begin_params_axis = -1`,
    ///  meaning that normalization is performed over all but the first axis
    ///  (the `HWC` if `inputs` is `NHWC`), while the `beta` and `gamma` trainable
    ///  parameters are calculated for the rightmost axis (the `C` if `inputs` is
    ///  `NHWC`).  Scaling and recentering is performed via broadcast of the
    ///  `beta` and `gamma` parameters with the normalized tensor.
    ///  The shapes of `beta` and `gamma` are `inputs.shape[begin_params_axis:]`,
    ///  and this part of the inputs' shape must be fully defined.
    /// </summary>
    /// <param name="inputs"> A tensor having rank `R`. The normalization is performed over axes
    ///  `begin_norm_axis ... R - 1` and centering and scaling parameters are
    ///  calculated over `begin_params_axis ... R - 1`.</param>
    /// <param name="center"> If True, add offset of `beta` to normalized tensor. If False, `beta`
    ///  is ignored. </param>
    /// <param name="scale"> If True, multiply by `gamma`. If False, `gamma` is not used. When the
    ///  next layer is linear (also e.g. `nn.relu`), this can be disabled since the
    ///  scaling can be done by the next layer. </param>
    /// <param name="activation_fn"> Activation function, default set to None to skip it and
    ///  maintain a linear activation. </param>
    /// <param name="reuse"> Whether or not the layer and its variables should be reused. To be
    ///  able to reuse the layer scope must be given.</param>
    /// <param name="variables_collections"> Optional collections for the variables.</param>
    /// <param name="outputs_collections"> Collections to add the outputs.</param>
    /// <param name="trainable"> If `True` also add variables to the graph collection
    ///   `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).</param>
    /// <param name="begin_norm_axis"> The first normalization dimension: normalization will be
    ///   performed along dimensions `begin_norm_axis: rank(inputs)` </param>
    /// <param name="begin_params_axis"> The first parameter (beta, gamma) dimension: scale and
    ///  centering parameters will have dimensions 
    ///  `begin_params_axis: rank(inputs)` and will be broadcast with the
    ///    normalized inputs accordingly. </param>
    /// <param name="scope"> Optional scope for `variable_scope`.</param>
    /// <returns>A `Tensor` representing the output of the operation, having the same
    /// shape and dtype as `inputs`.  </returns>
    /// <exception cdef="Tensorflow.ValueError">
    /// If the rank of `inputs` is not known at graph build time,
    /// or if `inputs.shape[begin_params_axis:]` is not fully defined at
    /// graph build time.  </exception>
    static member layer_norm(inputs: Tensor,
                    ?center: bool,
                    ?scale: bool,
                    ?activation_fn: IActivation,
                    ?reuse: bool, 
                    ?variables_collections: Map<string,string[]>, 
                    ?output_collections: string[], 
                    ?trainable: bool,
                    ?begin_norm_axis: int,
                    ?begin_params_axis: int,
                    ?scope: string,
                    ?name: string) = 

        //let scope = defaultArg scope "LayerNorm"
        let name = defaultArg name String.Empty
        let center = defaultArg center true
        let scale = defaultArg scale true
        let trainable = defaultArg trainable true
        let begin_norm_axis = defaultArg begin_norm_axis 1
        let begin_params_axis = defaultArg begin_params_axis 1
        let variables_collections = defaultArg variables_collections (Map(["beta",[||];"gamma",[||]]))
        
        Tensorflow.Binding.tf_with(
            tf.variable_scope(name, 
                              "LayerNorm", 
                              values = [|inputs|],
                              reuse = (reuse |> Option.toNullable)), fun (vs:variable_scope) ->

            //let inputs = ops.convert_to_tensor(inputs)
            // NOTE: TensorShape GetSlice is not defined which is needed for slicing
            let inputs_shape = inputs.TensorShape.dims
            let inputs_rank = inputs_shape.Length 
            if inputs_rank = 0 then
                raise (ValueError(sprintf "Inputs %s has undefined rank." inputs.name))

            let dtype = inputs.dtype.as_base_dtype()
            let begin_norm_axis = 
                if begin_norm_axis < 0 then
                    inputs_rank + begin_norm_axis
                else 
                    begin_norm_axis

            if begin_params_axis >= inputs_rank || begin_norm_axis >= inputs_rank then
                raise (ValueError(sprintf "begin_params_axis (%d) and begin_norm_axis (%d) must be < rank(inputs) (%d)"
                                    begin_params_axis begin_norm_axis inputs_rank))

            //params_shape = inputs_shape[begin_params_axis:]
            let params_shape = 
                inputs_shape.[(if begin_params_axis < 0 then begin_params_axis + inputs_shape.Length else begin_params_axis) .. ] 
                |> TensorShape
                
            if not(params_shape.is_fully_defined()) then
                raise (ValueError(sprintf "Inputs %s: shape(inputs)[%i:] is not fully defined: %i"
                                    inputs.name begin_params_axis inputs_rank))

            // Allocate parameters for the beta and gamma of the normalization.
            let beta = 
                if center then
                    //let beta_collections = utils.get_variable_collections(variables_collections,"beta")
                    let beta_collections = variables_collections.["beta"]
                    variables.model_variable("beta",
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=tf.zeros_initializer,
                                             collections = beta_collections,
                                             trainable = trainable )._AsTensor() |> Some
                else None

            let gamma = 
                if scale then
                    //let gamma_collections = utils.get_variable_collections(variables_collections,"gamma")
                    let gamma_collections = variables_collections.["gamma"]
                    variables.model_variable("gamma",
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=tf.ones_initializer,
                                             collections = gamma_collections,
                                             trainable = trainable )._AsTensor() |> Some
                    
                else None

            // By default, compute the moments across all the dimensions except the one with index 0.
            let norm_axes = [|begin_norm_axis .. inputs_rank - 1|] // todo, perhaps fix this??
            let (mean, variance) = tf.nn.moments(inputs, norm_axes, keep_dims = true).ToTuple()

            // Compute layer normalization using the batch_normalization function.
            // Note that epsilon must be increased for float16 due to the limited
            // representable range.
            let variance_epsilon = if dtype = dtypes.float16 then 1e-3 else 1e-12
            let outputs = Layers.batch_normalization(inputs, 
                                                    mean, 
                                                    variance, 
                                                    ?offset=beta, 
                                                    ?scale=gamma, 
                                                    epsilon=variance_epsilon)

            outputs.set_shape(inputs_shape)
            let outputs = match activation_fn with | None -> outputs | Some(f) -> f.Activate(outputs)
            utils.collect_named_outputs(defaultArg output_collections Array.empty<string>, vs.name, outputs)
            )


type Utils() =
    /// <summary>Perform dropout.</summary>
    /// <param name="input_tensor">float Tensor.</param>
    /// <param name="dropout_prob">float. The probability of dropping out a value 
    /// (NOT of *keeping* a value as in `tf.nn.dropout`).</param>
    /// <returns>A version of `input_tensor` with dropout applied</returns>
    static member dropout(input_tensor, dropout_prob: float32) =
        if dropout_prob = 0.0f
        then input_tensor
        else tf.nn.dropout(input_tensor, tf.constant(1.0f - dropout_prob))

    /// Run layer normalization on the last dimension of the tensor."""
    static member layer_norm(input_tensor) =
      Layers.layer_norm(inputs=input_tensor, begin_norm_axis = -1, begin_params_axis = -1)

    /// Runs layer normalization followed by dropout.
    static member layer_norm_and_dropout(input_tensor: Tensor, dropout_prob: float32) =
        Utils.dropout(Utils.layer_norm(input_tensor),dropout_prob)

    /// Creates a `truncated_normal_initializer` with the given range.
    static member create_initializer(?initializer_range: float32) =
        tf.truncated_normal_initializer(stddev = defaultArg initializer_range 0.02f)
