[<AutoOpen>]
module Tensorflow.OperationExtras

open System.Runtime.InteropServices
open System

type TF with
    /// <summary>
    /// Creates a constant operation from a Tensor or constant
    /// </summary>
    /// <param name="value">Value.</param>
    /// <param name="name">Oper name.</param>
    /// <remarks>
    /// Since Tensor have implicit conversion operators, you can call this method with
    /// a constant like this: graph.Const (23)
    /// </remarks>
    static member Const (value : Tensor, ?name : string) = TF.Const (value, value.DType, ?name = name)

    // Returns range(0, rank(x)) if reduction_indices is null
    static member ReduceDims (input : Output, ?axis : Output) =
        match axis with
        | Some(axis) -> axis
        | None ->
            // Fast path: avoid creating Rank and Range ops if ndims is known.
            let shape = TF.GetTensorShape (input)
            if shape.IsFullySpecified then
                // NOTE: The python code distinguishes between tensor and sparsetensor
                TF.Const (new Tensor([|0 .. shape.NumDimensions|]), DType.Int32)
            else
                // Otherwise, we rely on Range and Rank to do the right thing at run-time.
                TF.Range (TF.Const (new Tensor(0)), TF.Rank (input), TF.Const (new Tensor(1)))

    /// <summary>
    /// Computes the sum of elements across dimensions of a tensor.
    /// </summary>
    /// <returns>The reduced tensor.</returns>
    /// <param name="input">The tensor to reduce. Should have numeric type.</param>
    /// <param name="axis">The dimensions to reduce. If not se (the default), reduces all dimensions.</param>
    /// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
    /// <param name="name">A name for the operation, optional.</param>
    /// <remarks>
    ///   Reduces input_tensor along the dimensions given in axis.
    /// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
    /// entry in axis. If keep_dims is true, the reduced dimensions
    /// are retained with length 1.
    /// 
    /// If axis has no entries, all dimensions are reduced, and a
    /// tensor with a single element is returned.
    /// </remarks>
    static member ReduceSum (input : Output, ?axis : Output, ?keep_dims : bool, ?name : string) =
        let keep_dims = defaultArg keep_dims false
        TF.Sum (input, TF.ReduceDims (input, ?axis=axis), keep_dims, ?name=name)

    /// <summary>
    /// Computes the product of elements across dimensions of a tensor.
    /// </summary>
    /// <returns>The reduced tensor.</returns>
    /// <param name="input">The tensor to reduce. Should have numeric type.</param>
    /// <param name="axis">The dimensions to reduce. If not se (the default), reduces all dimensions.</param>
    /// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
    /// <param name="name">A name for the operation, optional.</param>
    /// <remarks>
    ///   Reduces input_tensor along the dimensions given in axis.
    /// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
    /// entry in axis. If keep_dims is true, the reduced dimensions
    /// are retained with length 1.
    /// 
    /// If axis has no entries, all dimensions are reduced, and a
    /// tensor with a single element is returned.
    /// </remarks>
    static member ReduceProd (input : Output, ?axis : Output, ?keep_dims : bool, ?name : string) =
        let keep_dims = defaultArg keep_dims false
        TF.Prod (input, TF.ReduceDims (input, ?axis=axis), keep_dims, ?name=name);
        
    /// <summary>
    /// Computes the mean of elements across dimensions of a tensor.
    /// </summary>
    /// <returns>The reduced tensor.</returns>
    /// <param name="input">The tensor to reduce. Should have numeric type.</param>
    /// <param name="axis">The dimensions to reduce. If not set (the default), reduces all dimensions.</param>
    /// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
    /// <param name="name">A name for the operation, optional.</param>
    /// <remarks>
    /// <para>
    ///   Reduces input_tensor along the dimensions given in axis.
    /// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
    /// entry in axis. If keep_dims is true, the reduced dimensions
    /// are retained with length 1.</para>
    /// 
    /// <para>
    /// If axis has no entries, all dimensions are reduced, and a
    /// tensor with a single element is returned.</para>
    /// </remarks>
    static member ReduceMean (input : Output, ?axis : Output, ?keep_dims : bool, ?name: string) =
        let keep_dims = defaultArg keep_dims false
        let boolToInt8Cast (x:Output) = if input.DType = DType.Bool then TF.Cast (x, DType.Int8) else x
        TF.Mean (input, TF.ReduceDims (boolToInt8Cast(input), ?axis=axis), keep_dims, ?name=name);


    // Helper method to create a variable and track it.
    static member MakeVariable (initialValue : Output, trainable : bool, ?name : string) : Variable =
        use newScope = TF.WithScope (TF.MakeName ("Variable", ?userName=name))
        let Type = initialValue.DType
        let variableHandle = TF.VarHandleOp (Type, new Shape (TF.GetShape (initialValue)))
        use aScope = TF.WithScope ("Assign")
        let assignOp = TF.AssignVariableOp (variableHandle, initialValue)
        use rScope = TF.WithScope ("Read")
        let readHandle = TF.ReadVariableOp (variableHandle, Type)
        let nv = new Variable (variableHandle.Struct, readHandle.Struct, assignOp.Handle)
        if trainable then TF.AddTrainableVariable (nv)
        TF.AddInitVariable (assignOp)
        nv

    /// <summary>
    /// Variable node, with a starting initial value.
    /// </summary>
    /// <param name="initialValue">Initial value.</param>
    /// <param name="trainable">If true, this add the variable to the graph's TrainableVariables, this collection is intended to be used by the Optimizer classes.</param>
    /// <param name="name">Operation name, optional.</param>
    /// <returns>The returning Variable contains the variable, with three nodes with the operations making up the variable assignment.</returns>
    /// <remarks>
    /// Variables need to be initialized before the main execution so you will typically want to
    /// run the session on the variable
    /// </remarks>
    static member Variable (initialValue : Output, ?trainable : bool, ?name : string) =
        let trainable = defaultArg trainable true
        TF.MakeVariable (initialValue, trainable, ?name=name)

    /// <summary>
    /// Registers a specified variable as an initialization variable.
    /// </summary>
    /// <param name="variable">Variable to register.</param>
    /// <remarks>
    /// <para>
    /// This is a convenience method to track the variables that need to be initialized in the graph,
    /// you can retrieve the list of all those variables by calling the <see cref="M:TensorFlow.Graph.GetGlobalVariablesInitializer"/>
    /// which will return this list and clear the state at that point.
    /// </para>
    /// <para>
    /// You typically use this method from helper methods to register all the variables that you want
    /// initialized, and a higher level method will retrieve all these variables and initialize them
    /// at their convenience.
    /// </para>
    /// </remarks>
    static member AddInitVariable (variable : Operation) =
        TF.DefaultGraph.PendingInitVariables.Add (variable)

    static member AddTrainableVariable (variable : Variable) =
        TF.DefaultGraph.TrainingVariables.Add (variable)

    /// <summary>
    /// Gets the list of all registered global variables.
    /// </summary>
    /// <returns>The array of variables that should be initialized.</returns>
    /// <remarks>
    /// After this method is invoked the list of pending initialization variables
    /// is cleared.
    /// </remarks>
    static member GetGlobalVariablesInitializer () : Operation [] =
        let res = TF.DefaultGraph.PendingInitVariables.ToArray ()
        TF.DefaultGraph.PendingInitVariables.Clear () // NOTE: (matt) I'm not sure about this, I suppose it makes sense
        res

    //
    // Converts a shape to a tensor, to a Output
    //
    static member ShapeTensorOutput (shape : Shape) =
        TF.Const (new Tensor(shape.ToArray ()), if shape.IsLongArray then DType.Int64 else DType.Int32)

    /// <summary>
    /// Computes dropout. 
    /// </summary>
    /// <param name="x">A tensor.</param>
    /// <param name="keep_prob">A scalar Tensor with the same type as x. The probability that each element is kept.</param>
    /// <param name="noise_shape">A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
    /// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
    /// <param name="name">Operation name, optional.</param>
    /// <remarks>
    /// With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
    /// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
    /// </remarks>
    static member Dropout (x : Output, keep_prob : Output, ?noise_shape : Shape, ?seed : int, ?name : string) =
        use newScope = TF.WithScope (TF.MakeName ("dropout", ?userName = name))
        let noiseShape = noise_shape |> Option.defaultWith (fun () -> new Shape (TF.GetShape (x)))
        let shapeTensor = TF.ShapeTensorOutput (noiseShape)
        // uniform [keep_prob, 1.0 + keep_prob)
        let random_tensor = keep_prob
        let random_tensor = TF.Add (random_tensor, TF.RandomUniform (shapeTensor, ?seed = (seed |> Option.map int64), dtype = x.DType))
        // 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        let binary_tensor = TF.Floor (random_tensor)
        let ret = TF.Mul (TF.Div (x, keep_prob), binary_tensor)
        TF.SetTensorShape (ret, TF.GetShape (x))
        ret

    /// <summary>
    /// Computes dropout. 
    /// </summary>
    /// <param name="x">A tensor.</param>
    /// <param name="keep_prob">A scalar Tensor with the same type as x. The probability that each element is kept.</param>
    /// <param name="noise_shape">A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
    /// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
    /// <param name="name">Operation name, optional.</param>
    /// <remarks>
    /// With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
    /// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
    /// </remarks>
    static member Dropout (x : Output, keep_prob : double, ?noise_shape : Shape, ?seed : int, ?name : string) =
        if (keep_prob < 0.0 || keep_prob >= 1.0) then raise(ArgumentOutOfRangeException (sprintf "keep_prob must be a scalar tensor or a float in the range (0, 1], got %f"  keep_prob))
        if keep_prob = 1.0 then x
        else
            use newScope = TF.WithScope (TF.MakeName ("dropout", ?userName = name))
            let tkeep_prob = TF.Const (new Tensor(keep_prob))
            TF.Dropout (x, tkeep_prob, ?noise_shape=noise_shape, ?seed=seed, ?name=name)

    /// <summary>
    /// Clips tensor values to a specified min and max.
    /// </summary>
    /// <remarks>
    /// Given a tensor <paramref name="x"/>, this operation returns a tensor of the same type and shape
    /// as <paramref name="x"/> with its values clipped to <paramref name="clip_value_min"/> and <paramref name="clip_value_max"/>.
    /// Any values less than <paramref name="clip_value_min"/> are set to <paramref name="clip_value_min"/>. Any values greater than 
    /// <paramref name="clip_value_max"/> are set to <paramref name="clip_value_max"/>.
    /// </remarks>
    /// <param name="x">The tensor.</param>
    /// <param name="clip_value_min">The minimum value to clip by. A 0 - D(scalar) tensor, or a tensor with the same shape as <paramref name="x"/>.</param>
    /// <param name="clip_value_max">The minimum value to clip by. A 0 - D(scalar) tensor, or a tensor with the same shape as <paramref name="x"/>.</param>
    /// <param name="name">Operation name, optional.</param>
    /// <returns>A clipped <see cref="Output">tensor</see>.</returns>
    static member ClipByValue2 (x : Output, clip_value_min : Output, clip_value_max : Output, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L33
        use newScope = TF.WithScope (TF.MakeName ("ClipByValue", ?userName=name))
        // Go through list of tensors, for each value in each tensor clip
        let t_min = TF.Minimum (x, clip_value_max)
        let t_max = TF.Maximum (t_min, clip_value_min, ?name = name)
        t_max

    /// <summary>
    /// Clips tensor values to a maximum L2-norm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Given a tensor <paramref name="x"/>, and a maximum clip value <paramref name="clip_norm"/>, this operation normalizes 
    /// <paramref name="x"/> so that its L2-norm is less than or equal to <paramref name="clip_norm"/>, along the dimensions 
    /// given in <paramref name="axes"/>. Specifically, in the default case where all dimensions are used for calculation, if
    /// the L2-norm of <paramref name="x"/> is already less than or equal to <paramref name="clip_norm"/>, then <paramref name="x"/>
    /// is not modified. If the L2-norm is greater than <paramref name="clip_norm"/>, then this operation returns a tensor of 
    /// the same type and shape as <paramref name="x"/> with its values set to: <c>t* clip_norm / l2norm(t)</c></para>
    /// </remarks>
    /// <param name="x">The tensor.</param>
    /// <param name="clip_norm">The minimum value to clip by. A 0 - D(scalar) tensor, or a tensor with the same shape as <paramref name="x"/>.</param>
    /// <param name="axes">The minimum value to clip by. A 0 - D(scalar) tensor, or a tensor with the same shape as <paramref name="x"/>.</param>
    /// <param name="name">Operation name, optional.</param>
    /// <returns>A clipped <see cref="Output">tensor</see>.</returns>
    static member ClipByNorm (x : Output, clip_norm :Output, ?axes : Output, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L73
        let scopeName = TF.MakeName ("ClipByNorm", ?userName = name)
        use newScope = TF.WithScope (scopeName)
        // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        let l2norm_inv = TF.Rsqrt (TF.ReduceSum (TF.Mul (x, x), ?axis = axes, keep_dims = true))
        let intermediate = TF.Mul (x, clip_norm)
        let tclip = TF.Identity (TF.Mul (intermediate, TF.Minimum (l2norm_inv, TF.Div (TF.Const (new Tensor (1.0)), clip_norm), ?name = name)))
        tclip

    /// <summary>
    /// Computes the global norm of multiple tensors.
    /// </summary>
    /// <remarks>
    /// <para>
    ///  Given a tuple or list of tensors <paramref name="tensors"/>, this operation returns the global norm of the elements in all tensors 
    ///  in <paramref name="tensors"/>. The global norm is computed as: <c>global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))</c>. Any 
    ///  entries in <paramref name="tensors"/> that are of type None are ignored.</para>
    /// </remarks>
    /// <param name="tensors">The input tensors.</param>
    /// <param name="name">Operation name, optional.</param>
    /// <returns>A clipped <see cref="Output">tensor</see>.</returns>
    static member GlobalNorm (tensors : Output [], ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L122
        use newScope = TF.WithScope (TF.MakeName ("GlobalNorm", ?userName = name))
        let half_squared_norms = Array.init tensors.Length (fun i -> TF.L2Loss (tensors.[i]))
        let half_squared_norm = TF.ReduceSum (TF.Stack (half_squared_norms))
        let norm = TF.Sqrt (TF.Mul (half_squared_norm, TF.Const (new Tensor(2.0))), name = "global_norm")
        norm

    /// <summary>
    /// Clips tensor values to a maximum average L2-norm.
    /// </summary>
    /// <remarks>
    /// Given a tensor <paramref name="x"/>, and a maximum clip value <paramref name="clip_norm"/>, this operation 
    /// normalizes <paramref name="x"/> so that its its average L2-norm is less than or equal to <paramref name="clip_norm"/>.
    /// Specifically, if the average L2-norm is already less than or equal to <paramref name="clip_norm"/>, then <paramref name="x"/>
    /// is not modified. If the average L2-norm is greater than <paramref name="clip_norm"/>, then this operation returns a tensor of the same
    /// type and shape as <paramref name="x"/> with its values set to: <c>t* clip_norm / l2norm_avg(t)</c>. In this case, 
    /// the average L2-norm of the output tensor is <paramref name="clip_norm"/>.
    /// </remarks>
    /// <param name="x">The input tensor.</param>
    /// <param name="clip_norm">A maximum clipping value.</param>
    /// <param name="name">Name of the oper.</param>
    static member ClipByAverageNorm (x : Output, clip_norm :Output , ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L251
        use newScope = TF.WithScope (TF.MakeName ("ClipByAverageNorm", ?userName = name))
        // Calculate L2-norm per element, clip elements by ratio of clip_norm to
        // L2-norm per element
        let n_element = TF.Cast (TF.Size (x), DType.Float32)
        let l2norm_inv = TF.Rsqrt (TF.ReduceSum (TF.Mul (x, x), TF.Range (TF.Rank (x))))
        let tclip = TF.Identity (TF.Mul (TF.Mul (x, clip_norm), TF.Minimum (TF.Mul (l2norm_inv, n_element), TF.Div (TF.Const (new Tensor (1.0)), clip_norm)), ?name = name))
        tclip

    /// <summary>
    ///   Computes sigmoid cross entropy given `logits`.
    /// </summary>
    /// 
    /// <remarks>
    ///    Measures the probability error in discrete classification tasks in which each
    ///    class is independent and not mutually exclusive.For instance, one could
    ///    perform multilabel classification where a picture can contain both an elephant
    ///    and a dog at the same time.
    /// </remarks>
    /// 
    static member SigmoidCrossEntropyWithLogits (labels : Output, logits : Output, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/nn_impl.py#L100
        use newScope = TF.WithScope (TF.MakeName ("logistic_loss", ?userName = name ))
        // Note: The following lines have not been ported from the original TF implementation since 
        // TensorFlowSharp API should guarantee that logits and labels are of type Output by design:
        //
        //   logits = ops.convert_to_tensor(logits, name: "logits");
        //   labels = ops.convert_to_tensor(labels, name: "labels");
        //   try
        //   {
        //       labels.get_shape().merge_with(logits.get_shape())
        //   }
        //   catch
        //   {
        //       throw new ArgumentException("logits and labels must have the same shape ({logits.get_shape()} vs {labels.get_shape()})");
        //   }

        // The logistic loss formula from above is
        // x - x * z + log(1 + exp(-x))
        // For x < 0, a more numerically stable formula is
        //   -x * z + log(1 + exp(x))
        // Note that these two expressions can be combined into the following:
        // max(x, 0) - x * z + log(1 + exp(-abs(x)))
        // To allow computing gradients at zero, we define custom versions of max and
        // abs functions.
        let zeros = TF.ZerosLike (logits)
        let cond = TF.GreaterEqual (logits, zeros)
        let relu_logits = TF.Where (cond, logits, zeros)
        let neg_abs_logits = TF.Where (cond, TF.Neg (logits), logits)
        TF.Add ( TF.Sub (relu_logits, TF.Mul (logits, labels)), TF.Log1p (TF.Exp (neg_abs_logits)), ?name = name)

    /// <summary>
    ///   Shuffle dimensions of x according to a permutation.
    /// </summary>
    /// <param name="x">
    /// </param>
    /// <param name="name">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Transpose'.
    /// </param>
    /// <returns>
    ///   The Operation can be fetched from the resulting Output, by fethching the Operation property from the result.
    /// </returns>
    /// <remarks>
    ///   The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    ///     `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
    /// </remarks>
    static member Transpose (x : Output, ?name : string) =
        let rank = TF.Rank (x);
        let perm = TF.Sub (TF.Sub (rank, TF.Const (new Tensor(1))), TF.Range (TF.Const (new Tensor(0)), rank, TF.Const (new Tensor(1))));
        TF.Transpose (x = x, perm = perm, ?name = name)

    /// <summary>
    ///   Returns <paramref name="true_fn"/> if the predicate <paramref name="pred"/> is <c>true</c> else <paramref name="false_fn"/>.
    /// </summary>
    /// <param name="pred">A scalar determining whether to return the result of true_fn or false_fn.</param>
    /// <param name="true_fn">The callable to be performed if pred is true.</param>
    /// <param name="false_fn">The callable to be performed if pred is false.</param>
    /// <param name="name">Optional name prefix for the returned tensors.</param>
    /// <returns>Output.</returns>
    static member Cond (pred : Output, true_fn : unit -> Output, false_fn : unit -> Output, ?name : string) =
        use newScope = TF.WithScope (TF.MakeName ("cond", ?userName = name))
        // Add the Switch to the graph.
        let (p_2, p_1) = TF.Switch (pred, pred);
        let pivot_t = TF.Identity (p_1, name = "switch_t");
        let pivot_f = TF.Identity (p_2, name = "switch_f");
        let pred = TF.Identity (pred, name= "pred_id");
        let res_t =
            // Build the graph for the true branch in a new context.
            use deps = TF.WithDependencies (pivot_t.Operation)
            true_fn ()
        let res_f = 
            // Build the graph for the false branch in a new context.
            use deps = TF.WithDependencies (pivot_f.Operation)
            false_fn ()

        // Add the final merge to the graph.
        let merges, idnex = TF.Merge ([|res_t; res_f|],2L)
        merges

    /// <summary>
    ///   Return elements from x or y depending on condition.
    /// </summary>
    /// 
    /// <param name="condition">LabeledTensor of type `bool`.</param>
    /// <param name="x">LabeledTensor for values where condition is true.</param>
    /// <param name="y">LabeledTensor for values where condition is false.</param>
    /// <param name="name">Optional op name.</param>
    /// 
    /// <returns>The labeled tensor with values according to condition.</returns>
    /// 
    static member Where (condition : Output , ?x : Output, ?y : Output, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/d4ce3b4681b3a550c095b2cd18a79494d1cc4039/tensorflow/python/ops/array_ops.py#L2342
        match x,y with 
        | None,None -> TF.Where (input = condition, ?name = name)
        | Some(x),Some(y) -> TF.Select(condition = condition, t = x, e = y, ?name = name)
        | _ -> raise(ArgumentException ("x and y must both be non-None or both be None."))

    /// <summary>
    /// Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
    /// </summary>
    /// <remarks>
    ///  Packs the list of tensors in <paramref name="values"/> into a tensor with rank one higher than
    ///  each tensor in <paramref name="values"/>, by packing them along the <paramref name="axis"/> dimension.
    ///  Given a list of length <c>N</c> of tensors of shape <c>(A, B, C)</c>: if <c>axis == 0</c> then the 
    ///  <c>output</c> tensor will have the shape <c>(N, A, B, C)</c>; if <c>axis == 1</c> then the <c>output</c>
    ///  tensor will have the shape <c>(A, N, B, C)</c>; etc.
    /// </remarks>
    /// 
    static member Stack (values : Output [], ?axis : int, ?name : string) =
        let axis = defaultArg axis 0
        let name = defaultArg name "stack"
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_ops.py#L804
        let ndims = TF.GetTensorNumDims (values.[0])
        let expanded_num_dims = ndims + 1
        if axis < -expanded_num_dims || axis >= expanded_num_dims then
            raise (InvalidOperationException (sprintf "axis = %i not in [-%i, %i]" axis expanded_num_dims expanded_num_dims))
        else
            failwith "todo figure out what the N parameter means and if we can make it optional"
            TF.Pack (values,-1L, int64 axis, name);

    /// <summary>
    /// Creates a sequence of numbers.
    /// </summary>
    /// <remarks>
    /// Creates a sequence of numbers that begins at `start` and extends by increments of `delta` up to but not including 
    /// `limit`. The dtype of the resulting tensor is inferred from the inputs unless it is provided explicitly.
    /// </remarks>
    /// <param name="start">A 0 - D `Tensor` (scalar).Acts as first entry in the range if `limit` is not None; otherwise, acts as range limit and first entry defaults to 0.</param>
    /// <param name="limit">A 0 - D `Tensor` (scalar).Upper limit of sequence, exclusive. If None, defaults to the value of `start` while the first entry of the range defaults to 0.</param>
    /// <param name="delta">A 0 - D `Tensor` (scalar).Number that increments `start`. Defaults to 1.</param>
    /// <param name="dataType">The type of the elements of the resulting tensor.</param>
    /// <param name="name">A name for the operation.Defaults to "range".</param>
    static member Range (start : Output, ?limit : Output, ?delta : Output, ?dataType : DType, ?name : string) =
        let name = defaultArg name "range"
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/math_ops.py#L1156
        let start,limit =
            match limit with 
            | None -> TF.Cast (TF.Const (new Tensor (0.0)), start.DType), start // TODO: Maybe add dataType as convenience in Const?
            | Some(limit) -> start,limit

        let delta = delta |> Option.defaultWith (fun () -> TF.Cast ( TF.Const (new Tensor (1.0)), start.DType))
        use newScope = TF.WithScope (TF.MakeName ("Range", name))
        // infer dtype if not explicitly provided
        let start, limit, delta =
            match dataType with 
            | Some(_) -> start, limit, delta
            | None -> 
                // TODO this type inference could be useful in other areas
                let dtype_hierarchy = [|DType.Int32; DType.Int64; DType.Float32; DType.Float64|]
                // NOTE; When this fails to type check we should extend Array2D<_> to include a contains method
                if not (dtype_hierarchy.Contains (start.DType))  ||
                   not(dtype_hierarchy.Contains (limit.DType)) ||
                   not(dtype_hierarchy.Contains (delta.DType)) then
                    raise( ArgumentException ("Unexpected type"))
                let dtypes = [|start.DType; limit.DType; delta.DType|]
                let imax = dtypes |> Array.map (fun x -> dtype_hierarchy |> Array.findIndex ((=) x)) |> Array.max
                let inferred_dtype = dtype_hierarchy.[imax]
                let start = TF.Cast (start, inferred_dtype)
                let limit = TF.Cast (limit, inferred_dtype)
                let delta = TF.Cast (delta, inferred_dtype)
                (start, limit, delta)
        TF.Range(start, limit, delta, name)

    /// <summary>
    ///    Concatenates tensors along one dimension.
    /// </summary>
    /// <param name="concat_dim">
    ///    0-D.  The dimension along which to concatenate.  Must be in the
    ///    range [0, rank(values)).
    /// </param>
    /// <param name="values">
    ///    The <c>N<c> Tensors to concatenate. Their ranks and types must match,
    ///    and their sizes must match in all dimensions except <c>concat_dim<c>.
    /// </param>
    /// <param name="name">
    /// If specified, the created operation in the graph will be this one, otherwise it will be named 'Concat'.
    /// <param>
    /// <param name="N">
    ///    Optional argument
    /// </param>
    /// <returns>
    ///    A <c>Tensor<c> with the concatenation of values stacked along the
    ///    <c>concat_dim<c> dimension.  This tensor's shape matches that of <c>values<c> except
    ///    in <c>concat_dim<c> where it has the sum of the sizes.
    ///    The Operation can be fetched from the resulting Output, by fetching the Operation property from the result.
    /// </returns>
    static member Concat (concat_dim : Output, values : Output[], ?N : int64,  ?name : string) : Output =
        let name = defaultArg name ""
        let desc = new OperationDesc (TF.DefaultGraph, "Concat", TF.DefaultGraph.MakeName ("Concat", name))
        desc.AddInput (concat_dim) |> ignore
        desc.AddInputs (values) |> ignore
        TF.DefaultGraph.CurrentDependencies |> Seq.iter (fun x -> desc.AddControlInput x |> ignore)
        N |> Option.iter (fun x -> desc.SetAttr ("N", x) |> ignore)
        let op = desc.FinishOperation ()
        let mutable _idx = 0
        let output = new Output (op, _idx)
        _idx <- _idx + 1
        output
