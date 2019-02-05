[<AutoOpen>]
module TensorFlow.FSharp.OperationExtras

open System

type TFGraph with
    /// <summary>
    /// Creates a constant operation from a TFTensor or constant
    /// </summary>
    /// <param name="value">Value.</param>
    /// <param name="name">Oper name.</param>
    /// <remarks>
    /// Since TFTensor have implicit conversion operators, you can call this method with
    /// a constant like this: graph.Const (23)
    /// </remarks>
    member graph.Const (value : TFTensor, ?name : string) = graph.Const (value, value.TFDataType, ?name = name)

    // Returns range(0, rank(x)) if reduction_indices is null
    member graph.ReduceDims (input : TFOutput, ?axis : TFOutput) =
        match axis with
        | Some(axis) -> axis
        | None ->
            // Fast path: avoid creating Rank and Range ops if ndims is known.
            let shape = graph.GetTensorShape (input)
            if shape.IsFullyDefined then
                // NOTE: The python code distinguishes between tensor and sparsetensor
                graph.Const (new TFTensor([|0 .. shape.NumDimensions - 1|]), TFDataType.Int32)
            else
                // Otherwise, we rely on Range and Rank to do the right thing at run-time.
                graph.Range (graph.Const (new TFTensor(0)), graph.Rank (input), graph.Const (new TFTensor(1)))

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
    member graph.ReduceSum (input : TFOutput, ?axis : TFOutput, ?keep_dims : bool, ?name : string) =
        let keep_dims = defaultArg keep_dims false
        graph.Sum (input, graph.ReduceDims (input, ?axis=axis), keep_dims, ?name=name)

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
    member graph.ReduceProd (input : TFOutput, ?axis : TFOutput, ?keep_dims : bool, ?name : string) =
        let keep_dims = defaultArg keep_dims false
        graph.Prod (input, graph.ReduceDims (input, ?axis=axis), keep_dims, ?name=name);
        
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
    member graph.ReduceMean (input : TFOutput, ?axis : TFOutput, ?keep_dims : bool, ?name: string) =
        let keep_dims = defaultArg keep_dims false
        let boolToInt8Cast (x:TFOutput) = if input.TFDataType = TFDataType.Bool then graph.Cast (x, TFDataType.Int8) else x
        graph.Mean (input, graph.ReduceDims (boolToInt8Cast(input), ?axis=axis), keep_dims, ?name=name);


    // Helper method to create a variable and track it.
    member graph.MakeVariable (initialValue : TFOutput, trainable : bool, ?name : string) : TFVariable =
        use newScope = graph.NameScope(name, "Variable", initialValue)
        let dtype = initialValue.TFDataType
        let shape = graph.GetShape (initialValue)
        let variableHandle = graph.VarHandleOp (dtype, new TFShape (shape))
        use aScope = graph.NameScope("Assign")
        let assignOp = graph.AssignVariableOp (variableHandle, initialValue)
        use rScope = graph.NameScope("Read")
        let readHandle = graph.ReadVariableOp (variableHandle, dtype)
        let nv = new TFVariable (variableHandle.Struct, readHandle.Struct, assignOp.Handle)
        if trainable then graph.AddTrainableVariable (nv)
        graph.AddInitVariable (assignOp)
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
    member graph.Variable (initialValue : TFOutput, ?trainable : bool, ?name : string) =
        let trainable = defaultArg trainable true
        graph.MakeVariable (initialValue, trainable, ?name=name)

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
    member graph.AddInitVariable (variable : TFOperation) =
        graph.PendingInitVariables.Add (variable)

    member graph.AddTrainableVariable (variable : TFVariable) =
        graph.TrainingVariables.Add (variable)

    /// <summary>
    /// Gets the list of all registered global variables.
    /// </summary>
    /// <returns>The array of variables that should be initialized.</returns>
    /// <remarks>
    /// After this method is invoked the list of pending initialization variables
    /// is cleared.
    /// </remarks>
    member graph.GetGlobalVariablesInitializer () : TFOperation [] =
        let res = graph.PendingInitVariables.ToArray ()
        graph.PendingInitVariables.Clear () // NOTE: (matt) I'm not sure about this, I suppose it makes sense
        res

    //
    // Converts a shape to a tensor, to a Output
    //
    member graph.ShapeTensorOutput (shape : TFShape) =
        graph.Const (new TFTensor(shape.ToArray ()), TFDataType.Int64)
        //  if shape.IsLongArray then TFDataType.Int64 else TFDataType.Int32)

    /// <summary>
    /// Computes dropout. 
    /// </summary>
    /// <param name="x">A tensor.</param>
    /// <param name="keep_prob">A scalar TFTensor with the same type as x. The probability that each element is kept.</param>
    /// <param name="noise_shape">A 1-D TFTensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
    /// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
    /// <param name="name">Operation name, optional.</param>
    /// <remarks>
    /// With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
    /// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
    /// </remarks>
    member graph.Dropout (x : TFOutput, keep_prob : TFOutput, ?noise_shape : TFShape, ?seed : int, ?name : string) =
        use newScope = graph.NameScope(name, "Dropout",x)
        let noiseShape = noise_shape |> Option.defaultWith (fun () -> new TFShape (graph.GetShape (x)))
        let shapeTensor = graph.ShapeTensorOutput (noiseShape)
        // uniform [keep_prob, 1.0 + keep_prob)
        let random_tensor = keep_prob
        let random_tensor = graph.Add (random_tensor, graph.RandomUniform (shapeTensor, ?seed = (seed |> Option.map int64), dtype = x.TFDataType))
        // 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        let binary_tensor = graph.Floor (random_tensor)
        let ret = graph.Mul (graph.Div (x, keep_prob), binary_tensor)
        graph.SetTensorShape (ret, graph.GetShape (x))
        ret

    /// <summary>
    /// Computes dropout. 
    /// </summary>
    /// <param name="x">A tensor.</param>
    /// <param name="keep_prob">A scalar TFTensor with the same type as x. The probability that each element is kept.</param>
    /// <param name="noise_shape">A 1-D TFTensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
    /// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
    /// <param name="name">Operation name, optional.</param>
    /// <remarks>
    /// With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
    /// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
    /// </remarks>
    member graph.Dropout (x : TFOutput, keep_prob : double, ?noise_shape : TFShape, ?seed : int, ?name : string) =
        if (keep_prob < 0.0 || keep_prob >= 1.0) then raise(ArgumentOutOfRangeException (sprintf "keep_prob must be a scalar tensor or a float in the range (0, 1], got %f"  keep_prob))
        if keep_prob = 1.0 then x
        else
            use newScope = graph.NameScope("Dropout", x)
            let tkeep_prob = graph.Const (new TFTensor(keep_prob))
            graph.Dropout (x, tkeep_prob, ?noise_shape=noise_shape, ?seed=seed, ?name=name)

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
    member graph.ClipByValue2 (x : TFOutput, clip_value_min : TFOutput, clip_value_max : TFOutput, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L33
        use newScope = graph.NameScope (name, "ClipByValue", x, clip_value_min, clip_value_max)
        // Go through list of tensors, for each value in each tensor clip
        let t_min = graph.Minimum (x, clip_value_max)
        let t_max = graph.Maximum (t_min, clip_value_min, ?name = name)
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
    member graph.ClipByNorm (x : TFOutput, clip_norm :TFOutput, ?axes : TFOutput, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L73
        use nameScope = graph.NameScope(name,"ClipByNorm", [|yield! [x;clip_norm]; yield! axes |> Option.toArray|])
        // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        let l2norm_inv = graph.Rsqrt (graph.ReduceSum (graph.Mul (x, x), ?axis = axes, keep_dims = true))
        let intermediate = graph.Mul (x, clip_norm)
        let tclip = graph.Identity (graph.Mul (intermediate, graph.Minimum (l2norm_inv, graph.Div (graph.Const (new TFTensor (1.0)), clip_norm), name = string nameScope)))
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
    member graph.GlobalNorm (tensors : TFOutput [], ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L122
        use nameScope = graph.NameScope(name, "GlobalNorm", tensors)
        let half_squared_norms = Array.init tensors.Length (fun i -> graph.L2Loss (tensors.[i]))
        let half_squared_norm = graph.ReduceSum (graph.Stack (half_squared_norms))
        graph.Sqrt (graph.Mul (half_squared_norm, graph.Const (new TFTensor(2.0))), name = string nameScope)

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
    member graph.ClipByAverageNorm (x : TFOutput, clip_norm :TFOutput , ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L251
        use nameScope = graph.NameScope(name, "ClipByAverageNorm", x, clip_norm)
        // Calculate L2-norm per element, clip elements by ratio of clip_norm to
        // L2-norm per element
        let n_element = graph.Cast (graph.Size (x), TFDataType.Float32)
        let l2norm_inv = graph.Rsqrt (graph.ReduceSum (graph.Mul (x, x), graph.Range (graph.Rank (x))))
        graph.Mul (graph.Mul (x, clip_norm), graph.Minimum (graph.Mul (l2norm_inv, n_element), graph.Div (graph.Const (new TFTensor (1.0)), clip_norm)), name = string nameScope) 

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
    member graph.SigmoidCrossEntropyWithLogits (labels : TFOutput, logits : TFOutput, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/nn_impl.py#L100
        use nameScope = graph.NameScope(name, "LogisticLoss", labels, logits)
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
        let zeros = graph.ZerosLike (logits)
        let cond = graph.GreaterEqual (logits, zeros)
        let relu_logits = graph.Where (cond, logits, zeros)
        let neg_abs_logits = graph.Where (cond, graph.Neg (logits), logits)
        graph.Add ( graph.Sub (relu_logits, graph.Mul (logits, labels)), graph.Log1p (graph.Exp (neg_abs_logits)), name = string nameScope)

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
    member graph.Transpose (x : TFOutput, ?name : string) =
        let rank = graph.Rank (x);
        let perm = graph.Sub (graph.Sub (rank, graph.Const (new TFTensor(1))), graph.Range (graph.Const (new TFTensor(0)), rank, graph.Const (new TFTensor(1))));
        graph.Transpose (x = x, perm = perm, ?name = name)

    /// <summary>
    ///   Returns <paramref name="true_fn"/> if the predicate <paramref name="pred"/> is <c>true</c> else <paramref name="false_fn"/>.
    /// </summary>
    /// <param name="pred">A scalar determining whether to return the result of true_fn or false_fn.</param>
    /// <param name="true_fn">The callable to be performed if pred is true.</param>
    /// <param name="false_fn">The callable to be performed if pred is false.</param>
    /// <param name="name">Optional name prefix for the returned tensors.</param>
    /// <returns>Output.</returns>
    member graph.Cond (pred : TFOutput, true_fn : unit -> TFOutput, false_fn : unit -> TFOutput, ?name : string) =
        use newScope = graph.NameScope(name, "Cond")
        // Add the Switch to the graph.
        let (p_2, p_1) = graph.Switch (pred, pred);
        let pivot_t = graph.Identity (p_1, name = "switch_t");
        let pivot_f = graph.Identity (p_2, name = "switch_f");
        let pred = graph.Identity (pred, name= "pred_id");
        let res_t =
            // Build the graph for the true branch in a new context.
            use deps = graph.WithDependencies (pivot_t.Operation)
            true_fn ()
        let res_f = 
            // Build the graph for the false branch in a new context.
            use deps = graph.WithDependencies (pivot_f.Operation)
            false_fn ()

        // Add the final merge to the graph.
        let merges, idnex = graph.Merge ([|res_t; res_f|], name= string newScope)
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
    member graph.Where (condition : TFOutput , ?x : TFOutput, ?y : TFOutput, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/d4ce3b4681b3a550c095b2cd18a79494d1cc4039/tensorflow/python/ops/array_ops.py#L2342
        match x,y with 
        | None,None -> graph.Where (input = condition, ?name = name)
        | Some(x),Some(y) -> graph.Select(condition = condition, t = x, e = y, ?name = name)
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
    member graph.Stack (values : TFOutput [], ?axis : int, ?name : string) =
        use nameScope = graph.NameScope(name, "Stack", values)
        let axis = defaultArg axis 0
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_ops.py#L804
        let ndims = graph.GetTensorNumDims (values.[0])
        let expanded_num_dims = ndims + 1
        if axis < -expanded_num_dims || axis >= expanded_num_dims then
            raise (InvalidOperationException (sprintf "axis = %i not in [-%i, %i]" axis expanded_num_dims expanded_num_dims))
        else
            graph.Pack (values, axis = int64 axis, name=string nameScope);

    /// <summary>
    /// Creates a sequence of numbers.
    /// </summary>
    /// <remarks>
    /// Creates a sequence of numbers that begins at `start` and extends by increments of `delta` up to but not including 
    /// `limit`. The dtype of the resulting tensor is inferred from the inputs unless it is provided explicitly.
    /// </remarks>
    /// <param name="start">A 0 - D `TFTensor` (scalar).Acts as first entry in the range if `limit` is not None; otherwise, acts as range limit and first entry defaults to 0.</param>
    /// <param name="limit">A 0 - D `TFTensor` (scalar).Upper limit of sequence, exclusive. If None, defaults to the value of `start` while the first entry of the range defaults to 0.</param>
    /// <param name="delta">A 0 - D `TFTensor` (scalar).Number that increments `start`. Defaults to 1.</param>
    /// <param name="dataType">The type of the elements of the resulting tensor.</param>
    /// <param name="name">A name for the operation.Defaults to "range".</param>
    member graph.Range (start : TFOutput, ?limit : TFOutput, ?delta : TFOutput, ?dataType : TFDataType, ?name : string) =
        let name = defaultArg name "range"
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/math_ops.py#L1156
        use newScope = graph.NameScope(name, "Range", [|yield start; yield! limit |> Option.toArray; yield! delta |> Option.toArray|]) 
        let start,limit =
            match limit with 
            | None -> graph.Cast (graph.Const (new TFTensor (0.0)), start.TFDataType), start // TODO: Maybe add dataType as convenience in Const?
            | Some(limit) -> start,limit

        let delta = delta |> Option.defaultWith (fun () -> graph.Cast ( graph.Const (new TFTensor (1.0)), start.TFDataType))
        // infer dtype if not explicitly provided
        let start, limit, delta =
            match dataType with 
            | Some(_) -> start, limit, delta
            | None -> 
                // TODO this type inference could be useful in other areas
                let dtype_hierarchy = [|TFDataType.Int32; TFDataType.Int64; TFDataType.Float32; TFDataType.Float64|]
                // NOTE; When this fails to type check we should extend Array2D<_> to include a contains method
                if not (dtype_hierarchy.Contains (start.TFDataType))  ||
                   not(dtype_hierarchy.Contains (limit.TFDataType)) ||
                   not(dtype_hierarchy.Contains (delta.TFDataType)) then
                    raise( ArgumentException ("Unexpected type"))
                let dtypes = [|start.TFDataType; limit.TFDataType; delta.TFDataType|]
                let imax = dtypes |> Array.map (fun x -> dtype_hierarchy |> Array.findIndex ((=) x)) |> Array.max
                let inferred_dtype = dtype_hierarchy.[imax]
                let start = graph.Cast (start, inferred_dtype)
                let limit = graph.Cast (limit, inferred_dtype)
                let delta = graph.Cast (delta, inferred_dtype)
                (start, limit, delta)
        graph.Range(start, limit, delta, name)

    /// <summary>
    /// Outputs random values from a normal distribution
    /// </summary>
    /// <returns>A tensor of the specified shape filled with random normal values.</returns>
    /// <param name="shape">Shape of the output tensor.</param>
    /// <param name="mean">The mean of the standard distribution.</param>
    /// <param name="stddev">The standard deviation of the normal distribution.</param>
    /// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
    /// <param name="operName">Operation name, optional.</param>
    member graph.RandomNormal (shape : TFShape, ?mean : float32, ?stddev : float32, ?seed : int, ?name : string) =
        let mean = defaultArg mean 0.f
        let stddev = defaultArg stddev 1.f
        use ns = graph.NameScope(name, "RandomNormal")
        let shapeTensor = graph.ShapeTensorOutput (shape)
        let tmean = graph.Const(new TFTensor(mean), "mean")
        let tstddev = graph.Const(new TFTensor(stddev), "stddev")
        let graphSeed, localSeed = graph.GetRandomSeeds(?operationSeed=seed)
        let rnd = graph.RandomStandardNormal(shapeTensor, TFDataType.Float32, int64 graphSeed, int64 localSeed)
        let mul = graph.Mul(rnd, tstddev)
        graph.Add(mul, tmean,name= string ns)

    /// <summary>
    /// Randoms the uniform.
    /// </summary>
    /// <returns>The uniform.</returns>
    /// <param name="shape">Shape.</param>
    /// <param name="minval">Minval.</param>
    /// <param name="maxval">Maxval.</param>
    /// <param name="seed">Seed.</param>
    /// <param name="operName">Oper name.</param>
    member graph.RandomUniform (shape : TFShape, ?minval : float32, ?maxval : float32, ?seed : int, ?name : string) =
        let minval = defaultArg minval 0.f
        let maxval = defaultArg maxval 1.f
        use ns = graph.NameScope(name, "RandomUniform")
        let shapeTensor = graph.ShapeTensorOutput (shape)
        let minvalTensor = graph.Const (new TFTensor(minval), "minval")
        let maxvalTensor = graph.Const (new TFTensor(maxval), "maxval")
        let graphSeed, localSeed = graph.GetRandomSeeds(?operationSeed=seed)
        let rnd = graph.RandomUniform(shapeTensor, TFDataType.Float32, int64 graphSeed, int64 localSeed)
        let mul = graph.Mul(rnd, graph.Sub(maxvalTensor, minvalTensor))
        graph.Add(mul, minvalTensor, name = string ns)
