namespace Tensorflow

type F() = 
    let x = 10

// type F with
//     member this.x = 10
//     member this.y = this.x

type TF with
    /// <summary>
    /// Creates a constant operation from a TFTensor or constant
    /// </summary>
    /// <param name="value">Value.</param>
    /// <param name="name">Oper name.</param>
    /// <remarks>
    /// Since TFTensor have implicit conversion operators, you can call this method with
    /// a constant like this: graph.Const (23)
    /// </remarks>
    /// TODO: given that tensor type is optional is this really needed?
    static member Const (value : TFTensor, ?name : string) = TF.Const (value, value.TensorType, name)

    // Returns range(0, rank(x)) if reduction_indices is null
    static member ReduceDims (input : TFOutput, ?axis : TFOutput) =
        match axis with
        | Some(axis) -> axis
        | None ->
            // Fast path: avoid creating Rank and Range ops if ndims is known.
            let shape = TF.GetTensorShape (input)
            if shape.IsFullySpecified then
                // The python code distinguishes between tensor and sparsetensor
                // TODO; this will need to be extended to enable subtypes
                TF.Const ([|0..array.Length|], TFDataType.Int32)
            else
                // Otherwise, we rely on Range and Rank to do the right thing at run-time.
                TF.Range (TF.Const (0), TF.Rank (input), TF.Const (1))

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
    static member ReduceSum (input : TFOutput, ?axis : TFOutput, ?keep_dims : bool, ?name : string) =
        let keep_dims = defaultArg keep_dims false
        TF.Sum (input, TF.ReduceDims (input, axis), keep_dims, name)

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
    static member ReduceProd (input : TFOutput, ?axis : TFOutput, ?keep_dims : bool, name : string) =
        let keep_dims = defaultArg keep_dims false
        TF.Prod (input, TF.ReduceDims (input, axis), keep_dims, name);
        

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
    static member ReduceMean (input : TFOutput, ?axis : TFOutput, ?keep_dims : bool, ?name: string) =
        let keep_dims = defaultArg keep_dims false
        let boolToInt8Cast (x:TFOutput) = if input.OutputType = TFDataType.Bool then TF.Cast (x, TFDataType.Int8) else x
        TF.Mean (input, this.ReduceDims (boolToInt8Cast(input), axis), keep_dims, name);


    // Helper method to create a variable and track it.
    static member MakeVariable (initialValue : TFOutput, trainable : bool, string name) : Variable =
        let scopeName = TF.MakeName ("Variable", name)
        use newScope = TF.WithScope (scopeName)
        let Type = initialValue.OutputType
        let variableHandle = VarHandleOp (Type, new TFShape (GetShape (initialValue)))
        use aScope = TF.WithScope ("Assign")
        let assignOp = TF.AssignVariableOp (variableHandle, initialValue)
        use rScope = TF.WithScope ("Read")
        let readHandle = TF.ReadVariableOp (variableHandle, Type)
        let nv = new Variable (variableHandle, readHandle, assignOp)
        if trainable then TF.AddTrainableVariable (nv)
        TF.AddInitVariable (assignOp);
        nv;

    /// <summary>
    /// Variable node, with a starting initial value.
    /// </summary>
    /// <param name="initialValue">Initial value.</param>
    /// <param name="init">Returns the operation that initializes the value of the variable.</param>
    /// <param name="value">Returns the value of the variable.</param>
    /// <param name="trainable">If true, this add the variable to the graph's TrainableVariables, this collection is intended to be used by the Optimizer classes.</param>
    /// <param name="name">Operation name, optional.</param>
    /// <returns>The returning Variable contains the variable, with three nodes with the operations making up the variable assignment.</returns>
    /// <remarks>
    /// Variables need to be initialized before the main execution so you will typically want to
    /// run the session on the variable
    /// </remarks>
    /// TODO: Check up on how init and value are used here, it may be best to 
    static member Variable (initialValue : TFOutput, [<Out>] init : TFOperation, [<Out>] value : TFOutput , ?trainable : bool, ?name : string) =
        let trainable = defaultArg trainable true
        let nv = TF.MakeVariable (initialValue, trainable, name)
        init <- nv.Assign
        value <- nv.Read
        nv


    /// <summary>
    /// Registers a specified variable as an initialization variable.
    /// </summary>
    /// <param name="variable">Variable to register.</param>
    /// <remarks>
    /// <para>
    /// This is a convenience method to track the variables that need to be initialized in the graph,
    /// you can retrieve the list of all those variables by calling the <see cref="M:TensorFlow.TFGraph.GetGlobalVariablesInitializer"/>
    /// which will return this list and clear the state at that point.
    /// </para>
    /// <para>
    /// You typically use this method from helper methods to register all the variables that you want
    /// initialized, and a higher level method will retrieve all these variables and initialize them
    /// at their convenience.
    /// </para>
    /// </remarks>
    static member AddInitVariable (variable : TFOperation) =
        defaultGraph.PendingInitVariables.Add (variable)

    // TODO: finalize semantics, when should we clear these?
    static member AddTrainableVariable (variable : Variable) =
        defaultGraph.TrainableVariables.Add (variable)

    /// <summary>
    /// Gets the list of all registered global variables.
    /// </summary>
    /// <returns>The array of variables that should be initialized.</returns>
    /// <remarks>
    /// After this method is invoked the list of pending initialization variables
    /// is cleared.
    /// </remarks>
    static member GetGlobalVariablesInitializer () : TFOperation [] =
        let res = defaultGraph.PendingInitVariables.ToArray ()
        defaultGraph.PendingInitVariables.Clear () // NOTE: (matt) I'm not sure about this, I suppose it makes sense
        res

    /// <summary>
    /// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
    /// </summary>
    /// <param name="initialValue">Initial value.</param>
    /// <param name="value">Returns the value of the variable.</param>
    /// <param name="trainable">If true, this add the variable to the graph's TrainableVariables, this collection is intended to be used by the Optimizer classes.</param>
    /// <param name="name">Operation name, optional.</param>
    /// <returns>The returning Variable contains the variable, with three nodes with the operations making up the variable assignment.</returns>
    /// <remarks>
    /// Variables need to be initialized before the main execution so you will typically want to
    /// run the session on the variable.
    /// 
    /// The init sequence for the variable is stored in the graph, you must manually initialize 
    /// those by running the session on the global variables.
    /// </remarks>
    /// TODO (matt): Thiss shold probably be converted into a tuple return
    static member Variable (initialValue : TFOutput, [<Out>] value : TFOutput , ?trainable : bool, ?name : string) : Variable =
        let trainable = defaultArg trainable
        let nv = TF.MakeVariable (initialValue, trainable, name)
        value <- nv.Read
        nv

    /// <summary>
    /// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
    /// </summary>
    /// <param name="initialValue">Initial value.</param>
    /// <param name="trainable">If true, this add the variable to the graph's TrainableVariables, this collection is intended to be used by the Optimizer classes.</param>
    /// <param name="name">Operation name, optional.</param>
    /// <returns>The returning Variable contains the variable, with three nodes with the operations making up the variable assignment.</returns>
    /// <remarks>
    /// Variables need to be initialized before the main execution so you will typically want to
    /// run the session on the variable.
    /// 
    /// The init sequence for the variable is stored in the graph, you must manually initialize 
    /// those by running the session on the global variables.
    /// </remarks>
    static member this.Variable (initialValue : TFOutput, ?trainable : bool, ?name : string) =
        let trainable = defaultArg trainable true
        TF.MakeVariable (initialValue, trainable, name);

    //
    // Converts a shape to a tensor, to a TFOutput
    //
    static member ShapeTensorOutput (shape : TFShape) =
        if shape.IsLongArray then TF.Const (shape.ToArray (), TFDataType.Int64)
        else TF.Const (shape.ToIntArray (), TFDataType.Int32);

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
    static member Dropout (x : TFOutput, keep_prob : TFOutput, ?noise_shape : TFShape, ?seed : int, ?name : string) =
        use newScope = TF.WithScope (TF.MakeName ("dropout", name))
        let noiseShape = noise_shape |> Option.orDelay (fun _ -> new TFShape (TF.GetShape (x)))
        let shapeTensor = TF.ShapeTensorOutput (noise_shape)

        // uniform [keep_prob, 1.0 + keep_prob)
        let random_tensor = keep_prob
        let random_tensor = TF.Add (random_tensor, TF.RandomUniform (shapeTensor, ?seed = seed, dtype = x.OutputType))

        // 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        let binary_tensor = TF.Floor (random_tensor)
        let ret = TF.Mul (TF.Div (x, keep_prob), binary_tensor)
        TF.SetTensorShape (ret, GetShape (x))
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
    static member Dropout (x : TFOutput, keep_prob : double, ?noise_shape : TFShape, seed : int, ?name : string) =
        if (keep_prob < 0 || keep_prob >= 1) then raise(ArgumentOutOfRangeException ("keep_prob must be a scalar tensor or a float in the range (0, 1], got " + keep_prob))
        if keep_prob = 1.0 then x
        else
            use newScope = TF.WithScope (TF.MakeName ("dropout", name))
            let tkeep_prob = TF.Const (keep_prob)
            TF.Dropout (x, tkeep_prob, noise_shape, seed, name)

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
    /// <returns>A clipped <see cref="TFOutput">tensor</see>.</returns>
    static member ClipByValue2 (x : TFOutput, clip_value_min : TFOutput, clip_value_max : TFOutput, name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L33
        use newScope = TF.WithScope (TF.MakeName ("ClipByValue", name))
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
    /// <returns>A clipped <see cref="TFOutput">tensor</see>.</returns>
    static member ClipByNorm (x : TFOutput, clip_norm :TFOutput, ?axes : TFOutput, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L73
        let scopeName = TF.MakeName ("ClipByNorm", name)
        use newScope = TF.WithScope (scopeName)
        // Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
        let l2norm_inv = TF.Rsqrt (TF.ReduceSum (TF.Mul (x, x), axes, keep_dims: true))
        let intermediate = TF.Mul (x, clip_norm)
        let tclip = TF.Identity (TF.Mul (intermediate, TF.Minimum (l2norm_inv, TF.Div (TF.Const (new TFTensor (1.0)), clip_norm), name: name)))
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
    /// <returns>A clipped <see cref="TFOutput">tensor</see>.</returns>
    static member GlobalNorm (tensors : TFOutput [], ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L122
        use newScope = WithScope (TF.MakeName ("GlobalNorm", name))
        let half_squared_norms = Array.init half_squared_norms.Length (fun i -> TF.L2Loss (tensors.[i]))
        let half_squared_norm = TF.ReduceSum (TF.Stack (half_squared_norms))
        let norm = TF.Sqrt (TF.Mul (half_squared_norm, TF.Const (2.0)), ?name = "global_norm")
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
    static member ClipByAverageNorm (x : TFOutput, clip_norm :TFOutput , ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L251
        use newScope = WithScope (TF.MakeName ("ClipByAverageNorm", name))
        // Calculate L2-norm per element, clip elements by ratio of clip_norm to
        // L2-norm per element
        let n_element = TF.Cast (TF.Size (x), TFDataType.Float)
        let l2norm_inv = TF.Rsqrt (TF.ReduceSum (TF.Mul (x, x), TF.Range (TF.Rank (x))))
        let tclip = TF.Identity (TF.Mul (TF.Mul (x, clip_norm), TF.Minimum (TF.Mul (l2norm_inv, n_element), TF.Div (TF.Const (new TFTensor (1.0)), clip_norm)), ?name = name))
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
    static member SigmoidCrossEntropyWithLogits (labels : TFOuput, logits : TFOutput, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/nn_impl.py#L100
        use newScope = this.WithScope (TF.MakeName ("logistic_loss", name ))
        // Note: The following lines have not been ported from the original TF implementation since 
        // TensorFlowSharp API should guarantee that logits and labels are of type TFOutput by design:
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
        return TF.Add ( TF.Sub (relu_logits, TF.Mul (logits, labels)), TF.Log1p (TF.Exp (neg_abs_logits)), ?name = name)

    /// <summary>
    ///   Shuffle dimensions of x according to a permutation.
    /// </summary>
    /// <param name="x">
    /// </param>
    /// <param name="name">
    ///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Transpose'.
    /// </param>
    /// <returns>
    ///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
    /// </returns>
    /// <remarks>
    ///   The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    ///     `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
    /// </remarks>
    static member Transpose (x : TFOutput, ?name : string) =
        let rank = TF.Rank (x);
        let perm = TF.Sub (TF.Sub (TF.rank, TF.Const (1)), TF.Range (TF.Const (0), rank, TF.Const (1)));
        TF.Transpose (x = x, perm = perm, ?name = name)

    /// <summary>
    ///   Returns <paramref name="true_fn"/> if the predicate <paramref name="pred"/> is <c>true</c> else <paramref name="false_fn"/>.
    /// </summary>
    /// <param name="pred">A scalar determining whether to return the result of true_fn or false_fn.</param>
    /// <param name="true_fn">The callable to be performed if pred is true.</param>
    /// <param name="false_fn">The callable to be performed if pred is false.</param>
    /// <param name="name">Optional name prefix for the returned tensors.</param>
    /// <returns>TFOutput.</returns>
    static member Cond (pred : TFOutput, true_fn : Func<TFOutput>, false_fn : Func<TFOutput>, ?name : string) =
        use newScope = TF.WithScope (TF.MakeName ("cond", name))
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
        let merges, idnex = TF.Merge ([|res_t; res_f|])
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
    static member Where (condition : TFOutput , ?x : TFOutput, ?y : TFOutput, ?name : string) =
        // https://github.com/tensorflow/tensorflow/blob/d4ce3b4681b3a550c095b2cd18a79494d1cc4039/tensorflow/python/ops/array_ops.py#L2342
        match x,y with 
        | None,None -> TF.Where (input = condition, ?name = name)
        | Some(x),Some(y) -> TF.Select(condition = condition, t = x, e = y, ?name = name)
        | _ -> raise(ArgumentException ("x and y must both be non-None or both be None."))
    }

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
    static member Stack (values : TFOutput [], ?axis : int, ?name : string) =
        let axis = defaultArg axis 0
        let stack = defaultArg name "stack"
        // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_ops.py#L804
        let ndims = TF.GetTensorNumDims (values. [0])
        let expanded_num_dims = ndims + 1
        if axis < -expanded_num_dims || axis >= expanded_num_dims then
            raise (InvalidOperationException (spritnf "axis = %i not in [-%i, %i]" axis expanded_num_dims expanded_num_dims))
        else
            TF.Pack (values, axis = axis, ?name = name);

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
    static member Range (start : TFOutput, ?limit : TFOutput, ?delta : TFOutput, ?dataType : DataType, ?name : string) =
        let name = defaultArg range 
        // https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/math_ops.py#L1156
        let start,limit =
            match limit with 
            | None -> TF.Cast (TF.Const (new TFTensor (0.0)), start.OutputType), start // TODO: Maybe add dataType as convenience in Const?
            | Some(limit) -> start,limit

        let delta = delta |> Option.orDelay (fun _ -> TF.Cast ( TF.Const (new TFTensor (1.0)), start.OutputType))
        use newScope = TF.WithScope (TF.MakeName ("Range", name))) =
        // infer dtype if not explicitly provided
        let start, limit, delta =
            match dataType with 
            | Some(_) -> start, limit, delta
            | None -> 
                // TODO this type inference could be useful in other areas
                let dtype_hierarchy = [|TFDataType.Int32; TFDataType.Int64; TFDataType.Float; TFDataType.Double |]
                if (!dtype_hierarchy.Contains (start.OutputType) // NOTE; When this fails to type check we should extend Array2D<_> to include a contains method
                 || !dtype_hierarchy.Contains (limit.Value.OutputType)
                 || !dtype_hierarchy.Contains (delta.Value.OutputType))
                    raise( ArgumentException ("Unexpected type"))

                let dtypes = [|start.OutputType; limit.Value.OutputType; delta.Value.OutputType|]
                let imax = dtypes |> Array.maxBy (fun x -> dtype_hierarchy |> Array.findIndex ((=) x))
                let inferred_dtype = dtype_hierarchy.[imax]

                let start = TF.Cast (start, inferred_dtype)
                let limit = TF.Cast (limit, inferred_dtype)
                let delta = TF.Cast (delta, inferred_dtype)
                (start, limit, delta)

        (start, limit.Value, delta.Value, ?name = name)