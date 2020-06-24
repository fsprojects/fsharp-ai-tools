namespace ModellingTests

open System.Text.RegularExpressions
open Microsoft.VisualStudio.TestTools.UnitTesting
open Tensorflow

type BertModelTester(?batch_size : int,
         ?seq_length : int,
         ?is_training : bool,
         ?use_input_mask : bool,
         ?use_token_type_ids : bool,
         ?vocab_size : int,
         ?hidden_size : int,
         ?num_hidden_layers : int,
         ?num_attention_heads : int,
         ?intermediate_size : int,
         ?hidden_act : Operations.Activation.IActivation, 
         ?hidden_dropout_prob : float32,
         ?attention_probs_dropout_prob : float32,
         ?max_position_embeddings : int,
         ?type_vocab_size : int,
         ?initializer_range : float32,
         ?scope : string) = 

    let batch_size = defaultArg batch_size 13
    let seq_length = defaultArg seq_length 7
    let is_training = defaultArg is_training true
    let use_input_mask = defaultArg use_input_mask true
    let use_token_type_ids = defaultArg use_token_type_ids true
    let vocab_size = defaultArg vocab_size 99
    let hidden_size = defaultArg hidden_size 32
    let num_hidden_layers = defaultArg num_hidden_layers 5
    let num_attention_heads = defaultArg num_attention_heads 4
    let intermediate_size = defaultArg intermediate_size 37
    let hidden_act = defaultArg hidden_act Modeling.Activation.Gelu
    let hidden_dropout_prob = defaultArg hidden_dropout_prob 0.1f
    let attention_probs_dropout_prob = defaultArg attention_probs_dropout_prob 0.1f
    let max_position_embeddings = defaultArg max_position_embeddings 512
    let type_vocab_size = defaultArg type_vocab_size 16
    let initializer_range = defaultArg initializer_range 0.02f
    static let tf = Tensorflow.Binding.tf

    /// Creates a random int32 tensor of the shape within the vocab size.
    static member ids_tensor(shape : int[], vocab_size, ?rng : System.Random, ?name : string) = 
        let rng = defaultArg rng (System.Random())
        let total_dims = shape |> Array.fold (*) 1
        let values = Array.init total_dims (fun _ -> rng.Next(0,vocab_size-1))
        // TODO fix C#/F# optional param bug
        match name with
        | None -> tf.constant(values, dtype=tf.int32, shape=shape)
        | Some(name) -> tf.constant(values, dtype=tf.int32, shape=shape, name=name)

    member this.create_model() = 
        let input_ids = BertModelTester.ids_tensor([|batch_size; seq_length|], vocab_size)
        let input_mask = 
            if use_input_mask 
            then Some(BertModelTester.ids_tensor([|batch_size; seq_length|], 2))
            else None
        let token_type_ids = 
            if use_token_type_ids 
            then Some(BertModelTester.ids_tensor([|batch_size; seq_length|], type_vocab_size))
            else None
        let config = { vocab_size = Some(vocab_size)
                       hidden_size = hidden_size
                       num_hidden_layers = num_hidden_layers
                       num_attention_heads = num_attention_heads
                       intermediate_size = intermediate_size
                       hidden_act = hidden_act
                       hidden_dropout_prob = hidden_dropout_prob
                       attention_probs_dropout_prob = attention_probs_dropout_prob
                       max_position_embeddings = max_position_embeddings
                       type_vocab_size = type_vocab_size
                       initializer_range = initializer_range } : Modeling.BertConfig

        let model = Modeling.BertModel(config = config,
                                       is_training = is_training,
                                       input_ids = input_ids,
                                       ?input_mask = input_mask,
                                       ?token_type_ids = token_type_ids,
                                       ?scope = scope)
        [| 
            "embedding_output", [|model.EmbeddingOutput|]
            "sequence_output", [|model.SequenceOutput|]
            "pooled_output", [|model.PooledOutput|]
            "all_encoder_layers", model.AllEncoderLayers
        |] |> Map.ofArray

    member this.check_output(result : Map<string,Tensor[]>) =
        Assert.AreEqual(result.["embedding_output"].[0].shape,
                       [|batch_size; seq_length; hidden_size|])
        Assert.AreEqual(result.["sequence_output"].[0].shape,
                       [|batch_size; seq_length; hidden_size|])
        Assert.AreEqual(result.["pooled_output"].[0].shape,
                       [|batch_size; hidden_size|])

[<TestClass>]
type BertModelTest()  =
    static let tf = Tensorflow.Binding.tf

    /// Finds all the tensors in graph that are unreachable from outputs.
    let get_unreachable_ops(graph : Tensorflow.Graph, outputs : ITensorOrOperation[][]) = 
        let outputs = outputs |> Array.collect id
        let ops = graph.get_operations()
        let groupByFirst (xs : ('a*'b)[])  = 
            xs |> Array.groupBy fst |> Array.map (fun (k,vs) -> (k,vs |> Array.map snd)) |> Map.ofArray
        let output_to_op = 
            [| for op in ops do for y in op.op.outputs do yield (y.name, op.name) |] |> groupByFirst
        let op_to_input = 
            [| for op in ops do for x in op.op.inputs do yield (op.name, x.name) |] |> groupByFirst
        let assign_out_to_in = 
            [| 
                for op in ops do 
                    if op.op.``type`` = "Assign" 
                    then 
                        for y in op.op.outputs do
                            for x in op.op.inputs do
                                yield (y.name,x.name)
            |] |> groupByFirst

        let op_to_all = (Map.toArray output_to_op, Map.toArray op_to_input) ||> Array.append |> Map.ofArray
        let assign_groups = 
            [| 
                for KeyValue(out_name,name_group) in assign_out_to_in do
                    for n1 in name_group do
                        yield (n1, out_name)
                        for n2 in name_group do
                            if n1 <> n2 then
                                yield (n1,n2)
            |] |> groupByFirst

        let seen_tensors = System.Collections.Generic.HashSet<string>()
        let stack = System.Collections.Generic.Stack<string>()
        while stack.Count > 0 do
            let name = stack.Pop()
            if seen_tensors.Contains(name) 
            then ()
            else
                seen_tensors.Add(name) |> ignore
                for op_name in output_to_op.TryFind(name) |> Option.defaultValue [||] do
                    for input_name in op_to_all.TryFind(op_name) |> Option.defaultValue [||] do
                        if not(stack.Contains(input_name)) then
                            stack.Push(input_name)

                for assign_name in assign_groups.TryFind(name) |> Option.defaultValue [||] do
                    if not(stack.Contains(assign_name)) then
                        stack.Push(assign_name)

        let unreachable_ops = 
            [|
                for op in graph.get_operations() do
                    let all_names = 
                        [| for x in op.op.inputs do yield x.name; 
                           for x in op.outputs do yield x.name |] 
                    if all_names |> Array.exists (fun name -> seen_tensors.Contains(name) |> not) then
                        yield op
            |]

        unreachable_ops

    /// Checks that the tensors in the graph are reachable from outputs.
    let assert_all_tensors_reachable(sess : Session, outputs : ITensorOrOperation[][]) = 
        let graph = sess.graph
        let ignore_strings = 
            [|
                @"^.*/dilation_rate$"
                @"^.*/Tensordot/concat$"
                @"^.*/Tensordot/concat/axis$"
                @"^testing/.*$"
            |] |> Array.map (fun x -> Regex(x, RegexOptions.Compiled))

        // TODO double check this regex behavior
        let unreachable = 
            get_unreachable_ops(graph, outputs)
            |> Array.filter (fun x -> ignore_strings |> Array.exists (fun r -> r.IsMatch(x.name)) |> not)
        Assert.AreNotEqual(unreachable.Length,0)

    let run_tester(tester : BertModelTester) = 
        use sess = tf.Session()
        let ops = tester.create_model()
        // TODO get local variable initializer, this uses GraphKeys.LOCAL_VARIABLES
        let init_op = tf.group([|tf.global_variables_initializer(); (* tf.local_variables_initializer() *)|])
        sess.run(init_op)
        // TODO check that we have the concept of using dictionaries as parameters and getting them back out
        //let output_result = sess.run(ops |> Map.toArray |> Array.map snd |> Array.collect id)
        //tester.check_output(output_result)
        // Test via script predicting_movie_reviews_with_bert.fsx


    [<TestMethod>]
    member this.test_default() = 
        run_tester(BertModelTester())

