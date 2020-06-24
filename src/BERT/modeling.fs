// Apache License, Version 2.0
// Converted to F# from https://github.com/google-research/bert/blob/cc7051dc592802f501e8a6f71f8fb3cf9de95dc9/modeling.py
/// The main BERT model and related functions.
module Modeling

// TODO tensor alias
// TODO properly manage GraphKeys
// TODO Variance in tf.nn.moments is incorrectly capitalized
// TODO tf.nn.moments has extra Const ops

open NumSharp
open System
open System.Linq
open Tensorflow
open Tensorflow.Operations.Activation
open Newtonsoft.Json.Linq

/// <summary>Gaussian Error Linear Unit.
///  This is a smoother version of the RELU.
///  Original paper: https://arxiv.org/abs/1606.08415
/// </summary>
/// <param name="x">float Tensor to perform activation.</param>
/// <returns>  `x` with the GELU activation applied.</returns>
let gelu(x: Tensor) =
    let cdf = 0.5 * (1.0 + tf.tanh(0.7978845608 * (x + 0.044715 * tf.pow(x, 3))))
    x * cdf

module Activation = 
    let Gelu = {new Operations.Activation.IActivation with member _.Activate(x,name) = tf.identity(gelu(x),name)}
    let Relu  = Operations.Activation.relu()
    let Tanh  = Operations.Activation.tanh()
    let Linear = Operations.Activation.linear()

/// Configuration for `BertModel`.
type BertConfig = 
    { 
        /// Vocabulary size of `inputs_ids` in `BertModel`.
        vocab_size: int option
        /// Size of the encoder layers and the pooler layer.
        hidden_size: int
        /// Number of hidden layers in the Transformer encoder.
        num_hidden_layers: int
        /// Number of attention heads for each attention layer in
        /// the Transformer encoder.
        num_attention_heads: int
        /// The size of the "intermediate" (i.e., feed-forward)
        /// layer in the Transformer encoder.
        intermediate_size: int
        /// The non-linear activation function (function or string) in the
        /// encoder and pooler.
        hidden_act: IActivation
        /// The dropout probability for all fully connected
        /// layers in the embeddings, encoder, and pooler.
        hidden_dropout_prob: float32
        ///  The dropout ratio for the attention
        ///  probabilities.
        attention_probs_dropout_prob: float32
        ///  The maximum sequence length that this model might
        ///    ever be used with. Typically set this to something large just in case
        ///    (e.g., 512 or 1024 or 2048).
        max_position_embeddings: int
        /// The vocabulary size of the `token_type_ids` passed into
        /// `BertModel`.
        type_vocab_size: int
        /// The stdev of the truncated_normal_initializer for
        initializer_range: float32
    }

    static member Default = 
        {
            vocab_size = None
            hidden_size = 768
            num_hidden_layers = 12
            num_attention_heads = 12
            intermediate_size = 3072
            hidden_act = Activation.Gelu
            hidden_dropout_prob = 0.1f
            attention_probs_dropout_prob = 0.1f
            max_position_embeddings = 512
            type_vocab_size = 16
            initializer_range = 0.02f
         }

    /// Constructs a `BertConfig` from a Python dictionary of parameters.
    static member from_json_string(json_object: string) = 
        let bertObj = JObject.Parse(json_object)
        let getInt32 name = bertObj.[name].ToString() |> Int32.Parse
        let getFloat32 name = bertObj.[name].ToString() |> Single.Parse
        let getActivation(name: string) = 
            match bertObj.[name].ToString().ToLower() with
            | "gelu" -> Activation.Gelu
            | "relu" -> Activation.Relu :> IActivation
            | "tanh" -> Activation.Tanh :> IActivation
            | "linear" -> Activation.Linear :> IActivation
            | _ -> failwithf "Activation %s is not supported" name

        {
            vocab_size = getInt32 "vocab_size" |> Some
            hidden_size = getInt32 "hidden_size"
            num_hidden_layers = getInt32 "num_hidden_layers"
            num_attention_heads = getInt32 "num_attention_heads"
            intermediate_size = getInt32 "intermediate_size"
            hidden_act= getActivation "hidden_act"
            hidden_dropout_prob = getFloat32 "hidden_dropout_prob"
            attention_probs_dropout_prob = getFloat32 "attention_probs_dropout_prob"
            max_position_embeddings = getInt32 "max_position_embeddings"
            type_vocab_size = getInt32 "type_vocab_size"
            initializer_range = getFloat32 "initializer_range"
        }

    /// Serializes this instance to a JSON string.
    member this.to_json_string(): string = 
        let hidden_act = 
            if this.hidden_act = Activation.Gelu then "gelu"
            elif this.hidden_act = upcast Activation.Relu then "relu"
            elif this.hidden_act = upcast Activation.Tanh then "tanh"
            elif this.hidden_act = upcast Activation.Linear then "linear"
            else failwithf "activation is unrecognized"

        [|
            JProperty("vocab_size",this.vocab_size)
            JProperty("hidden_size",this.hidden_size)
            JProperty("num_hidden_layers", this.num_hidden_layers)
            JProperty("num_attention_heads", this.num_attention_heads)
            JProperty("intermediate_size", this.intermediate_size)
            JProperty("hidden_act", hidden_act)
            JProperty("hidden_dropout_prob", this.hidden_dropout_prob)
            JProperty("attention_probs_dropout_prob", this.attention_probs_dropout_prob)
            JProperty("max_position_embeddings",this.max_position_embeddings)
            JProperty("type_vocab_size", this.type_vocab_size)
            JProperty("initializer_range", this.initializer_range)
        |] 
        |> JObject |> string

/// <summary>
/// BERT model ("Bidirectional Encoder Representations from Transformers").
///  Example usage:
///  ```python
///  # Already been converted into WordPiece token ids
///  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
///  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
///  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
///  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
///    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)
///  model = modeling.BertModel(config=config, is_training=True,
///    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
///  label_embeddings = tf.get_variable(...)
///  pooled_output = model.get_pooled_output()
///  logits = tf.matmul(pooled_output, label_embeddings)
///  ...
/// </summary>
/// <param name="config"> `BertConfig` instance.</param>
/// <param name="is_training"> bool. true for training model, false for eval model. Controls 
/// whether dropout will be applied.</param>
/// <param name="input_ids">int32 Tensor of shape [batch_size, seq_length].</param>
/// <param name="input_mask"> (optional) int32 Tensor of shape [batch_size, seq_length].</param>
/// <param name="token_type_ids"> (optional) int32 Tensor of shape [batch_size, seq_length].</param>
/// <param name="use_one_hot_embeddings"> (optional) bool. Whether to use one-hot word
///        embeddings or tf.embedding_lookup() for the word embeddings.</param>
/// <param name="scope"> (optional) variable scope. Defaults to "bert".</param>
/// <exception cref="Tensorflow.ValueError"> The config is invalid or one of the input tensor shapes
/// is invalid </exception>
type BertModel(config: BertConfig, 
        is_training: bool, 
        input_ids: Tensor,
        ?input_mask: Tensor,
        ?token_type_ids: Tensor,
        ?use_one_hot_embeddings: bool,
        ?scope: string) = 

    let scope = defaultArg scope "bert"
    let use_one_hot_embeddings = defaultArg use_one_hot_embeddings false
    let config = 
        if not is_training 
        then {config with hidden_dropout_prob = 0.0f; attention_probs_dropout_prob = 0.0f}
        else config

    let input_shape: int[] = BertModel.get_shape_list(input_ids, expected_rank=2)
    let batch_size = input_shape.[0]
    let seq_length = input_shape.[1]

    let input_mask = defaultArg input_mask (tf.ones(TensorShape(batch_size, seq_length),dtype=tf.int32))
    let token_type_ids = defaultArg token_type_ids (tf.zeros(TensorShape(batch_size, seq_length),dtype=tf.int32))
    let vocab_size = defaultArg config.vocab_size -1 // TODO figure out if this should be an error

    let (pooled_output, sequence_output, all_encoder_layers, embedding_table, embedding_output) =
        use _bert = vs.variable_scope(scope)
        let (embedding_output, embedding_table) = 
            use _embeddings = vs.variable_scope("embeddings")
            let (embedding_output, embedding_table) =
                BertModel.embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

            let embedding_output = 
                BertModel.embedding_postprocessor(
                    input_tensor=embedding_output,
                    use_token_type=true,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=true,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)

            (embedding_output, embedding_table)

        //This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        // mask of shape [batch_size, seq_length, seq_length] which is used
        // for the attention scores.
        let all_encoder_layers =
            use _encoder = vs.variable_scope("encoder")
            let attention_mask = BertModel.create_attention_mask_from_input_mask(input_ids, input_mask)
            // Run the stacked transformer.
            // `sequence_output` shape = [batch_size, seq_length, hidden_size].
            BertModel.transformer_model(input_tensor=embedding_output,
                attention_mask=attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=config.hidden_act,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=true)

        let sequence_output = all_encoder_layers |> Seq.last
        // The "pooler" converts the encoded sequence tensor of shape
        // [batch_size, seq_length, hidden_size] to a tensor of shape
        // [batch_size, hidden_size]. This is necessary for segment-level
        // (or segment-pair-level) classification tasks where we need a fixed
        // dimensional representation of the segment.
        // We "pool" the model by simply taking the hidden state corresponding
        // to the first token. We assume that this has been pre-trained
        let pooled_output = 
            use _pooler = vs.variable_scope("pooler")
            let first_token_tensor = tf.squeeze(sequence_output.[Slice.All, Slice(Nullable(0),Nullable(1)), Slice.All], axis=[|1|])
            Layers.dense(first_token_tensor,
                config.hidden_size,
                activation=tanh(),
                kernel_initializer=Utils.create_initializer(config.initializer_range))
        (pooled_output, sequence_output, all_encoder_layers, embedding_table, embedding_output)

    member _.PooledOutput = pooled_output

    /// <summary>Gets final hidden layer of encoder</summary>
    /// <returns>float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    /// to the final hidden of the transformer encoder.</returns>
    member _.SequenceOutput = sequence_output

    member _.AllEncoderLayers = all_encoder_layers

    /// <summary>Gets output of the embedding lookup (i.e., input to the transformer).</summary>
    /// <returns> float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    /// to the output of the embedding layer, after summing the word
    /// embeddings with the positional embeddings and the token type embeddings,
    /// then performing layer normalization. This is the input to the transformer.  </returns>
    member _.EmbeddingOutput = embedding_output

    member _.EmbeddingTable = embedding_table

    /// Compute the union of the current variables and checkpoint variables.
    static member get_assignment_map_from_checkpoint(tvars:RefVariable[], init_checkpoint) =

        let re = System.Text.RegularExpressions.Regex("^(.*):\\d+$")
            
        let name_to_variable = 
            [|
                for var in tvars do
                    let name = var.name
                    let m = re.Match(name)
                    if m.Groups.Count > 1 then 
                        yield (m.Groups.[1].Value,var)
            |] 
            //|> Map.ofArray // NOTE the Map to RefVariable is not used     
            |> Array.map fst |> Set


        //let init_vars = tf.train.list_variables(init_checkpoint)
        //let xs = init_vars |> Array.map fst |> Array.filter name_to_variable.ContainsKey

        // NOTE: Expects full path to *.ckpt.meta file, this may not be correct
        let list_variables(path: string) =
            use f = System.IO.File.OpenRead(path)
            let metaGraph = Tensorflow.MetaGraphDef.Parser.ParseFrom(f)
            metaGraph.GraphDef.Node 
            |> Seq.choose (fun x -> if x.Op = "VariableV2" then Some(x.Name) else None)
            |> Seq.toArray

        let variable_names: string[] = list_variables(init_checkpoint)
        let xs = variable_names |> Array.filter name_to_variable.Contains
        let assignment_map = Map([| for x in xs -> (x,x)|])
        let initialized_variable_names = Map([| for x in xs do yield (x,1); yield (x + ":0",1)|])
            
        (assignment_map, initialized_variable_names)

    /// <summary>Create 3D attention mask from a 2D tensor mask.</summary>
    /// <param name="from_tensor"> 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].</param>
    /// <param name="to_mask"> int32 Tensor of shape [batch_size, to_seq_length].</param>
    /// <returns>float Tensor of shape [batch_size, from_seq_length, to_seq_length].</returns>
    static member create_attention_mask_from_input_mask(from_tensor, to_mask) =
        let from_shape = BertModel.get_shape_list(from_tensor, expected_rank=[|2; 3|])
        let batch_size = from_shape.[0]
        let from_seq_length = from_shape.[1]
            
        let to_shape = BertModel.get_shape_list(to_mask, expected_rank=2)
        let to_seq_length = to_shape.[1]
            
        let to_mask = 
            tf.cast(tf.reshape(to_mask, [|batch_size; 1; to_seq_length|]), tf.float32)
            
        //  We don't assume that `from_tensor` is a mask (although it could be). We
        //  don't actually care if we attend *from* padding tokens (only *to* padding)
        //  tokens so we create a tensor of all ones.
        // 
        // `broadcast_ones` = [batch_size, from_seq_length, 1]
        let broadcast_ones = tf.ones(shape=TensorShape(batch_size, from_seq_length, 1), dtype=tf.float32)
        //
        //   Here we broadcast along two dimensions to create the mask.
        let mask = broadcast_ones * to_mask
        mask

    /// <summary>Multi-headed, multi-layer Transformer from "Attention is All You Need".
    ///  This is almost an exact implementation of the original Transformer encoder.
    ///  See the original paper:
    ///  https://arxiv.org/abs/1706.03762
    ///  Also see:
    ///  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    /// </summary>
    /// <param name="input_tensor"> float Tensor of shape [batch_size, seq_length, hidden_size].</param>
    /// <param name="attention_mask"> (optional) int32 Tensor of shape [batch_size, seq_length,
    ///      seq_length], with 1 for positions that can be attended to and 0 in
    ///      positions that should not be.</param>
    /// <param name="hidden_size"> int. Hidden size of the Transformer.</param>
    /// <param name="num_hidden_layers"> int. Number of layers (blocks) in the Transformer.</param>
    /// <param name="num_attention_heads"> int. Number of attention heads in the Transformer.</param>
    /// <param name="intermediate_size"> int. The size of the "intermediate" (a.k.a., feed
    ///      forward) layer.</param>
    /// <param name="intermediate_act_fn"> function. The non-linear activation function to apply
    ///      to the output of the intermediate/feed-forward layer.</param>
    /// <param name="hidden_dropout_prob"> float. Dropout probability for the hidden layers.
    ///    attention_probs_dropout_prob: float. Dropout probability of the attention
    ///      probabilities.</param>
    /// <param name="initializer_range"> float. Range of the initializer (stddev of truncated
    ///      normal).</param>
    /// <param name="do_return_all_layers"> Whether to also return all layers or just the final
    ///      layer.</param>
    ///  <returns>float Tensor of shape [batch_size, seq_length, hidden_size], the final
    ///    hidden layer of the Transformer.</returns>
    /// <exception cdef="Tensorflow.ValueError">A Tensor shape or paramter is invalid</exception>
    static member transformer_model(input_tensor: Tensor,
            ?attention_mask,
            ?hidden_size: int,
            ?num_hidden_layers: int,
            ?num_attention_heads: int,
            ?intermediate_size: int,
            ?intermediate_act_fn: Operations.Activation.IActivation,
            ?hidden_dropout_prob: float32,
            ?attention_probs_dropout_prob: float32,
            ?initializer_range: float32,
            ?do_return_all_layers: bool): Tensor[] = 

        let hidden_size = defaultArg hidden_size 768
        let num_hidden_layers = defaultArg num_hidden_layers 12
        let num_attention_heads = defaultArg num_attention_heads 12
        let intermediate_size = defaultArg intermediate_size 3072
        let intermediate_act_fn = defaultArg intermediate_act_fn (Activation.Gelu)
        let hidden_dropout_prob = defaultArg hidden_dropout_prob 0.1f
        let attention_probs_dropout_prob = defaultArg attention_probs_dropout_prob 0.1f
        let initializer_range = defaultArg initializer_range 0.02f
        let do_return_all_layers = defaultArg do_return_all_layers false
        if hidden_size % num_attention_heads <> 0 then
            raise (ValueError( sprintf "The hidden size (%d) is not a multiple of the number of attention heads (%d)" hidden_size num_attention_heads )) 

        let attention_head_size = hidden_size / num_attention_heads

        let input_shape = BertModel.get_shape_list(input_tensor, expected_rank=3)
        let batch_size = input_shape.[0]
        let seq_length = input_shape.[1]
        let input_width = input_shape.[2]

        /// The Transformer performs sum residuals on all layers so the input needs
        ///  to be the same as the hidden size.
        if input_width <> hidden_size then
            raise (ValueError(sprintf "The width of the input tensor (%d) != hidden size (%d)" input_width hidden_size))

        // We keep the representation as a 2D tensor to avoid re-shaping it back and
        // forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        // the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        // help the optimizer.
        let mutable prev_output = BertModel.reshape_to_matrix(input_tensor)

        let makeLayer(layer_idx: int) = 
            let name = sprintf "layer_%d" layer_idx
            use _layer = vs.variable_scope(name)
            let layer_input = prev_output
            let attention_output = 
                use _attention = vs.variable_scope("attention")
                // TODO/WARN The python code does not seem to be able to return
                // more than one attention head here so either I'm wrong about that
                // or a fair amount of the original code here is moot
                let attention_output = 
                    use _self = vs.variable_scope("self")
                    BertModel.attention_layer(from_tensor=layer_input,
                        to_tensor=layer_input,
                        ?attention_mask=attention_mask,
                        num_attention_heads=num_attention_heads,
                        size_per_head=attention_head_size,
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=true,
                        batch_size=batch_size,
                        from_seq_length=seq_length,
                        to_seq_length=seq_length)
                    // Run a linear projection of `hidden_size` then add a residual
                // with `layer_input`. 
                let attention_output = 
                    use _output = vs.variable_scope("output")
                    let attention_output = 
                        Layers.dense(attention_output, 
                            hidden_size,
                            kernel_initializer=Utils.create_initializer(initializer_range))
                    let attention_output = Utils.dropout(attention_output, hidden_dropout_prob)
                    let attention_output = Utils.layer_norm(attention_output + layer_input)
                    attention_output
                attention_output

            let intermediate_output = 
                use _intermediate = vs.variable_scope("intermediate")
                Layers.dense(attention_output, 
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=Utils.create_initializer(initializer_range))

            let layer_output = 
                // Down-project back to `hidden_size` then add the residual.
                use output = vs.variable_scope("output")
                let layer_output = 
                    Layers.dense(intermediate_output, 
                        hidden_size,
                        kernel_initializer=Utils.create_initializer(initializer_range))
                let layer_output = Utils.dropout(layer_output, hidden_dropout_prob)
                let layer_output = Utils.layer_norm(layer_output + attention_output)
                layer_output
            prev_output <- layer_output
            layer_output

        let all_layer_outputs = [| for layer_idx in 0..num_hidden_layers - 1 -> makeLayer(layer_idx) |]

        if do_return_all_layers then
            [| for x in all_layer_outputs -> BertModel.reshape_from_matrix(x, input_shape)|]
        else
            [|BertModel.reshape_from_matrix(prev_output, input_shape)|]

    /// Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix).
    static member reshape_to_matrix(input_tensor) = 
        let ndims = input_tensor.TensorShape.ndim
        if ndims < 2 then 
            raise (ValueError(sprintf "Input tensor must have a least rank 2. Shape = %A" input_tensor.shape))
        elif ndims = 2 then input_tensor
        else 
            let width = input_tensor.shape |> Seq.last
            let output_tensor = tf.reshape(input_tensor, [|-1;width|])
            output_tensor

    /// Reshapes a rank 2 tensor back to its original rank >= 2 tensor.
    static member reshape_from_matrix(output_tensor, orig_shape_list) = 
        if orig_shape_list.Length = 2 then 
            output_tensor
        else
            let output_shape = BertModel.get_shape_list(output_tensor)
            let orig_dims = orig_shape_list.[0..orig_shape_list.Length - 2]
            let width = output_shape |> Seq.last
            tf.reshape(output_tensor, Array.append orig_dims [|width|])

    ///<summary>Looks up word embeddings for id tensor.</summary>
    ///<param name="input_ids">int32 Tensor of shape [batch_size, seq_length] containing word ids.</param>
    ///<param name="vocab_size">int. Size of the embedding vocabulary.</param>
    ///<param name="embedding_size">int. Width of the word embeddings.</param>
    ///<param name="initializer_range">float. Embedding initialization range.</param>
    ///<param name="word_embedding_name">string. Name of the embedding table.</param>
    ///<param name="use_one_hot_embeddings">bool. If True, use one-hot method for word 
    /// embeddings. If False, use `tf.gather()`.</param>
    /// <returns> float Tensor of shape [batch_size, seq_length, embedding_size]. </returns>
    static member embedding_lookup(input_ids: Tensor, 
            vocab_size: int,
            ?embedding_size: int,
            ?initializer_range: float32,
            ?word_embedding_name: string,
            ?use_one_hot_embeddings: bool) = 

        let embedding_size = defaultArg embedding_size 128
        let initializer_range = defaultArg initializer_range 0.02f
        let word_embedding_name = defaultArg word_embedding_name "word_embeddings"
        let use_one_hot_embeddings = defaultArg use_one_hot_embeddings false
        // This function assumes that the input is of shape [batch_size, seq_length,
        // num_inputs].
        //
        // If the input is a 2D tensor of shape [batch_size, seq_length], we
        // reshape to [batch_size, seq_length, 1]. 
        let input_ids = 
            if input_ids.TensorShape.ndim = 2 then 
                tf.expand_dims(input_ids, axis = -1) 
            else
                input_ids

        let embedding_table = 
            tf.get_variable(name=word_embedding_name, 
                shape=TensorShape(vocab_size, embedding_size),
                initializer=Utils.create_initializer(initializer_range))._AsTensor()

        let flat_input_ids = tf.reshape(input_ids, [|-1|])

        let output = 
            if use_one_hot_embeddings then
                tf.matmul2(tf.one_hot(flat_input_ids, depth=vocab_size), embedding_table)
            else
                gen_ops.gather_v2(embedding_table, flat_input_ids,tf.constant(0))

        let input_shape = BertModel.get_shape_list(input_ids)

        let output = tf.reshape(output, [|yield! input_shape.[0..input_shape.Length-2]; yield input_shape.[input_shape.Length-1] * embedding_size|])

        (output, embedding_table)


    /// <summary>Performs various post-processing on a word embedding tensor.</summary>
    /// <param name="input_tensor">float Tensor of shape [batch_size, seq_length, 
    /// embedding_size].</param>
    /// <param name="use_token_type"> bool. Whether to add embeddings for 
    /// `token_type_ids`.</param>
    /// <param name="token_type_ids">(optional) int32 Tensor of shape [batch_size, seq_length].
    /// Must be specified if `use_token_type` is True. </param>
    /// <param name="token_type_vocab_size"> int. The vocabulary size of `token_type_ids`.</param>
    /// <param name="token_type_embedding_name"> string. The name of the embedding table variable
    /// for token type ids.</param>
    /// <param name="use_position_embeddings"> bool. Whether to add position embeddings for the
    /// position of each token in the sequence.</param>
    /// <param name="position_embedding_name"> string. The name of the embedding table variable
    /// for positional embeddings.</param>
    /// <param name="initializer_range"> float. Range of the weight initialization.</param>
    /// <param name="max_position_embeddings"> int. Maximum sequence length that might ever be
    /// used with this model. This can be longer than the sequence length of
    /// input_tensor, but cannot be shorter.</param>
    /// <param name="dropout_prob"> float. Dropout probability applied to the final output tensor.</param>
    /// <returns>float tensor with same shape as `input_tensor`</returns>
    /// <exception cdef="Tensorflow.ValueError">One of the tensor shapes or input values is invalid.</exception>
    static member embedding_postprocessor(input_tensor: Tensor,
            ?use_token_type: bool,
            ?token_type_ids: Tensor,
            ?token_type_vocab_size: int,
            ?token_type_embedding_name: string,
            ?use_position_embeddings: bool,
            ?position_embedding_name: string,
            ?initializer_range: float32,
            ?max_position_embeddings: int,
            ?dropout_prob: float32) = 
                
        let use_token_type = defaultArg use_token_type false
        let token_type_vocab_size = defaultArg token_type_vocab_size 16
        let token_type_embedding_name = defaultArg token_type_embedding_name "token_type_embeddings"
        let use_position_embeddings = defaultArg use_position_embeddings  true
        let position_embedding_name = defaultArg position_embedding_name  "position_embeddings"
        let initializer_range = defaultArg initializer_range 0.02f
        let max_position_embeddings = defaultArg max_position_embeddings 512
        let dropout_prob = defaultArg dropout_prob 0.1f
        let input_shape = BertModel.get_shape_list(input_tensor, expected_rank=3)
        let batch_size = input_shape.[0] 
        let seq_length = input_shape.[1] 
        let width = input_shape.[2] 

        let output = input_tensor

        let output = 
            if use_token_type then
                match token_type_ids with
                | None -> raise (ValueError("`token_type_ids` must be specified if `use_token_type` is true."))
                | Some(token_type_ids) ->
                    let token_type_table = 
                        tf.get_variable(token_type_embedding_name,
                            dtype = tf.float32,
                            shape = TensorShape(token_type_vocab_size, width),
                            initializer = Utils.create_initializer(initializer_range))._AsTensor()
 
                    // This vocab will be small so we always do one-hot here, since it is always
                    // faster for a small vocabulary.
                    let flat_token_type_ids = tf.reshape(token_type_ids, [|-1|])
                    let one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
                    let token_type_embeddings = tf.matmul2(one_hot_ids, token_type_table)
                    let token_type_embeddings = 
                        tf.reshape(token_type_embeddings, [|batch_size; seq_length; width|])
                    output + token_type_embeddings
            else output

        let output = 
            if use_position_embeddings then
                // TODO add an assert_less_equal
                // TODO ControlFlowDependencies not implemented
                //let assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
                //let assert_op = tf.assert_equal(seq_length, max_position_embeddings)
                //use _cd = tf.control_dependencies([|assert_op|])
                let full_position_embeddings = 
                    tf.get_variable(name = position_embedding_name,
                        shape = TensorShape(max_position_embeddings, width),
                        initializer = Utils.create_initializer(initializer_range)
                        )._AsTensor()
                // Since the position embedding table is a learned variable, we create it
                // using a (long) sequence length `max_position_embeddings`. The actual
                // sequence length might be shorter than this, for faster training of
                // tasks that do not have long sequences.
                //
                // So `full_position_embeddings` is effectively an embedding table
                // for position [0, 1, 2, ..., max_position_embeddings-1], and the current
                // sequence has positions [0, 1, 2, ... seq_length-1], so we can just
                // perform a slice.
                let position_embeddings = tf.slice(full_position_embeddings, [|0;0|], [|seq_length;-1|])

                let num_dims = output.TensorShape.as_list().Length

                // Only the last two dimensions are relevant (`seq_length` and `width`), so
                // we broadcast among the first dimensions, which is typically just
                // the batch size.
                let position_broadcast_shape = 
                    [| for _i in 0..num_dims - 3 do yield 1; yield seq_length; yield width |]
                let position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
                output + position_embeddings
            else output
        Utils.layer_norm_and_dropout(output, dropout_prob)

    /// <summary>Performs multi-headed attention from `from_tensor` to `to_tensor`.
    ///  This is an implementation of multi-headed attention based on "Attention
    ///  is all you Need". If `from_tensor` and `to_tensor` are the same, then
    ///  this is self-attention. Each timestep in `from_tensor` attends to the
    ///  corresponding sequence in `to_tensor`, and returns a fixed-with vector.
    ///  This function first projects `from_tensor` into a "query" tensor and
    ///  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    ///  of tensors of length `num_attention_heads`, where each tensor is of shape
    ///  [batch_size, seq_length, size_per_head].
    ///  Then, the query and key tensors are dot-producted and scaled. These are
    ///  softmaxed to obtain attention probabilities. The value tensors are then
    ///  interpolated by these probabilities, then concatenated back to a single
    ///  tensor and returned.
    ///  In practice, the multi-headed attention are done with transposes and
    ///  reshapes rather than actual separate tensors.</summary>
    ///  <param name="from_tensor"> float Tensor of shape [batch_size, from_seq_length,
    ///      from_width].</param>
    ///  <param name="to_tensor"> float Tensor of shape [batch_size, to_seq_length, to_width].</param>
    ///  <param name="attention_mask"> (optional) int32 Tensor of shape [batch_size,
    ///      from_seq_length, to_seq_length]. The values should be 1 or 0. The
    ///      attention scores will effectively be set to -infinity for any positions in
    ///      the mask that are 0, and will be unchanged for positions that are 1.</param>
    ///  <param name="num_attention_heads"> int. Number of attention heads.</param>
    ///  <param name="size_per_head"> int. Size of each attention head.</param>
    ///  <param name="query_act"> (optional) Activation function for the query transform.</param>
    ///  <param name="key_act"> (optional) Activation function for the key transform.</param>
    ///  <param name="value_act"> (optional) Activation function for the value transform.</param>
    ///  <param name="attention_probs_dropout_prob"> (optional) float. Dropout probability of the
    ///      attention probabilities. </param>
    ///  <param name="initializer_range"> float. Range of the weight initializer.</param>
    ///  <param name="do_return_2d_tensor"> bool. If True, the output will be of shape [batch_size
    ///      * from_seq_length, num_attention_heads * size_per_head]. If False, the
    ///      output will be of shape [batch_size, from_seq_length, num_attention_heads
    ///      * size_per_head].
    ///    batch_size: (Optional) int. If the input is 2D, this might be the batch size
    ///      of the 3D version of the `from_tensor` and `to_tensor`.
    ///    from_seq_length: (Optional) If the input is 2D, this might be the seq length
    ///      of the 3D version of the `from_tensor`.
    ///    to_seq_length: (Optional) If the input is 2D, this might be the seq length
    ///      of the 3D version of the `to_tensor`. </param>
    ///  <returns>
    ///    float Tensor of shape [batch_size, from_seq_length,
    ///      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
    ///      true, this will be of shape [batch_size * from_seq_length,
    ///      num_attention_heads * size_per_head]). </returns>
    /// <exception cdef="Tensorflow.ValueError">Any of the arguments or tensor shapes are invalid.</exception>
    static member attention_layer(from_tensor: Tensor, 
            to_tensor: Tensor, 
            ?attention_mask: Tensor,
            ?num_attention_heads: int,
            ?size_per_head: int,
            ?query_act: Operations.Activation.IActivation,
            ?key_act: Operations.Activation.IActivation,
            ?value_act: Operations.Activation.IActivation,
            ?attention_probs_dropout_prob: float32,
            ?initializer_range: float32,
            ?do_return_2d_tensor: bool,
            ?batch_size: int,
            ?from_seq_length: int,
            ?to_seq_length: int) = 
            
        let num_attention_heads = defaultArg num_attention_heads 1
        let size_per_head = defaultArg size_per_head 512
        let attention_probs_dropout_prob = defaultArg attention_probs_dropout_prob 0.0f
        let initializer_range = defaultArg initializer_range 0.02f
        let do_return_2d_tensor = defaultArg do_return_2d_tensor false
        let transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width) =
            let output_tensor = tf.reshape(input_tensor, [|batch_size; seq_length; num_attention_heads; width|])
            let output_tensor = tf.transpose(output_tensor, [|0; 2; 1; 3|])
            output_tensor

        let from_shape = BertModel.get_shape_list(from_tensor, expected_rank=[|2;3|])
        let to_shape = BertModel.get_shape_list(to_tensor, expected_rank=[|2;3|])
        if from_shape.Length <> to_shape.Length then
            raise (ValueError("The rank of `from_tensor` must match the rank of `to_tensor`."))

        let (batch_size, from_seq_length, to_seq_length) =
            match from_shape.Length with
            | 3 -> from_shape.[0], from_shape.[1], to_shape.[1]
            | 2 -> 
                match batch_size, from_seq_length, to_seq_length with
                | Some x, Some y, Some z -> (x,y,z)
                | _ -> 
                    raise (ValueError("When passing in a rank 2 tensors to attention_layer, the values " +
                                       "for `batch_size`, `from_seq_length`, and `to_seq_length` " +
                                       "must all be specified."))
            | n -> failwithf "from_shape is expected to be 3 or 2 but is %i" n

        // Scalar dimensions referenced here:
        //   B = batch size (number of sequences)
        //   F = `from_tensor` sequence length
        //   T = `to_tensor` sequence length
        //   N = `num_attention_heads`
        //   H = `size_per_head`

        // TODO: Need to fix the ability to easily pass in an optional to a C# function
        //       e.g. tf.layers.dense(...,?activation=query)
        //       Then we can remove the identity activations from this code

        let identity = 
            { new Operations.Activation.IActivation with 
                member _.Activate(x:Tensor,name:string) = tf.identity(x,name) }

        let query_act = defaultArg query_act identity
        let key_act = defaultArg key_act identity
        let value_act = defaultArg value_act identity

        let from_tensor_2d = BertModel.reshape_to_matrix(from_tensor)
        let to_tensor_2d = BertModel.reshape_to_matrix(to_tensor)

        // `query_layer` = [B*F, N*H]
        let query_layer = 
            Layers.dense(from_tensor_2d,
                num_attention_heads * size_per_head,
                activation=query_act,
                name="query",
                kernel_initializer=Utils.create_initializer(initializer_range))

        // `key_layer` = [B*T, N*H]
        let key_layer =
            Layers.dense(to_tensor_2d,
                num_attention_heads * size_per_head,
                activation=key_act,
                name="key",
                kernel_initializer=Utils.create_initializer(initializer_range))

        // `value_layer` = [B*T, N*H]
        let value_layer =
            Layers.dense(to_tensor_2d,
                num_attention_heads * size_per_head,
                activation=value_act,
                name="value",
                kernel_initializer=Utils.create_initializer(initializer_range))

        // `query_layer` = [B, N, F, H]
        let query_layer =
            transpose_for_scores(query_layer, batch_size, num_attention_heads,
                from_seq_length, size_per_head)

        // `key_layer` = [B, N, T, H]
        let key_layer =
            transpose_for_scores(key_layer, batch_size, num_attention_heads,
                to_seq_length, size_per_head)

        // Take the dot product between "query" and "key" to get the raw
        // attention scores.
        // `attention_scores` = [B, N, F, T]
        let attention_scores = tf.matmul2(query_layer, key_layer, transpose_b=true)
        let attention_scores = tf.multiply(attention_scores, 1.0 / Math.Sqrt(float size_per_head))

        let attention_scores = 
            match attention_mask with
            | None -> attention_scores
            | Some attention_mask ->
                // `attention mask` = [B, 1, F, T]
                // TODO We might be able to do something like attention_maks.[:,None]
                let attention_mask = tf.expand_dims(attention_mask, axis=1)

                // Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                // masked positions, this operation will create a tensor which is 0.0 for
                // positions we want to attend and -10000.0 for masked positions.
                let adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

                // Since we are adding it to the raw scores before the softmax, this is
                // effectively the same as removing these entirely.
                attention_scores + adder

        // Normalize the attention scores to probabilities.
        // `attention_probs` = [B, N, F, T]
        let attention_probs = tf.nn.softmax(attention_scores)

        // This is actually dropping out entire tokens to attend to, which might
        // seem a bit unusual, but is taken from the original Transformer paper
        let attention_probs = Utils.dropout(attention_probs, attention_probs_dropout_prob)

        // `value_layer` = [B, T, N, H]
        let value_layer = 
            tf.reshape(value_layer, 
                [|batch_size; to_seq_length; num_attention_heads; size_per_head|])

        // `value_layer` = [B, N, T, H]
        let value_layer = tf.transpose(value_layer, [|0; 2; 1; 3|])

        // `context_layer` = [B, N, F, H]
        let context_layer = tf.matmul2(attention_probs, value_layer)

        // `context_layer` = [B, F, N, H]
        let context_layer = tf.transpose(context_layer, [|0; 2; 1; 3|])

        let context_layer = 
            if do_return_2d_tensor then
                // `context_layer` = [B*F, N*H]
                tf.reshape(context_layer,
                    [|batch_size * from_seq_length; num_attention_heads * size_per_head|])
            else
                // `context_layer` = [B, F, N*H]
                tf.reshape(context_layer,
                    [|batch_size; from_seq_length; num_attention_heads * size_per_head|])

        context_layer

    /// <summary>Raises an exception if the tensor rank is not of the expected rank.</summary>
    /// <param name="tensor">A tf.Tensor to check the rank of.</param>
    /// <param name="expected_rank">list of integers, expected rank.</param>
    /// <param name="name">Optional name of the tensor for the error message.<param>
    /// <exception cdef="Tensorflow.ValueError">If the expected shape doesn't match the actual shape.</exception>
    static member assert_rank(tensor: Tensor, expected_rank: int[], ?name: string) =
        let name = defaultArg name tensor.name
        let actual_rank = tensor.TensorShape.ndim
        if not(expected_rank.Contains(actual_rank)) then
            let scope_name = tf.get_variable_scope().name
            raise (ValueError(sprintf "For the tensor.`%s` in scope `%s`, the actual rank `%d` (shape = %O) is not equal to the expected rank `%A`"
                name scope_name actual_rank tensor.TensorShape expected_rank))

    /// <summary>Raises an exception if the tensor rank is not of the expected rank.</summary>
    /// <param name="tensor">A tf.Tensor to check the rank of.</param>
    /// <param name="expected_rank">int, expected rank.</param>
    /// <param name="name">Optional name of the tensor for the error message.<param>
    /// <exception cdef="Tensorflow.ValueError">If the expected shape doesn't match the actual shape.</exception>
    static member assert_rank(tensor: Tensor, expected_rank: int, ?name: string) =
        BertModel.assert_rank(tensor, [|expected_rank|],?name=name)

    /// <summary>Returns a list of the shape of tensor, preferring static dimensions.</summary>
    /// <param name="tensor">A tf.Tensor object to find the shape of.</param>
    /// <param name="expected_rank"> (optional) int. The expected rank of `tensor`. If this is
    ///   specified and the `tensor` has a different rank, and exception will be
    ///   thrown.</param>
    /// <param name="name"> Optional name of the tensor for the error message.</param>
    /// <returns>
    /// A list of dimensions of the shape of tensor. All static dimensions will
    /// be returned as python integers, and dynamic dimensions will be returned
    /// as tf.Tensor scalars.
    /// </returns>
    static member get_shape_list(tensor: Tensor, ?expected_rank: int, ?name: string): int[] =
        BertModel.get_shape_list(tensor, expected_rank |> Option.toArray, ?name = name)

    /// <summary>Returns a list of the shape of tensor, preferring static dimensions.</summary>
    /// <param name="tensor">A tf.Tensor object to find the shape of.</param>
    /// <param name="expected_rank"> (optional) int. The expected rank of `tensor`. If this is
    ///   specified and the `tensor` has a different rank, and exception will be
    ///   thrown.</param>
    /// <param name="name"> Optional name of the tensor for the error message.</param>
    /// <returns>
    /// A list of dimensions of the shape of tensor. All static dimensions will
    /// be returned as python integers, and dynamic dimensions will be returned
    /// as tf.Tensor scalars.
    /// </returns>
    static member get_shape_list(tensor: Tensor, expected_rank: int[], ?name: string): int[] =
        let name = defaultArg name tensor.name
        //expected_rank |> Option.iter (fun expected_rank -> BertModel.assert_rank(tensor, expected_rank,name))
        if expected_rank.Length > 0 then BertModel.assert_rank(tensor, expected_rank,name)
        let shape = tensor.TensorShape.as_list()
        if shape |> Array.exists (fun x -> x < 0) then
            //let dyn_shape = tf.shape(tensor)
            //shape |> Array.mapi (fun i x -> if x < 0 then Choice2Of2(dyn_shape.[i]) else Choice1Of2(i))
            failwith "Non-static shapes are not supported at this time"
        else 
            //shape |> Array.map Choice1Of2
            shape

