// Apache 2.0 from https://github.com/google-research/bert/blob/e13c1f3459cc254f7abbabfc5a286a3304d573e4/run_pretraining.py
/// Run masked LM/next sentence masked_lm pre-training for BERT.

module RunPretraining
let x = 10

(*

open Argu
open System
open System.IO
open Modeling
open Optimization
open Tensorflow

type Arguements =
    | [<Mandatory>] Bert_Config_File of path:string
    | [<Mandatory>] Input_File of path:string
    | [<Mandatory>] Output_Dir of dir:string
    | Init_Checkpoint of string
    | Max_Seq_Length of int
    | Max_Predictions_Per_Seq of int
    | Do_Train of bool
    | Do_Eval of bool
    | Train_Batch_Size of int
    | Eval_Batch_Size of int
    | Learning_Rate of float32
    | Num_Train_Steps of int
    | Num_Warmup_Steps of int
    | Save_Checkpoints_Steps of int
    | Iterations_Per_Loop of int
    | Max_Eval_Steps of int
    | Use_Tpu of bool
    | TPU_Name of string
    | TPU_Zone of string
    | GCP_Project of string option
    | Master of string option
    | Num_TPU_Cores of int option
    interface Argu.IArgParserTemplate with
        member this.Usage =
            match this with
            | Bert_Config_File _ -> 
                "The config json file corresponding to the pre-trained BERT model. " +
                "This specifies the model architecture."
            | Input_File _ -> 
                "Input TF example files (can be a glob or comma separated)."
            | Output_Dir _ -> 
                "The output directory where the model checkpoints will be written."
            | Init_Checkpoint _ -> 
                "Initial checkpoint (usually from a pre-trained BERT model)."
            | Max_Seq_Length _ -> 
                "The maximum total input sequence length after WordPiece tokenization. " +
                "Sequences longer than this will be truncated, and sequences shorter " + 
                "than this will be padded. Must match data generation."
            | Max_Predictions_Per_Seq _ -> 
                "Maximum number of masked LM predictions per sequence. " +
                "Must match data generation."
            | Do_Train _ -> "Whether to run training."
            | Do_Eval _ -> "Whether to run eval on the dev set."
            | Train_Batch_Size _ -> "Total batch size for training."
            | Eval_Batch_Size _ -> "Total batch size for eval."
            | Learning_Rate _ -> "The initial learning rate for Adam."
            | Num_Train_Steps _ -> "Number of training steps."
            | Num_Warmup_Steps _ -> "Number of warmup steps."
            | Save_Checkpoints_Steps _ -> "How often to save the model checkpoint."
            | Iterations_Per_Loop _ -> "How many steps to make in each estimator call."
            | Max_Eval_Steps _ -> "Maximum number of eval steps."
            | Use_Tpu _ -> "Whether to use TPU or GPU/CPU."
            | TPU_Name _ -> 
                "The Cloud TPU to use for training. This should be either the name " +
                "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url."
            | TPU_Zone _ -> 
                "[Optional] GCE zone where the Cloud TPU is located in. If not " +
                "specified, we will attempt to automatically detect the GCE project from " +
                "metadata."
            | GCP_Project _ -> 
                "[Optional] Project name for the Cloud TPU-enabled project. If not " + 
                "specified, we will attempt to automatically detect the GCE project from " +
                "metadata."
            | Master _ -> "[Optional] TensorFlow master URL."
            | Num_TPU_Cores _ -> "Only used if `use_tpu` is True. Total number of TPU cores to use."

// TODO defaults
// max_seq_length 128
// max_predictions_per_seq 20
// do_train false
// do_eval false
// train_batch_size 32
// eval_batch_size 8
// learning_rate 5e-5
// num_train_steps 100000
// num_warmup_steps 10000
// save_checkpoints_steps 1000
// iterations_per_loop 1000
// max_eval_steps 100
// use_tpu false
// num_tpu_cores

// TODO figure out tf.estimator.ModeKeys.TRAIN
type ModeKeys = 
    | Train
    | Eval

/// Gathers the vectors at the specific positions over a minibatch.
let gather_indexes(sequence_tensor, positions) = 
    let sequence_shape = Modeling.BertModel.get_shape_list(sequence_tensor, expected_rank=3)
    let batch_size = sequence_shape.[0]
    let seq_length = sequence_shape.[1]
    let width = sequence_shape.[2]
    let flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [|-1, 1|])
    let flat_positions = tf.reshape(positions + flat_offsets, [|-1|])
    let flat_sequence_tensor = tf.reshape(sequence_tensor, [|batch_size * seq_length; width|])
    let output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    output_tensor

let create_initializer(x:float32) : IInitializer = failwith "todo"
    
/// Get loss and log probs for the masked LM.
let get_masked_lm_output(bert_config, 
                         input_tensor, output_weights, positions,
                         label_ids, label_weights) = 
    let input_tensor = gather_indexes(input_tensor, positions)
    use _clsPredictions = vs.variable_scope("cls/predictions")
    //Binding.tf_with(tf.variable_scope(), fun _ -> 
    // We apply one more non-linear transformation before the output layer.
    // This matrix is not used after pre-training.
    let input_tensor = 
        use _transform = vs.variable_scope("transform")
        //Binding.tf_with(tf.variable_scope("transform"), fun _ -> 
        let input_tensor = 
            Layers.dense(input_tensor,
                units=bert_config.hidden_size,
                activation = bert_config.hidden_act,
                kernel_initializer = create_initializer(bert_config.initializer_range))
        let input_tensor = Layers.layer_norm(input_tensor)
        input_tensor
    // The output weights are the same as the input embeddings, but there is
    // an output-only bias for each token.
    let output_bias = tf.get_variable("output_bias", 
                                      shape = TensorShape(bert_config.vocab_size.Value),
                                      initializer = tf.zeros_initializer)
    let logits = tf.matmul2(input_tensor, output_weights, transpose_b = true)
    let logits = tf.nn.bias_add(logits, output_bias)
    //Tensorflow.Operations.gen_nn_ops.log_softmax() // missing an axis
    let log_probs = tf.log(tf.nn.softmax(logits, axis = -1))

    let label_ids = tf.reshape(label_ids,[|-1|])
    let label_weights = tf.reshape(label_weights,[|-1|])

    let one_hot_labels = tf.one_hot(label_ids, 
                                    depth=bert_config.vocab_size.Value, 
                                    dtype=tf.float32)
    // The `positions` tensor might be zero-padded (if the sequence is too
    // short to have the maximum number of predictions). The `label_weights`
    // tensor has a value of 1.0 for every real prediction and 0.0 for the
    // padding predictions.
    let per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis= Nullable(-1))
    let numerator = tf.reduce_sum(label_weights * per_example_loss)
    let denominator = tf.reduce_sum(label_weights) + 1e-5
    let loss = numerator / denominator
    (loss, per_example_loss, log_probs)

/// Get loss and log probs for the next sentence prediction."""
let get_next_sentence_output(bert_config, input_tensor, labels) = 
    // Simple binary classification. Note that 0 is "next sentence" and 1 is
    // "random sentence". This weight matrix is not used after pre-training.
    let output_weights, output_bias = 
        use _clsSeqRelationship = vs.variable_scope("cls/seq_relationship")
        //Binding.tf_with(tf.variable_scope("cls/seq_relationship"), fun _ -> 
        let output_weights = 
            tf.get_variable("output_weights",
                            shape=TensorShape([|2; bert_config.hidden_size|]),
                            initializer= create_initializer(bert_config.initializer_range))
        let output_bias = 
            tf.get_variable("output_bias", 
                            shape= TensorShape([|2|]), 
                            initializer=tf.zeros_initializer)
        (output_weights, output_bias)
    let logits = tf.matmul2(input_tensor, output_weights._AsTensor(), transpose_b = true)
    let logits = tf.nn.bias_add(logits, output_bias)
    let log_probs = tf.log(tf.nn.softmax(logits, axis = -1))
    let labels = tf.reshape(labels, [|-1|])
    let one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    let per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis = Nullable(-1))
    let loss = tf.reduce_mean(per_example_loss)
    (loss, per_example_loss, log_probs)

*)

