// Apache 2.0 from https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

#I @"C:/Users/moloneymb/.nuget/packages/"
#I @"../BertInFSharp/"
//#I @"/home/moloneymb/.nuget/packages/"

#r @"system.runtime.compilerservices.unsafe/4.5.2/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r @"numsharp/0.20.5/lib/netstandard2.0/NumSharp.Core.dll"
#r @"tensorflow.net/0.14.0/lib/netstandard2.0/TensorFlow.NET.dll"
#r @"system.memory/4.5.3/lib/netstandard2.0/System.Memory.dll"
#r @"google.protobuf/3.10.1/lib/netstandard2.0/Google.Protobuf.dll"
#r @"argu/6.0.0/lib/netstandard2.0/Argu.dll"
#r @"csvhelper/12.2.3/lib/net47/CsvHelper.dll"
#r @"newtonsoft.json/12.0.2/lib/net45/Newtonsoft.Json.dll"
#r @"sharpziplib/1.2.0/lib/net45/ICSharpCode.SharpZipLib.dll"
#r @"System.IO.Compression"
#load "common.fs"
#load "utils.fs"
#load "tokenization.fs"
#load "run_classifier.fs"
#load "modeling.fs"
#load "optimization.fs"

open Modeling
open NumSharp
open RunClassifier
open System
open System.IO
open Tensorflow

#time "on"

Common.setup() // This sets up the correct
Utils.setup() // This patches some of the gradient logic

//open Tokenization
//open System
//open System.IO
//open Newtonsoft.Json.Linq
//open Modeling
//open Tensorflow.Operations.Activation
//open Modeling.Activation
//open NumSharp
//open Tensorflow
//open System.Collections.Generic
//open RunClassifier

let do_lower_case = true
let tokenizer =   Tokenization.FullTokenizer(vocab_file=Common.vocab_file, do_lower_case=do_lower_case)
let bert_config = BertConfig.from_json_string(File.ReadAllText(Common.bert_config_file))
// Compute train and warmup steps from batch size
// These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)

let BATCH_SIZE = 2
let NUM_LABELS = 2
let LEARNING_RATE = 0.000001f
let MAX_SEQ_LENGTH = 128
let NUM_TRAIN_EPOCHS = 3.0f
// Warmup is a period of time where hte learning rate 
// is small and gradually increases--usually helps training.
let WARMUP_PROPORTION = 3.0f
// Model configs
let SAVE_CHECKPOINTS_STEPS = 500
let SAVE_SUMMARY_STEPS = 100

let vocab = File.ReadAllLines(Common.vocab_file)

let getTrainTest limit = 
    let vocab_map = vocab |> Array.mapi (fun i x -> (x,i)) |> Map.ofArray
    let f x y v = 
        Directory.GetFiles(Path.Combine(Common.data, "aclImdb",x,y)) 
        |> Array.truncate limit
        |> Async.mapiChunkBySize 200 (fun _ x -> InputExample(text_a = File.ReadAllText(x), label = string v) :> IExample)
    let g x = 
        let mm = [| yield! f x "pos" 1; yield! f x "neg" 0|] |> Array.shuffle
        convert_examples_to_features(mm,vocab_map,MAX_SEQ_LENGTH, tokenizer :> Tokenization.ITokenizer)
    (g "train", g "test")

let train,test = getTrainTest 2500

let input_ids = tf.placeholder(tf.int32,TensorShape([|BATCH_SIZE; MAX_SEQ_LENGTH|]))
let input_mask = tf.placeholder(tf.int32,TensorShape([|BATCH_SIZE; MAX_SEQ_LENGTH|]))
let labels = tf.placeholder(tf.int32,TensorShape([|BATCH_SIZE|]))
let bertModel = BertModel(bert_config, false, input_ids = input_ids, input_mask = input_mask)

let ops = tf.get_default_graph().get_operations() 

// create the restore op before the other ops
let restore = tf.restore(Common.bert_chkpt)

// Use "pooled_output" for classification tasks on an entire sentence.
// Use "sequence_outputs" for token-level output.
let output_layer = bertModel.PooledOutput

let hidden_size = output_layer.shape |> Seq.last

let output_weights = tf.get_variable("output_weights", 
                                     TensorShape([|hidden_size; NUM_LABELS|]), 
                                     initializer=tf.truncated_normal_initializer(stddev=0.02f))

let output_bias = tf.get_variable("output_bias", 
                                  TensorShape(NUM_LABELS), 
                                  initializer=tf.zeros_initializer)

let (loss, predicted_labels, log_probs) =
    use _loss = vs.variable_scope("loss")
    // Dropout helps prevent overfitting
    let output_layer = tf.nn.dropout(output_layer, keep_prob=tf.constant(0.9f))
    let logits = tf.matmul(output_layer, output_weights._AsTensor()) // trained in transpose
    let logits = tf.nn.bias_add(logits, output_bias)
    let log_probs = tf.log(tf.nn.softmax(logits, axis = -1))
    // Convert Labels into one-hot encoding
    let one_hot_labels = tf.one_hot(labels, depth=NUM_LABELS, dtype=tf.float32)
    let predicted_labels = tf.squeeze(tf.argmax(log_probs, axis = -1, output_type = tf.int32))
    /// If we're predicting, we want predicted labels and the probabiltiies.
    //if is_predicting:
    //  return (predicted_labels, log_probs)
    // If we're train/eval, compute loss between predicted and actual label
    let per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis= Nullable(-1))
    let loss = tf.reduce_mean(per_example_loss)
    (loss, predicted_labels, log_probs)

let num_train_steps = int(float32 train.Length / float32 BATCH_SIZE * NUM_TRAIN_EPOCHS)
let num_warmup_steps = int(float32 num_train_steps * WARMUP_PROPORTION)

// Train steps should be 400
// This should be ~43

//num_train_steps
//1404
//let train_op = Optimization.create_optimizer(loss, LEARNING_RATE, num_train_steps, Some(num_warmup_steps))

let train_op = Optimization.create_optimizer(loss, LEARNING_RATE, num_train_steps, None)

let sess = tf.Session(ConfigProto(GpuOptions = GPUOptions(AllowGrowth = true)))

let init = tf.global_variables_initializer()
sess.run(init)
// load weights from checkpoint
sess.run(restore) 

System.Diagnostics.Debug.WriteLine(sprintf "Training with batch size %i" BATCH_SIZE)

let fetchOps = [|
    train_op :> ITensorOrOperation
    loss :> ITensorOrOperation
  |]

let accuracies = 
  [|
    for i in 0..400 do
      let subsample = train |> Array.subSample BATCH_SIZE 
      let t1 = NDArray(subsample |> Array.map (fun x -> x.input_ids))
      let t2 = NDArray(subsample |> Array.map (fun x -> x.input_mask))
      let t3 = NDArray(subsample |> Array.map (fun x -> match x.label_id with | 1014 -> 0 | 1015 -> 1 | _ -> failwith "err"))
      let res = sess.run(fetchOps, [|FeedItem(input_ids,t1); FeedItem(input_mask,t2); FeedItem(labels,t3)|])
      let acc = (res.[1].Data<float32>().[0])
      printfn "%i %f" i acc
      yield acc
  |]

