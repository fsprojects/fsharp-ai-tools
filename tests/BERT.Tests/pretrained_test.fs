/// Tests BERT inferences with known values
module PretrainedTests

open Modeling
open NUnit.Framework
open NumSharp
open RunClassifier
open System
open System.IO
open Tensorflow

[<Test>]
let ``test pretrained BERT run``() =
    Utils.init.Force()
    Common.setup()
    let tf = Tensorflow.Binding.tf
    tf.reset_default_graph()

    let do_lower_case = true
    let tokenizer =   Tokenization.FullTokenizer(vocab_file=Common.vocab_file, do_lower_case=do_lower_case)
    let bert_config = BertConfig.from_json_string(File.ReadAllText(Common.bert_config_file))
    let BATCH_SIZE = 2
    let MAX_SEQ_LENGTH = 128
    let vocab_map = File.ReadAllLines(Common.vocab_file) |> Array.mapi (fun i x -> (x,i)) |> Map.ofArray
    let movie_reviews = 
        [|
            "a stirring , funny and finally transporting re imagining of beauty and the beast and 1930s horror films",1
            "apparently reassembled from the cutting room floor of any given daytime soap",0
            "they presume their audience won't sit still for a sociology lesson",0
            "this is a visually stunning rumination on love , memory , history and the war between art and commerce",1
            "jonathan parker 's bartleby should have been the be all end all of the modern office anomie films",1
        |]

    let examples = 
        let mm = movie_reviews |> Array.map (fun (x,y) -> InputExample(text_a = x, label = string y) :> IExample)
        RunClassifier.convert_examples_to_features(mm,vocab_map,MAX_SEQ_LENGTH, tokenizer :> Tokenization.ITokenizer)

    let input_ids = tf.placeholder(tf.int32,TensorShape([|BATCH_SIZE; MAX_SEQ_LENGTH|]))
    let input_mask = tf.placeholder(tf.int32,TensorShape([|BATCH_SIZE; MAX_SEQ_LENGTH|]))

    let bertModel = BertModel(bert_config, false, input_ids = input_ids, input_mask = input_mask)
    let restore = tf.restore(Common.bert_chkpt)

    use sess = tf.Session()
    sess.run(restore)

    let subsample = examples.[0..1]
    let t1 = NDArray(subsample |> Array.map (fun x -> x.input_ids))
    let t2 = NDArray(subsample |> Array.map (fun x -> x.input_mask))

    let expected = [|-0.791169226f; -0.372503042f; -0.784386933f; 0.597510815f |]
    let res = sess.run(bertModel.PooledOutput,[|FeedItem(input_ids,t1); FeedItem(input_mask,t2)|])
                  .Data<float32>().ToArray().[0..3]
    if (expected,res) ||> Array.zip |> Array.exists (fun (x,y) -> System.Math.Abs(x-y) > 1e-2f ) 
    then Assert.Fail(sprintf "fail test adam: expected %A, got %A" expected res)


[<Test>]
let ``test pretrained BERT train``() =
    Utils.init.Force()
    Common.setup()
    let tf = Tensorflow.Binding.tf
    tf.reset_default_graph()

    let do_lower_case = true
    let tokenizer =   Tokenization.FullTokenizer(vocab_file=Common.vocab_file, do_lower_case=do_lower_case)
    let bert_config = BertConfig.from_json_string(File.ReadAllText(Common.bert_config_file))
    // Compute train and warmup steps from batch size
    // These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)

    let BATCH_SIZE = 2 // when not on laptop bump up to 32
    let NUM_LABELS = 2
    let LEARNING_RATE = 2e-5f
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
        let logits = tf.matmul(output_layer, output_weights._AsTensor())
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

    let train_op = Optimization.create_optimizer(loss, LEARNING_RATE, num_train_steps, Some(num_warmup_steps))

    let sess = tf.Session()
    let init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(restore) // load weights

    System.Diagnostics.Debug.WriteLine(sprintf "Training with batch size %i" BATCH_SIZE)

    for i in 0..3 do
        let subsample = train |> Array.subSample BATCH_SIZE 
        let t1 = NDArray(subsample |> Array.map (fun x -> x.input_ids))
        let t2 = NDArray(subsample |> Array.map (fun x -> x.input_mask))
        let t3 = NDArray(subsample |> Array.map (fun x -> x.label_id))
        let res = sess.run([|train_op :> ITensorOrOperation;loss :> ITensorOrOperation|], [|FeedItem(input_ids,t1); FeedItem(input_mask,t2); FeedItem(labels,t3)|])
        System.Diagnostics.Debug.WriteLine(sprintf "%i %f" i (float res.[1]))

