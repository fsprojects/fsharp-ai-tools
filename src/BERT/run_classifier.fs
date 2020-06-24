// Apache 2.0 https://github.com/google-research/bert/blob/master/run_classifier.py
module RunClassifier

open Argu
open System
open Tokenization

type Arguments =
    | [<Mandatory>] Data_Dir of path:string
    | [<Mandatory>] Bert_Config_File of path:string
    | [<Mandatory>] Task_Name of string
    | [<Mandatory>] Vocab_File of path:string
    | [<Mandatory>] Output_Dir of path:string
    | Init_Checkpoint of path:string
    | Do_Lower_Case of bool
    | Max_Seq_Length of int
    | Do_Train of bool
    | Do_Eval of bool
    | Do_Predict of bool
    | Train_Batch_Size of int
    | Eval_Batch_Size of int
    | Predict_Batch_Size of int
    | Learning_Rate of float32
    | Num_Train_Epochs of float32
    | Warmup_Proportion of float32
    | Save_Checkpoints_Steps of int
    | Iterations_Per_Loop of int
    | Use_TPU of bool
    | TPU_Name of string
    | TPU_Zone of string
    | GCP_Project of string
    | Master of string
    | Num_TPU_Cores of int
    interface Argu.IArgParserTemplate with
        member this.Usage =
            match this with
            | Data_Dir _ -> "The input data dir. Should contain the .tsv files (or other data files) for the task."
            | Bert_Config_File _ -> "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture."
            | Task_Name _ -> "The name of the task to train."
            | Vocab_File _ -> "The vocabulary file that the BERT model was trained on."
            | Output_Dir _ -> "The output directory where the model checkpoints will be written."
            | Init_Checkpoint _ -> "Initial checkpoint (usually from a pre-trained BERT model)."
            | Do_Lower_Case _ -> "Whether to lower case the input text. Should be True for uncased models and False for cased models."
            | Max_Seq_Length _ -> 
                "The maximum total input sequence length after WordPiece tokenization. " + 
                "Sequences longer than this will be truncated, and sequences shorter " + 
                "than this will be padded."
            | Do_Train _ -> "Whether to run training."
            | Do_Eval _ -> "Whether to run eval on the dev set."
            | Do_Predict _ -> "Whether to run the model in inference mode on the test set."
            | Train_Batch_Size _ -> "Total batch size for training."
            | Eval_Batch_Size _ -> "Total batch size for eval."
            | Predict_Batch_Size _ -> "Total batch size for predict."
            | Learning_Rate _ -> "The initial learning rate for Adam."
            | Num_Train_Epochs _ -> "Total number of training epochs to perform."
            | Warmup_Proportion _ -> "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."
            | Save_Checkpoints_Steps _ -> "How often to save the model checkpoint."
            | Iterations_Per_Loop _ -> "How many steps to make in each estimator call."
            | Use_TPU _ -> "Whether to use TPU or GPU/CPU."
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

// TODO Defaults 
// do_lower_case true
// max_seq_length 128
// do_train false
// do_eval false
// do_predict false
// train_batch_size 32
// eval_batch_size 8
// predict_batch_size 8
// learning_rate 5e-5
// num_train_epochs 3.0
// warmup_proportion 0.1
// save_checkpoints_steps 1000
// iterations_per_loop 1000
// use_tpu false
// num_tpu_cores 8

type IExample = 
    abstract member guid : Guid option
    abstract member text_a : string
    abstract member text_b : string option
    abstract member label : string

// <summary>A single training/test example for simple sequence classification </summary>
// <param name="guid">Unique id for the example.</param>
// <param name="text_a"> string. The untokenized text of the first sequence. For single
// sequence tasks, only this sequence must be specified.</param>
// <param name="text_b"> (Optional) string. The untokenized text of the second sequence.
//  Only must be specified for sequence pair tasks.</param>
// <param name="label"> (Optional) string. The label of the example. This should be
// specified for train and dev examples, but not for test examples </param>
type InputExample(?guid : System.Guid, ?text_a : string, ?text_b : string, ?label : string) = 
    interface IExample with
        member this.guid = guid 
        member this.text_a = defaultArg text_a String.Empty
        member this.text_b = text_b
        member this.label = defaultArg label String.Empty

// Fake example so the num input examples is a multiple of the batch size.
//  When running eval/predict on the TPU, we need to pad the number of examples
//  to be a multiple of the batch size, because the TPU requires a fixed batch
//  size. The alternative is to drop the last batch, which is bad because it means
//  the entire output data won't be generated.
//  We use this class instead of `None` because treating `None` as padding
//  battches could cause silent errors.
type PaddingInputExample() = 
    class
    end

/// A single set of features of data.
type InputFeatures(input_ids : int[],
                   input_mask : int[],
                   segment_ids : int[],
                   label_id : int,
                   ?is_real_example) =
    member this.input_ids = input_ids
    member this.input_mask = input_mask
    member this.segment_ids = segment_ids
    member this.label_id = label_id
    member this.is_real_example = defaultArg is_real_example false

/// NOTE: Should I get rid of the default methods
/// Base class for data converters for sequence classification data sets.
type DataProcessor() = 
    abstract member get_train_examples : string -> string
    /// Gets a collection of `InputExample`s for the train set.
    default this.get_train_examples(data_dir) = raise (System.NotImplementedException())

    abstract member get_dev_examples : string -> string
    /// Gets a collection of `InputExample`s for the dev set.
    default this.get_dev_examples(data_dir) = raise (System.NotImplementedException())

    abstract member get_test_examples : string -> string
    /// Gets a collection of `InputExample`s for prediction.
    default this.get_test_examples (data_dir) = raise (System.NotImplementedException())

    abstract member get_labels : unit -> string []
    /// Gets the list of labels for this data set."""
    default this.get_labels () = raise (System.NotImplementedException())

    /// Reads a tab separated value file.
    static member private read_tsv(input_file : string, ?quotechar : char)  =
        let reader = new System.IO.StreamReader(input_file)
        let config = CsvHelper.Configuration.Configuration(Delimiter = "\t", Quote = (defaultArg quotechar '"'))
        use csv = new CsvHelper.CsvReader(reader, config)
        if not(csv.Read()) then [||]
        else 
            // NOTE this is a hack because csv.Context.ColumnCount always returned 0 in testing
            // and there seemed to be no other way to get the column count
            let rec getColumns(col : int) = 
                try 
                    csv.[col] |> ignore
                    getColumns(col+1)
                with
                | :? CsvHelper.MissingFieldException -> col
            let colCount = getColumns(0)
            let getRow() = [|for i in 0 .. colCount - 1 -> csv.[i]|]
            [|
                yield getRow()
                while csv.Read() do
                    yield getRow()
            |]


/// Truncates a sequence pair in place to the maximum length.
let truncate_seq_pair(tokens_a : 'a[], tokens_b : 'a[], max_length : int) = 
  // This is a simple heuristic which will always truncate the longer sequence
  // one token at a time. This makes more sense than truncating an equal percent
  // of tokens from each, since if one sequence is very short then each token
  // that's truncated likely contains more information than a longer sequence.
    let rec f(length_a,length_b) = 
        if length_a + length_b <= max_length then
            tokens_a.[0..length_a-1],
            tokens_b.[0..length_b-1]
        else
            if length_a > length_b then
                f(length_a-1,length_b)
            else 
                f(length_a,length_b-1)
    f(tokens_a.Length, tokens_b.Length )

let padding_input_example_to_iexample(example : PaddingInputExample, max_seq_length : int) = 
            let zeroVector = Array.create max_seq_length 0
            InputFeatures(input_ids = zeroVector,
                          input_mask = zeroVector,
                          segment_ids = zeroVector,
                          label_id = 0,
                          is_real_example = false)  

/// Converts a single `InputExample` into a single `InputFeatures`.
let convert_single_example (ex_index : int, example : IExample, label_map : Map<string,int>, max_seq_length : int, tokenizer : ITokenizer) = 
    //let example = 
//        match example with
//        | :? PaddingInputExample as x -> 
//            // I am currently confused by this... 
//            // TODO This will likely need refactoring
//        | _ -> example
  //      :?> IExample // TODO obviously fix this

    let tokens_a = tokenizer.tokenize(example.text_a)
    let tokens_b = example.text_b |> Option.map tokenizer.tokenize

    let tokens_a, tokens_b = 
        match tokens_b with
        | Some(tokens_b) -> 
            let (x,y) = truncate_seq_pair(tokens_a, tokens_b,max_seq_length)
            (x,Some(y))
        | None -> 
            // Account for [CLS] and [SEP] with "-2"
            if tokens_a.Length > max_seq_length - 2 then
                (tokens_a.[0 .. (max_seq_length - 3)],None)
            else (tokens_a,None)

    // The convention in BERT is:
    // (a) For sequence pairs:
    //  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    //  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    // (b) For single sequences:
    //  tokens:   [CLS] the dog is hairy . [SEP]
    //  type_ids: 0     0   0   0  0     0 0
    // 
    // Where "type_ids" are used to indicate whether this is the first
    // sequence or the second sequence. The embedding vectors for `type=0` and
    // `type=1` were learned during pre-training and are added to the wordpiece
    // embedding vector (and position vector). This is not *strictly* necessary
    // since the [SEP] token unambiguously separates the sequences, but it makes
    // it easier for the model to learn the concept of sequences.
   // 
    // For classification tasks, the first vector (corresponding to [CLS]) is
    // used as the "sentence vector". Note that this only makes sense because
    // the entire model is fine-tuned.

    let tokens,segment_ids = 
        [|
            yield ("[CLS]",0)
            yield! tokens_a |> Array.map (fun x -> (x,0))
            yield ("[SEP]",0)
            match tokens_b with
            | Some(tokens_b) ->
                yield! tokens_b |> Array.map (fun x -> (x,1))
                yield ("[SEP]",1)
            | None -> ()
        |] |> Array.unzip
    
    let input_ids = tokenizer.convert_tokens_to_ids(tokens)

    // The mask has 1 fo real tokens and 0 for padding tokens. Only real
    let input_mask = Array.create input_ids.Length 1

    // Zero-pad up to the sequence length.
    let padVector  = Array.create (max_seq_length - input_ids.Length) 0
    let input_ids  = Array.append input_ids padVector
    let input_mask = Array.append input_mask padVector
    let segment_ids = Array.append segment_ids padVector

    assert (input_ids.Length = max_seq_length)
    assert (input_mask.Length = max_seq_length)
    assert (segment_ids.Length = max_seq_length)

    let label_id = label_map.[example.label]
    if ex_index < 5 then
        loggingf "*** Example ***"
        example.guid |> Option.iter (fun guid -> loggingf "guid: %s" (guid.ToString("N")))
        loggingf "tokens: %s" (tokens |> Array.map tokenizer.printable_text |> String.concat " ")
        for name,xs in [("input_ids",input_ids);("input_mask",input_mask);("segment_ids",segment_ids)] do
            loggingf "%s: %s" name (xs |> Array.map string |> String.concat " ")
        loggingf "label: %s (id = %d)" example.label label_id

    InputFeatures(input_ids = input_ids,
                  input_mask = input_mask,
                  segment_ids = segment_ids,
                  label_id = label_id,
                  is_real_example = true)


// NOTE: This function is not used by this file but is still used by the Colab and
// people who depend on it.
/// Convert a set of `InputExample`s to a list of `InputFeatures`.
let convert_examples_to_features(examples : IExample[], label_map, max_seq_length : int, tokenizer : ITokenizer) =
    examples |> Async.mapiChunkBySize 200 (fun ex_index example -> 
        if ex_index % 10000 = 0 then
            loggingf "Writing example %d of %d" ex_index examples.Length
        convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer))

// This function is not used by this file but is still used by the Colab and
// people who depend on it.

//Tensorflow.Binding.tf.data.Datasets

/// Creates an `input_fn` closure to be passed to TPUEstimator. 
let input_fn_build(features, seq_length, is_training, drop_remainder) =
    failwith "todo - a lot of the Dataset library is missing...."

