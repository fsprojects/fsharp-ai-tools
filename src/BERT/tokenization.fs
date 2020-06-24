// Apache License, Version 2.0
// Converted to F# from https://github.com/google-research/bert/blob/master/tokenization.py
module Tokenization

open System
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Text.RegularExpressions
open Tensorflow

type ITokenizer =
    abstract member tokenize: string -> string[]
    abstract member convert_tokens_to_ids: string[] -> int[]
    abstract member printable_text: string -> string

/// Checks whether the casing config is consistent with the checkpoint name.
let validate_case_matches_checkpoint(do_lower_case: bool, init_checkpoint: string) = 

    // The casing has to be passed in by the user and there is no explicit check
    // as to whether it matches the checkpoint. The casing information probably
    // should have been stored in the bert_config.json file, but it's not, so
    // we have to heuristically detect it to validate. 

    let m = Regex.Match(init_checkpoint,@"^.*?([A-Za-z0-9_-]+)/bert_model.ckpt")
    if m.Success && m.Groups.Count = 1 then 
        let model_name = m.Groups.[0].Value
        let lower_models = set [ "uncased_L-24_H-1024_A-16"; "uncased_L-12_H-768_A-12"; "multilingual_L-12_H-768_A-12"; "chinese_L-12_H-768_A-12" ]
        let cased_models = set [ "cased_L-12_H-768_A-12"; "cased_L-24_H-1024_A-16"; "multi_cased_L-12_H-768_A-12" ]
        let (is_bad_config, actual_flag, case_name, opposite_flag) =
            if lower_models.Contains model_name && not do_lower_case then
                (true,"False","lowercase","True")
            elif cased_models.Contains model_name && not do_lower_case then
                (true,"True","cased","False")
            else
                (false,"","","")

        if is_bad_config then
            sprintf """You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`.
However, `%s` seems to be a %s model, so you
should pass in `--do_lower_case=%s` so taht the fine-tuning matches 
how the model was pre-training. If this error is wrong, please 
just comment out this check.""" actual_flag init_checkpoint model_name case_name opposite_flag
            |> ValueError |> raise

type Vocab = IDictionary<string,int>
type InvVocab = IDictionary<int,string>

/// Loads a vocabulary file into a dictionary.
let load_vocab(vocab_file: string) = 
    File.ReadAllLines(vocab_file) |> Array.mapi (fun i x -> (x.Trim(),i)) |> dict
  
let convert_tokens_to_ids(vocab: Vocab, items: string[]) = items |> Array.map (fun x -> vocab.[x])
let convert_ids_to_tokens(inv_vocab: InvVocab, items: int[]) = items |> Array.map (fun x -> inv_vocab.[x])

/// Runs basic whitespace cleaning and splitting on a piece of text
let whitespace_tokenize(text: string) = text.Trim().Split([|' '|], StringSplitOptions.RemoveEmptyEntries)


/// Checks whether `chars` is a punctuation character.
let is_punctuation(char: Char) = 
    let cp = int char
    // We treat all non-letter/number ASCII as punctuation.
    // Characters such as "^", "$", and "`" are not in the Unicode
    // Punctuation class but we treat them as punctuation anyways, for
    // consistency.
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) then
        true
    else
        match Char.GetUnicodeCategory(char) with
        | UnicodeCategory.ClosePunctuation
        | UnicodeCategory.ConnectorPunctuation
        | UnicodeCategory.DashPunctuation
        | UnicodeCategory.FinalQuotePunctuation
        | UnicodeCategory.InitialQuotePunctuation
        | UnicodeCategory.OpenPunctuation
        | UnicodeCategory.OtherPunctuation -> true
        | _ -> false

/// Checks whether `char` is a control cahracter.
let is_control(char: Char) =
    // These are technically control characters but we count them as whitespace
    // characters
    match char with
    | '\t' | '\n' | '\r' -> false
    | _ ->
        match Char.GetUnicodeCategory(char) with
        | UnicodeCategory.Control
        | UnicodeCategory.Format -> true
        | _ -> false

/// Checks whether `chars` is a whitespace character.
let is_whitespace(char: Char) = 
    // \t, \n, and \r are technically control characters but we treat them
    // as whitespace since they are generally considered as such. 
    match char with 
    | ' '  | '\t' | '\n' | '\r' -> true
    | _ -> 
        match Char.GetUnicodeCategory(char) with
        | UnicodeCategory.SpaceSeparator -> true
        | _ -> false

/// Performs invalid character removal and whitespace cleanup on text.
let clean_text(text: string) = 
    text.ToCharArray() |> Array.choose (fun char -> 
        match int char with 
        | 0 | 0xfffd | _ when is_control(char) -> None 
        | _ when is_whitespace(char) -> Some(' ')
        | _ -> Some(char))
    |> String

/// Checks whether CP is the codepoint of a CJK character.
let is_chinese_char(cp: int) = 
    // This defines a "chinese character" as anything in the CJK Unicode block:
    //   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    //
    // Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    // despite its name. The modern Korean Hangul alphabet is a different block,
    // as is Japanese Hiragana and Katakana. Those alphabets are used to write
    // space-separated words, so they are not treated specially and handled        
    // like all of the other languages.
    (cp >= 0x4E00  && cp <= 0x9FFF ) ||
    (cp >= 0x3400  && cp <= 0x4DBF ) ||
    (cp >= 0x20000 && cp <= 0x2A6DF) ||
    (cp >= 0x2A700 && cp <= 0x2B73F) ||
    (cp >= 0x2B740 && cp <= 0x2B81F) ||
    (cp >= 0x2B820 && cp <= 0x2CEAF) ||
    (cp >= 0xF900  && cp <= 0xFAFF ) ||
    (cp >= 0x2F800 && cp <= 0x2FA1F)


// <summary>Runs basic tokenization (punctuation splitting, lower casing, etc.).</summary>
// <param name="do_lower_case">Whether to lower case the input</param>
type BasicTokenizer(?do_lower_case: bool) =
    let do_lower_case = defaultArg do_lower_case true

    /// Adds whitespace around any CJK character.
    let tokenize_chinese_chars(text: string) = 
        text |> String.collect (fun c -> 
            if is_chinese_char(int c) then sprintf " %c " c else string c)
        
    /// Strips accents from a piece of text.
    let run_strip_accents(text: string) = 
        text.Normalize(Text.NormalizationForm.FormD) // "NFD"
        |> String.filter (fun c -> not(Char.GetUnicodeCategory(c) = Globalization.UnicodeCategory.NonSpacingMark)) // "Mn"

    /// Splits punctuation on a piece of text.
    let run_split_on_punc(text: string) = 
        text.ToCharArray() 
        |> Array.fold (fun s c ->
            if is_punctuation(c) then [] :: [c] :: s
            else match s with | [] -> [[c]] | head::tail -> (c::head)::tail) []
        |> List.map (fun x -> String(x |> List.rev |> List.toArray))
        |> List.rev |> List.toArray

    /// Tokenizes a piece of text.
    member _.tokenize(text: string) = 
        let text = clean_text(text)
        // This was added on November 1st, 2018 for the multilingual and Chinese
        // models. This is also applied to the English models now, but it doesn't
        // matter since the English models were not trained on any Chinese data
        // and generally don't have any Chinese data in them (there are Chinese
        // characters in the vocabulary because Wikipedia does have some Chinese
        // words in the English Wikipedia.).
        let text = tokenize_chinese_chars(text)

        let orig_tokens = whitespace_tokenize(text)
        let split_tokens = 
            orig_tokens 
            |> Array.map (fun x -> 
                (if do_lower_case then x.ToLowerInvariant() else x) 
                |> run_strip_accents |> run_split_on_punc)
            |> Array.collect id
        // NOTE: This seems weird https://github.com/google-research/bert/blob/cc7051dc592802f501e8a6f71f8fb3cf9de95dc9/tokenization.py#L217
        whitespace_tokenize(split_tokens |> String.concat " ") 

type RoseTree<'a> = Node of 'a * RoseTree<'a>[]

// Runs WordPiece tokenziation.
type WordpieceTokenizer(vocab: Vocab,
                        ?unk_token: string,
                        ?max_input_chars_per_word: int) = 

    let unk_token = defaultArg unk_token "[UNK]"
    let max_input_chars_per_word = defaultArg max_input_chars_per_word 200
    
    let makePrefixTree(tokens: string[]) = 
        let rec makePrefixTree(tokens: char list []): RoseTree<char*bool>[] =
            [| 
                for (c,xs) in tokens |> Array.groupBy List.tryHead do
                    match c with
                    | Some(c) ->
                        // TODO could be more efficient?
                        let term = xs |> Array.exists (function | _::[] -> true | _ -> false)
                        yield Node((c,term),xs |> Array.map List.tail |> makePrefixTree)
                    | None -> ()
            |]
        tokens 
        |> Array.map (fun x -> x.ToCharArray() |> Array.toList) 
        |> makePrefixTree

    let findLongestMatch(text: string, prefixTree: RoseTree<char*bool>[]) =
        let rec f (text: char list, prefixTree: RoseTree<char*bool>[], depth: int, lastTerm: int) = 
            match text with
            | [] -> lastTerm
            | head::tail -> 
                match prefixTree |> Array.tryFind (function | Node((c,_),_) -> head = c) with
                | None -> lastTerm
                | Some(Node((_,term),xs)) -> 
                    let depth = depth + 1
                    f(tail,xs,depth,if term then depth else lastTerm)

        f(text.ToCharArray() |> Array.toList, prefixTree,0,0)

    let prefixTree = vocab.Keys |> Seq.toArray |> makePrefixTree

    member _.Vocab = vocab
    member _.Unk_token = unk_token
    member _.Max_input_chars_per_word = max_input_chars_per_word
    //<summary>Tokenizes a piece of text into its word pieces.
    //This uses a greedy longest-match-first algorithm to perform tokenization
    //using the given vocabulary.
    //For example:
    //  input = "unaffable"
    //  output = ["un", "##aff", "##able"]</summary>
    //<param name="text"> 
    // A single token or whitespace separated tokens. This should have
    //    already been passed through `BasicTokenizer. </param>
    //<returns> A list of wordpiece tokens.</returns>
    member _.tokenize(text: string) =
        [|
            for token in whitespace_tokenize(text) do
                if token.Length > max_input_chars_per_word then yield unk_token
                else 
                    let rec getSubtoken(token: string) =
                        if token = "##" 
                        then Some([||]) 
                        else
                            match findLongestMatch(token, prefixTree) with
                            | 0 -> if token.Length = 0 then Some([||]) else None
                            | n -> 
                                getSubtoken("##" + token.[n..]) 
                                |> Option.map (fun xs -> [|yield token.[0..n-1]; yield! xs|])
                    match getSubtoken(token) with
                    | None -> yield unk_token
                    | Some(xs) -> yield! xs
        |]

/// Runs end-to-end tokenization
type FullTokenizer(vocab_file: string, ?do_lower_case: bool) = 
    let do_lower_case = defaultArg do_lower_case true
    let vocab = load_vocab(vocab_file)
    let inv_vocab = dict [ for KeyValue(k,v) in vocab -> (v,k) ]
    let basic_tokenizer = BasicTokenizer(do_lower_case)
    let wordpiece_tokenizer = WordpieceTokenizer(vocab = vocab)

    member _.Vocab = vocab
    member _.InvVocab = inv_vocab
    member _.BasicTokenizer = basic_tokenizer
    member _.WordpieceTokenizer = wordpiece_tokenizer

    member _.tokenize(text: string) = 
        basic_tokenizer.tokenize(text) |> Array.collect wordpiece_tokenizer.tokenize

    member _.convert_tokens_to_ids(tokens) = convert_tokens_to_ids(vocab, tokens)
    
    member _.convert_ids_to_tokens(tokens) = convert_ids_to_tokens(inv_vocab, tokens)

    interface ITokenizer with
        member this.tokenize(text) = this.tokenize(text) 
        member this.convert_tokens_to_ids(tokens) = this.convert_tokens_to_ids(tokens) 

        /// Returns text encoded in a way suitable for print or logging
        /// Since all text is encoded utf-16 this just returns an identity
        member _.printable_text(text) = text
