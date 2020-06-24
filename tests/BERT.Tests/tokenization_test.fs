module TokenizationTests

open System
open System.IO
open NUnit.Framework

let shouldEqual (msg: string) (v1: 'T) (v2: 'T) = 
    if v1 <> v2 then 
        Assert.Fail(sprintf "fail %s: expected %A, got %A" msg v1 v2)

[<Test>]
let ``test full tokenizer``() =
    let vocab_tokens = [| "[UNK]"; "[CLS]"; "[SEP]"; "want"; "##want"; "##ed"; "wa"; "un"; "runn"; "##ing"; "," |]
    let vocab_filename = IO.Path.GetTempFileName()
    File.WriteAllLines(vocab_filename, vocab_tokens)
    let tokenizer = Tokenization.FullTokenizer(vocab_filename)
    File.Delete(vocab_filename)
    let tokens = tokenizer.tokenize("UNwant\u00E9d,running")
    tokens |> shouldEqual "A" [|"un"; "##want"; "##ed"; ","; "runn"; "##ing"|]
    (tokenizer.convert_tokens_to_ids(tokens)) |> shouldEqual "B"  [|7; 4; 5; 10; 8; 9|]

[<Test>]
let ``test chinese``() = 
    let tokenizer = Tokenization.BasicTokenizer()
    (tokenizer.tokenize("ah\u535A\u63A8zz")) 
    |> shouldEqual "A"  [|"ah"; "\u535A"; "\u63A8"; "zz"|]

[<Test>]
let ``test basic tokenizer lower``() =
    let tokenizer = Tokenization.BasicTokenizer(do_lower_case = true)
    tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  ")
    |> shouldEqual "A" [|"hello"; "!"; "how"; "are"; "you"; "?"|]
    (tokenizer.tokenize("H\u00E9llo")) |> shouldEqual "B"  [|"hello"|]

[<Test>]
let ``test basic tokenizer no lower``() = 
    let tokenizer = Tokenization.BasicTokenizer(do_lower_case = false)
    (tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  "))
    |> shouldEqual "A" [|"HeLLo"; "!"; "how"; "Are"; "yoU"; "?"|]

[<Test>]
let ``test_wordpiece_tokenizer``() = 
    let vocab_tokens = [| "[UNK]"; "[CLS]"; "[SEP]"; "want"; "##want"; "##ed"; "wa"; "un"; "runn"; "##ing" |] 
    let vocab = vocab_tokens |> Array.mapi (fun i x -> (x,i)) |> dict
    let tokenizer = Tokenization.WordpieceTokenizer(vocab=vocab)
    (tokenizer.tokenize("")) |> shouldEqual "A"  [||]
    (tokenizer.tokenize("unwanted running")) 
    |> shouldEqual "B"  [|"un"; "##want"; "##ed"; "runn"; "##ing"|]
    (tokenizer.tokenize("unwantedX running")) 
    |> shouldEqual "C"  [|"[UNK]"; "runn"; "##ing"|]
    
[<Test>]
let ``test convert tokens to ids``() = 
    let vocab_tokens = [| "[UNK]"; "[CLS]"; "[SEP]"; "want"; "##want"; "##ed"; "wa"; "un"; "runn"; "##ing" |] 
    let vocab = vocab_tokens |> Array.mapi (fun i x -> (x,i)) |> dict
    (Tokenization.convert_tokens_to_ids(vocab, [|"un"; "##want"; "##ed"; "runn"; "##ing"|])) 
    |> shouldEqual "A"  [|7; 4; 5; 8; 9|]

[<Test>]
let ``test is whitespace``() = 
    for c in [|' '; '\t'; '\r'; '\n'; '\u00A0'|] do
        if not(Tokenization.is_whitespace(c)) then
            failwithf "%c is whitespace" c
    for c in [|'A'; '-'|] do
        if Tokenization.is_whitespace(c) then
            failwithf "%c is not whitespace" c

[<Test>]
let ``test is control``() = 
    if not(Tokenization.is_control('\u0005')) then
        failwith "the given character should be control"
    // TODO how to handle ? '\U0001F4A9'
    for c in [|'A'; ' '; '\t'; '\r'|] do
        if Tokenization.is_control(c) then
            failwith "given character is not a control" 

[<Test>]
let ``test is punctuation``() = 
    for c in [|'-'; '$'; '`'; '.'|] do  
        if not(Tokenization.is_punctuation(c)) then
            failwith "given character should be punctuation"
    for c in [|'A'; ' '|] do 
        if Tokenization.is_punctuation(c) then
            failwith "given character is not punctuation"
