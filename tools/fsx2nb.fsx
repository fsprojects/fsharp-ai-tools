// TODO get #r nuget from project file

open System
open System.IO
open System.Text.RegularExpressions

let escapeAndQuote(txt: string) = 
    System.Web.HttpUtility.JavaScriptStringEncode(txt,true)

type Cell =
    {
        cell_type : string
        execution_count : int option
        metadata : string
        outputs : string[]
        source : string[]
    }
    static member Default = 
        {
            cell_type = "code"
            execution_count = None
            metadata = ""
            outputs = [||]
            source = [||]
        }

    override this.ToString() =
        sprintf """
          {
           "cell_type": %s,
           "metadata": {%s},
           %s
           "source": [%s]
          }""" (escapeAndQuote this.cell_type)
              
              this.metadata
              (if this.cell_type <> "code" then "" else
                  (sprintf """ "execution_count": %s, "outputs": [%s], """
                        (match this.execution_count with | None -> "null" | Some(x) -> string x) 
                        (this.outputs |> Array.map escapeAndQuote |> String.concat ",\n")))
              (this.source |> Array.map escapeAndQuote |> String.concat ",\n")

type Kernelspec = 
    {
        display_name : string
        language : string
        name : string
    }
    static member Default = {display_name = ".NET (F#)"; language = "F#"; name = ".net-fsharp"}
    override this.ToString() = 
        sprintf """{"display_name": %s, "language": %s, "name": %s}""" 
            (escapeAndQuote this.display_name) 
            (escapeAndQuote this.language) 
            (escapeAndQuote this.name)

type LanguageInfo = 
    {
        file_extension : string
        mimetype : string
        name : string
        pygments_lexer : string
        version : string
    }
    static member Default = 
        {file_extension = ".fs"; mimetype = "text/x-fsharp"; name = "C#"; pygments_lexer = "fsharp"; version = "4.5"}
    override this.ToString() = 
        sprintf """{
        "file_extension": %s,
        "mimetype": %s,
        "name": %s,
        "pygments_lexer": %s,
        "version": %s
        }""" 
            (escapeAndQuote this.file_extension)
            (escapeAndQuote this.mimetype)
            (escapeAndQuote this.name)
            (escapeAndQuote this.pygments_lexer)
            (escapeAndQuote this.version)

type Metadata = 
    {
        kernelspec : Kernelspec
        language_info : LanguageInfo
    }
    static member Default = 
        {kernelspec = Kernelspec.Default; language_info = LanguageInfo.Default}
    override this.ToString() =
        sprintf """{
            "kernelspec": %O,
            "langauge_info": %O
        }""" this.kernelspec this.language_info

type Notebook =
    {
        nbformat : int
        nbformat_minor : int
        metadata : Metadata
        cells : Cell[]
    }
    static member Default = 
        {nbformat = 4; nbformat_minor = 1; metadata = Metadata.Default; cells = [||]}
    override this.ToString() = 
        sprintf """
        {
            "cells": [%s],
            "metadata": %O,
            "nbformat": %i,
            "nbformat_minor": %i
        }
        """ 
            (this.cells |> Array.map string |> String.concat "\n,") 
            this.metadata this.nbformat this.nbformat_minor


module Array = 
    let filteri (f:int*'a->bool) (xs:'a[]) = 
        xs
        |> Array.mapi (fun i x -> (i,x))
        |> Array.filter f
        |> Array.map snd

type Entry =
    | StartSection
    | EndSection

type SectionType = 
    | Code
    | Cell
    | Markdown
    | Raw
    | Ydec

let linesToNotebook(lines: string[]) =
    // Strip simple lines, remove #if INTERACTIVE (or COMPILED) and clean up #if NOTEBOOK
    let text2 = 
        lines 
        |> Array.filter (fun x -> not(x.StartsWith("#time") || x.StartsWith("#nowarn"))) 
        |> Array.fold (fun ((isInter,isNote),acc) line -> 
            if isInter then
                if line.StartsWith("#endif") then ((false,false),acc)
                else ((isInter,isNote),acc)
            elif isNote then
                if line.StartsWith("#endif") then ((false,false),acc)
                else ((isInter,isNote),line::acc)
            else
                if line.StartsWith("#if INTERACTIVE") ||  line.StartsWith("#if COMPILED") 
                then ((true,false),acc)
                elif line.StartsWith("#if NOTEBOOK") then ((false,true),acc)
                else ((false,false),line::acc)
            ) ((false,false),[])
        |> fun (_,z) -> z
        |> List.toArray |> Array.rev |> String.concat "\n"

    let codeFromSource(src: string) = 
        {Cell.Default with cell_type = "code"; source = (src.Split([|'\n';'\r'|]) |> Array.map (sprintf "%s\n"))}

    // Handle (**... *) notation
    let sections = 
        [|
            yield Regex(@"^\(\*\*",RegexOptions.Multiline), StartSection
            yield Regex(@"\*\)", RegexOptions.Multiline), EndSection
        |] 
        |> Array.collect (fun (x,y) -> [|for m in x.Matches(text2) -> (y, m.Index, m.Length)|])
        |> Array.sortBy(fun (_,index,_) -> index)
        |> Array.fold (fun ((x,xIndex,xLength),acc) (y,yIndex,yLength)-> 
            if x = y then ((x,xIndex,xLength),acc)
            else ((y,yIndex,yLength),(y,(yIndex,yLength))::acc)
        ) ((EndSection,0,0),[])
        |> snd |> List.rev 
        |> List.toArray 
        |> Array.pairwise
        |> Array.filteri (fun (i,_) -> i % 2 = 0)
        |> Array.map (
            function 
            | (StartSection,(xIndex,xLength)),(EndSection,(yIndex,yLength)) ->  
                let substring = text2.Substring(xIndex + xLength)
                let sectionType,offset = 
                    if substring.StartsWith("cell") then Cell,4
                    elif substring.StartsWith("markdown") then Markdown,8
                    elif substring.StartsWith("raw") then Raw,4
                    elif substring.StartsWith("ydec") then Ydec,5
                    else failwithf "Section %s" (substring.Substring(0,100))
                (sectionType,xIndex,xLength + offset,yIndex,yLength) 
            | _ -> failwith "should not happen")
        |> Array.fold (fun (index,acc) (sectionType,xIndex,xLength,yIndex,yLength) ->
            (yIndex + yLength,(sectionType,xIndex,xLength,yIndex,yLength)::(Code,index,0,xIndex,0)::acc)
        ) (0,[]) |> snd |> List.rev |> List.toArray
        |> function 
        | [||] -> [|(Code,0,0,text2.Length,0)|]
        | xs -> 
            match xs |> Array.last with
            | (_,_,_,yIndex,yLength) -> [|yield! xs; yield (Code,yIndex+yLength,0,text2.Length,0)|]
        |> Array.map (fun (t,xIndex,xLength,yIndex,yLength) -> (t,text2.Substring(xIndex + xLength,yIndex - (xIndex + xLength))))

    // Merge ydec into code, filter out cell breaks, map into Cell type
    let cells = 
        sections
        |> Array.filter (fun (t,s) -> not (t = Code && System.String.IsNullOrWhiteSpace s))
        |> Array.map (fun (t,s) -> (t, s.Trim(' ', '\n', '\r')))
        |> Array.fold (fun (state,acc) (t,s) -> 
            let addOf(cellType: string) = 
                let x = {Cell.Default with cell_type = cellType; source = s.Split([|'\n';'\r'|])}
                match state with 
                | None -> x::acc 
                | Some(s1: string) -> x::(codeFromSource s1):: acc
            match t with
            | Ydec
            | Code -> (Some((match state with | None -> "" | Some(s) -> s) + "\n" + s),acc)
            | Cell -> 
                match state with 
                | None -> (None,acc)
                | Some(s1) -> (None,codeFromSource s1 :: acc)
            | Markdown -> (None,addOf("markdown"))
            | Raw -> (None,addOf("raw"))) (None,[])
        |> function | (None,xs) -> xs | (Some(s),xs) -> codeFromSource(s) :: xs
        |> List.rev |> List.toArray

    {Notebook.Default with cells = cells}


let cwd = System.Environment.CurrentDirectory
let getRooted(path: string) = if Path.IsPathRooted(path) then path else Path.Combine(cwd,path)


let input, output =
    match fsi.CommandLineArgs with
    | [|_;"-i";input;"-o";output|] ->
        let input = getRooted input
        let output = getRooted output
        (input, output)
    | [|_;"-i";input|] ->
        let input = getRooted input
        let output = Path.ChangeExtension(input, "ipynb")
        (input, output)
    | _ -> 
        printfn "Expected format of \"-i input [-o output]\". Input command line args were %s" (fsi.CommandLineArgs |> String.concat " ")
        exit 100

if not (input.EndsWith ".fsx" || input.EndsWith ".fs") then 
    eprintfn "Unknown input %s. Input should be .fsx or .fs" input
    exit 100
    
if not(File.Exists(input)) then 
    eprintfn "Input file %s does not exist" input
    exit 100
let lines = File.ReadAllLines(input)
let notebook = lines |> linesToNotebook
if File.Exists(output) then
    printfn "Overwriting %s..." output
else
    printfn "Writing %s..." output
File.WriteAllText(output,notebook.ToString())

