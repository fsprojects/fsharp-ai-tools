module TensorFlow.FSharp.Logging
// TODO introduce logging framework
// TODO make formatable with infof, vlogf, warningf etc.

open System

type LogLevel =
| FATAL
| ERROR
| WARN
| INFO
| DEBUG

let info(x:string) = Console.WriteLine(x)
let vlog(v:int,x:string) = Console.WriteLine(x)
let warning(x:string) = Console.WriteLine(x)
