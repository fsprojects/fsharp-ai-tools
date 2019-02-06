module TensorFlow.FSharp.Platform.TFLogging

// NOTE: I'm not 100% on this design, but it's stubbed in in any case
// TODO: log counting... only if needed
// NOTE: AFAIK Verbosity can only be changed by nlog.config or builder.SetMinmumLevel
// TODO: Make us of Google2LogPrefix instead of NLog prefix??
// NOTE: If NLog is overkill then sub in commented out logger below

open System
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open NLog.Extensions.Logging // NOTE: An extension method is actually used from here...

/// Get id of current thread, suitable for logging as an unsinged qunatity.
let private getThreadId() = 
    // NOTE: Could use native process id instead?
    //Process.GetCurrentProcess().Threads  |> Array.tryFind (fun x -> x..ThreadState = "Running")
    System.Threading.Thread.CurrentThread.ManagedThreadId

/// Assemble a logline prefix using the google2 format.
let Google2LogPrefix(level : LogLevel, timestamp : DateTime option, fileAndLine : (string*int) option) : string =
    let now = timestamp |> Option.defaultWith (fun _ -> DateTime.Now)
    let nowMicrosecond = now.Ticks / (TimeSpan.TicksPerMillisecond / 1000L)
    failwith "todo - get file and line form stack"
    let (filename, line ) = failwith "todo"
    let basename = System.IO.Path.GetFileName(filename : string)
    let severity = (string level).Chars(0)

    sprintf "%c%2i%2i %2i:%2i:%2i.%6i %5i %s:%i]" 
        severity 
        now.Month 
        now.Day 
        now.Hour 
        now.Minute 
        now.Second 
        nowMicrosecond 
        (getThreadId()) 
        basename 
        line

type TFLogger(logger : ILogger<TFLogger>) =
    member this.Log(logLevel : LogLevel, msg : string) =
        logger.Log(logLevel, msg)
    member this.Error(msg : string, error:exn) =
        logger.LogError(error,msg)

let private logger = 
    lazy
        let serviceProvider = 
            ServiceCollection()
                .AddLogging(fun builder ->
                    builder.SetMinimumLevel(LogLevel.Trace) |> ignore
                    builder.AddNLog(new NLogProviderOptions(CaptureMessageTemplates = true, CaptureMessageProperties = true)) |> ignore )
                .AddTransient<TFLogger>()
                .BuildServiceProvider()
        let logger = serviceProvider.GetRequiredService<TFLogger>()
        AppDomain.CurrentDomain.ProcessExit.Add(fun _ -> NLog.LogManager.Shutdown())
        logger


let log(msg:string)       = logger.Force().Log(LogLevel.None,msg)
let debug(msg:string)     = logger.Force().Log(LogLevel.Debug,msg)
let error (msg:string) (error:exn)     = logger.Force().Error(msg,error)
let fatal(msg:string)     = logger.Force().Log(LogLevel.Critical,msg)
let info(msg:string)      = logger.Force().Log(LogLevel.Information,msg)
let warn(msg:string)      = logger.Force().Log(LogLevel.Warning,msg)
let warning(msg:string)   = warn(msg)


//
//[<AutoOpen>]
//module Logging =
//    open System
//
//    type Log =
//            | Log of string
//            | Warning of string
//            | Debug of string
//            | Error of string*exn
//           // | Catastrophe of int*string /// Escalation Level (higher value is lower level)
//
//    let mutable logger : Log -> unit = 
//        function
//        | Log(msg) -> System.Console.WriteLine(sprintf "Log: %s" msg)
//        | Warning(msg) -> System.Console.WriteLine(sprintf "Warning: %s" msg)
//        | Debug(msg) -> System.Diagnostics.Debug.WriteLine(msg)
//        | Error(msg,e) -> System.Console.Error.WriteLine(sprintf "%s %s" msg e.Message)
//        
//            let getColor =
//                function
//                | Log(_) -> ConsoleColor.Green
//                | Warning(_) -> ConsoleColor.Yellow
//                | Error(_) -> ConsoleColor.Red
//                | Catastrophe(_) -> ConsoleColor.Magenta
//
//            let agent = 
//                MailboxProcessor.Start(fun (inbox:MailboxProcessor<Log>) ->
//                    async {
//                        while true do
//                           let! msg = inbox.Receive()
//                           let color = getColor msg
//                           let text = 
//                               match msg with
//                               | Log(s) -> "Message: " + s
//                               | Warning(s) -> "Warning: " + s
//                               | Error(s,e,_) -> "Error: " + s + " " + e 
//                               | Catastrophe(l,s) -> "Catastrophe: " + l.ToString() + s
//                           Console.ForegroundColor <- color
//                           Console.WriteLine(text)
//                           Console.ResetColor()
//                           
//
//                    } )
//            fun msg -> agent.Post(msg)
//    
//    let log (message:string) = Log(message) |> logger
//    let warning (message:string) = Warning(message) |> logger
//    let error (message:string) (e:exn) = Error(message,e) |> logger
//    let debug (message:string) = Debug(message) |> logger
//    let writef<'a> msgFactory : Microsoft.FSharp.Core.Printf.StringFormat<'a, unit> -> 'a =
//        Microsoft.FSharp.Core.Printf.kprintf (fun message ->
//            msgFactory message |> logger)
//    let logf fmt = writef Log fmt
//    let warnf fmt = writef Warning fmt
//    let debugf fmt = writef Debug fmt
//
