#r "System.IO.Compression"
#r "System.Runtime.InteropServices.RuntimeInformation"
#r "netstandard"
open System
open System.IO.Compression
open System.IO
open System.Net
open System.Runtime.InteropServices


type OS = | Windows  | Linux | OSX

(*
let os = 
    if RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then Linux 
    elif RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then Windows
    elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then OSX
    else failwithf "Unsupported OS %s" RuntimeInformation.OSDescription
*)

let os = 
    let platformId = System.Environment.OSVersion.Platform
    if platformId = PlatformID.MacOSX then OSX
    elif platformId = PlatformID.Unix then Linux
    else Windows

let libPath = Path.Combine(__SOURCE_DIRECTORY__,"..","lib")
let nugetPath = Path.Combine(__SOURCE_DIRECTORY__,"..","nuget")

let downloadFile(url:string,path:string) =
    Directory.CreateDirectory(Path.GetDirectoryName(path)) |> ignore
    if not(File.Exists(path)) then
        use wc = new System.Net.WebClient()
        printfn "Downloading %s -> %s" url path
        wc.DownloadFile(url,path)
        printfn "Completed %s -> %s" url path
    else 
        printfn "File %s already exists"  path


let private tryExtractEntry(entry:ZipArchiveEntry,dest:string) =
    if not (File.Exists(dest)) then
            use entryStream = entry.Open()
            use entryFileStream = new FileStream(dest,FileMode.CreateNew)
            entryStream.CopyTo(entryFileStream)
    else
        printfn "File %s already exists" dest

let extractZipFile(zipFile:string,entries:(string*string)[]) =
    use fs = new FileStream(zipFile,FileMode.Open)
    use zip = new ZipArchive(fs,ZipArchiveMode.Read)
    let zipEntryNameMap = zip.Entries |> Seq.mapi (fun i e -> (e.FullName,i)) |> Map.ofSeq
    for (entry,dest) in entries do
        match zipEntryNameMap.TryFind(entry) with
        | None -> failwithf "file %s not found in nuget archive %s" entry dest 
        | Some(i) -> 
            let entry = zip.Entries.[i]
            tryExtractEntry(entry,dest)

let extractZipFileAll(zipFile:string,destDirectory:string) =
    if File.Exists(zipFile) then
        use fs = new FileStream(zipFile,FileMode.Open)
        use zip = new ZipArchive(fs,ZipArchiveMode.Read)
        Directory.CreateDirectory(destDirectory) |> ignore
        for entry in zip.Entries do
            let dest = Path.Combine(destDirectory, entry.FullName)
            tryExtractEntry(entry,dest)
    else 
        printfn "File %s does not exist" zipFile

let downloadAndExtractNugetFiles(nugetFiles:(string*string[])[]) =
    Directory.CreateDirectory(libPath) |> ignore
    Directory.CreateDirectory(nugetPath) |> ignore
    for (package,files) in nugetFiles do
        let url = sprintf "https://www.nuget.org/api/v2/package/%s" package
        let nugetFileName = package.Replace("/",".") + ".nupkg"
        let nugetFullFileName = Path.Combine(nugetPath,nugetFileName)
        downloadFile(url,nugetFullFileName)
        let archiveFiles = [|for file in files -> (file, Path.Combine(libPath,Path.GetFileName(file)))|]
        extractZipFile(nugetFullFileName,archiveFiles)


open System.Diagnostics

let runProcess  (name:string) (args:string)  =
        printfn "Running: %s %s" name args
        let psi = ProcessStartInfo()
        psi.FileName <- name
        psi.Arguments <- args
        psi.ErrorDialog <- true
        psi.UseShellExecute <-false
        let p = new Process()
        p.StartInfo <- psi
        p.ErrorDataReceived.Add(fun e -> printfn "Error %s" e.Data)
        p.OutputDataReceived.Add(fun e -> printfn "Output %s" e.Data)
        p.Start() |> ignore
        p.WaitForExit()
        p.ExitCode



// TODO extend this to other versions of Visual Studio
let fsharpFolderCandidates = 
    [|
        yield! ["Community";"Professional"; "Enterprise"] 
        |> Seq.map (sprintf @"C:\Program Files (x86)\Microsoft Visual Studio\2017\%s\Common7\IDE\CommonExtensions\Microsoft\FSharp")
    |]

let private exeInDirCandiates exe candidates args = 
   candidates 
    |> Array.tryFind (fun x -> File.Exists(Path.Combine(x,exe)))
    |> function
    | None -> failwithf "%s not found" exe
    | Some(dir) ->
        runProcess (Path.Combine(dir, exe)) args

let runFSI (args:string) = exeInDirCandiates "fsiAnyCpu.exe" fsharpFolderCandidates args

let runFSC (args:string) = exeInDirCandiates "fsc.exe" fsharpFolderCandidates args

