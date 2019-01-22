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
