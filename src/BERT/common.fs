module Common

open ICSharpCode.SharpZipLib.GZip
open ICSharpCode.SharpZipLib.Tar
open System.IO.Compression
open System.IO
open System

type OS = | Windows  | Linux | OSX

let os = 
    let platformId = System.Environment.OSVersion.Platform
    if platformId = PlatformID.MacOSX then OSX
    elif platformId = PlatformID.Unix then Linux
    else Windows

module WinNative = 
    open System.Runtime.InteropServices
    [<DllImport("Kernel32.dll")>]
    extern IntPtr GetModuleHandle(string lpModuleName);

    let IsDllLoaded(path : string) =
        GetModuleHandle(path) <> IntPtr.Zero

    [<DllImport("kernel32.dll")>]
    extern IntPtr LoadLibrary(string dllToLoad);

    [<DllImport("kernel32.dll")>]
    extern IntPtr GetProcAddress(IntPtr hModule, string procedureName);

    [<DllImport("kernel32.dll")>]
    extern bool FreeLibrary(IntPtr hModule);


/// TODO Find a better way to detect CUDA in a cross platform way
let UseGPU = 
    match os with
    | OSX -> false
    | Linux ->
        not(String.IsNullOrWhiteSpace(System.Environment.GetEnvironmentVariable("CUDA_HOME")))
    | Windows ->
        // This is untested and probably does not work
        WinNative.IsDllLoaded("cudart.dll") 
//        try 
//            // Try to load a cuda dll in windows
//        with
//        | :? DllNotFoundException -> false

let redistPackage = 
    if UseGPU then
        match os with 
        | Linux -> "scisharp.tensorflow.redist-linux-gpu","1.15.1"  
        | Windows -> "scisharp.tensorflow.redist-windows-gpu","1.14.1"  
        | OSX -> failwith "no available GPU redist at this time"
    else
        "scisharp.tensorflow.redist", "1.14.1"


let dir = Path.Combine(__SOURCE_DIRECTORY__, "..")
let data = Path.Combine(dir, "data")
let chkpt = Path.Combine(dir, "chkpt")
let home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
let nuget = Path.Combine(home,".nuget","packages")
let pretrainedVersion = "uncased_L-12_H-768_A-12"
let chkptUrl = sprintf "https://storage.googleapis.com/bert_models/2018_10_18/%s.zip" pretrainedVersion
let dataUrl = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz" 

let extractPackagesFromPaths(files : string[]) = 
  [| for file in files -> file.Split('\\') |> fun xs -> (xs.[0],xs.[1])|] |> Array.distinct


type System.IO.Compression.ZipArchive with
  member this.ExtractContents(folder : string) = 
      for entry in this.Entries do
        let t = Path.Combine(folder, entry.FullName)
        if not(entry.FullName.EndsWith("/")) then
          if not(Directory.Exists(t)) then
            Directory.CreateDirectory(Path.GetDirectoryName(t)) |> ignore
          use f = File.OpenWrite(t)
          use g = entry.Open()
          g.CopyTo(f)

let downloadPackages(packages : (string*string)[]) = 
  for package, version in packages do
    use wc = new System.Net.WebClient()
    let targetDir = Path.Combine(nuget,package,version)
    if Directory.Exists(targetDir) then
      printfn "Package %s\\%s found" package version
    else
      let url = sprintf "https://www.nuget.org/api/v2/package/%s/%s" package version
      printfn "Downloading %s\\%s" package version
      let data = wc.DownloadData(url)
      use za = new ZipArchive(new MemoryStream(data))
      za.ExtractContents(targetDir)

let fetchAndExtract(url : string, file : string, dir : string) =
    if not(File.Exists(file)) then
        use wc = new Net.WebClient()
        wc.DownloadFile(url,file) 
    use fs = File.OpenRead(file)
    match Path.GetExtension(file) with
    | ".gz" -> 
        use gs = new GZipInputStream(fs)
        match Path.GetExtension(Path.GetFileNameWithoutExtension(file)) with
        | ".tar" -> 
            use tarArchive = TarArchive.CreateInputTarArchive(gs)
            tarArchive.ExtractContents(dir)
        | x -> failwithf "unexpected extension %s for file %s" x file
    | ".zip" -> 
        use zipArchive = new System.IO.Compression.ZipArchive(fs)
        zipArchive.ExtractContents(dir)
    | x -> failwithf "unexpected extension %s for file %s" x file

let setup() = 
    printfn "Fetching Nuget Packages"
    [| redistPackage |] |> downloadPackages
    do
        printfn "Copy native dlls from redist nuget to tensorflow nuget"
        let tgtDir = Path.Combine(nuget,@"tensorflow.net","0.14.0","lib","netstandard2.0")
        let srcDir = Path.Combine(nuget,fst redistPackage ,snd redistPackage)

        let redistFiles = [|
            match os with
            | Linux -> 
                yield "runtimes/linux-x64/native/libtensorflow.so"
                yield "runtimes/linux-x64/native/libtensorflow_framework.so.1"
            | Windows -> yield "runtimes/win-x64/native/tensorflow.dll"
            | _ -> failwith "todo"
        |]

        for file in redistFiles do
            let tgtFile = Path.Combine(tgtDir,Path.GetFileName(file))
            if not(File.Exists(tgtFile)) then
                File.Copy(Path.Combine(srcDir,file),tgtFile)

    Directory.CreateDirectory(chkpt) |> ignore
    let chkptFile = Path.Combine(chkpt, pretrainedVersion + ".zip")
    if not(File.Exists(chkptFile)) then
        fetchAndExtract(chkptUrl,chkptFile,chkpt)
    Directory.CreateDirectory(data) |> ignore
    let dataFile = Path.Combine(data,"aclImdb_v1.tar.gz")
    if not(File.Exists(dataFile)) then
        fetchAndExtract(dataUrl,dataFile,data)

let vocab_file = Path.Combine(chkpt,pretrainedVersion, "vocab.txt")
let bert_config_file = Path.Combine(chkpt, pretrainedVersion, "bert_config.json")
let bert_chkpt = Path.Combine(chkpt, pretrainedVersion, "bert_model.ckpt")

