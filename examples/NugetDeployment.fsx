// This will likely be depricated with 
(*
#r "System.Xml.Linq"
#r "System.IO.Compression"

open System
open System.IO
open System.Net
open System.Windows
open System.Windows.Input
open System.Xml.Linq
open System.Xml

module Zip =
    open System.IO.Compression
    let createArchive(entries:(string*byte[])[]) =
        use ms = new MemoryStream()
        let zipArchive = new ZipArchive(ms,ZipArchiveMode.Create)
        for (name,data) in entries do
            let entry = zipArchive.CreateEntry(name)
            use str = entry.Open()
            str.Write(data,0,data.Length)
            str.Close()
        zipArchive.Dispose()
        ms.ToArray()


module String =
    let toBytes(str:string) = System.Text.Encoding.UTF8.GetBytes(str)
    let ofBytes(bytes:byte[]) = System.Text.Encoding.UTF8.GetString(bytes)
    let ofXNode(x:XNode) = x.ToString(SaveOptions.OmitDuplicateNamespaces)

// TODO change this to use computer secret
let nugetApiKey = "todo"

let postNuget(filename:string) (data:byte[]) =
    let req = HttpWebRequest.Create("https://www.nuget.org/api/v2/package")
    req.Method <- "PUT"
    let boundary = "---------------------------" + DateTime.Now.Ticks.ToString("x");
    let boundarybytes = System.Text.Encoding.ASCII.GetBytes("\r\n--" + boundary + "\r\n");
    req.ContentType <- "multipart/form-data; boundary=" + boundary
    req.Headers.["X-NuGet-ApiKey"] <- nugetApiKey
    let rs = req.GetRequestStream()
    let toBytes (str:string) = System.Text.Encoding.ASCII.GetBytes(str)
    let data =
        [|
            sprintf "--%s\r\n" boundary |> toBytes
            sprintf "Content-Disposition: form-data; name=\"%s\";
filename=\"%s\"\r\n" filename filename |> toBytes
            "Content-Type: application/octet-stream\r\n\r\n" |> toBytes
            data
            "\r\n" |> toBytes
            sprintf "--%s--" boundary |> toBytes
        |] |> Array.collect id
    rs.Write(data,0,data.Length)
    rs.Close()
    let resp = req.GetResponse()
    let respStream = resp.GetResponseStream()
    use ms = new MemoryStream()
    respStream.CopyTo(ms)
    System.Text.Encoding.ASCII.GetString(ms.ToArray())

#r "WindowsBase"
#r "Microsoft.CSharp"
#r "System.Security"
#r "System.ServiceModel"
#r "System.Runtime.Serialization"
#r "System.ComponentModel.DataAnnotations"
#r @"..\lib\Microsoft.Web.XmlTransform.dll"
#r @"..\lib\NuGet.Core.dll"

let name = "TensorFlow.FSharp"
let version = "0.0.0.1"
let description = "TensorFlow bindings for F#. Code ported from TensorFlowSharp (C#) and from TensorFlow (python)"
let releaseNotes = "Alpha"

let mm = NuGet.ManifestMetadata(
            Id = name,
            Version = version,
            Title = name,
            RequireLicenseAcceptance = false ,
            Description = description,
            Summary = description,
            ReleaseNotes = releaseNotes,
            ProjectUrl = "https://github.com/fsprojects/TensorFlow.FSharp",
            Copyright = "Microsoft and contributors",
            Authors = "Microsoft",
            Owners = "moloneymb, dsyme",
            Tags = "fsharp tensorflow machine deep learning "
    )


let pb = NuGet.PackageBuilder()

type MPackageFile(
                    effectivePath:string, path:string,
                    targetFramework:Runtime.Versioning.FrameworkName,
                    supportedFrameworks:Runtime.Versioning.FrameworkName seq,
                    data:byte[]) =
    interface NuGet.IPackageFile  with
        member this.EffectivePath = effectivePath
        member this.GetStream() = new MemoryStream(data) :> Stream
        member this.Path = path
        member this.TargetFramework = targetFramework
        member this.SupportedFrameworks = supportedFrameworks

let net40 = Runtime.Versioning.FrameworkName(".NETFramework",new
System.Version(4,7))


let files = ["FCell.ManagedXll.dll"; "FCell.Interop.dll";
"Desaware.MachineLicense40.DLL"]

sprintf @"C:\Users\moloneymb\AppData\Local\Statfactory\FCell\%s"


// HDF.PInvoke
// protobuf-net
// Google.Protobuf
let dependecies =  [ ("ionic.zlib","1.9.1.5"); ("HDF.PInvoke.NETStandard","1.10.200"); ("Argu","5.1.0");("Google.Protobuf","3.6.1"); ("protobuf-net","2.4.0")]

[|
    for file in ["TensorFlow.FSharp.dll";"TensorFlow.FSharp.XML"; "libtensorflow.dll";
"TensorFlow.FSharp.Proto.dll";"TensorFlow.FSharp.Proto.xml"]  ->
        MPackageFile(file,@"lib\net40\" + file, net40,[|net40|], File.ReadAllBytes(__SOURCE_DIRECTORY__ + sprintf "\\bin\\Debug\\%s"
file))
|] |> Seq.iter pb.Files.Add

//pb.DependencySets.Add(NuGet.PackageDependencySet())


pb.Populate(mm)
let ms = new MemoryStream()
pb.Save(ms)



ms.ToArray()
|> postNuget (name + version)


//ms.ToArray()
//pb.Files.Add(file)
//let zp = NuGet.ZipPackage(@"C:\Lz4.Net.zip")
//let xs = zp.GetFiles() |> Seq.toArray


//let filename = "MyPackageMatt123.1.0.3"
//let data = File.ReadAllBytes(@"C:\MyPackageMatt123.1.0.3.nupkg")

//postNuget(filename,data)

//let nugetXml =
//    let nuget = XNamespace.Get
"http://schemas.microsoft.com/packaging/2011/08/nuspec.xsd"
//    let xmlHead = """<?xml version="1.0" encoding="utf-8"?>"""
//    XElement(nuget + "package",
//        XElement(nuget + "metadata",
//            [|
//                "id","Hive.TypeProvider"
//                "version","1.0.0.1"
//                "title","Hive.TypeProvider"
//                "authors","moloneymb"
//                "owners", "moloneymb"
//                //"licenseUrl", "https://lz4net.codeplex.com/license"
//                //"projectUrl", "https://lz4net.codeplex.com/"
//                "requireLicenseAcceptance", "false"
//                "description", "Interactivly explore query your Hive database"
//                "summary", "Interactivly explore query your Hive database"
//                "releaseNotes", "Initial Upload"
//                "copyright", "moloneymb"
//                "tags", "fsharp typeprovider bigdata hive hadoop"
//            |] |> Array.map (fun (name,value) -> XElement(nuget +
//name, value))
//            )
//        ) |> String.ofXNode |> sprintf "%s\n%s" xmlHead


//open System.IO.Packaging
//
//let pkg = Package.Open(@"C:\lz4net.nupkg")
//pkg.GetRelationships()
//pkg.GetParts()
//
//let uri (str:string) = Uri(str,UriKind.Relative)
//
//let ms = new MemoryStream()
//let zp = ZipPackage.Open(ms, FileMode.CreateNew)
//zp.CreateRelationship(uri("/lz4net.nuspec"),TargetMode.Internal,
//"http://schemas.microsoft.com/packaging/2010/07/manifest")
//zp.CreateRelationship(uri("/package/services/metadata/core-properties/e5d13f30ad0746ff8b295bba93550fb0.psmdcp"),TargetMode.Internal,
//"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties")
//
//let part = zp.CreatePart(uri "/lib/net40-client/LZ4.dll",
//"application/octet", CompressionOption.Maximum)
*)