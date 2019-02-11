module ImageWriter

// This is a lightweight library for saving bitmap files in order to view the output of image base machine learning
// The point is is have minimal dependencies

#r "netstandard"
#I __SOURCE_DIRECTORY__
#r "../../tests/bin/Debug/net461/Ionic.Zlib.Core.dll"
#r "../../tests/bin/Debug/net461/System.IO.Compression.dll"
#nowarn "9"

open System
open System.IO
open Ionic.Zlib

module Endian =
    module Little = 
        let writeInt(buf:byte[],offset:int,value:int) =
            buf.[offset+0] <- byte value
            buf.[offset+1] <- byte (value >>> 8)
            buf.[offset+2] <- byte (value >>> 16)
            buf.[offset+3] <- byte (value >>> 24)

    module Big =
        let writeInt(buf:byte[],offset:int,value:int) =
            buf.[offset+0] <- byte (value >>> 24)
            buf.[offset+1] <- byte (value >>> 16)
            buf.[offset+2] <- byte (value >>> 8)
            buf.[offset+3] <- byte value

        let readInt(buf:byte[],offset:int) =
            ((int32 buf.[offset+0]) <<< 24) |||
            ((int32 buf.[offset+1]) <<< 16) |||
            ((int32 buf.[offset+2]) <<<  8) |||
            (int32 buf.[offset+3])

/// Spec from https://en.wikipedia.org/wiki/BMP_file_format
/// NOTE: 
let RGBAToBitmap(height:int, width:int, pixels:int[]) = 
    let header =
        [|
            0x42; 0x4D;                 // BM                           ID field
            0x00; 0x00; 0x00; 0x00;     //                              Size of the BMP file
            0x00; 0x00;                 // Unused
            0x00; 0x00;                 // Unused 
            0x7A; 0x00; 0x00; 0x00      // 122 bytes (14 + 108)         Offset where the pixel arrah can be found                 
            0x6C; 0x00; 0x00; 0x00      // 108 bytes                    Number of bytes in the DIB header (from this point)
            0x00; 0x00; 0x00; 0x00      //                              Width of the bitmap in pixels
            0x00; 0x00; 0x00; 0x00      //                              Height of the bitmap in pixels
            0x01; 0x00;                 //                              Number of color panes being used
            0x20; 0x00;                 // 32 bits                      Number of bits per pixel
            0x03; 0x00; 0x00; 0x00      // 1 plane                      BI_BITFIELDS, no pixel array compression used
            0x20; 0x00; 0x00; 0x00      // 32 bits                      Size of the rab bitmap data (including padding)
            0x13; 0x0b; 0x00; 0x00      // 2835 pixels/metre horizontal Print resolution of the image,
            0x13; 0x0b; 0x00; 0x00      // 2835 pixels/metre vertical   72 DPI 
            0x00; 0x00; 0x00; 0x00      // 0 colors                     Number of colors in the palette
            0x00; 0x00; 0x00; 0x00      // 0 important colors           0 means all colors are important
            0x00; 0x00; 0xFF; 0x00      // 00FF0000  in big-endian      Red channel bit mask
            0x00; 0xFF; 0x00; 0x00      // 0000FF00  in big-endian      Green channel bit mask
            0xFF; 0x00; 0x00; 0x00      // 000000FF  in big-endian      Blue channel bit mask
            0x00; 0x00; 0x00; 0xFF      // FF000000  in big-endian      Alpha channel bit mask
            0x20; 0x6E; 0x69; 0x57      // little-ending "Win "         LCS_WINDOWS_COLOR_SPACE
        |] |> Array.map byte
    // NOTE: CIEXYZTRIPLE are unused for LCS "Win " and are initialized to zero
    let sizeBmpFile = 122 + pixels.Length * 4
    let buffer = Array.zeroCreate<byte> sizeBmpFile
    Buffer.BlockCopy(header,0,buffer,0,74)
    [sizeBmpFile, 0x02; width, 0x12; height, 0x16] |> Seq.iter (fun (i,offset) ->  do Endian.Little.writeInt(buffer,offset,i) )
    Buffer.BlockCopy(pixels,0,buffer,122,pixels.Length*4)
    buffer

// Spec from https://en.wikipedia.org/wiki/Portable_Network_Graphics
// and http://www.libpng.org/pub/png/spec/1.2/PNG-Compression.html
// Inspriation from https://gist.github.com/mmalex/908299/0b61f8a64842e413f030a3d8d46e253aa5808267
// NOTE: ZLib buffers generally starts with byte 120uy, the built in DeflateStream is a streaming version of zlib which is different



[<AutoOpen>]
module (*private*) PNG =
    open System.IO.Compression

    let crcTable : Lazy<uint32[]> =
        Lazy(fun () -> 
            [|
                for n in 0..255 do
                    let mutable c = uint32 n
                    for k in 0..7 do
                        if ((c &&& 1u) <> 0u) then c <- uint32 (0xedb88320L ^^^ ((int64 c) >>> 1)) else c  <- c >>> 1
                    yield c
           |]         )

    let getCrc(buf:byte[],start:int,finish:int) =
        let crcTable = crcTable.Value
        let mutable c = 0xffffffffu 
        for n in start..finish-1 do 
            c <- crcTable.[int(c ^^^ uint32 buf.[n] &&& 0xffu)] ^^^ (c >>> 8)
        c ^^^ 0xffffffffu 

    type ColorOption =
        | Indexed         = 0uy
        | Grayscale       = 2uy
        | GrayscaleAlphha = 3uy
        | Truecolor       = 4uy
        | TruecolorAlpha  = 6uy

    let colorOptionsToChannels(x:ColorOption) =
        match x with
        | ColorOption.Indexed
        | ColorOption.Grayscale       -> 1
        | ColorOption.GrayscaleAlphha -> 2
        | ColorOption.Truecolor       -> 3
        | ColorOption.TruecolorAlpha  -> 4
        | _ -> failwith "unsupported"


    module ChunkType =
        /// must be the first chunk; it contains (in this order) the image's width (4 bytes), height (4 bytes), 
        /// bit depth (1 byte), color type (1 byte), compression method (1 byte), 
        /// filter method (1 byte), and interlace method (1 byte) (13 data bytes total)
        let IHDR = 1229472850 // [|73uy; 72uy; 68uy; 82uy|]
        /// contains the palette; list of colors.
        /// This is essential for color type 3 and optional for types 2 and 6 and must not appear for color types 0 and 4
        let PLTE = 1347179589 // [|80uy; 76uy; 84uy; 69uy|]
        // contains the image, which may be split among multiple IDAT chunks. Such splitting increases filesize slightly, but makes it possible to generate a PNG in a streaming manner. 
        // The IDAT chunk contains the actual image data, which is the output stream of the compression algorithm.
        let IDAT = 1229209940 // [|73uy; 68uy; 65uy; 84uy|]
        /// marks the image end.
        let IEND = 1229278788 // [|73uy; 69uy; 78uy; 68uy|]       

        // TODO Ancillary chunks

    /// It is unfortunate that this is not built into .Net
    let decompress = ZlibStream.UncompressBuffer
    let compress   = ZlibStream.CompressBuffer


open FSharp.NativeInterop

/// This only supports TruecolorAlpha
let RGBAToPNG(height:int, width:int, pixels:int[]) : byte[] =
    let writeInt = Endian.Big.writeInt
    let readInt  = Endian.Big.readInt
    let pixelBytes =
        //generateRandomForTesting
        let bufPixels = Array.zeroCreate<byte> (pixels.Length * 4)
        Buffer.BlockCopy(pixels,0,bufPixels,0,bufPixels.Length)
        let bufOutput = Array.zeroCreate<byte> (pixels.Length * 4 + height)
        for h in 0..height - 1 do
            let spanOffsetIn  = h*width*4
            let spanOffsetOut = h*(width*4+1) + 1
            for w in 0..width - 1 do
                let outOffset = spanOffsetOut + w*4
                let inOffset  = spanOffsetIn + w*4
                bufOutput.[outOffset    ] <- bufPixels.[inOffset    ]
                bufOutput.[outOffset + 1] <- bufPixels.[inOffset + 1]
                bufOutput.[outOffset + 2] <- bufPixels.[inOffset + 2]
                bufOutput.[outOffset + 3] <- bufPixels.[inOffset + 3]
        bufOutput |> compress
    let buf : byte[] = Array.zeroCreate<byte> (pixelBytes.Length + 53)
    writeInt(buf,0,-1991225785); // Magic number "PNG" and line endings
    writeInt(buf,4,218765834); 
    writeInt(buf,8,13) // IHDR Length
    writeInt(buf,12,ChunkType.IHDR)
    writeInt(buf,16,width)
    writeInt(buf,20,height)
    buf.[24] <- 8uy // Bit Depth
    buf.[25] <- byte ColorOption.TruecolorAlpha // Color Type
    buf.[26] <- 0uy // Default Compression Method
    buf.[27] <- 0uy // None. 1uy is Paeth filtered
    buf.[28] <- 0uy // Inetlace method
    writeInt(buf,29,int32 <| getCrc(buf,12,29))
    writeInt(buf,33,pixelBytes.Length)
    writeInt(buf,37,ChunkType.IDAT) 
    Buffer.BlockCopy(pixelBytes,0,buf,41,pixelBytes.Length)
    let IDATCRCStart = pixelBytes.Length + 41
    writeInt(buf, IDATCRCStart, int32 <| getCrc(buf,37,IDATCRCStart))
    writeInt(buf,pixelBytes.Length + 45,ChunkType.IEND)
    writeInt(buf,pixelBytes.Length + 49,-1371381630) // CRC of IEND
    buf



