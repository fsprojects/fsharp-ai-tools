// This is to work around https://github.com/fsharp/fsharp/issues/794
//fsharpc LinuxNativeWorkaround.fs -o LinuxNativeWorkaround.dll
open System
open System.Runtime.InteropServices


[<Struct>]
type LLBuffer = {
    data             : IntPtr
    length           : IntPtr
    data_deallocator : IntPtr
}

let [<Literal>] libtensorflow = "libtensorflow"

module Native = 
    [<DllImport (libtensorflow)>]
    extern IntPtr TF_NewStatus ()

    [<DllImport (libtensorflow)>]
    extern void TF_DeleteStatus (IntPtr status);

    [<DllImport (libtensorflow)>]
    extern int TF_GetCode (IntPtr s);

    [<DllImport (libtensorflow)>]
    extern IntPtr TF_NewApiDefMap (IntPtr buffer, IntPtr status);

    [<DllImport (libtensorflow)>]
    extern void TF_DeleteApiDefMap (IntPtr handle);

    [<DllImport (libtensorflow)>]
    extern void TF_ApiDefMapPut (IntPtr handle, string text, IntPtr textLen, IntPtr status);

    [<DllImport (libtensorflow)>]
    extern LLBuffer *TF_ApiDefMapGet (IntPtr handle, string name, IntPtr nameLen, IntPtr status);

    [<DllImport (libtensorflow)>]
    extern IntPtr TF_Version ();

    [<DllImport (libtensorflow)>]
    extern LLBuffer *TF_GetAllOpList ();



