Current source version of tensorflow is 1.11

TODO
 1) Record Reader / Checkpoint Reader
 2) File System Shim
 3) Custom F# Ops
 4) TF_Server functions
 5) Missing header files

The record reader is needed for loading checkpoints, events, and other records. This is the initial reason for setting up this native wrapper. 

The File System Shim may not be needed as .Net already has functions for reading/writing/memory mapping etc. Tensorflow does have an environment context which has the location of files and relative location of files built in. This will probably be needed further down the line.

Custom F# Ops will require some native shim to work. As this project managed calling into native the mechanism for this is usually Reverse P/Invoke or callback delegates. There is also UnmanagedExports using the DllExport attribute. Assuming Callbacks are sufficient  there will be no need for using UnmanagedExports. A prerequisite for this is the ability to load custom ops from native libraries. This is a function we haven't gotten to yet.

Server Functions appears to be fairly straight forward and will probably not even need a native shim as the functions accessible from P/Invoke. That said the pre-built native tensorflowlib version 1.11 does not appear to have these functions. This may be a build option. Will require further examination.

There are numerous header files that are not in tensorflow repository as first checked out, they might be added by bazel. We may not need these.


Files that I expect to port
 * checkpoint_reader.cc
 * checkpoint_reader.h
 * exception.h
 * file_io.cc
 * file_io.h
 * record_reader.cc
 * record_reader.h
 * server.cc
 * server.h
 * utilities.h

NOTE: CMake for Visual Studio does not seem to support include_directory(..). CppProperties.json is supposed to work around this. I have not got this to work yet.
