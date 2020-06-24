# Protobuf compilation notes

The DLL is checked in so this step will not need to be automated at this time

// https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protoc-3.11.4-win64.zip
// protoc --csharp_out="csharp" onnx.proto3
//protoc --csharp_out="onnx\csharp" onnx\onnx-operators.proto3
// /r:Growl.Connector.dll,Growl.CoreLibrary.dll /out:test.exe
//C:\Users\moloneymb\.nuget\packages\google.protobuf\3.11.2\lib\netstandard2.0\Google.Protobuf.dll
//C:\Users\moloneymb\source\repos\ONNXBackend\onnx\csharp>csc.exe Onnx.cs OnnxMl.cs OnnxOperators.cs OnnxOperatorsMl.cs /r:Google.Protobuf.dll /r:netstandard.dll -target:library -out:OnnxProto.dll
// C:\Users\moloneymb\source\repos\ONNXBackend\onnx\csharp>csc.exe OnnxMl.cs OnnxOperatorsMl.cs /r:Google.Protobuf.dll /r:netstandard.dll -target:library -out:OnnxMLProto.dll
