// TODO/WARN: Dataset loading does not work yet

// NOTES:
// * The need to use the static Model name is a bit verbose
// * The shape logic is simpler as they are determined entirely at runtime
// * The tensor lifetime is managed with a series of disposable tensors.

#if INTERACTIVE
#I __SOURCE_DIRECTORY__
#I "../../tests/bin/Debug/netcoreapp2.0/"
#r "TorchSharp.dll"
#endif
#if NOTEBOOK
#r "nuget: TODO"
#endif

open TorchSharp
open TorchSharp.Tensor
open TorchSharp.NN
open System.Diagnostics
open System.Collections.Generic

type Loss = TorchSharp.NN.LossFunction.Loss

type Model() as this=
    inherit Module()
    let conv1 = Model.conv2d(1L,10L,5L)
    let conv2 = Model.conv2d(10L,20L,5L)
    let fc1 = Model.Linear(320L,50L)
    let fc2 = Model.Linear(50L,10L)
    do
        this.RegisterModule(fc1)
        this.RegisterModule(fc2)
        this.RegisterModule(conv1)
        this.RegisterModule(conv2)

    override this.Forward(input: TorchTensor): TorchTensor =
        use l11 = conv1.Forward(input)
        use l12 = Model.MaxPool2D(l11,[|2L|])
        use l21 = conv2.Forward(l12)
        use l23 = Model.FeatureDropout(l21)
        use l24 = Model.FeatureDropout(l21)
        use x = l24.View([|-1L;320L|])
        use l31 = fc1.Forward(x)
        use l32 = Model.Relu(l31)
        use l33 = Model.Dropout(l32,this.IsTraining(),0.5)
        use l41 = fc2.Forward(l33)
        Model.LogSoftMax(l41,1L)

let Train(model: Module, 
          optimizer: Optimizer, 
          loss: Loss,
          dataLoader:  IEnumerable<struct (TorchTensor*TorchTensor)>,
          epoch: int,
          batchSize: int,
          size: int64) = 
    model.Train()
    let mutable batchId = 1
    for struct (data,target) in dataLoader do
        optimizer.ZeroGrad()
        use prediction = model.Forward(data)
        use output = loss.Invoke(prediction,target)
        output.Backward()
        optimizer.Step()
        if batchId % 10 = 0 then
            printfn "Train: epoch %i [%i % %i] Loss: %f" epoch (batchId * batchSize) (output.DataItem<float>())
        batchId <- batchId + 1
        data.Dispose()
        target.Dispose()

let Test(model: Module, loss: Loss, dataLoader: IEnumerable<struct (TorchTensor*TorchTensor)>, size: int64) =
    model.eval()
    let mutable testLoss = 0.1
    let mutable correct = 0
    for struct (data,target) in dataLoader do
        use prediction = model.Forward(data)
        use output = loss.Invoke(prediction,target)
        testLoss <- testLoss + 0.1 // TODO output.Item()
        let pred = output.Argmax(1L)
        correct <- correct + pred.Eq(target).Sum().DataItem<int>()
        data.Dispose()
        target.Dispose()
        pred.Dispose()
        printfn "Test set: Average loss %f | Accuracy %f" (testLoss / float size) (float correct / float size) 

Torch.SetSeed(0L)

let dataLocation = @"C:\EE\Data\MNist\"
let train = Data.Loader.MNIST(dataLocation,64L)
let test = Data.Loader.MNIST(dataLocation,1000L,isTrain = false)
let model = new Model()
let optimizer = NN.Optimizer.SGD(model.Parameters(), 0.01,0.5)
let sw = Stopwatch()
sw.Start()

for epoch in 1..10 do
    Train(model,optimizer,LossFunction.NLL(),train,epoch,64,train.Size())
    Test(model,LossFunction.NLL(reduction = NN.Reduction.Sum),test,test.Size())

#if COMPILED

let v = sprintf "running test in %s at %A" __SOURCE_FILE__ System.DateTime.Now
open NUnit.Framework
[<Test>]
let ``run test`` () = 
    v |> ignore
#endif


