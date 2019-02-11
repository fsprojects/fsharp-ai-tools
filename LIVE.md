


git clone https://github.com/fsproject/FSharp.Compiler.PortaCode
cd FSharp.Compiler.PortaCode
dotnet test
dotnet pack

dotnet pack c:\GitHub\dsyme\FSharp.Compiler.PortaCode\FsLive.Cli\FsLive.Cli.fsproj
dotnet tool uninstall --global fslive-cli
dotnet tool install --global --add-source c:\GitHub\dsyme\FSharp.Compiler.PortaCode\FsLive.Cli\nupkg fslive-cli
