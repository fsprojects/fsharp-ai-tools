FROM mcr.microsoft.com/dotnet/core/sdk:3.1-bionic

RUN apt-get update && apt-get install -y \
    python3-pip

RUN pip3 install jupyter jupyterlab
ENV PATH="$PATH:~/.local/bin"
ENV PATH="$PATH:/root/.dotnet/tools"
RUN dotnet tool install --global dotnet-try
RUN dotnet try jupyter install 

COPY /tools /tools
COPY /scripts /scripts

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

