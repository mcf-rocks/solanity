FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y build-essential ocaml ocamlbuild automake autoconf libtool wget python libssl-dev libcurl4-openssl-dev protobuf-compiler libprotobuf-dev sudo kmod vim curl git-core libprotobuf-c0-dev libboost-thread-dev libboost-system-dev liblog4cpp5-dev libjsoncpp-dev alien uuid-dev libxml2-dev cmake pkg-config expect


RUN mkdir /root/sgx && mkdir /etc/init/ && \
    wget -O /root/sgx/sdk.bin https://download.01.org/intel-sgx/linux-2.3.1/ubuntu18.04/sgx_linux_x64_sdk_2.3.101.46683.bin && \
    wget -O /root/sgx/psw.deb https://download.01.org/intel-sgx/linux-2.3.1/ubuntu18.04/libsgx-enclave-common_2.3.101.46683-1_amd64.deb && \
    cd /root/sgx && \
    dpkg -i /root/sgx/psw.deb && \
    chmod +x /root/sgx/sdk.bin && \
    echo -e 'no\n/opt' | /root/sgx/sdk.bin && \
    echo 'source /opt/sgxsdk/environment' >> /root/.bashrc && \
    rm -rf /root/sgx/*

WORKDIR /root

