#!/bin/bash
apt-get update
apt-get install -y openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
pip install -r requirements.txt
