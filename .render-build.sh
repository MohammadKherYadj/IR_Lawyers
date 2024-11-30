#!/bin/bash
apt-get update
apt-get install -y openjdk-22-jdk
export JAVA_HOME=/usr/lib/jvm/java-22-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
pip install -r requirements.txt
