#!/bin/zsh

SOURCEPATH="./src/main/java"
OUTPUTPATH="./target/classes"

javac -sourcepath $SOURCEPATH -d $OUTPUTPATH $SOURCEPATH/Main.java
