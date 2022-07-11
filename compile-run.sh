#!/bin/zsh

SOURCEPATH="./src/main/java"
OUTPUTPATH="./target/classes"
MAINCLASS="Main"

javac -sourcepath $SOURCEPATH -d $OUTPUTPATH $SOURCEPATH/Main.java
java -classpath $OUTPUTPATH $MAINCLASS

