@echo off

set SOURCEPATH=".\src\main\java"
set OUTPUTPATH=".\target\classes"
set MAINCLASS="Main"

java -classpath %OUTPUTPATH% %MAINCLASS%
