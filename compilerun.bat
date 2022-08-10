@echo off

set SOURCEPATH=".\src\main\java"
set OUTPUTPATH=".\target\classes"
set MAINCLASS="Main"

javac -encoding UTF-8 -sourcepath %SOURCEPATH% -d %OUTPUTPATH% %SOURCEPATH%\Main.java
java -classpath %OUTPUTPATH% %MAINCLASS%
