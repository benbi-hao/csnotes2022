@echo off

set SOURCEPATH=".\src\main\java"
set OUTPUTPATH=".\target\classes"

javac -encoding UTF-8 -sourcepath %SOURCEPATH% -d %OUTPUTPATH% %SOURCEPATH%\Main.java
