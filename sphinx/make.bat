@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

REM Check if SPHINXBUILD variable is set, otherwise use the default value
if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

REM If no argument is given, display help
if "%1" == "" goto help

REM Check if the first argument is "githb", if so, set ACTION to "html" and COPYGITHB to 1
if "%1" == "github" (
    set ACTION=html
    set COPYGITHB=1
) else (
    set ACTION=%1
    set COPYGITHB=0
)

REM Check if sphinx-build is available, display an error message and exit if not found
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
    echo.installed, then set the SPHINXBUILD environment variable to point
    echo.to the full path of the 'sphinx-build' executable. Alternatively you
    echo.may add the Sphinx directory to PATH.
    echo.
    echo.If you don't have Sphinx installed, grab it from
    echo.http://sphinx-doc.org/
    exit /b 1
)

REM Run sphinx-build with the specified action and options
%SPHINXBUILD% -M %ACTION% %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

REM If COPYGITHB is set to 1, copy the contents of _build\html\ to ..\docs
if %COPYGITHB% == 1 (
    echo Copying _build\html\ to ..\docs
    xcopy /E /Y /I "%BUILDDIR%\html\*" "..\docs"
)

REM Jump to the end of the script
goto end

:help
REM Display help by running sphinx-build with the help action
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
REM Return to the original directory and exit the script
popd
