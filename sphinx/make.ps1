# Sphinx build PowerShell script

# Set SPHINXBUILD variable if not set
if (!$env:SPHINXBUILD) {
    $env:SPHINXBUILD = 'sphinx-build'
}
$SOURCEDIR = "."
$BUILDDIR = "_build"

# If no argument is given, display help
if (!$args[0]) {
    & $env:SPHINXBUILD -M help $SOURCEDIR $BUILDDIR
    exit
}

# Check if the first argument is "githb", if so, set ACTION to "html" and COPYGITHB to 1
if ($args[0] -eq "github") {
    $ACTION = "html"
    $COPYGITHB = $true
} else {
    $ACTION = $args[0]
    $COPYGITHB = $false
}

# Check if sphinx-build is available, display an error message and exit if not found
try {
    & $env:SPHINXBUILD --version > $null 2>&1
} catch {
    Write-Host @"
    
    The 'sphinx-build' command was not found. Make sure you have Sphinx
    installed, then set the SPHINXBUILD environment variable to point
    to the full path of the 'sphinx-build' executable. Alternatively you
    may add the Sphinx directory to PATH.
    
    If you don't have Sphinx installed, grab it from
    http://sphinx-doc.org/
"@
    exit 1
}

# Run sphinx-build with the specified action and options
& $env:SPHINXBUILD -M $ACTION $SOURCEDIR $BUILDDIR

# If COPYGITHB is set to true, copy the contents of _build\html\ to ..\docs
if ($COPYGITHB) {
    Write-Host "Copying _build\html\ to ..\docs"
    Copy-Item -Recurse -Force "$BUILDDIR\html\*" "..\docs"
}
