$ErrorActionPreference = 'Stop'
$stage = Join-Path $PWD 'dist-windows'
$runtime = Join-Path $stage 'runtime'
if (Test-Path $stage) { throw 'Build output must start empty.' }
New-Item -ItemType Directory -Path $runtime | Out-Null
$base = (python -c "import sys;print(sys.base_prefix)").Trim()
# Copy the interpreter, standard library, native extensions and Tcl/Tk only.
foreach ($item in 'python.exe','pythonw.exe','LICENSE.txt','Lib','DLLs','tcl') {
    if ($item -eq 'Lib') {
        robocopy (Join-Path $base 'Lib') (Join-Path $runtime 'Lib') /E /XD site-packages __pycache__ test idlelib /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -gt 7) { throw 'Could not copy Python standard library.' }
    } else {
        Copy-Item (Join-Path $base $item) $runtime -Recurse
    }
}
Copy-Item (Join-Path $base '*.dll') $runtime
$python = Join-Path $runtime 'python.exe'
$site = Join-Path $runtime 'Lib/site-packages'
uv pip install --python $python --target $site .
if ($LASTEXITCODE) { throw 'Application installation failed.' }
# Library wheels are already installed. Remove generated commands containing build paths.
foreach ($subdir in 'Scripts','bin') {
    $scripts = Join-Path $site $subdir
    if (Test-Path $scripts) { Remove-Item -Recurse -Force $scripts }
}
Get-ChildItem $runtime -Recurse -Directory -Filter __pycache__ | Sort-Object FullName -Descending | Remove-Item -Recurse -Force
$csc = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
& $csc /nologo /target:winexe /platform:x64 /reference:System.Windows.Forms.dll "/out:$stage/QMT Desktop.exe" desktop/Launcher.cs
if ($LASTEXITCODE) { throw 'Desktop launcher compilation failed.' }
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
& $iscc packaging/windows.iss
if ($LASTEXITCODE) { throw 'Installer build failed.' }
