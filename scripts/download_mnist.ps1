# Download and extract MNIST dataset into data\mnist\raw\ (Windows PowerShell).
# Uses .NET GZipStream to decompress since gunzip is not available on Windows.
# Skips files that already exist.
$ErrorActionPreference = "Stop"

$LogDir = "results\logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
Start-Transcript -Path "$LogDir\download_mnist.log" -Force

$BaseUrl = "https://yann.lecun.com/exdb/mnist"
$OutDir  = "data\mnist\raw"

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

# The four MNIST data files (train/test images and labels)
$Files = @(
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
)

foreach ($f in $Files) {
    $outFile   = Join-Path $OutDir $f
    $extracted = Join-Path $OutDir ($f -replace '\.gz$', '')

    if (Test-Path $extracted) {
        Write-Host "$extracted already exists, skipping."
        continue
    }

    Write-Host "Downloading $f ..."
    Invoke-WebRequest -Uri "$BaseUrl/$f" -OutFile $outFile

    # Decompress .gz using .NET GZipStream
    Write-Host "Extracting $f ..."
    $inStream  = [System.IO.File]::OpenRead($outFile)
    $gzip      = New-Object System.IO.Compression.GZipStream($inStream, [System.IO.Compression.CompressionMode]::Decompress)
    $outStream = [System.IO.File]::Create($extracted)
    $gzip.CopyTo($outStream)
    $outStream.Close()
    $gzip.Close()
    $inStream.Close()
    Remove-Item $outFile  # Clean up the .gz archive
}

Write-Host "MNIST download complete."

Stop-Transcript
