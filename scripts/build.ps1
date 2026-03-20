param(
    [switch]$Dev
)

$ErrorActionPreference = 'Stop'

if ($Dev) {
    pip install -e .[dev]
} else {
    pip install -e .
}
