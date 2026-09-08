param([Parameter(Mandatory=$true)][string]$Path)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$signature = Get-AuthenticodeSignature -LiteralPath $Path
[ordered]@{
    status = [string]$signature.Status
    subject = [string]$signature.SignerCertificate.Subject
} | ConvertTo-Json -Compress