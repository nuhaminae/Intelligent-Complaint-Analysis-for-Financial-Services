# Windows PowerShell version (format.ps1)
# Run .\format.ps1 in bash
.\.chatvenv\Scripts\black . --exclude .chatvenv
.\.chatvenv\Scripts\isort . --skip .chatvenv
.\.chatvenv\Scripts\python.exe -m flake8 . --exclude=.chatvenv
if (Get-ChildItem -Recurse -Filter *.ipynb) {
  Write-Host "Formatting notebooks with black and isort via nbqa..."
  .\.chatvenv\Scripts\python.exe -m nbqa black . --line-length=88 --exclude .chatvenv
  .\.chatvenv\Scripts\python.exe -m nbqa isort . --skip .chatvenv
  Write-Host "Linting notebooks with flake8 via nbqa..."
  .\.chatvenv\Scripts\python.exe -m nbqa flake8 . --exclude .chatvenv --exit-zero
} else {
  Write-Host "No notebooks found - skipping nbQA."
}
