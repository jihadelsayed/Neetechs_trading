param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("setup", "run-small", "test")]
    [string]$Task
)

switch ($Task) {
    "setup" {
        python -m pip install -r requirements.txt
    }
    "run-small" {
        python -m app --universe small --refresh-data false
    }
    "test" {
        pytest -q
    }
}
