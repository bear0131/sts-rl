<#
.SYNOPSIS
    Preflight, start, inspect, and verify a four-rank NOSL replay run.

.DESCRIPTION
    This local Windows entry point invokes launch_nosl_replay.sh on four
    remote ranks. It never stores passwords; the configured SSH private key
    must already be authorized by the current machines.

.EXAMPLE
    .\experiments\launchers\manage_nosl_replay_4way.ps1 -Action status

.EXAMPLE
    .\experiments\launchers\manage_nosl_replay_4way.ps1 -Action start `
      -ResumeStep 2113536 -TargetStep 2162688 -WaitForTarget
#>

[CmdletBinding()]
param(
    [ValidateSet("plan", "preflight", "start", "status", "wait-update")]
    [string]$Action = "status",
    [ValidatePattern("^[A-Za-z0-9_-]+$")]
    [string]$RunId = "u40x4-20260915",
    [ValidatePattern("^[A-Za-z0-9_-]+$")]
    [string]$WandbId = "u40x4s15",
    [string]$WandbName = "u40x4-20260915",
    [string]$RemoteRepo = "/root/autodl-tmp/sts-rl",
    [string]$ArgsFile = "configs/train/nosl_paper_replay_fresh10_u40_4x_20260915.args",
    [string]$OverridesFile = "",
    [string]$Checkpoint = "migration/nosl-az/nosl_az_random_start_s512_boundary_step0.zip",
    [long]$ResumeStep = 0,
    [string]$ResumeRoot = "",
    [switch]$ForkResume,
    [switch]$PreserveForkEnvironments,
    [ValidatePattern("^[A-Za-z0-9_-]+$")]
    [string]$ResumeSourceRunId = "",
    [ValidatePattern("^[A-Za-z0-9_-]+$")]
    [string]$ResumeSourceWandbId = "",
    [long]$TargetStep = 0,
    [switch]$WaitForTarget,
    [int]$TimeoutSeconds = 1200,
    [int]$PollSeconds = 15,
    [int]$RingPort = 29789,
    [ValidateRange(2, 100)]
    [int]$MaxCheckpoints = 2,
    [ValidateRange(1, 1000000)]
    [int]$SaveEveryUpdates = 1,
    [int[]]$Ports = @(22324, 50418, 25377, 17011),
    # The accepted native build lives under the repository's runs directory.
    # Keep this explicit so a fresh launch cannot silently select a mutable
    # or unvalidated build-gumbel-* directory.
    [string[]]$BuildDirs = @(
        "../../runs/local/search-perf04-20260919/accepted-build-latest-rules",
        "../../runs/local/search-perf04-20260919/accepted-build-latest-rules",
        "../../runs/local/search-perf04-20260919/accepted-build-latest-rules",
        "../../runs/local/search-perf04-20260919/accepted-build-latest-rules"
    ),
    [string]$SshKey = "$env:USERPROFILE\.ssh\sts_dist_shared_ed25519"
)

$ErrorActionPreference = "Stop"
$HostName = "connect.bjb2.seetacloud.com"
$WorldSize = 4

if ($Ports.Count -ne $WorldSize) {
    throw "Ports must contain exactly $WorldSize entries."
}
if ($BuildDirs.Count -ne $WorldSize) {
    throw "BuildDirs must contain exactly $WorldSize entries."
}
if ($ResumeStep -lt 0 -or $TargetStep -lt 0) {
    throw "ResumeStep and TargetStep must be non-negative."
}
if ($ForkResume -and ($ResumeStep -le 0 -or -not $ResumeRoot -or -not $ResumeSourceRunId -or -not $ResumeSourceWandbId)) {
    throw "ForkResume requires ResumeStep, ResumeRoot, ResumeSourceRunId, and ResumeSourceWandbId."
}
if (-not $ForkResume -and $ResumeRoot) {
    throw "ResumeRoot is only supported with ForkResume."
}
if ($PreserveForkEnvironments -and -not $ForkResume) {
    throw "PreserveForkEnvironments requires ForkResume."
}
if ($TimeoutSeconds -le 0 -or $PollSeconds -le 0) {
    throw "TimeoutSeconds and PollSeconds must be positive."
}
if ($WaitForTarget -and $TargetStep -le 0) {
    throw "WaitForTarget requires a positive TargetStep."
}
if ($Action -eq "wait-update" -and $TargetStep -le 0) {
    throw "wait-update requires a positive TargetStep."
}
if (-not (Test-Path -LiteralPath $SshKey)) {
    throw "SSH key not found: $SshKey"
}
$SshExe = (Get-Command ssh -ErrorAction Stop).Source

$Peers = for ($rank = 0; $rank -lt $WorldSize; $rank++) {
    [pscustomobject]@{
        Rank = $rank
        Port = $Ports[$rank]
        BuildDir = $BuildDirs[$rank]
    }
}

function ConvertTo-BashQuote([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Resolve-RemotePath([string]$Path) {
    if ($Path.StartsWith("/")) {
        return $Path
    }
    return "$RemoteRepo/$Path"
}

function Invoke-Remote {
    param(
        [Parameter(Mandatory = $true)]$Peer,
        [Parameter(Mandatory = $true)][string]$Command,
        [int]$Attempts = 3
    )
    # PowerShell here-strings inherit this file's CRLF.  Normalize at the
    # transport boundary so Bash does not treat trailing CR as part of a
    # command (for example, `set +e` becomes an invalid `set +` option).
    $normalizedCommand = $Command -replace "`r`n", "`n" -replace "`r", "`n"
    $payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalizedCommand))
    $remoteCommand = "printf '%s' '$payload' | base64 -d | bash"
    $sshArgs = @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-p", [string]$Peer.Port,
        "root@$HostName",
        $remoteCommand
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        $output = (& $SshExe @sshArgs 2>&1 | Out-String).TrimEnd()
        if ($LASTEXITCODE -eq 0) {
            return $output
        }
        if ($attempt -eq $Attempts) {
            throw "rank $($Peer.Rank) SSH command failed after $Attempts attempts: $output"
        }
        Start-Sleep -Seconds ([Math]::Min(3 * $attempt, 9))
    }
}

function Get-PreflightCommand($Peer) {
    $repo = ConvertTo-BashQuote $RemoteRepo
    $args = ConvertTo-BashQuote (Resolve-RemotePath $ArgsFile)
    $overridesCheck = ""
    if ($OverridesFile) {
        $overrides = ConvertTo-BashQuote (Resolve-RemotePath $OverridesFile)
        $overridesCheck = "test -f $overrides"
    }
    $checkpoint = ConvertTo-BashQuote (Resolve-RemotePath $Checkpoint)
    $build = ConvertTo-BashQuote "$RemoteRepo/sim/sts_lightspeed/$($Peer.BuildDir)"
    $runDir = ConvertTo-BashQuote "$RemoteRepo/runs/$RunId"
    $recoveryBlock = @"
if find "`$run_dir/recovery/rank$($Peer.Rank)" -mindepth 2 -maxdepth 2 -name manifest.json -print -quit 2>/dev/null | grep -q .; then
  echo 'fresh launch refused: this run already has recovery generations' >&2
  exit 3
fi
"@
    if ($ResumeStep -gt 0) {
        $generation = "step-{0:D12}" -f $ResumeStep
        $sourceRunId = if ($ForkResume) { $ResumeSourceRunId } else { $RunId }
        $sourceWandbId = if ($ForkResume) { $ResumeSourceWandbId } else { $WandbId }
        $rawRecoveryPath = if ($ForkResume) {
            "$(Resolve-RemotePath $ResumeRoot)/rank$($Peer.Rank)/$generation"
        } else {
            "$RemoteRepo/runs/$RunId/recovery/rank$($Peer.Rank)/$generation"
        }
        $recoveryPath = ConvertTo-BashQuote $rawRecoveryPath
        $recoveryBlock = @"
generation=$recoveryPath
PYTHONPATH="`$repo" "`$python" - "`$generation" $($Peer.Rank) $WorldSize $(ConvertTo-BashQuote $sourceRunId) $(ConvertTo-BashQuote $sourceWandbId) $ResumeStep <<'PY'
import sys
from rl.training.checkpoints import read_bundle

path, rank, world_size, run_id, wandb_id, step = sys.argv[1:]
manifest = read_bundle(path, rank=int(rank), world_size=int(world_size))
expected = (run_id, wandb_id, int(step))
actual = (manifest['run_id'], manifest['wandb_id'], manifest['global_step'])
if actual != expected:
    raise SystemExit(f'recovery identity mismatch: expected={expected!r} actual={actual!r}')
print(f'recovery_ok step={manifest["global_step"]}')
PY
"@
    }
    return @"
set -euo pipefail
repo=$repo
args_file=$args
checkpoint=$checkpoint
build=$build
run_dir=$runDir
python="`$repo/.venv/bin/python"
launcher="`$repo/experiments/launchers/launch_nosl_replay.sh"
cd "`$repo"
test -x "`$python"
test -f "`$args_file"
$overridesCheck
test -f "`$checkpoint"
test -f "`$launcher"
compgen -G "`$build/slaythespire*.so" >/dev/null
test ! -e "`$run_dir/STOP" || { echo 'STOP exists' >&2; exit 3; }
if pgrep -f '[r]l.training.distributed_sb3' >/dev/null; then
  echo 'a distributed trainer is already running' >&2
  pgrep -af '[r]l.training.distributed_sb3' >&2
  exit 3
fi
$recoveryBlock
echo "preflight_ok rank=$($Peer.Rank) build=$($Peer.BuildDir) git=`$(git rev-parse --short HEAD 2>/dev/null || echo no-git)"
sha256sum "`$build"/slaythespire*.so | head -1
"@
}

function Get-PrepareTransportCommand($Peer) {
    $remoteRoot = ConvertTo-BashQuote "$RemoteRepo/.tmp-dist-sb3-$RunId"
    return @"
set -euo pipefail
remote_root=$remoteRoot
if pgrep -f '[r]l.training.distributed_sb3' >/dev/null 2>&1; then
  echo 'a distributed trainer started during preflight; refusing transport cleanup' >&2
  exit 3
fi
rm -rf -- "`$remote_root"
mkdir -p "`$remote_root"
echo "transport_root_prepared=`$remote_root"
"@
}

function Get-LaunchCommand($Peer) {
    $repo = ConvertTo-BashQuote $RemoteRepo
    $args = ConvertTo-BashQuote (Resolve-RemotePath $ArgsFile)
    $checkpoint = ConvertTo-BashQuote (Resolve-RemotePath $Checkpoint)
    $build = ConvertTo-BashQuote "$RemoteRepo/sim/sts_lightspeed/$($Peer.BuildDir)"
    $ports = ConvertTo-BashQuote ($Ports -join " ")
    $resumeLine = "unset NOSL_RESUME_STEP"
    if ($ResumeStep -gt 0) {
        $resumeLine = "export NOSL_RESUME_STEP=$ResumeStep"
    }
    $resumeRootLine = "unset NOSL_RESUME_ROOT"
    $forkResumeLine = "unset NOSL_FORK_RESUME"
    $preserveForkEnvironmentsLine = "unset NOSL_FORK_PRESERVE_ENVIRONMENTS"
    if ($ForkResume) {
        $resumeRootLine = "export NOSL_RESUME_ROOT=$(ConvertTo-BashQuote (Resolve-RemotePath $ResumeRoot))"
        $forkResumeLine = "export NOSL_FORK_RESUME=1"
        if ($PreserveForkEnvironments) {
            $preserveForkEnvironmentsLine = "export NOSL_FORK_PRESERVE_ENVIRONMENTS=1"
        }
    }
    $overridesLine = "unset NOSL_OVERRIDES_FILE"
    if ($OverridesFile) {
        $overridesLine = "export NOSL_OVERRIDES_FILE=$(ConvertTo-BashQuote (Resolve-RemotePath $OverridesFile))"
    }
    return @"
set -euo pipefail
repo=$repo
cd "`$repo"
# Windows worktrees can carry CRLF even though the remote runtime is Bash.
sed -i 's/\r$//' experiments/launchers/launch_nosl_replay.sh
export NOSL_RUN_ID=$(ConvertTo-BashQuote $RunId)
export NOSL_WANDB_ID=$(ConvertTo-BashQuote $WandbId)
export NOSL_RUN_NAME=$(ConvertTo-BashQuote $WandbName)
export NOSL_ARGS_FILE=$args
export NOSL_CHECKPOINT=$checkpoint
export NOSL_PEER_PORTS=$ports
export NOSL_RING_PORT=$RingPort
export NOSL_MAX_CHECKPOINTS=$MaxCheckpoints
export NOSL_SAVE_EVERY_UPDATES=$SaveEveryUpdates
# The manager cleans the shared transport root once for all ranks before
# launching. Per-rank cleanup would delete peer ready markers.
export NOSL_SKIP_TRANSPORT_CLEANUP=1
export STS_LIGHTSPEED_BUILD_DIR=$build
$resumeLine
$resumeRootLine
$forkResumeLine
$preserveForkEnvironmentsLine
$overridesLine
bash experiments/launchers/launch_nosl_replay.sh $($Peer.Rank)
"@
}

function Get-StatusCommand($Peer) {
    $runDir = ConvertTo-BashQuote "$RemoteRepo/runs/$RunId"
    return @"
set +e
run_dir=$runDir
pid=`$(cat "`$run_dir/rank$($Peer.Rank).pid" 2>/dev/null)
running=0
if [ -n "`$pid" ] && kill -0 "`$pid" 2>/dev/null; then
  cmdline=`$(tr '\0' ' ' < "/proc/`$pid/cmdline" 2>/dev/null)
  case "`$cmdline" in *rl.training.distributed_sb3*) running=1 ;; esac
  case "`$cmdline" in *"--run-id $RunId"*) ;; *) running=0 ;; esac
  case "`$cmdline" in *"--rank $($Peer.Rank)"*) ;; *) running=0 ;; esac
fi
latest=`$(cat "`$run_dir/recovery/rank$($Peer.Rank)/latest" 2>/dev/null)
echo "rank=$($Peer.Rank) running=`$running pid=`${pid:-none} latest=`${latest:-none}"
if [ "`$running" -eq 1 ]; then
  ps -o etime=,stat=,pcpu=,pmem= -p "`$pid"
fi
metric=`$(grep -E '^\{.*"global_timesteps":' "`$run_dir/rank$($Peer.Rank).log" 2>/dev/null | tail -n 1)
if [ -n "`$metric" ]; then
  step=`$(printf '%s\n' "`$metric" | grep -o '"global_timesteps": [0-9]*' | head -1 | tr -cd '0-9')
  wall=`$(printf '%s\n' "`$metric" | grep -o '"iteration_wall_sec": [0-9.]*' | head -1 | cut -d' ' -f2)
  replay=`$(printf '%s\n' "`$metric" | grep -o '"nosl_is_az/replay_size": [0-9.]*' | head -1 | cut -d' ' -f2)
  local_sps=`$(printf '%s\n' "`$metric" | grep -o '"rollout_steps_per_sec_local": [0-9.]*' | head -1 | cut -d' ' -f2)
  echo "metric_step=`${step:-none} wall_sec=`${wall:-none} replay_size=`${replay:-none} local_rollout_sps=`${local_sps:-none}"
fi
tail -n 24 "`$run_dir/rank$($Peer.Rank).log" 2>/dev/null | grep -Ev '^\{.*\}$' | tail -n 5
"@
}

function Show-Status {
    foreach ($peer in $Peers) {
        Write-Host "===== rank $($peer.Rank) $HostName`:$($peer.Port) ====="
        Write-Host (Invoke-Remote $peer (Get-StatusCommand $peer))
    }
}

function Wait-ForUpdate([long]$Step) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    do {
        $ready = 0
        foreach ($peer in $Peers) {
            $runDir = ConvertTo-BashQuote "$RemoteRepo/runs/$RunId"
            $command = @"
set +e
run_dir=$runDir
pid=`$(cat "`$run_dir/rank$($Peer.Rank).pid" 2>/dev/null)
running=0
if [ -n "`$pid" ] && kill -0 "`$pid" 2>/dev/null; then
  cmdline=`$(tr '\0' ' ' < "/proc/`$pid/cmdline" 2>/dev/null)
  case "`$cmdline" in *rl.training.distributed_sb3*) running=1 ;; esac
  case "`$cmdline" in *"--run-id $RunId"*) ;; *) running=0 ;; esac
  case "`$cmdline" in *"--rank $($Peer.Rank)"*) ;; *) running=0 ;; esac
fi
latest=`$(cat "`$run_dir/recovery/rank$($Peer.Rank)/latest" 2>/dev/null)
echo "running=`$running latest=`${latest:-none}"
"@
            $result = Invoke-Remote $peer $command
            Write-Host "rank=$($peer.Rank) $result"
            if ($result -notmatch 'running=1') {
                throw "rank $($peer.Rank) stopped before target step $Step"
            }
            if ($result -match 'latest=step-([0-9]+)' -and [long]$Matches[1] -ge $Step) {
                $ready++
            }
        }
        if ($ready -eq $WorldSize) {
            Write-Host "All ranks published a complete recovery generation at or beyond step $Step."
            return
        }
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "Timed out waiting for target step $Step; training was left running."
        }
        Start-Sleep -Seconds $PollSeconds
    } while ($true)
}

if ($Action -eq "plan") {
    Write-Host "run=$RunId wandb=$WandbId resume=$ResumeStep fork=$ForkResume preserve_fork_environments=$PreserveForkEnvironments resume_root=$ResumeRoot source_run=$ResumeSourceRunId source_wandb=$ResumeSourceWandbId target=$TargetStep args=$ArgsFile overrides=$OverridesFile max_checkpoints=$MaxCheckpoints save_every_updates=$SaveEveryUpdates"
    foreach ($peer in $Peers) {
        Write-Host "rank=$($peer.Rank) endpoint=$HostName`:$($peer.Port) build=$($peer.BuildDir)"
    }
    exit 0
}

if ($Action -eq "status") {
    Show-Status
    exit 0
}

if ($Action -eq "wait-update") {
    Wait-ForUpdate $TargetStep
    exit 0
}

foreach ($peer in $Peers) {
    Write-Host "===== rank $($peer.Rank) preflight ====="
    Write-Host (Invoke-Remote $peer (Get-PreflightCommand $peer))
}
if ($Action -eq "preflight") {
    exit 0
}

foreach ($peer in $Peers) {
    Write-Host "===== rank $($peer.Rank) transport cleanup ====="
    Write-Host (Invoke-Remote $peer (Get-PrepareTransportCommand $peer))
}

foreach ($peer in $Peers) {
    Write-Host "===== rank $($peer.Rank) launch ====="
    Write-Host (Invoke-Remote $peer (Get-LaunchCommand $peer))
}
Start-Sleep -Seconds 5
Show-Status
if ($WaitForTarget) {
    Wait-ForUpdate $TargetStep
}
