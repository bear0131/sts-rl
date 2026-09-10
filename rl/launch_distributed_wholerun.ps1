<#
.SYNOPSIS
    Prepare the trace-enabled simulator and launch the four-rank WholeRun PPO job.

.DESCRIPTION
    This is the single entry point for the current four-machine setup. It keeps
    remote Python/build paths, peer topology, rollout geometry, and detached
    logging in one place. Re-running with -Action status or -Action monitor is
    read-only; -Action start refuses to create a second process for the same run.

.EXAMPLE
    .\rl\launch_distributed_wholerun.ps1 -Action status -RunId ne44g
.EXAMPLE
    .\rl\launch_distributed_wholerun.ps1 -Action monitor -RunId ne44g
.EXAMPLE
    .\rl\launch_distributed_wholerun.ps1 -Action start -RunId ne44h
.EXAMPLE
    .\rl\launch_distributed_wholerun.ps1 -Action pause -RunId ne44h
.EXAMPLE
    .\rl\launch_distributed_wholerun.ps1 -Action resume -RunId ne44h
.EXAMPLE
    .\rl\launch_distributed_wholerun.ps1 -Action stop -RunId ne44h
#>

[CmdletBinding()]
param(
    [ValidateSet("start", "status", "monitor", "plan", "pause", "resume", "stop")]
    [string]$Action = "start",
    [ValidatePattern("^[A-Za-z0-9._-]+$")]
    [string]$RunId = "wrcv10r",
    [string]$Checkpoint = "/root/autodl-tmp/sts-rl/.tmp-wholerun-cardvec-random-step0.zip",
    [string]$CardVectorCheckpoint = "/root/autodl-tmp/sts-rl/.tmp-vector-rule-v7-contract-v10-step10000/last.pt",
    [string]$RemoteRepo = "/root/autodl-tmp/sts-rl",
    [int64]$TotalSteps = 10000000000,
    [int]$NEnvs = 128,
    [int]$NSteps = 2048,
    # The 8-layer whole-run policy can hit an occasional dynamic-width peak
    # at 16384 samples on a 32 GiB card. Keep the same global effective batch
    # with smaller forwards and eight accumulated micro-batches per rank.
    [int]$BatchSize = 8192,
    [int]$GradientAccumulationSteps = 8,
    [int]$NEpochs = 10,
    [double]$LearningRate = 1e-4,
    [switch]$HierarchicalCombatHp,
    # Keep one forward pass within the 32 GiB card limit.  The large number
    # of updates supplies the target reasoning compute budget; accumulation
    # raises the effective optimizer batch without increasing activation
    # memory or the single-forward footprint.
    [int]$ReasoningBatchSize = 1,
    # Keep the per-forward batch bounded, but train on enough microbatches
    # for reasoning to occupy roughly one third of each update.  Accumulation
    # keeps the extra optimizer synchronization from dominating that budget.
    [int]$ReasoningUpdates = 0,
    [int]$ReasoningGradientAccumulationSteps = 1,
    [double]$ReasoningCoef = 0.0,
    [int]$RingPort = 29763,
    [string]$WandbProject = "sts-rl",
    [string]$WandbName = "",
    # Keep one W&B run across checkpoint continuations unless explicitly
    # overridden for a deliberately new experiment.
    [string]$WandbId = "",
    [int]$MonitorSamples = 6,
    [int]$MonitorIntervalSeconds = 10,
    [int]$MinGpuUtil = 35,
    [switch]$ForceRestart
)

$ErrorActionPreference = "Stop"
$SshKey = Join-Path $env:USERPROFILE ".ssh\sts_dist_shared_ed25519"
# distributed_sb3 appends run_id/rank to this directory. Keep the base short
# so OpenSSH's AF_UNIX ControlPath remains below its 108-byte limit even when
# a descriptive W&B/run name is used.
$ControlDir = "/tmp/sts-launch"
$RemoteBuild = "$RemoteRepo/sim/sts_lightspeed/build-handonly-v4"
$RemotePython = "$RemoteRepo/.venv/bin/python"
$RemoteRoot = "$RemoteRepo/.tmp-dist-sb3-numberencoder-262144"
if ([string]::IsNullOrWhiteSpace($WandbName)) {
    $WandbName = "wholerun-cardvec-step10k-random-bs262144-e$NEpochs-n$NEnvs"
}
$RemoteLogDir = "$RemoteRepo/logs/$WandbName"
$RemoteCheckpointDir = "$RemoteRepo/checkpoints/$WandbName"

$Peers = @(
    [pscustomobject]@{ Rank = 0; Host = "connect.bjb2.seetacloud.com"; Port = 21496 },
    [pscustomobject]@{ Rank = 1; Host = "connect.bjb2.seetacloud.com"; Port = 20862 },
    [pscustomobject]@{ Rank = 2; Host = "connect.bjb2.seetacloud.com"; Port = 15557 },
    [pscustomobject]@{ Rank = 3; Host = "connect.bjb2.seetacloud.com"; Port = 46144
    }
)

if (-not (Test-Path -LiteralPath $SshKey)) {
    throw "SSH key not found: $SshKey"
}
if ($NEnvs * $NSteps -ne 262144) {
    throw "NEnvs * NSteps must remain 262144 per rank (got $($NEnvs * $NSteps))."
}
if (($NEnvs * $NSteps) % $BatchSize -ne 0) {
    throw "BatchSize must divide NEnvs * NSteps."
}
if (($NEnvs * $NSteps / $BatchSize) % $GradientAccumulationSteps -ne 0) {
    throw "GradientAccumulationSteps must divide the number of micro-batches."
}
if ($LearningRate -le 0) {
    throw "LearningRate must be positive."
}

function ConvertTo-BashQuote([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Invoke-RemoteRetry {
    param(
        [Parameter(Mandatory = $true)]$Peer,
        [Parameter(Mandatory = $true)][string]$Command,
        [int]$Attempts = 3
    )
    $sshArgs = @(
        "-i", $SshKey,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=15",
        "-p", [string]$Peer.Port,
        "root@$($Peer.Host)",
        $Command
    )
    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $output = & ssh @sshArgs 2>&1
            if ($LASTEXITCODE -ne 0) {
                $detail = ($output -join " ").Trim()
                if ($detail.Length -gt 300) { $detail = $detail.Substring(0, 300) }
                throw "ssh exited with code ${LASTEXITCODE}: $detail"
            }
            return ($output -join [Environment]::NewLine)
        } catch {
            if ($attempt -eq $Attempts) {
                throw "SSH failed for rank $($Peer.Rank) after $Attempts attempts: $($_.Exception.Message)"
            }
            Start-Sleep -Seconds ([Math]::Min(5 * $attempt, 15))
        }
    }
}

function Get-PrepareCommand {
    $repo = ConvertTo-BashQuote $RemoteRepo
    $python = ConvertTo-BashQuote $RemotePython
    $build = ConvertTo-BashQuote $RemoteBuild
    $root = ConvertTo-BashQuote $RemoteRoot
    $runRoot = ConvertTo-BashQuote "$RemoteRoot/$RunId"
    $logDir = ConvertTo-BashQuote $RemoteLogDir
    $checkpointDir = ConvertTo-BashQuote $RemoteCheckpointDir
    return @"
set -eu
repo=$repo
python=$python
build=$build
cd "`$repo"
test -x "`$python" || { echo "missing virtualenv Python: `$python" >&2; exit 2; }
needs_build=0
test -f "`$build/CMakeCache.txt" || needs_build=1
ls "`$build"/slaythespire*.so >/dev/null 2>&1 || needs_build=1
grep -q '^STS_ENABLE_RULE_TRACE:BOOL=ON$' "`$build/CMakeCache.txt" 2>/dev/null || needs_build=1
if [ "`$needs_build" -eq 1 ]; then
  generator_args=()
  if command -v ninja >/dev/null 2>&1; then
    generator_args=(-G Ninja)
  fi
  cmake -S sim/sts_lightspeed -B "`$build" "`${generator_args[@]}" \
    -DSTS_ENABLE_RULE_TRACE=ON \
    -DPYTHON_EXECUTABLE="`$python" \
    -DPython3_EXECUTABLE="`$python"
  cmake --build "`$build" --target slaythespire -j
fi
STS_ENABLE_RULE_TRACE=ON STS_LIGHTSPEED_BUILD_DIR="`$build" PYTHONPATH="`$build:`$repo" \
  "`$python" -c 'import slaythespire; print(1)'
mkdir -p $root $runRoot $logDir $checkpointDir
echo "prepared build=`$build python=`$python"
"@
}

function Get-PeerArguments($Rank) {
    $parts = @()
    foreach ($peer in $Peers) {
        if ($peer.Rank -ne $Rank) {
            $endpoint = "$($peer.Rank)=root@$($peer.Host):$($peer.Port)"
            $parts += "--peer $(ConvertTo-BashQuote $endpoint)"
        }
    }
    return ($parts -join " ")
}

function Get-TrainingCommand($Peer) {
    $repo = ConvertTo-BashQuote $RemoteRepo
    $python = ConvertTo-BashQuote $RemotePython
    $build = ConvertTo-BashQuote $RemoteBuild
    $root = ConvertTo-BashQuote $RemoteRoot
    $checkpoint = ConvertTo-BashQuote $Checkpoint
    $cardVectorCheckpoint = ConvertTo-BashQuote $CardVectorCheckpoint
    $logDir = ConvertTo-BashQuote $RemoteLogDir
    $checkpointDir = ConvertTo-BashQuote $RemoteCheckpointDir
    $wandbProject = ConvertTo-BashQuote $WandbProject
    $wandbName = ConvertTo-BashQuote $WandbName
    $wandbIdArg = ""
    if (-not [string]::IsNullOrWhiteSpace($WandbId)) {
        $wandbIdArg = "--wandb-id " + (ConvertTo-BashQuote $WandbId)
    }
    $hierarchicalCombatHpArg = ""
    if ($HierarchicalCombatHp) {
        $hierarchicalCombatHpArg = "--hierarchical-combat-hp"
    }
    $peerArgs = Get-PeerArguments $Peer.Rank
    $pidFile = ConvertTo-BashQuote "$RemoteLogDir/rank$($Peer.Rank).pid"
    $logFile = ConvertTo-BashQuote "$RemoteLogDir/rank$($Peer.Rank).log"
    return @"
set -eu
repo=$repo
cd "`$repo"
ulimit -n 65535 2>/dev/null || true
mkdir -p $logDir $checkpointDir $root/$RunId/rank$($Peer.Rank)
if [ -s $pidFile ] && kill -0 "`$(cat $pidFile)" 2>/dev/null; then
  echo "already running pid=`$(cat $pidFile)"
  exit 0
fi
cmd=(env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  STS_ENABLE_RULE_TRACE=ON STS_LIGHTSPEED_BUILD_DIR=$build PYTHONPATH=${build}:${repo} \
  $python -u -m rl.distributed_sb3 --rank $($Peer.Rank) --world-size 4 \
  --ssh-key /root/.ssh/sts_dist_shared_ed25519 --ssh-control-dir $ControlDir \
  --remote-root $root --run-id $RunId --checkpoint $checkpoint \
  --card-vector-checkpoint $cardVectorCheckpoint --no-interactive-reasoning \
  $hierarchicalCombatHpArg \
  --total-steps $TotalSteps --n-envs $NEnvs --n-steps $NSteps \
  --batch-size $BatchSize --gradient-accumulation-steps $GradientAccumulationSteps \
  --n-epochs $NEpochs --learning-rate $LearningRate \
  --amp-mode bf16 --gradient-transport ssh-ring \
  --ring-port $RingPort --reasoning-batch-size $ReasoningBatchSize \
  --reasoning-updates $ReasoningUpdates \
  --reasoning-gradient-accumulation-steps $ReasoningGradientAccumulationSteps \
  --reasoning-coef $ReasoningCoef \
  --wandb --wandb-project $wandbProject --wandb-name $wandbName $wandbIdArg \
  --save-every-updates 25 --checkpoint-dir $checkpointDir \
  --checkpoint-prefix hier_hp_ppo --max-checkpoints 2 \
  --barrier-timeout 900 --float32-matmul-precision high $peerArgs)
nohup "`${cmd[@]}" > $logFile 2>&1 < /dev/null &
pid=`$!
echo "`$pid" > $pidFile
echo "started pid=`$pid log=$logFile"
"@
}

function Get-StatusCommand {
    $logDir = ConvertTo-BashQuote $RemoteLogDir
    return @"
set +e
echo "--- processes ---"
for pid_file in $logDir/rank*.pid; do
  test -f "`$pid_file" || continue
  pid=`$(cat "`$pid_file")
  if kill -0 "`$pid" 2>/dev/null; then
    state=`$(awk '{print `$3}' /proc/`$pid/stat 2>/dev/null)
    if [ "`$state" = "T" ] || [ "`$state" = "t" ]; then
      echo "`$pid_file: paused pid=`$pid"
    else
      echo "`$pid_file: running pid=`$pid state=`$state"
    fi
  else
    echo "`$pid_file: stopped pid=`$pid"
  fi
done
echo "--- gpu ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || nvidia-smi -q -d UTILIZATION | grep -E 'GPU[[:space:]]+:|Avg[[:space:]]+:' | head -4
echo "--- log tail ---"
for log_file in $logDir/rank*.log; do
  test -f "`$log_file" || continue
  echo "`$log_file"
  tail -n 2 "`$log_file"
done
"@
}

function Get-StopCommand {
    $logDir = ConvertTo-BashQuote $RemoteLogDir
    $runId = ConvertTo-BashQuote $RunId
    return @"
set +e
log_dir=$logDir
run_id=$runId
for pid_file in "`$log_dir"/rank*.pid; do
  test -f "`$pid_file" || continue
  pid=`$(cat "`$pid_file" 2>/dev/null)
  case "`$pid" in
    ''|*[!0-9]*) echo "`$pid_file: invalid pid" >&2; rm -f "`$pid_file"; continue ;;
  esac
  if ! kill -0 "`$pid" 2>/dev/null; then
    echo "`$pid_file: stale pid=`$pid"
    rm -f "`$pid_file"
    continue
  fi
  cmdline=`$(tr '\0' ' ' < "/proc/`$pid/cmdline" 2>/dev/null)
  case "`$cmdline" in
    *rl.distributed_sb3*--run-id*"`$run_id"*) ;;
    *) echo "`$pid_file: refusing unverified pid=`$pid" >&2; continue ;;
  esac
  pgid=`$(ps -o pgid= -p "`$pid" 2>/dev/null | tr -d ' ')
  sid=`$(ps -o sid= -p "`$pid" 2>/dev/null | tr -d ' ')
  if [ -n "`$pgid" ] && [ "`$pgid" != "1" ] && [ "`$pgid" = "`$sid" ]; then
    echo "`$pid_file: stopping process-group pgid=`$pgid"
    kill -TERM -- -"`$pgid" 2>/dev/null || true
    sleep 2
    kill -KILL -- -"`$pgid" 2>/dev/null || true
  else
    echo "`$pid_file: stopping validated process tree pid=`$pid"
    # The normal nohup launch has one process group. This fallback handles
    # shells that detach workers into separate groups without broad pkill.
    stop_tree() {
      parent="`$1"
      for child in `$(pgrep -P "`$parent" 2>/dev/null); do
        stop_tree "`$child"
      done
      kill -TERM "`$parent" 2>/dev/null || true
    }
    stop_tree "`$pid"
    sleep 2
    kill -KILL "`$pid" 2>/dev/null || true
  fi
  rm -f "`$pid_file"
done
# A crashed rank can leave detached environment workers with no live PID file.
# Scan only this run's exact distributed command line; [r] avoids matching pgrep.
for candidate in `$(pgrep -f "[r]l.distributed_sb3.*--run-id.*`$run_id" 2>/dev/null); do
  [ "`$candidate" = "`$BASHPID" ] && continue
  [ -r "/proc/`$candidate/cmdline" ] || continue
  cmdline=`$(tr '\0' ' ' < "/proc/`$candidate/cmdline" 2>/dev/null)
  case "`$cmdline" in
    *rl.distributed_sb3*--run-id*"`$run_id"*) ;;
    *) continue ;;
  esac
  pgid=`$(ps -o pgid= -p "`$candidate" 2>/dev/null | tr -d ' ')
  sid=`$(ps -o sid= -p "`$candidate" 2>/dev/null | tr -d ' ')
  if [ -n "`$pgid" ] && [ "`$pgid" != "1" ] && [ "`$pgid" = "`$sid" ]; then
    echo "orphan candidate=`$candidate: stopping process-group pgid=`$pgid"
    kill -TERM -- -"`$pgid" 2>/dev/null || true
    sleep 1
    kill -KILL -- -"`$pgid" 2>/dev/null || true
  else
    echo "orphan candidate=`$candidate: stopping pid"
    kill -TERM "`$candidate" 2>/dev/null || true
    sleep 1
    kill -KILL "`$candidate" 2>/dev/null || true
  fi
done
# Ring transport clients can outlive a crashed rank. They do not contain the
# distributed command line, so clean only SSH clients using this run's control
# directory after the rank process cleanup above.
for candidate in `$(pgrep -f "[s]sh.*/tmp/sts-launch/`$run_id/" 2>/dev/null); do
  [ "`$candidate" = "`$BASHPID" ] && continue
  echo "stale ring ssh candidate=`$candidate: stopping"
  kill -TERM "`$candidate" 2>/dev/null || true
done
sleep 1
for candidate in `$(pgrep -f "[s]sh.*/tmp/sts-launch/`$run_id/" 2>/dev/null); do
  [ "`$candidate" = "`$BASHPID" ] && continue
  kill -KILL "`$candidate" 2>/dev/null || true
done
rm -rf "/tmp/sts-launch/`$run_id"
"@
}

if ($Action -eq "plan") {
    Write-Host "--- prepare command ---"
    Get-PrepareCommand | Write-Host
    foreach ($peer in $Peers) {
        Write-Host "--- rank $($peer.Rank) command ---"
        Get-TrainingCommand $peer | Write-Host
    }
    exit 0
}

if ($Action -eq "start") {
    Write-Host "Preparing trace-enabled simulator on $($Peers.Count) ranks..."
    foreach ($peer in $Peers) {
        $result = Invoke-RemoteRetry $peer (Get-PrepareCommand)
        Write-Host "rank $($peer.Rank): $result"
    }
    if ($ForceRestart) {
        Write-Host "Stopping only processes recorded for run $RunId..."
        foreach ($peer in $Peers) {
            Invoke-RemoteRetry $peer (Get-StopCommand) | Out-Host
        }
        Start-Sleep -Seconds 3
    }
    Write-Host "Launching run=$RunId n_envs=$NEnvs n_steps=$NSteps local_rollout=$($NEnvs * $NSteps) global_rollout=$($NEnvs * $NSteps * $Peers.Count) learning_rate=$LearningRate reasoning_updates=$ReasoningUpdates reasoning_acc=$ReasoningGradientAccumulationSteps..."
    foreach ($peer in $Peers) {
        $result = Invoke-RemoteRetry $peer (Get-TrainingCommand $peer)
        Write-Host "rank $($peer.Rank): $result"
    }
    Write-Host "Training launched. Use -Action monitor -RunId $RunId to sample GPU utilization."
    exit 0
}

if ($Action -eq "status") {
    foreach ($peer in $Peers) {
        Write-Host "===== rank $($peer.Rank) $($peer.Host):$($peer.Port) ====="
        Invoke-RemoteRetry $peer (Get-StatusCommand) | Write-Host
    }
    exit 0
}

if ($Action -eq "stop") {
    Write-Host "Stopping only validated processes for run $RunId (including environment workers)..."
    foreach ($peer in $Peers) {
        Write-Host "===== rank $($peer.Rank) stop ====="
        Invoke-RemoteRetry $peer (Get-StopCommand) | Write-Host
    }
    exit 0
}

if ($Action -eq "pause" -or $Action -eq "resume") {
    $signal = if ($Action -eq "pause") { "STOP" } else { "CONT" }
    $verb = if ($Action -eq "pause") { "paused" } else { "resumed" }
    foreach ($peer in $Peers) {
        Write-Host "===== rank $($peer.Rank) $verb ====="
        $logDir = ConvertTo-BashQuote $RemoteLogDir
        $command = @"
set +e
for pid_file in $logDir/rank*.pid; do
  test -f "`$pid_file" || continue
  pid=`$(cat "`$pid_file")
  if kill -0 "`$pid" 2>/dev/null; then
    pgid=`$(ps -o pgid= -p "`$pid" 2>/dev/null | tr -d ' ')
    sid=`$(ps -o sid= -p "`$pid" 2>/dev/null | tr -d ' ')
    target="`$pid"
    if [ -n "`$pgid" ] && [ "`$pgid" != "1" ] && [ "`$pgid" = "`$sid" ]; then
      target="-`$pgid"
    fi
    if kill -$signal -- "`$target" 2>/dev/null; then
      echo "`$pid_file: $verb process-group=`$target"
    else
      echo "`$pid_file: failed to $verb pid=`$pid" >&2
    fi
  else
    echo "`$pid_file: stopped pid=`$pid" >&2
  fi
done
"@
        Invoke-RemoteRetry $peer $command | Write-Host
    }
    exit 0
}

Write-Host "Monitoring $RunId for $MonitorSamples samples (interval $MonitorIntervalSeconds s)..."
$samples = @()
for ($sample = 1; $sample -le $MonitorSamples; $sample++) {
    $utilValues = @()
    foreach ($peer in $Peers) {
        $command = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null"
        $value = Invoke-RemoteRetry $peer $command
        $number = 0
        $raw = ($value -join "").Trim().Split(" ")[0]
        if ([int]::TryParse($raw, [ref]$number)) {
            $utilValues += $number
        }
        Write-Host "sample=$sample rank=$($peer.Rank) gpu=$($value -join ' ')"
    }
    if ($utilValues.Count -gt 0) {
        $average = ($utilValues | Measure-Object -Average).Average
        $samples += $average
        Write-Host ("sample=$sample average_gpu={0:N1}%" -f $average)
    }
    if ($sample -lt $MonitorSamples) {
        Start-Sleep -Seconds $MonitorIntervalSeconds
    }
}
if ($samples.Count -gt 0) {
    $overall = ($samples | Measure-Object -Average).Average
    Write-Host ("overall average_gpu={0:N1}% threshold={1}%" -f $overall, $MinGpuUtil)
    if ($overall -lt $MinGpuUtil) {
        if ($NEnvs -lt 512) {
            $nextEnvs = [Math]::Min($NEnvs * 2, 512)
            $nextSteps = [int](262144 / $nextEnvs)
            Write-Warning "GPU utilization remains low. Recommended geometry: -NEnvs $nextEnvs -NSteps $nextSteps (same 262144 transitions/rank); restart from the newest checkpoint."
        } else {
            Write-Warning "n_envs=512 is the tested process limit for this setup. Low point samples are expected while CPU environments are stepping; use rollout_steps_per_sec_local and the 14-second nvidia-smi average before increasing further."
        }
    } else {
        Write-Host "GPU utilization is above the configured threshold."
    }
}
