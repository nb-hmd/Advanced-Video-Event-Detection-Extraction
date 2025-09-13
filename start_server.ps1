# Robust Streamlit Server Startup Script
# This script ensures reliable startup with error handling and restart capabilities

param(
    [switch]$Debug = $false,
    [int]$MaxRetries = 3,
    [int]$RetryDelay = 5
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Function to log messages with timestamp
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] [$Level] $Message" -ForegroundColor $(if($Level -eq "ERROR") {"Red"} elseif($Level -eq "WARN") {"Yellow"} else {"Green"})
}

# Function to check if port is available
function Test-Port {
    param([int]$Port = 8501)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Function to kill existing Streamlit processes
function Stop-StreamlitProcesses {
    Write-Log "Checking for existing Streamlit processes..."
    $processes = Get-Process | Where-Object {$_.ProcessName -like '*python*' -and $_.CommandLine -like '*streamlit*'} -ErrorAction SilentlyContinue
    if ($processes) {
        Write-Log "Found $($processes.Count) existing Streamlit process(es). Terminating..."
        $processes | Stop-Process -Force
        Start-Sleep -Seconds 2
    }
}

# Function to start Streamlit with proper error handling
function Start-StreamlitServer {
    param([bool]$DebugMode = $false)
    
    try {
        Write-Log "Activating virtual environment..."
        & .\venv\Scripts\Activate.ps1
        
        Write-Log "Adding FFmpeg to PATH..."
        $env:PATH += ";C:\ffmpeg\bin"
        
        Write-Log "Starting Streamlit server..."
        $logLevel = if ($DebugMode) { "debug" } else { "info" }
        
        # Start Streamlit with proper error handling
        $process = Start-Process -FilePath "python" -ArgumentList "-m", "streamlit", "run", "src/web/streamlit_app.py", "--logger.level", $logLevel, "--server.headless", "true" -PassThru -NoNewWindow
        
        # Wait for server to start
        Write-Log "Waiting for server to start..."
        $timeout = 30
        $elapsed = 0
        
        while ($elapsed -lt $timeout) {
            if (Test-Port -Port 8501) {
                Write-Log "Streamlit server started successfully at http://localhost:8501"
                return $process
            }
            Start-Sleep -Seconds 1
            $elapsed++
        }
        
        throw "Server failed to start within $timeout seconds"
        
    } catch {
        Write-Log "Error starting Streamlit server: $($_.Exception.Message)" "ERROR"
        throw
    }
}

# Main execution
try {
    Write-Log "=== Streamlit Server Startup Script ==="
    Write-Log "Debug Mode: $Debug"
    Write-Log "Max Retries: $MaxRetries"
    
    # Stop any existing processes
    Stop-StreamlitProcesses
    
    # Check if port is already in use
    if (Test-Port -Port 8501) {
        Write-Log "Port 8501 is already in use. Attempting to free it..." "WARN"
        Stop-StreamlitProcesses
        Start-Sleep -Seconds 3
    }
    
    # Attempt to start server with retries
    $attempt = 1
    $success = $false
    
    while ($attempt -le $MaxRetries -and -not $success) {
        try {
            Write-Log "Startup attempt $attempt of $MaxRetries"
            $serverProcess = Start-StreamlitServer -DebugMode $Debug
            $success = $true
            
            Write-Log "Server startup successful!" "INFO"
            Write-Log "Access your application at: http://localhost:8501"
            Write-Log "Press Ctrl+C to stop the server"
            
            # Keep script running and monitor server
            while ($serverProcess -and -not $serverProcess.HasExited) {
                Start-Sleep -Seconds 5
                if (-not (Test-Port -Port 8501)) {
                    Write-Log "Server appears to have stopped unexpectedly" "WARN"
                    break
                }
            }
            
        } catch {
            Write-Log "Startup attempt $attempt failed: $($_.Exception.Message)" "ERROR"
            $attempt++
            
            if ($attempt -le $MaxRetries) {
                Write-Log "Retrying in $RetryDelay seconds..." "WARN"
                Start-Sleep -Seconds $RetryDelay
            }
        }
    }
    
    if (-not $success) {
        Write-Log "Failed to start server after $MaxRetries attempts" "ERROR"
        exit 1
    }
    
} catch {
    Write-Log "Critical error: $($_.Exception.Message)" "ERROR"
    exit 1
} finally {
    Write-Log "Cleaning up..."
    Stop-StreamlitProcesses
}