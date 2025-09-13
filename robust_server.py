#!/usr/bin/env python3
"""
Robust Streamlit Server with Auto-Restart and Memory Management
"""

import os
import sys
import time
import subprocess
import psutil
import signal
from pathlib import Path
import logging
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustStreamlitServer:
    """Robust Streamlit server with auto-restart and memory monitoring."""
    
    def __init__(self, app_path: str, port: int = 8501, max_retries: int = 5):
        self.app_path = app_path
        self.port = port
        self.max_retries = max_retries
        self.process: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.last_restart_time = 0
        self.min_restart_interval = 30  # Minimum 30 seconds between restarts
        
    def check_port_available(self) -> bool:
        """Check if the port is available."""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == self.port:
                    return False
            return True
        except Exception as e:
            logger.warning(f"Failed to check port availability: {e}")
            return True
    
    def kill_existing_processes(self) -> None:
        """Kill any existing processes on the port."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if ('streamlit' in cmdline.lower() and 
                        str(self.port) in cmdline):
                        logger.info(f"Killing existing Streamlit process: {proc.info['pid']}")
                        proc.kill()
                        proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
        except Exception as e:
            logger.warning(f"Failed to kill existing processes: {e}")
    
    def get_memory_info(self) -> dict:
        """Get current memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {'total_gb': 0, 'available_gb': 0, 'used_percent': 0}
    
    def check_memory_sufficient(self) -> bool:
        """Check if there's sufficient memory to start the server."""
        memory_info = self.get_memory_info()
        available_gb = memory_info['available_gb']
        
        # Require at least 1.5GB available memory for stable operation
        if available_gb < 1.5:
            logger.warning(f"Insufficient memory: {available_gb:.1f}GB available, need at least 1.5GB")
            return False
        
        logger.info(f"Memory check passed: {available_gb:.1f}GB available")
        return True
    
    def cleanup_memory(self) -> None:
        """Perform memory cleanup before starting server."""
        try:
            import gc
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            
            # Clear any cached models if possible
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared")
            except ImportError:
                pass
                
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")
    
    def start_server(self) -> bool:
        """Start the Streamlit server with robust error handling."""
        try:
            # Pre-flight checks
            if not self.check_memory_sufficient():
                logger.error("Insufficient memory to start server")
                return False
            
            # Cleanup before starting
            self.cleanup_memory()
            self.kill_existing_processes()
            time.sleep(2)  # Wait for cleanup
            
            # Prepare environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path.cwd())
            env['STREAMLIT_SERVER_HEADLESS'] = 'true'
            env['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
            env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
            
            # Start server with memory-optimized settings
            cmd = [
                sys.executable, '-m', 'streamlit', 'run',
                self.app_path,
                '--server.port', str(self.port),
                '--server.headless', 'true',
                '--logger.level', 'info',
                '--server.maxUploadSize', '200',  # Limit upload size
                '--server.maxMessageSize', '200'   # Limit message size
            ]
            
            logger.info(f"Starting Streamlit server: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait for server to start
            start_time = time.time()
            while time.time() - start_time < 60:  # 60 second timeout
                if self.process.poll() is not None:
                    # Process exited
                    output, _ = self.process.communicate()
                    logger.error(f"Server failed to start: {output}")
                    return False
                
                # Check if server is responding
                if not self.check_port_available():
                    logger.info(f"Server started successfully on port {self.port}")
                    return True
                
                time.sleep(1)
            
            logger.error("Server startup timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def monitor_server(self) -> None:
        """Monitor the server and restart if necessary."""
        logger.info("Starting server monitoring...")
        
        while self.restart_count < self.max_retries:
            try:
                # Start server
                if self.start_server():
                    logger.info(f"Server running (attempt {self.restart_count + 1}/{self.max_retries})")
                    
                    # Monitor server health
                    while True:
                        if self.process and self.process.poll() is not None:
                            # Server crashed
                            logger.warning("Server process terminated unexpectedly")
                            break
                        
                        # Check memory usage
                        memory_info = self.get_memory_info()
                        if memory_info['used_percent'] > 90:
                            logger.warning(f"High memory usage: {memory_info['used_percent']:.1f}%")
                        
                        time.sleep(10)  # Check every 10 seconds
                else:
                    logger.error("Failed to start server")
                
                # Server stopped, prepare for restart
                self.restart_count += 1
                current_time = time.time()
                
                if current_time - self.last_restart_time < self.min_restart_interval:
                    wait_time = self.min_restart_interval - (current_time - self.last_restart_time)
                    logger.info(f"Waiting {wait_time:.1f} seconds before restart...")
                    time.sleep(wait_time)
                
                self.last_restart_time = time.time()
                logger.info(f"Attempting restart {self.restart_count}/{self.max_retries}...")
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)
        
        logger.error(f"Maximum restart attempts ({self.max_retries}) reached")
    
    def stop_server(self) -> None:
        """Stop the server gracefully."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                logger.info("Server stopped gracefully")
            except subprocess.TimeoutExpired:
                self.process.kill()
                logger.info("Server force killed")
            except Exception as e:
                logger.error(f"Failed to stop server: {e}")

def main():
    """Main entry point."""
    app_path = "src/web/streamlit_app.py"
    
    if not Path(app_path).exists():
        logger.error(f"App file not found: {app_path}")
        sys.exit(1)
    
    server = RobustStreamlitServer(app_path, port=8501, max_retries=5)
    
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        server.stop_server()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.monitor_server()
    except Exception as e:
        logger.error(f"Server monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()