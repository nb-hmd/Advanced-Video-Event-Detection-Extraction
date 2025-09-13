#!/usr/bin/env python3
"""
Robust Streamlit Server Startup Script
Provides reliable startup with error handling, monitoring, and automatic restart capabilities.
"""

import os
import sys
import time
import socket
import subprocess
import signal
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('streamlit_server.log')
    ]
)
logger = logging.getLogger(__name__)

class StreamlitServerManager:
    def __init__(self, port: int = 8501, max_retries: int = 3, retry_delay: int = 5):
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.process: Optional[subprocess.Popen] = None
        self.project_root = Path(__file__).parent
        
    def is_port_available(self) -> bool:
        """Check if the specified port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', self.port))
                return result != 0
        except Exception as e:
            logger.warning(f"Error checking port availability: {e}")
            return True
    
    def kill_existing_processes(self):
        """Kill any existing Streamlit processes."""
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                             capture_output=True, check=False)
            else:  # Unix-like
                subprocess.run(['pkill', '-f', 'streamlit'], 
                             capture_output=True, check=False)
            time.sleep(2)
            logger.info("Cleaned up existing processes")
        except Exception as e:
            logger.warning(f"Error cleaning up processes: {e}")
    
    def setup_environment(self):
        """Setup the environment for Streamlit."""
        try:
            # Add FFmpeg to PATH
            ffmpeg_path = "C:\\ffmpeg\\bin"
            if ffmpeg_path not in os.environ.get('PATH', ''):
                os.environ['PATH'] += f";{ffmpeg_path}"
                logger.info("Added FFmpeg to PATH")
            
            # Set working directory
            os.chdir(self.project_root)
            logger.info(f"Set working directory to: {self.project_root}")
            
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            raise
    
    def start_server(self, debug_mode: bool = False) -> subprocess.Popen:
        """Start the Streamlit server."""
        try:
            # Prepare command
            venv_python = self.project_root / "venv" / "Scripts" / "python.exe"
            if not venv_python.exists():
                venv_python = "python"  # Fallback to system python
            
            log_level = "debug" if debug_mode else "info"
            cmd = [
                str(venv_python), "-m", "streamlit", "run", 
                "src/web/streamlit_app.py",
                "--logger.level", log_level,
                "--server.headless", "true",
                "--server.port", str(self.port)
            ]
            
            logger.info(f"Starting Streamlit with command: {' '.join(cmd)}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Wait for server to start
            logger.info("Waiting for server to start...")
            timeout = 30
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if not self.is_port_available():
                    logger.info(f"Streamlit server started successfully at http://localhost:{self.port}")
                    return process
                
                # Check if process is still running
                if process.poll() is not None:
                    stdout, _ = process.communicate()
                    logger.error(f"Process exited early with code {process.returncode}")
                    logger.error(f"Output: {stdout}")
                    raise RuntimeError(f"Server process exited with code {process.returncode}")
                
                time.sleep(1)
            
            raise TimeoutError(f"Server failed to start within {timeout} seconds")
            
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            raise
    
    def monitor_server(self, process: subprocess.Popen):
        """Monitor the server and handle unexpected shutdowns."""
        logger.info("Monitoring server... Press Ctrl+C to stop")
        
        try:
            while True:
                # Check if process is still running
                if process.poll() is not None:
                    logger.warning("Server process has stopped unexpectedly")
                    break
                
                # Check if port is still accessible
                if self.is_port_available():
                    logger.warning("Server port is no longer accessible")
                    break
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            self.cleanup(process)
    
    def cleanup(self, process: Optional[subprocess.Popen] = None):
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        self.kill_existing_processes()
        logger.info("Cleanup completed")
    
    def run(self, debug_mode: bool = False):
        """Main execution method with retry logic."""
        logger.info("=== Streamlit Server Manager ===")
        logger.info(f"Port: {self.port}")
        logger.info(f"Debug Mode: {debug_mode}")
        logger.info(f"Max Retries: {self.max_retries}")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        
        try:
            # Setup environment
            self.setup_environment()
            
            # Clean up any existing processes
            self.kill_existing_processes()
            
            # Attempt to start server with retries
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(f"Startup attempt {attempt} of {self.max_retries}")
                    
                    if not self.is_port_available():
                        logger.warning(f"Port {self.port} is in use. Cleaning up...")
                        self.kill_existing_processes()
                        time.sleep(3)
                    
                    process = self.start_server(debug_mode)
                    self.process = process
                    
                    logger.info("Server startup successful!")
                    logger.info(f"Access your application at: http://localhost:{self.port}")
                    
                    # Monitor server
                    self.monitor_server(process)
                    break
                    
                except Exception as e:
                    logger.error(f"Startup attempt {attempt} failed: {e}")
                    
                    if attempt < self.max_retries:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Failed to start server after {self.max_retries} attempts")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Critical error: {e}")
            return False
        finally:
            self.cleanup(self.process)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Streamlit Server Manager')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--port', type=int, default=8501, help='Server port')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retry attempts')
    parser.add_argument('--retry-delay', type=int, default=5, help='Delay between retries')
    
    args = parser.parse_args()
    
    manager = StreamlitServerManager(
        port=args.port,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )
    
    success = manager.run(debug_mode=args.debug)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()