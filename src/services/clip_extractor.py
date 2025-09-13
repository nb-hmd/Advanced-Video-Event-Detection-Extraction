import ffmpeg
import uuid
from pathlib import Path
import os
import shutil
import tempfile
from typing import Optional
from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)

class ClipExtractor:
    def __init__(self):
        self.output_dir = settings.DATA_DIR / "clips"
        self.output_dir.mkdir(exist_ok=True)
        self._validate_ffmpeg()
    
    def _validate_ffmpeg(self) -> bool:
        """Validate FFmpeg installation and availability."""
        try:
            # Check if ffmpeg is available in PATH
            ffmpeg_path = shutil.which('ffmpeg')
            if ffmpeg_path:
                logger.info(f"FFmpeg found at: {ffmpeg_path}")
                return True
            else:
                logger.warning("FFmpeg not found in PATH")
                # Try common installation paths on Windows
                common_paths = [
                    r"C:\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
                ]
                
                for path in common_paths:
                    if os.path.exists(path):
                        logger.info(f"FFmpeg found at: {path}")
                        # Add to PATH for this session
                        os.environ['PATH'] = os.path.dirname(path) + os.pathsep + os.environ['PATH']
                        return True
                
                logger.error("FFmpeg not found. Please install FFmpeg and add it to PATH.")
                return False
                
        except Exception as e:
            logger.error(f"Error validating FFmpeg: {e}")
            return False
    
    def _validate_video_file(self, video_path: str) -> bool:
        """Validate that the video file exists and is accessible."""
        try:
            video_file = Path(video_path)
            
            if not video_file.exists():
                logger.error(f"Video file does not exist: {video_path}")
                return False
            
            if not video_file.is_file():
                logger.error(f"Path is not a file: {video_path}")
                return False
            
            # Check if file is readable
            try:
                with open(video_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            except Exception as e:
                logger.error(f"Cannot read video file {video_path}: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating video file {video_path}: {e}")
            return False
    
    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration using ffprobe."""
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            return None
    
    def extract_clip(self, video_path: str, start_time: float, end_time: float) -> str:
        """Extract video clip using ffmpeg with improved error handling."""
        try:
            # Validate inputs
            if not self._validate_video_file(video_path):
                raise ValueError(f"Invalid video file: {video_path}")
            
            if start_time < 0:
                logger.warning(f"Start time {start_time} is negative, setting to 0")
                start_time = 0
            
            if end_time <= start_time:
                logger.warning(f"End time {end_time} <= start time {start_time}, adjusting")
                end_time = start_time + 5.0  # Default 5 second clip
            
            # Get video duration to validate timestamps
            duration = self._get_video_duration(video_path)
            if duration:
                if start_time >= duration:
                    logger.warning(f"Start time {start_time} >= video duration {duration}, adjusting")
                    start_time = max(0, duration - 5.0)
                    end_time = duration
                elif end_time > duration:
                    logger.warning(f"End time {end_time} > video duration {duration}, adjusting")
                    end_time = duration
            
            # Generate unique filename
            clip_id = str(uuid.uuid4())
            output_path = self.output_dir / f"clip_{clip_id}.mp4"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Calculate duration
            clip_duration = end_time - start_time
            
            logger.info(f"Extracting clip from {start_time:.2f}s to {end_time:.2f}s (duration: {clip_duration:.2f}s)")
            
            # Extract clip using ffmpeg with error handling
            try:
                (
                    ffmpeg
                    .input(video_path, ss=start_time, t=clip_duration)
                    .output(
                        str(output_path), 
                        vcodec='libx264', 
                        acodec='aac',
                        preset='fast',  # Faster encoding
                        crf=23,  # Good quality/size balance
                        movflags='faststart'  # Web optimization
                    )
                    .overwrite_output()
                    .run(quiet=True, capture_stdout=True, capture_stderr=True)
                )
            except ffmpeg.Error as e:
                # Log FFmpeg error details
                stderr = e.stderr.decode() if e.stderr else "No error details"
                logger.error(f"FFmpeg error: {stderr}")
                
                # Try fallback with simpler parameters
                logger.info("Attempting fallback extraction with simpler parameters")
                try:
                    (
                        ffmpeg
                        .input(video_path, ss=start_time, t=clip_duration)
                        .output(str(output_path), c='copy')  # Copy streams without re-encoding
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                except ffmpeg.Error as fallback_error:
                    fallback_stderr = fallback_error.stderr.decode() if fallback_error.stderr else "No error details"
                    logger.error(f"Fallback extraction also failed: {fallback_stderr}")
                    raise Exception(f"Both primary and fallback extraction failed. Last error: {fallback_stderr}")
            
            # Verify output file was created
            if not output_path.exists():
                raise Exception(f"Output file was not created: {output_path}")
            
            if output_path.stat().st_size == 0:
                raise Exception(f"Output file is empty: {output_path}")
            
            logger.info(f"Successfully extracted clip: {output_path} ({output_path.stat().st_size} bytes)")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error extracting clip from {video_path} ({start_time}-{end_time}s): {e}")
            raise
    
    def extract_clip_with_padding(self, video_path: str, timestamp: float, duration: float = None) -> str:
        """Extract clip with padding around timestamp."""
        if duration is None:
            duration = settings.CLIP_DURATION
        
        start_time = max(0, timestamp - duration / 2)
        end_time = timestamp + duration / 2
        
        return self.extract_clip(video_path, start_time, end_time)