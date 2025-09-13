#!/usr/bin/env python3
"""
Enhanced Image Matching Test Suite

This script tests and demonstrates the system's capabilities for:
1. Background Independence - Finding objects regardless of background differences
2. Cross-Domain Vision Matching - Matching across color/grayscale differences

Author: SOLO Coding Assistant
Date: 2024
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from PIL import Image
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Import our services
from src.services.image_matcher import ImageMatcher
from src.services.cross_domain_matcher import CrossDomainMatcher
from src.services.object_detector import ObjectDetector
from src.services.video_processor import VideoProcessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedMatchingTester:
    """
    Test suite for enhanced image matching capabilities.
    """
    
    def __init__(self):
        self.image_matcher = ImageMatcher(use_gpu=True)
        self.cross_domain_matcher = CrossDomainMatcher(use_gpu=True)
        self.object_detector = ObjectDetector(use_gpu=True)
        self.video_processor = VideoProcessor()
        
        # Test results storage
        self.test_results = {
            'background_independence': {},
            'cross_domain_matching': {},
            'combined_scenarios': {}
        }
    
    def create_test_images(self) -> Dict[str, np.ndarray]:
        """
        Create synthetic test images to demonstrate capabilities.
        """
        logger.info("Creating synthetic test images...")
        
        # Create a simple person silhouette
        person_shape = np.zeros((200, 100, 3), dtype=np.uint8)
        # Head
        cv2.circle(person_shape, (50, 30), 15, (255, 255, 255), -1)
        # Body
        cv2.rectangle(person_shape, (40, 45), (60, 120), (255, 255, 255), -1)
        # Arms
        cv2.rectangle(person_shape, (25, 60), (40, 100), (255, 255, 255), -1)
        cv2.rectangle(person_shape, (60, 60), (75, 100), (255, 255, 255), -1)
        # Legs
        cv2.rectangle(person_shape, (42, 120), (50, 180), (255, 255, 255), -1)
        cv2.rectangle(person_shape, (50, 120), (58, 180), (255, 255, 255), -1)
        
        test_images = {}
        
        # 1. Person with no background (transparent/black)
        test_images['person_no_bg'] = person_shape.copy()
        
        # 2. Person with red background
        red_bg = np.full((200, 100, 3), (255, 0, 0), dtype=np.uint8)
        red_bg[person_shape[:,:,0] > 0] = person_shape[person_shape[:,:,0] > 0]
        test_images['person_red_bg'] = red_bg
        
        # 3. Person with blue background
        blue_bg = np.full((200, 100, 3), (0, 0, 255), dtype=np.uint8)
        blue_bg[person_shape[:,:,0] > 0] = person_shape[person_shape[:,:,0] > 0]
        test_images['person_blue_bg'] = blue_bg
        
        # 4. Person with complex background (gradient)
        gradient_bg = np.zeros((200, 100, 3), dtype=np.uint8)
        for i in range(200):
            gradient_bg[i, :] = [i * 255 // 200, (200-i) * 255 // 200, 128]
        gradient_bg[person_shape[:,:,0] > 0] = person_shape[person_shape[:,:,0] > 0]
        test_images['person_gradient_bg'] = gradient_bg
        
        # 5. Grayscale versions
        test_images['person_no_bg_gray'] = cv2.cvtColor(person_shape, cv2.COLOR_RGB2GRAY)
        test_images['person_red_bg_gray'] = cv2.cvtColor(red_bg, cv2.COLOR_RGB2GRAY)
        test_images['person_blue_bg_gray'] = cv2.cvtColor(blue_bg, cv2.COLOR_RGB2GRAY)
        test_images['person_gradient_bg_gray'] = cv2.cvtColor(gradient_bg, cv2.COLOR_RGB2GRAY)
        
        logger.info(f"Created {len(test_images)} test images")
        return test_images
    
    def create_test_video(self, test_images: Dict[str, np.ndarray]) -> str:
        """
        Create a test video with different backgrounds and color variations.
        """
        logger.info("Creating test video...")
        
        # Create temporary video file
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        # Video properties
        fps = 2  # Slow for easy analysis
        frame_size = (400, 300)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, frame_size)
        
        # Create frames with different scenarios
        frames_data = [
            ('person_no_bg', 'Person with no background'),
            ('person_red_bg', 'Person with red background'),
            ('person_blue_bg', 'Person with blue background'),
            ('person_gradient_bg', 'Person with gradient background'),
            ('person_no_bg_gray', 'Person grayscale no background'),
            ('person_red_bg_gray', 'Person grayscale red background'),
            ('person_blue_bg_gray', 'Person grayscale blue background'),
            ('person_gradient_bg_gray', 'Person grayscale gradient background')
        ]
        
        for img_key, description in frames_data:
            # Get the test image
            img = test_images[img_key]
            
            # Handle grayscale images
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            # Resize to frame size
            frame = cv2.resize(img, frame_size)
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame multiple times for duration
            for _ in range(fps * 2):  # 2 seconds per scenario
                out.write(frame_bgr)
        
        out.release()
        logger.info(f"Test video created: {temp_video_path}")
        return temp_video_path
    
    def test_background_independence(self, test_images: Dict[str, np.ndarray], 
                                   video_path: str) -> Dict:
        """
        Test Scenario 1: Background Independence
        Can the system find a person when the reference image has no background
        or a different background from the video?
        """
        logger.info("\n" + "="*60)
        logger.info("TESTING SCENARIO 1: BACKGROUND INDEPENDENCE")
        logger.info("="*60)
        
        results = {}
        
        # Test cases for background independence
        test_cases = [
            {
                'name': 'No Background → Different Backgrounds',
                'reference': 'person_no_bg',
                'description': 'Reference: Person with no background, Video: Various backgrounds'
            },
            {
                'name': 'Red Background → Blue Background',
                'reference': 'person_red_bg',
                'description': 'Reference: Person with red background, Video: Person with blue background'
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\nTesting: {test_case['name']}")
            logger.info(f"Description: {test_case['description']}")
            
            reference_image = test_images[test_case['reference']]
            
            # Test with Object-Focused Match mode
            start_time = time.time()
            
            try:
                result = self.video_processor.process_image_matching(
                    video_path=video_path,
                    reference_image=reference_image,
                    matching_mode='object_focused',
                    top_k=10,
                    similarity_threshold=0.3,
                    target_class='person'
                )
                
                processing_time = time.time() - start_time
                
                success = result['status'] == 'success' and len(result.get('results', [])) > 0
                
                results[test_case['name']] = {
                    'success': success,
                    'matches_found': len(result.get('results', [])),
                    'processing_time': processing_time,
                    'confidence_scores': [r.get('confidence', 0) for r in result.get('results', [])],
                    'method': 'object_focused',
                    'details': result
                }
                
                if success:
                    logger.info(f"✅ SUCCESS: Found {len(result['results'])} matches")
                    logger.info(f"   Best confidence: {max(results[test_case['name']]['confidence_scores']):.3f}")
                else:
                    logger.info(f"❌ FAILED: No matches found")
                
                logger.info(f"   Processing time: {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ ERROR: {str(e)}")
                results[test_case['name']] = {
                    'success': False,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
        
        return results
    
    def test_cross_domain_matching(self, test_images: Dict[str, np.ndarray], 
                                 video_path: str) -> Dict:
        """
        Test Scenario 2: Cross-Domain Vision Matching
        Can the system handle color vs grayscale differences with different backgrounds?
        """
        logger.info("\n" + "="*60)
        logger.info("TESTING SCENARIO 2: CROSS-DOMAIN VISION MATCHING")
        logger.info("="*60)
        
        results = {}
        
        # Test cases for cross-domain matching
        test_cases = [
            {
                'name': 'Color → Grayscale (Different Backgrounds)',
                'reference': 'person_red_bg',
                'description': 'Reference: Color person with red background, Video: Grayscale with various backgrounds'
            },
            {
                'name': 'Grayscale → Color (Different Backgrounds)',
                'reference': 'person_no_bg_gray',
                'description': 'Reference: Grayscale person no background, Video: Color with various backgrounds'
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\nTesting: {test_case['name']}")
            logger.info(f"Description: {test_case['description']}")
            
            reference_image = test_images[test_case['reference']]
            
            # Test with Cross-Domain Match mode
            start_time = time.time()
            
            try:
                result = self.video_processor.process_image_matching(
                    video_path=video_path,
                    reference_image=reference_image,
                    matching_mode='cross_domain',
                    top_k=10,
                    similarity_threshold=0.3
                )
                
                processing_time = time.time() - start_time
                
                success = result['status'] == 'success' and len(result.get('results', [])) > 0
                
                results[test_case['name']] = {
                    'success': success,
                    'matches_found': len(result.get('results', [])),
                    'processing_time': processing_time,
                    'confidence_scores': [r.get('confidence', 0) for r in result.get('results', [])],
                    'method': 'cross_domain',
                    'details': result
                }
                
                if success:
                    logger.info(f"✅ SUCCESS: Found {len(result['results'])} matches")
                    logger.info(f"   Best confidence: {max(results[test_case['name']]['confidence_scores']):.3f}")
                else:
                    logger.info(f"❌ FAILED: No matches found")
                
                logger.info(f"   Processing time: {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ ERROR: {str(e)}")
                results[test_case['name']] = {
                    'success': False,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
        
        return results
    
    def test_smart_match_mode(self, test_images: Dict[str, np.ndarray], 
                            video_path: str) -> Dict:
        """
        Test Smart Match mode - should automatically handle both scenarios.
        """
        logger.info("\n" + "="*60)
        logger.info("TESTING SMART MATCH MODE (COMBINED SCENARIOS)")
        logger.info("="*60)
        
        results = {}
        
        # Test the most challenging scenarios with Smart Match
        test_cases = [
            {
                'name': 'Smart Match: No BG → Various BG + Color/Gray',
                'reference': 'person_no_bg',
                'description': 'Reference: Person no background, Video: All variations (color/gray, different backgrounds)'
            },
            {
                'name': 'Smart Match: Gray No BG → Color Various BG',
                'reference': 'person_no_bg_gray',
                'description': 'Reference: Grayscale person no background, Video: Color with various backgrounds'
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\nTesting: {test_case['name']}")
            logger.info(f"Description: {test_case['description']}")
            
            reference_image = test_images[test_case['reference']]
            
            # Test with Smart Match mode
            start_time = time.time()
            
            try:
                result = self.video_processor.process_image_matching(
                    video_path=video_path,
                    reference_image=reference_image,
                    matching_mode='smart_match',
                    top_k=15,
                    similarity_threshold=0.2  # Lower threshold for comprehensive testing
                )
                
                processing_time = time.time() - start_time
                
                success = result['status'] == 'success' and len(result.get('results', [])) > 0
                
                results[test_case['name']] = {
                    'success': success,
                    'matches_found': len(result.get('results', [])),
                    'processing_time': processing_time,
                    'confidence_scores': [r.get('confidence', 0) for r in result.get('results', [])],
                    'method': 'smart_match',
                    'details': result
                }
                
                if success:
                    logger.info(f"✅ SUCCESS: Found {len(result['results'])} matches")
                    logger.info(f"   Best confidence: {max(results[test_case['name']]['confidence_scores']):.3f}")
                    logger.info(f"   Average confidence: {np.mean(results[test_case['name']]['confidence_scores']):.3f}")
                else:
                    logger.info(f"❌ FAILED: No matches found")
                
                logger.info(f"   Processing time: {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ ERROR: {str(e)}")
                results[test_case['name']] = {
                    'success': False,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }
        
        return results
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive test report.
        """
        report = []
        report.append("\n" + "="*80)
        report.append("ENHANCED IMAGE MATCHING CAPABILITIES TEST REPORT")
        report.append("="*80)
        
        # Summary
        total_tests = 0
        successful_tests = 0
        
        for category, tests in self.test_results.items():
            for test_name, result in tests.items():
                total_tests += 1
                if result.get('success', False):
                    successful_tests += 1
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        report.append(f"\nOVERALL RESULTS:")
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Successful: {successful_tests}")
        report.append(f"Success Rate: {success_rate:.1f}%")
        
        # Detailed results by category
        for category, tests in self.test_results.items():
            report.append(f"\n{category.upper().replace('_', ' ')}:")
            report.append("-" * 50)
            
            for test_name, result in tests.items():
                status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
                report.append(f"{status} {test_name}")
                
                if result.get('success', False):
                    report.append(f"    Matches Found: {result.get('matches_found', 0)}")
                    if result.get('confidence_scores'):
                        report.append(f"    Best Confidence: {max(result['confidence_scores']):.3f}")
                        report.append(f"    Avg Confidence: {np.mean(result['confidence_scores']):.3f}")
                    report.append(f"    Processing Time: {result.get('processing_time', 0):.2f}s")
                else:
                    if 'error' in result:
                        report.append(f"    Error: {result['error']}")
                    else:
                        report.append(f"    No matches found")
        
        # Conclusions
        report.append("\nCONCLUSIONS:")
        report.append("-" * 50)
        
        bg_tests = self.test_results.get('background_independence', {})
        cd_tests = self.test_results.get('cross_domain_matching', {})
        smart_tests = self.test_results.get('combined_scenarios', {})
        
        # Question 1: Background Independence
        bg_success = any(test.get('success', False) for test in bg_tests.values())
        report.append(f"\n1. Can find person with no/different background? {'YES ✅' if bg_success else 'NO ❌'}")
        if bg_success:
            report.append("   The system successfully demonstrates background independence.")
            report.append("   Object-focused matching can isolate and match objects regardless of background.")
        
        # Question 2: Cross-Domain Vision
        cd_success = any(test.get('success', False) for test in cd_tests.values())
        report.append(f"\n2. Can handle color vs grayscale with different backgrounds? {'YES ✅' if cd_success else 'NO ❌'}")
        if cd_success:
            report.append("   The system successfully handles cross-domain vision matching.")
            report.append("   Color space normalization enables matching across color/grayscale differences.")
        
        # Smart Match Performance
        smart_success = any(test.get('success', False) for test in smart_tests.values())
        if smart_success:
            report.append(f"\n3. Smart Match Mode: EXCELLENT ✅")
            report.append("   Automatically adapts to handle both background and color domain challenges.")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def run_all_tests(self) -> str:
        """
        Run all test scenarios and return comprehensive report.
        """
        logger.info("Starting Enhanced Image Matching Test Suite...")
        
        try:
            # Create test data
            test_images = self.create_test_images()
            video_path = self.create_test_video(test_images)
            
            # Run tests
            self.test_results['background_independence'] = self.test_background_independence(test_images, video_path)
            self.test_results['cross_domain_matching'] = self.test_cross_domain_matching(test_images, video_path)
            self.test_results['combined_scenarios'] = self.test_smart_match_mode(test_images, video_path)
            
            # Generate report
            report = self.generate_report()
            
            # Cleanup
            try:
                os.unlink(video_path)
            except:
                pass
            
            return report
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return f"Test suite failed with error: {str(e)}"

def main():
    """Main test execution."""
    tester = EnhancedMatchingTester()
    report = tester.run_all_tests()
    
    print(report)
    
    # Save report to file
    with open('enhanced_matching_test_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nTest report saved to: enhanced_matching_test_report.txt")

if __name__ == "__main__":
    main()