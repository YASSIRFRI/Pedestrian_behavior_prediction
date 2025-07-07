"""Visualization utilities for the behavior classifier."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from typing import Dict, Optional, Union
from matplotlib.patches import Patch

from ..constants import BEHAVIOR_CLASSES


class VisualizationUtils:
    """Utilities for creating visualizations and saving results."""
    
    def __init__(self, verbose: bool = True):
        """Initialize visualization utilities."""
        self.verbose = verbose

    def create_comprehensive_visualization(
        self,
        image: Image.Image,
        person_box: np.ndarray,
        expanded_box: np.ndarray,
        crop: Image.Image,
        caption: str,
        caption_variations: Dict[str, str],
        behavior: str,
        scores: np.ndarray,
        confidence: float,
        person_idx: int,
        image_name: str = "",
        output_dir: str = "./outputs",
        model_name: str = "BLIP2",
        show_plots: bool = False
    ) -> Optional[str]:
        """Create a single comprehensive visualization showing everything for one person."""

        if self.verbose:
            print(f"   🎨 Creating comprehensive BLIP-2 visualization for Person {person_idx}...")

        try:
            # Create figure with 4 subplots: Original+Boxes, Crop, Caption+Info, Histogram
            fig = plt.figure(figsize=(22, 14))

            # Define colors for different boxes
            original_color = (255, 0, 0)  # Red
            expanded_color = (0, 255, 0)  # Green

            # Convert PIL to OpenCV if needed
            if isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image.copy()

            # 1. Original image with bounding boxes (top-left)
            ax1 = plt.subplot(2, 3, (1, 2))  # Span 2 columns

            # Draw original bounding box
            x1, y1, x2, y2 = map(int, person_box)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), original_color, 3)
            cv2.putText(cv_image, "Original Box", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, original_color, 2)

            # Draw expanded bounding box
            ex1, ey1, ex2, ey2 = map(int, expanded_box)
            cv2.rectangle(cv_image, (ex1, ey1), (ex2, ey2), expanded_color, 3)
            cv2.putText(cv_image, "Expanded Box", (ex1, ey1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, expanded_color, 2)

            # Add person label
            cv2.putText(cv_image, f"Person {person_idx}", (x1, y2+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(cv_image, f"Person {person_idx}", (x1, y2+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

            ax1.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'BLIP-2 Detection Boxes - {image_name}', fontsize=14, fontweight='bold')
            ax1.axis('off')

            # Add legend
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label='Original Detection Box'),
                Patch(facecolor='green', alpha=0.7, label='Expanded Context Box')
            ]
            ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

            # 2. Cropped person (top-right)
            ax2 = plt.subplot(2, 3, 3)
            ax2.imshow(crop)
            ax2.set_title(f'Cropped Person {person_idx}', fontsize=14, fontweight='bold')
            ax2.axis('off')

            # Add crop info
            crop_info = f"Size: {crop.size[0]}x{crop.size[1]}px"
            ax2.text(0.02, 0.98, crop_info, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=10, fontweight='bold')

            # 3. Caption and Information (bottom-left)
            ax3 = plt.subplot(2, 3, 4)
            ax3.axis('off')

            # Create info text
            info_text = f"""
PERSON {person_idx} BLIP-2 ANALYSIS

🏷️  PREDICTED BEHAVIOR:
    {behavior.upper()}

🎯 CONFIDENCE:
    {confidence:.3f}

🤖 BLIP-2 MODEL:
    {model_name.split('/')[-1]}

📝 MAIN CAPTION:
    "{caption}"

📋 CAPTION VARIATIONS:
"""
            # Add caption variations
            for prompt_type, cap in caption_variations.items():
                if prompt_type != "unconditional":
                    info_text += f"    {prompt_type}: \"{cap}\"\n"

            info_text += f"""
📊 TOP 3 PREDICTIONS:
"""
            # Add top 3 predictions
            top_indices = np.argsort(scores)[::-1][:3]
            for i, idx in enumerate(top_indices):
                info_text += f"    {i+1}. {BEHAVIOR_CLASSES[idx]}: {scores[idx]:.3f}\n"

            # Box dimensions info
            orig_w = person_box[2] - person_box[0]
            orig_h = person_box[3] - person_box[1]
            exp_w = expanded_box[2] - expanded_box[0]
            exp_h = expanded_box[3] - expanded_box[1]

            info_text += f"""
📐 BOUNDING BOX INFO:
    Original: {orig_w:.0f}x{orig_h:.0f}px
    Expanded: {exp_w:.0f}x{exp_h:.0f}px
    Expansion: {((exp_w*exp_h)/(orig_w*orig_h)):.1f}x area
"""

            ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
                    verticalalignment='top', fontsize=10, fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

            ax3.set_title('BLIP-2 Detection Information', fontsize=14, fontweight='bold')

            # 4. Behavior Probability Histogram (bottom-center and bottom-right)
            ax4 = plt.subplot(2, 3, (5, 6))  # Span 2 columns

            # Create horizontal bar chart
            y_pos = np.arange(len(BEHAVIOR_CLASSES))
            colors = ['#ff4444' if i == np.argmax(scores) else '#44aaff' for i in range(len(scores))]

            bars = ax4.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

            # Customize the plot
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(BEHAVIOR_CLASSES, fontsize=11)
            ax4.set_xlabel('Probability Score', fontsize=12, fontweight='bold')
            ax4.set_title(f'Behavior Classification Probabilities - Person {person_idx} (BLIP-2)',
                         fontsize=14, fontweight='bold')

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)

            # Add grid and formatting
            ax4.grid(axis='x', alpha=0.3)
            ax4.set_xlim(0, max(scores) * 1.15)  # Add some space for labels

            # Highlight the predicted behavior
            predicted_idx = np.argmax(scores)
            ax4.axhline(y=predicted_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)

            # Add overall title
            fig.suptitle(f'BLIP-2 Comprehensive Analysis - Person {person_idx} in {image_name}',
                        fontsize=16, fontweight='bold', y=0.95)

            plt.tight_layout()
            plt.subplots_adjust(top=0.90)  # Make room for suptitle

            # Save the comprehensive visualization
            filepath = None
            if output_dir:
                timestamp = datetime.now().strftime("%H%M%S")
                safe_image_name = image_name.replace('.', '_').replace(' ', '_') if image_name else "unknown"
                filename = f"blip2_person_{person_idx:03d}_{safe_image_name}_{timestamp}_comprehensive.png"
                filepath = os.path.join(output_dir, filename)

                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')

                if self.verbose:
                    print(f"      💾 Saved comprehensive visualization: {filename}")
                    print(f"      📁 Location: {filepath}")

            # Show the plot only if show_plots is True
            if show_plots:
                plt.show()
            else:
                plt.close()  # Close without showing to save memory

            return filepath

        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error creating comprehensive visualization: {e}")
            return None

    def draw_bbox(
        self, 
        image: np.ndarray, 
        box: np.ndarray, 
        behavior: str, 
        confidence: float, 
        color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        """Draw bounding box with behavior label."""
        x1, y1, x2, y2 = map(int, box)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{behavior}: {confidence:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)

        # Draw label text
        cv2.putText(image, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def create_summary_plot(
        self, 
        results: list, 
        output_path: str = None, 
        show_plot: bool = False
    ) -> Optional[str]:
        """Create a summary plot of all results."""
        if not results:
            print("No results to plot")
            return None

        # Count behaviors
        behavior_counts = {}
        for result in results:
            for detection in result.get('detections', []):
                behavior = detection['behavior']
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        behaviors = list(behavior_counts.keys())
        counts = list(behavior_counts.values())
        
        bars = ax.bar(behaviors, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Customize plot
        ax.set_xlabel('Behavior', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Behavior Detection Summary', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Summary plot saved to: {output_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return output_path