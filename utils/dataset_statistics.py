import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import itertools
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class DatasetAnalyzer:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.stats_dir = os.path.join(dataset_dir, 'statistics')
        self.data = []
        self.all_classes = set()
        self.bbox_data = []
        
        # Create statistics directory
        os.makedirs(self.stats_dir, exist_ok=True)
        
    def parse_all_xml_files(self):
        """Parse all XML files and extract comprehensive data"""
        xml_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.xml')]
        
        print(f"Processing {len(xml_files)} XML files...")
        
        for xml_file in xml_files:
            xml_path = os.path.join(self.dataset_dir, xml_file)
            try:
                self._parse_single_xml(xml_path)
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue
        
        print(f"Successfully processed {len(self.data)} files")
        print(f"Found {len(self.all_classes)} unique classes: {sorted(self.all_classes)}")
        
    def _parse_single_xml(self, xml_path):
        """Parse single XML file"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract basic info
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Extract objects
        objects = root.findall('object')
        classes_in_image = []
        bboxes_in_image = []
        
        for obj in objects:
            class_name = obj.find('name').text
            bbox = obj.find('bndbox')
            
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            bbox_area = bbox_width * bbox_height
            bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
            
            # Center coordinates (normalized)
            center_x = (xmin + xmax) / 2 / width
            center_y = (ymin + ymax) / 2 / height
            
            # Relative size
            relative_area = bbox_area / (width * height)
            
            classes_in_image.append(class_name)
            self.all_classes.add(class_name)
            
            bbox_info = {
                'filename': filename,
                'class': class_name,
                'width': bbox_width,
                'height': bbox_height,
                'area': bbox_area,
                'aspect_ratio': bbox_aspect_ratio,
                'center_x': center_x,
                'center_y': center_y,
                'relative_area': relative_area,
                'image_width': width,
                'image_height': height,
                'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax
            }
            
            bboxes_in_image.append(bbox_info)
            self.bbox_data.append(bbox_info)
        
        # Store image-level data
        image_data = {
            'filename': filename,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'aspect_ratio': width / height,
            'total_area': width * height,
            'num_objects': len(objects),
            'classes': classes_in_image,
            'unique_classes': list(set(classes_in_image))
        }
        
        self.data.append(image_data)
    
    def plot_class_distribution(self):
        """Plot class distribution"""
        all_classes = []
        for item in self.data:
            all_classes.extend(item['classes'])
        
        class_counts = Counter(all_classes)
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(classes)), counts, color=sns.color_palette("husl", len(classes)))
        plt.xlabel('Classes')
        plt.ylabel('Number of Instances')
        plt.title('Class Distribution in Dataset')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create percentage pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution (Percentage)')
        plt.axis('equal')
        plt.savefig(os.path.join(self.stats_dir, 'class_distribution_pie.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cooccurrence_matrix(self):
        """Plot class co-occurrence matrix"""
        classes = sorted(list(self.all_classes))
        cooccurrence = np.zeros((len(classes), len(classes)))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        for item in self.data:
            unique_classes = item['unique_classes']
            for cls1, cls2 in itertools.combinations_with_replacement(unique_classes, 2):
                i, j = class_to_idx[cls1], class_to_idx[cls2]
                cooccurrence[i][j] += 1
                if i != j:
                    cooccurrence[j][i] += 1
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cooccurrence, xticklabels=classes, yticklabels=classes, 
                   annot=True, fmt='g', cmap='Blues', square=True)
        plt.title('Class Co-occurrence Matrix')
        plt.xlabel('Classes')
        plt.ylabel('Classes')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_dir, 'cooccurrence_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_resolution_analysis(self):
        """Plot resolution distribution and analysis"""
        widths = [item['width'] for item in self.data]
        heights = [item['height'] for item in self.data]
        resolutions = [item['resolution'] for item in self.data]
        aspect_ratios = [item['aspect_ratio'] for item in self.data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Width distribution
        axes[0, 0].hist(widths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Image Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Image Width Distribution')
        axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
        axes[0, 0].legend()
        
        # Height distribution
        axes[0, 1].hist(heights, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Image Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Image Height Distribution')
        axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
        axes[0, 1].legend()
        
        # Resolution scatter plot
        axes[1, 0].scatter(widths, heights, alpha=0.6, color='purple')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Height (pixels)')
        axes[1, 0].set_title('Width vs Height Distribution')
        
        # Aspect ratio distribution
        axes[1, 1].hist(aspect_ratios, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Aspect Ratio (Width/Height)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Aspect Ratio Distribution')
        axes[1, 1].axvline(np.mean(aspect_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(aspect_ratios):.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_dir, 'resolution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_bbox_statistics(self):
        """Plot comprehensive bounding box statistics"""
        df_bbox = pd.DataFrame(self.bbox_data)
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # Bbox width distribution
        axes[0, 0].hist(df_bbox['width'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Bounding Box Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Bounding Box Width Distribution')
        
        # Bbox height distribution
        axes[0, 1].hist(df_bbox['height'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('Bounding Box Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Bounding Box Height Distribution')
        
        # Bbox area distribution
        axes[0, 2].hist(df_bbox['area'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
        axes[0, 2].set_xlabel('Bounding Box Area (pixels²)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Bounding Box Area Distribution')
        
        # Aspect ratio distribution
        axes[1, 0].hist(df_bbox['aspect_ratio'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Bounding Box Aspect Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Bounding Box Aspect Ratio Distribution')
        
        # Relative area distribution
        axes[1, 1].hist(df_bbox['relative_area'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_xlabel('Relative Area (bbox/image)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Relative Bounding Box Area Distribution')
        
        # Center position heatmap
        axes[1, 2].hist2d(df_bbox['center_x'], df_bbox['center_y'], bins=20, cmap='Blues')
        axes[1, 2].set_xlabel('Normalized Center X')
        axes[1, 2].set_ylabel('Normalized Center Y')
        axes[1, 2].set_title('Bounding Box Center Position Heatmap')
        
        # Width vs Height scatter
        axes[2, 0].scatter(df_bbox['width'], df_bbox['height'], alpha=0.5, color='green')
        axes[2, 0].set_xlabel('Bounding Box Width')
        axes[2, 0].set_ylabel('Bounding Box Height')
        axes[2, 0].set_title('Bounding Box Width vs Height')
        
        # Box plot of area by class
        if len(df_bbox['class'].unique()) <= 10:
            df_bbox.boxplot(column='area', by='class', ax=axes[2, 1])
            axes[2, 1].set_xlabel('Class')
            axes[2, 1].set_ylabel('Bounding Box Area')
            axes[2, 1].set_title('Bounding Box Area by Class')
            axes[2, 1].tick_params(axis='x', rotation=45)
        else:
            axes[2, 1].text(0.5, 0.5, 'Too many classes\nfor box plot', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
        
        # Objects per image distribution
        objects_per_image = [item['num_objects'] for item in self.data]
        axes[2, 2].hist(objects_per_image, bins=range(max(objects_per_image)+2), 
                       alpha=0.7, color='gold', edgecolor='black')
        axes[2, 2].set_xlabel('Number of Objects per Image')
        axes[2, 2].set_ylabel('Frequency')
        axes[2, 2].set_title('Objects per Image Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_dir, 'bbox_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_specific_analysis(self):
        """Plot class-specific bounding box analysis"""
        df_bbox = pd.DataFrame(self.bbox_data)
        classes = sorted(df_bbox['class'].unique())
        
        if len(classes) > 12:  # Too many classes for detailed analysis
            print("Too many classes for detailed class-specific analysis. Skipping...")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Average bbox size per class
        avg_sizes = df_bbox.groupby('class')[['width', 'height', 'area']].mean()
        
        axes[0, 0].bar(range(len(avg_sizes)), avg_sizes['area'], 
                      color=sns.color_palette("husl", len(avg_sizes)))
        axes[0, 0].set_xlabel('Classes')
        axes[0, 0].set_ylabel('Average Area (pixels²)')
        axes[0, 0].set_title('Average Bounding Box Area by Class')
        axes[0, 0].set_xticks(range(len(avg_sizes)))
        axes[0, 0].set_xticklabels(avg_sizes.index, rotation=45, ha='right')
        
        # Average aspect ratio per class
        avg_aspect = df_bbox.groupby('class')['aspect_ratio'].mean()
        axes[0, 1].bar(range(len(avg_aspect)), avg_aspect.values, 
                      color=sns.color_palette("husl", len(avg_aspect)))
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Average Aspect Ratio')
        axes[0, 1].set_title('Average Aspect Ratio by Class')
        axes[0, 1].set_xticks(range(len(avg_aspect)))
        axes[0, 1].set_xticklabels(avg_aspect.index, rotation=45, ha='right')
        
        # Relative size per class
        avg_rel_area = df_bbox.groupby('class')['relative_area'].mean()
        axes[1, 0].bar(range(len(avg_rel_area)), avg_rel_area.values, 
                      color=sns.color_palette("husl", len(avg_rel_area)))
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('Average Relative Area')
        axes[1, 0].set_title('Average Relative Area by Class')
        axes[1, 0].set_xticks(range(len(avg_rel_area)))
        axes[1, 0].set_xticklabels(avg_rel_area.index, rotation=45, ha='right')
        
        # Instance count per class
        instance_counts = df_bbox['class'].value_counts().sort_index()
        axes[1, 1].bar(range(len(instance_counts)), instance_counts.values, 
                      color=sns.color_palette("husl", len(instance_counts)))
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Number of Instances')
        axes[1, 1].set_title('Number of Instances by Class')
        axes[1, 1].set_xticks(range(len(instance_counts)))
        axes[1, 1].set_xticklabels(instance_counts.index, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.stats_dir, 'class_specific_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a text summary report"""
        df_bbox = pd.DataFrame(self.bbox_data)
        
        report = []
        report.append("=" * 60)
        report.append("DATASET ANALYSIS SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic statistics
        report.append("DATASET OVERVIEW:")
        report.append(f"Total Images: {len(self.data)}")
        report.append(f"Total Annotations: {len(self.bbox_data)}")
        report.append(f"Unique Classes: {len(self.all_classes)}")
        report.append(f"Classes: {', '.join(sorted(self.all_classes))}")
        report.append("")
        
        # Image statistics
        widths = [item['width'] for item in self.data]
        heights = [item['height'] for item in self.data]
        aspect_ratios = [item['aspect_ratio'] for item in self.data]
        
        report.append("IMAGE STATISTICS:")
        report.append(f"Width - Mean: {np.mean(widths):.1f}, Std: {np.std(widths):.1f}, Range: {min(widths)}-{max(widths)}")
        report.append(f"Height - Mean: {np.mean(heights):.1f}, Std: {np.std(heights):.1f}, Range: {min(heights)}-{max(heights)}")
        report.append(f"Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Std: {np.std(aspect_ratios):.2f}")
        report.append("")
        
        # Bounding box statistics
        report.append("BOUNDING BOX STATISTICS:")
        report.append(f"Average bbox width: {df_bbox['width'].mean():.1f} ± {df_bbox['width'].std():.1f}")
        report.append(f"Average bbox height: {df_bbox['height'].mean():.1f} ± {df_bbox['height'].std():.1f}")
        report.append(f"Average bbox area: {df_bbox['area'].mean():.1f} ± {df_bbox['area'].std():.1f}")
        report.append(f"Average aspect ratio: {df_bbox['aspect_ratio'].mean():.2f} ± {df_bbox['aspect_ratio'].std():.2f}")
        report.append(f"Average relative area: {df_bbox['relative_area'].mean():.3f} ± {df_bbox['relative_area'].std():.3f}")
        report.append("")
        
        # Objects per image
        objects_per_image = [item['num_objects'] for item in self.data]
        report.append("OBJECTS PER IMAGE:")
        report.append(f"Mean: {np.mean(objects_per_image):.1f}")
        report.append(f"Median: {np.median(objects_per_image):.1f}")
        report.append(f"Range: {min(objects_per_image)}-{max(objects_per_image)}")
        report.append("")
        
        # Class distribution
        all_classes = []
        for item in self.data:
            all_classes.extend(item['classes'])
        class_counts = Counter(all_classes)
        
        report.append("CLASS DISTRIBUTION:")
        for class_name, count in class_counts.most_common():
            percentage = (count / len(self.bbox_data)) * 100
            report.append(f"{class_name}: {count} instances ({percentage:.1f}%)")
        
        # Save report
        with open(os.path.join(self.stats_dir, 'summary_report.txt'), 'w') as f:
            f.write('\n'.join(report))
        
        # Print to console
        print('\n'.join(report))
    
    def analyze_dataset(self):
        """Run complete dataset analysis"""
        print("Starting comprehensive dataset analysis...")
        
        # Parse data
        self.parse_all_xml_files()
        
        if not self.data:
            print("No data found! Please check your dataset directory.")
            return
        
        # Generate all plots
        print("\nGenerating class distribution plots...")
        self.plot_class_distribution()
        
        print("Generating co-occurrence matrix...")
        self.plot_cooccurrence_matrix()
        
        print("Generating resolution analysis...")
        self.plot_resolution_analysis()
        
        print("Generating bounding box statistics...")
        self.plot_bbox_statistics()
        
        print("Generating class-specific analysis...")
        self.plot_class_specific_analysis()
        
        print("Generating summary report...")
        self.generate_summary_report()
        
        print(f"\nAnalysis complete! All plots and reports saved to: {self.stats_dir}")
        
        # List generated files
        generated_files = os.listdir(self.stats_dir)
        print("\nGenerated files:")
        for file in sorted(generated_files):
            print(f"  - {file}")

# Main execution
if __name__ == "__main__":
    dataset_directory = r"D:\cropped_persons\dataset\test_dataset"  # Change this to your actual path
    
    if not os.path.exists(dataset_directory):
        print(f"Directory '{dataset_directory}' not found!")
        print("Please update the 'dataset_directory' variable with your actual dataset path.")
    else:
        analyzer = DatasetAnalyzer(dataset_directory)
        analyzer.analyze_dataset()