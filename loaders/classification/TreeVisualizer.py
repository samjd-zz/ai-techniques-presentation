#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
from sklearn import tree as sklearn_tree
from matplotlib.patches import Rectangle, Ellipse
import textwrap


class DecisionTreeVisualizer:
    """
    Handles visualization of trained Decision Tree models.
    
    Provides both standard sklearn-based visualizations and custom RapidMiner-style
    visualizations with decoded category names for better interpretability.
    
    This class is separate from the data loader to maintain separation of concerns
    and keep the codebase modular and maintainable.
    """
    
    def __init__(self, model, X_train, label_encoders, target_encoder):
        """
        Initialize the visualizer with required components from a trained model.
        
        Args:
            model: Trained DecisionTreeClassifier from sklearn
            X_train: Training features DataFrame (used to get feature names)
            label_encoders: Dict mapping feature names to their LabelEncoders
            target_encoder: LabelEncoder for the target variable
            
        Raises:
            ValueError: If model is None or required data is missing
        """
        if model is None:
            raise ValueError("Model cannot be None. Please provide a trained Decision Tree model.")
        if X_train is None:
            raise ValueError("X_train cannot be None. Feature names are required for visualization.")
            
        self.model = model
        self.X_train = X_train
        self.label_encoders = label_encoders
        self.target_encoder = target_encoder

    def visualize_sklearn_style(self, save_path='../../plots/decision_tree.png', max_depth=3):
        """
        Visualize the decision tree using sklearn's standard plot_tree with comprehensive legend.
        
        Creates a visual representation showing how the model makes decisions at each split point,
        with a detailed legend explaining all node components.
        
        Args:
            save_path (str): Path to save the visualization (default: '../../plots/decision_tree.png')
            max_depth (int): Maximum depth to display for readability (default: 3)
        """
        print(f"\n" + "="*60)
        print("VISUALIZING DECISION TREE (SKLEARN STYLE)")
        print("="*60)
        
        try:
            from sklearn import tree as sklearn_tree
            from matplotlib.patches import Rectangle
        except ImportError:
            print("Required modules not available for tree visualization")
            return
        
        # Get feature and class names for better readability
        feature_names = list(self.X_train.columns)
        class_names = list(self.target_encoder.classes_)
        
        # Create figure with extra space for legend
        fig = plt.figure(figsize=(28, 16))
        
        # Create main axes for tree (left side, taking most space)
        ax_tree = plt.subplot2grid((1, 4), (0, 0), colspan=3, fig=fig)
        
        # Create axes for legend (right side, narrower)
        ax_legend = plt.subplot2grid((1, 4), (0, 3), fig=fig)
        ax_legend.axis('off')  # Hide axes for legend
        
        # Plot the tree
        sklearn_tree.plot_tree(
            self.model,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,           # Color nodes by class
            rounded=True,          # Rounded corners on boxes
            fontsize=9,
            ax=ax_tree,
            max_depth=max_depth    # Limit depth for readability
        )
        
        # Add main title
        ax_tree.set_title(
            f"Decision Tree Classification: Healthcare Charges\n"
            f"(Showing depth 1-{max_depth} of {self.model.get_depth()} total)",
            fontsize=18, fontweight='bold', pad=20
        )
        
        # Create comprehensive legend
        legend_text = """
  DECISION TREE LEGEND

[*] TREE STRUCTURE:
  • Nodes (boxes): Decision points
  • Branches (lines): Paths based on
    whether condition is True/False
    
    Left branch  -> Condition TRUE (<=)
    Right branch -> Condition FALSE (>)


[#] WHAT'S IN EACH NODE:

[1] SPLIT RULE (top line)
    Example: "smoker <= 0.5"
    • Feature being tested
    • Threshold value
    • Only ONE feature per node!
   
[2] GINI/ENTROPY (line 2)
    Example: "gini = 0.245"
    • Measures node impurity
    • Lower = purer (better)
    • Range: 0.0 (pure) to ~0.9
   
[3] SAMPLES (line 3)
    Example: "samples = 450"
    • # training samples at node
    • Decreases down the tree
   
[4] VALUE (line 4)
    Example: "value = [10, 5, 78, ...]"
    • Array of sample counts
    • One count per class
    • Shows class distribution
   
[5] CLASS (line 5)
    Example: "class = charges_bin_7"
    • Predicted class (majority)
    • What we'd predict if stopped
   

[~] NODE COLORS:
  • Different colors = different
    predicted classes
  • Intensity = prediction confidence
    (darker = more confident)


[+] FEATURES IN THIS MODEL:
  • age: Customer age (18-64)
  • sex: Gender (0=F, 1=M)
  • bmi: Body Mass Index
  • children: # of dependents
  • smoker: Status (0=no, 1=yes)
  • region: Geography (0-3)
  • age_group: Age category
  • bmi_group: BMI category


[>] TARGET (What We Predict):
  charges_group: Healthcare cost
  tier (10 bins from low to high)


[!] HOW TO READ THE TREE:
  1. Start at top (root) node
  2. Check the condition
  3. Follow left if <=, right if >
  4. Repeat until leaf node
  5. Leaf's class = prediction!

"""
        
        # Add legend text to the legend axes
        ax_legend.text(
            0.05, 0.98, legend_text,
            transform=ax_legend.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save visualization
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sklearn-style visualization saved to: {save_path}")
        
        # Close figure to free memory
        plt.close(fig)
        
        # Print tree statistics
        print(f"\nTree Statistics:")
        print(f"  Tree Depth: {self.model.get_depth()}")
        print(f"  Number of Leaves: {self.model.get_n_leaves()}")
        print(f"  Number of Features Used: {self.model.n_features_in_}")
        print(f"  Number of Classes: {len(class_names)}")
        print(f"\nVisualization Details:")
        print(f"  • Showing depth: 1-{max_depth} (of {self.model.get_depth()} total)")
        print(f"  • Legend: Comprehensive guide on right side")
        print(f"  • Each node tests ONE feature at a time")
        print(f"  • Node info: split rule, gini, samples, value, class")

    def visualize_rapidminer_style(self, save_path='../../plots/decision_tree_decoded.png', max_depth=3):
        """
        Create RapidMiner-style tree visualization with decoded category names.
        
        This method builds a completely custom tree visualization that properly decodes
        categorical features, showing intuitive labels like "smoker = yes" instead of "smoker <= 0.5".
        
        Args:
            save_path (str): Path to save the visualization (default: '../../plots/decision_tree_decoded.png')
            max_depth (int): Maximum depth to display (default: 3)
        """
        print(f"\n" + "="*60)
        print("CREATING CUSTOM RAPIDMINER-STYLE VISUALIZATION")
        print("="*60)
        
        try:
            from sklearn import tree as sklearn_tree
            import matplotlib.patches as FancyBboxPatch
        except ImportError:
            print("Required modules not available for visualization")
            return
        
        # Get tree structure and data
        tree_structure = self.model.tree_
        feature_names = list(self.X_train.columns)
        class_names = list(self.target_encoder.classes_)
        
        # Helper function to abbreviate long category names for display
        def abbreviate_category(cat_name):
            """Abbreviate long category names for compact display."""
            abbrev_map = {
                'Normal weight': 'Normal',
                'Overweight': 'Overweight',
                'Obese Class I': 'Obese I',
                'Obese Class II': 'Obese II',
                'Obese Class III': 'Obese III',
                'Underweight: BMI < 18.5': 'Underweight',
                'young_adult': 'Young',
                'middle_aged': 'Middle',
                'senior_adult': 'Senior'
            }
            return abbrev_map.get(cat_name, cat_name)
        
        # Helper function to decode split condition
        def get_split_label(feature_idx, threshold, direction='left'):
            """
            Get human-readable split label for a node.
            
            Args:
                feature_idx: Index of the feature being split on
                threshold: Numeric threshold from the tree
                direction: 'left' for <=, 'right' for >
            
            Returns:
                str: Human-readable label
            """
            feature_name = feature_names[feature_idx]
            
            # Check if this is a categorical feature
            if feature_name in self.label_encoders:
                encoder = self.label_encoders[feature_name]
                categories = encoder.classes_
                
                if len(categories) == 2:
                    # Binary categorical: map directly to category
                    if direction == 'left':
                        # <= 0.5 means first category (index 0)
                        return f"{feature_name}\n= {categories[0]}"
                    else:
                        # > 0.5 means second category (index 1)
                        return f"{feature_name}\n= {categories[1]}"
                else:
                    # Multi-class categorical - abbreviate and wrap
                    threshold_int = int(threshold)
                    if direction == 'left':
                        # <= threshold means categories 0 through threshold_int
                        if threshold_int == 0:
                            abbrev = abbreviate_category(categories[0])
                            return f"{feature_name}\n= {abbrev}"
                        else:
                            # Abbreviate all categories
                            cats_list = [abbreviate_category(c) for c in categories[:threshold_int+1]]
                            # Wrap text: max 2 categories per line
                            if len(cats_list) <= 2:
                                cats = ', '.join(cats_list)
                                return f"{feature_name}\nin [{cats}]"
                            else:
                                # Split into multiple lines
                                lines = []
                                for i in range(0, len(cats_list), 2):
                                    chunk = cats_list[i:i+2]
                                    lines.append(', '.join(chunk))
                                cats = '\n'.join(lines)
                                return f"{feature_name} in\n[{cats}]"
                    else:
                        # > threshold means categories threshold_int+1 onwards
                        # Abbreviate all categories
                        cats_list = [abbreviate_category(c) for c in categories[threshold_int+1:]]
                        # Wrap text: max 2 categories per line
                        if len(cats_list) <= 2:
                            cats = ', '.join(cats_list)
                            return f"{feature_name}\nin [{cats}]"
                        else:
                            # Split into multiple lines
                            lines = []
                            for i in range(0, len(cats_list), 2):
                                chunk = cats_list[i:i+2]
                                lines.append(', '.join(chunk))
                            cats = '\n'.join(lines)
                            return f"{feature_name} in\n[{cats}]"
            else:
                # Numeric feature: show threshold
                if direction == 'left':
                    return f"{feature_name}\n≤ {threshold:.1f}"
                else:
                    return f"{feature_name}\n> {threshold:.1f}"
        
        # Collect node positions and information
        # Calculate vertical spacing based on max_depth
        y_spacing = 1.0 / (max_depth + 1)  # Divide height evenly across levels
        
        def traverse_tree(node_id, depth, x, y, x_offset, nodes_info, edges_info):
            """
            Recursively traverse tree and collect node positions and labels.
            """
            # Get node information
            feature_idx = tree_structure.feature[node_id]
            is_actual_leaf = feature_idx == sklearn_tree._tree.TREE_UNDEFINED
            
            # Treat nodes at max_depth as "leaf-like" for visualization purposes
            is_display_leaf = is_actual_leaf or (depth >= max_depth)
            
            if is_display_leaf:
                # Leaf node or node at max depth - display as leaf
                samples = tree_structure.value[node_id][0]
                predicted_class_idx = np.argmax(samples)
                predicted_class = class_names[predicted_class_idx]
                confidence = samples[predicted_class_idx] / np.sum(samples)
                n_samples = int(np.sum(samples))
                
                label = f"{predicted_class}\n{confidence:.1%}\nn={n_samples}"
                color = plt.cm.RdYlGn(confidence)  # Color by confidence
                
                nodes_info.append({
                    'id': node_id,
                    'x': x,
                    'y': y,
                    'label': label,
                    'is_leaf': True,  # Display as leaf
                    'color': color
                })
            else:
                # Internal node (not at max depth yet)
                feature_name = feature_names[feature_idx]
                threshold = tree_structure.threshold[node_id]
                n_samples = int(tree_structure.n_node_samples[node_id])
                gini = tree_structure.impurity[node_id]
                
                label = f"{feature_name}\ngini={gini:.3f}\nn={n_samples}"
                color = '#ffffff'  # White for internal nodes
                
                nodes_info.append({
                    'id': node_id,
                    'x': x,
                    'y': y,
                    'label': label,
                    'is_leaf': False,
                    'color': color,
                    'feature_idx': feature_idx,
                    'threshold': threshold
                })
                
                # Only process children if we haven't exceeded max depth
                if depth < max_depth:
                    # Calculate child positions
                    left_child = tree_structure.children_left[node_id]
                    right_child = tree_structure.children_right[node_id]
                    
                    new_x_offset = x_offset / 2
                    left_x = x - x_offset
                    right_x = x + x_offset
                    child_y = y - y_spacing  # Use calculated spacing
                    
                    # Add left edge and node
                    if left_child != sklearn_tree._tree.TREE_LEAF:
                        edge_label = get_split_label(feature_idx, threshold, 'left')
                        edges_info.append({
                            'from_x': x,
                            'from_y': y,
                            'to_x': left_x,
                            'to_y': child_y,
                            'label': edge_label
                        })
                        traverse_tree(left_child, depth + 1, left_x, child_y, new_x_offset, nodes_info, edges_info)
                    
                    # Add right edge and node
                    if right_child != sklearn_tree._tree.TREE_LEAF:
                        edge_label = get_split_label(feature_idx, threshold, 'right')
                        edges_info.append({
                            'from_x': x,
                            'from_y': y,
                            'to_x': right_x,
                            'to_y': child_y,
                            'label': edge_label
                        })
                        traverse_tree(right_child, depth + 1, right_x, child_y, new_x_offset, nodes_info, edges_info)
        
        # Start traversal from root at top of visible area
        nodes_info = []
        edges_info = []
        traverse_tree(0, 0, 0.5, 0.95, 0.25, nodes_info, edges_info)
        
        # Create figure with gradient background
        fig, ax = plt.subplots(figsize=(26, 16))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, 1.15)
        ax.axis('off')
        
        # Add subtle background gradient
        ax.set_facecolor('#f8f9fa')
        
        # Draw edges with gradients and arrows
        for edge in edges_info:
            # Draw main edge line with gradient effect (thicker line underneath)
            ax.plot([edge['from_x'], edge['to_x']], 
                   [edge['from_y'], edge['to_y']], 
                   color='#495057', linewidth=4, alpha=0.3, zorder=1)
            ax.plot([edge['from_x'], edge['to_x']], 
                   [edge['from_y'], edge['to_y']], 
                   color='#343a40', linewidth=2.5, zorder=2)
            
            # Add arrow at the end of edge
            dx = edge['to_x'] - edge['from_x']
            dy = edge['to_y'] - edge['from_y']
            ax.arrow(edge['from_x'] + dx*0.85, edge['from_y'] + dy*0.85, 
                    dx*0.1, dy*0.1,
                    head_width=0.015, head_length=0.015, 
                    fc='#343a40', ec='#343a40', zorder=2)
            
            # Add edge label with enhanced styling and width constraint
            mid_x = (edge['from_x'] + edge['to_x']) / 2
            mid_y = (edge['from_y'] + edge['to_y']) / 2
            
            # Determine if this is a "Yes" (left/true) or "No" (right/false) branch
            is_left = edge['to_x'] < edge['from_x']
            edge_color = '#28a745' if is_left else '#dc3545'  # Green for left/yes, red for right/no
            
            # Wrap text to constrain width (max ~15 characters per line)
            max_chars_per_line = 15
            wrapped_label = '\n'.join(textwrap.wrap(edge['label'].replace('\n', ' '), 
                                                     width=max_chars_per_line,
                                                     break_long_words=False,
                                                     break_on_hyphens=False))
            
            ax.text(mid_x, mid_y, wrapped_label, 
                   ha='center', va='center',
                   fontsize=9, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=edge_color, 
                            edgecolor='white',
                            linewidth=2,
                            alpha=0.95),
                   zorder=3,
                   multialignment='center')
        
        # Draw nodes with enhanced styling
        for node in nodes_info:
            if node['is_leaf']:
                # Leaf nodes: ELLIPSES with gradient colors
                width, height = 0.10, 0.07
                
                # Create shadow effect (ellipse)
                shadow = Ellipse((node['x'] + 0.003, node['y'] - 0.003), 
                                width, height,
                                facecolor='#000000', alpha=0.2,
                                zorder=3)
                ax.add_patch(shadow)
                
                # Main ellipse with gradient colors
                ellipse = Ellipse((node['x'], node['y']), 
                                 width, height,
                                 facecolor=node['color'], 
                                 edgecolor='#2c3e50',
                                 linewidth=3,
                                 zorder=4)
                ax.add_patch(ellipse)
                
                # Add a subtle inner border for depth
                inner_ellipse = Ellipse((node['x'], node['y']), 
                                       width * 0.9, height * 0.9,
                                       facecolor='none', 
                                       edgecolor='white',
                                       linewidth=1.5,
                                       alpha=0.5,
                                       zorder=5)
                ax.add_patch(inner_ellipse)
                
                # Add label with better formatting
                ax.text(node['x'], node['y'], node['label'],
                       ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       color='#1a1a1a',
                       zorder=6)
            else:
                # Internal nodes: rounded rectangles with gradient
                width, height = 0.14, 0.09
                
                # Create shadow effect
                shadow = Rectangle((node['x'] - width/2 + 0.003, node['y'] - height/2 - 0.003), 
                                      width, height,
                                      facecolor='#000000', alpha=0.2,
                                      transform=ax.transData, zorder=3)
                ax.add_patch(shadow)
                
                # Main box with gradient effect (light blue to white)
                box = Rectangle((node['x'] - width/2, node['y'] - height/2), 
                                   width, height,
                                   facecolor='#e3f2fd',
                                   edgecolor='#1976d2',
                                   linewidth=3,
                                   transform=ax.transData,
                                   zorder=4)
                ax.add_patch(box)
                
                # Add inner highlight
                inner_box = Rectangle((node['x'] - width/2 + 0.005, node['y'] - height/2 + 0.005), 
                                         width - 0.01, height - 0.01,
                                         facecolor='none', 
                                         edgecolor='white',
                                         linewidth=2,
                                         alpha=0.7,
                                         transform=ax.transData,
                                         zorder=5)
                ax.add_patch(inner_box)
                
                # Add label with better formatting
                ax.text(node['x'], node['y'], node['label'],
                       ha='center', va='center',
                       fontsize=10, fontweight='bold',
                       color='#0d47a1',
                       zorder=6)
        
        # Add title
        ax.set_title(
            f"Custom Decision Tree with Decoded Categories (RapidMiner-style)\n"
            f"Depth 1-{max_depth} of {self.model.get_depth()} total",
            fontsize=18, fontweight='bold', pad=20
        )
        
        plt.tight_layout()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save visualization
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Custom decoded visualization saved to: {save_path}")
        
        plt.close(fig)
        
        # Print summary
        print(f"\nVisualization Details:")
        print(f"  • Fully custom tree plotter (not sklearn's plot_tree)")
        print(f"  • Showing depth: 1-{max_depth} (of {self.model.get_depth()} total)")
        print(f"  • Edge labels show decoded conditions")
        print(f"  • Binary features: 'feature = category'")
        print(f"  • Multi-class features: 'feature in [categories]'")
        print(f"  • Numeric features: 'feature ≤ threshold'")
        print(f"\nDecoded Features:")
        for feature_name, encoder in self.label_encoders.items():
            categories = ', '.join(encoder.classes_)
            print(f"  • {feature_name}: {categories}")
