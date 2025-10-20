#!/usr/bin/env python3
"""
Data Aggregation Module for Insurance Data

This module provides utilities for aggregating and visualizing insurance data
across different grouping variables and regions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class DataAggregator:
    """
    Handles data aggregation and visualization for insurance dataset analysis.
    
    Provides methods to:
    - Aggregate data by various grouping variables
    - Compare aggregations across regions
    - Generate visualizations of aggregated data
    """
    
    def __init__(self, regional_dataframes):
        """
        Initialize the aggregator with regional dataframes.
        
        Args:
            regional_dataframes (dict): Dictionary with region names as keys and
                                       DataFrames as values. Expected keys:
                                       'northeast', 'southeast', 'northwest', 'southwest'
        """
        self.northeast_df = regional_dataframes.get('northeast')
        self.southeast_df = regional_dataframes.get('southeast')
        self.northwest_df = regional_dataframes.get('northwest')
        self.southwest_df = regional_dataframes.get('southwest')
        
        # Validate that all regions are provided
        if any(df is None for df in [self.northeast_df, self.southeast_df, 
                                      self.northwest_df, self.southwest_df]):
            raise ValueError("All four regions must be provided: northeast, southeast, northwest, southwest")
    
    def aggregate_group_by(self, df, group_col, agg_col='charges', agg='mean'):
        """
        Group dataframe by a column and compute aggregated statistics.
        
        Args:
            df (pd.DataFrame): Input dataframe
            group_col (str): Column name to group by (e.g., 'children', 'smoker')
            agg_col (str): Column to aggregate (default: 'charges')
            agg (str): Aggregation function to apply (default: 'mean')
            
        Returns:
            pd.DataFrame: Aggregated dataframe with group_col and aggregated values
        """
        df = df.groupby(group_col)[agg_col].agg(agg).round(2)
        # After groupby, group_col becomes the INDEX, so it must be reset
        df = df.reset_index()
        df.columns = [group_col, agg_col]
        return df
    
    def __aggregate_regions(self, grp_by='children', agg_by='charges'):
        """
        Aggregate data across all four regions and pivot to wide format.
        
        Creates a comparison table where each region's aggregated values
        are shown side-by-side for easy comparison.
        
        Args:
            grp_by (str): Column to group by (default: 'children')
            agg_by (str): Column to aggregate (default: 'charges')
            
        Returns:
            pd.DataFrame: Pivoted dataframe with regions as columns and grp_by values as rows
        """
        northeast_agg = self.aggregate_group_by(self.northeast_df, 
                                                grp_by, agg_col=agg_by)
        southeast_agg = self.aggregate_group_by(self.southeast_df, 
                                                grp_by, agg_col=agg_by)
        northwest_agg = self.aggregate_group_by(self.northwest_df, 
                                                grp_by, agg_col=agg_by)
        southwest_agg = self.aggregate_group_by(self.southwest_df, 
                                                grp_by, agg_col=agg_by)
        
        # Add region column to each
        northeast_agg['REGION->'] = 'northeast_avg_'+agg_by 
        southeast_agg['REGION->'] = 'southeast_avg_'+agg_by
        northwest_agg['REGION->'] = 'northwest_avg_'+agg_by
        southwest_agg['REGION->'] = 'southwest_avg_'+agg_by
        
        # Concatenate
        combined = pd.concat([northeast_agg, southeast_agg, 
                              northwest_agg, southwest_agg], ignore_index=True)
        
        # Pivot to wide format
        return combined.pivot(index=grp_by, columns='REGION->', values=agg_by)
    
    def get_aggregate_table(self, grp_by='children', agg_by='charges'):
        """
        Get aggregated comparison table across all regions.
        
        Public interface method for regional aggregation.
        
        Args:
            grp_by (str): Column to group by (default: 'children')
            agg_by (str): Column to aggregate (default: 'charges')
            
        Returns:
            pd.DataFrame: Pivoted table comparing aggregated values across regions
        """
        return self.__aggregate_regions(grp_by, agg_by)
    
    def get_grouped_bar_chart(self, grp_by='children', agg_by='charges'):
        """
        Generate grouped bar chart comparing regions by aggregated values.
        
        Visualizes how different regions compare across grouping categories,
        useful for identifying regional patterns in insurance charges.
        
        Args:
            grp_by (str): Column to group by on x-axis (default: 'children')
            agg_by (str): Column to aggregate for y-axis values (default: 'charges')
            
        Returns:
            matplotlib.pyplot: Configured plot object ready to display with plt.show()
        """
        df = self.__aggregate_regions(grp_by, agg_by)        
        ax = df.plot(kind='bar', figsize=(10,6))
        
        plt.title("Insurance "+agg_by+" by Number of "+grp_by+" per Region", 
                  fontsize=14, fontweight='bold')
        plt.xlabel(grp_by, fontsize=15)
        plt.ylabel("Average $", fontsize=12)
        plt.legend(title='Region', loc='best')
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        plt.tight_layout()
        
        # Format y-axis tick labels with comma separators (e.g., 12000 → 12,000)
        # FuncFormatter accepts a lambda function that receives two arguments:
        #   - x: the tick value (the actual number to format, e.g., 12345.67)
        #   - pos: the tick position/index on the axis (which we don't need)
        # The underscore (_) is Python convention for "I'm required to accept this 
        # parameter but won't use it" - it satisfies the function signature
        # The lambda converts to int and uses :, format specifier for thousand separators
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, _: f'{int(x):,}')
        )

        return plt


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import os
    
    print("=" * 60)
    print("Data Aggregator - Testing Module")
    print("=" * 60)
    
    # Load sample data
    df = pd.read_csv('data/insurance.csv')
    
    # Filter by regions (simulating what LoadDecisionTree does)
    regional_dfs = {
        'northeast': df[df['region'] == 'northeast'].copy(),
        'southeast': df[df['region'] == 'southeast'].copy(),
        'northwest': df[df['region'] == 'northwest'].copy(),
        'southwest': df[df['region'] == 'southwest'].copy()
    }
    
    print(f"\nRegional data loaded:")
    for region, region_df in regional_dfs.items():
        print(f"  {region}: {len(region_df)} records")
    
    # Create aggregator
    aggregator = DataAggregator(regional_dfs)
    
    # Test aggregation by children
    print("\nAggregate by children:")
    children_agg = aggregator.get_aggregate_table(grp_by='children', agg_by='charges')
    print(children_agg)
    
    # Test aggregation by smoker
    print("\nAggregate by smoker:")
    smoker_agg = aggregator.get_aggregate_table(grp_by='smoker', agg_by='charges')
    print(smoker_agg)
    
    # Generate visualization
    print("\nGenerating visualization...")
    aggregator.get_grouped_bar_chart(grp_by='children', agg_by='charges')
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/test_aggregator.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to plots/test_aggregator.png")
    
    print("\n" + "=" * 60)
    print("Data Aggregation Complete!")
    print("=" * 60)
