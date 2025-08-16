import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ShoppingPlannerVisualizer:
    def __init__(self, output_dir: str = "output/visualizations"):
        self.output_dir = output_dir
        self.ensure_output_directory()
        
    def ensure_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("output", exist_ok=True)
    
    def create_all_visualizations(self, plan: Dict, preferences: Dict):
        print("Creating visualizations...")
        
        try:
            # 1. Budget utilization pie chart
            self.plot_budget_utilization(plan, preferences)
            
            # 2. Nutritional breakdown
            self.plot_nutritional_breakdown(plan)
            
            # 3. Category distribution
            self.plot_category_distribution(plan['selected_items'])
            
            # 4. Price vs nutrition scatter plot
            self.plot_price_vs_nutrition(plan['selected_items'])
            
            # 5. Expiration timeline
            self.plot_expiration_timeline(plan['selected_items'])
            
            # 6. Algorithm performance comparison
            self.plot_algorithm_performance(plan['algorithm_performance'])
            
            # 7. Daily shopping schedule
            self.plot_shopping_schedule(plan['shopping_schedule'])
            
            # 8. Macronutrient balance
            self.plot_macronutrient_balance(plan)
            
            print(f"All visualizations saved to {self.output_dir}/")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def plot_budget_utilization(self, plan: Dict, preferences: Dict):
        total_budget = preferences['budget']
        used_budget = plan['total_cost']
        remaining_budget = total_budget - used_budget
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Data for pie chart
        sizes = [used_budget, remaining_budget]
        labels = [f'Used (${used_budget:.2f})', f'Remaining (${remaining_budget:.2f})']
        colors = ['#ff9999', '#66b3ff']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 12})
        
        # Customize the chart
        ax.set_title(f'Budget Utilization\nTotal Budget: ${total_budget:.2f}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add budget efficiency info
        efficiency = (used_budget / total_budget) * 100
        ax.text(0, -1.3, f'Budget Efficiency: {efficiency:.1f}%', 
                ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/budget_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_nutritional_breakdown(self, plan: Dict):
        nutrition_data = plan['nutritional_summary']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total nutritional content
        nutrients = ['Calories', 'Protein (g)', 'Carbs (g)', 'Fat (g)']
        values = [
            nutrition_data['total_calories'],
            nutrition_data['total_protein'],
            nutrition_data.get('total_carbs', 0),
            nutrition_data.get('total_fat', 0)
        ]
        
        bars1 = ax1.bar(nutrients, values, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_title('Total Nutritional Content', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Amount')
        
        # Add value labels on bars
        for bar, value in zip(bars1, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # Daily average nutritional content
        daily_nutrients = ['Daily Calories', 'Daily Protein (g)']
        daily_values = [
            nutrition_data['daily_average_calories'],
            nutrition_data['daily_average_protein']
        ]
        
        bars2 = ax2.bar(daily_nutrients, daily_values, color=['#ff7f0e', '#2ca02c'])
        ax2.set_title('Daily Average Nutrition', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Amount per Day')
        
        # Add value labels on bars
        for bar, value in zip(bars2, daily_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/nutritional_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_category_distribution(self, items: List[Dict]):
        if not items:
            return
        
        # Count items by category
        categories = {}
        category_costs = {}
        
        for item in items:
            category = item.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            category_costs[category] = category_costs.get(category, 0) + item.get('price', 0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Item count by category
        cats = list(categories.keys())
        counts = list(categories.values())
        
        bars1 = ax1.bar(cats, counts, color=plt.cm.Set3(np.linspace(0, 1, len(cats))))
        ax1.set_title('Number of Items by Category', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Items')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars1, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}', ha='center', va='bottom')
        
        # Cost by category
        costs = [category_costs[cat] for cat in cats]
        bars2 = ax2.bar(cats, costs, color=plt.cm.Set3(np.linspace(0, 1, len(cats))))
        ax2.set_title('Total Cost by Category', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Total Cost ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, cost in zip(bars2, costs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${cost:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_price_vs_nutrition(self, items: List[Dict]):
        if not items:
            return
        
        prices = []
        calories = []
        proteins = []
        categories = []
        names = []
        
        for item in items:
            prices.append(item.get('price', 0))
            calories.append(item.get('calories', 0))
            proteins.append(item.get('protein', 0))
            categories.append(item.get('category', 'unknown'))
            names.append(item.get('name', 'Unknown')[:15])  # Truncate long names
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Price vs Calories
        unique_categories = list(set(categories))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        category_colors = {cat: color for cat, color in zip(unique_categories, colors)}
        
        for i, (price, cal, cat, name) in enumerate(zip(prices, calories, categories, names)):
            ax1.scatter(price, cal, c=[category_colors[cat]], s=60, alpha=0.7)
            if i < 10:  # Label only first 10 items to avoid clutter
                ax1.annotate(name, (price, cal), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Price ($)')
        ax1.set_ylabel('Calories')
        ax1.set_title('Price vs Calories', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Price vs Protein
        for i, (price, prot, cat, name) in enumerate(zip(prices, proteins, categories, names)):
            ax2.scatter(price, prot, c=[category_colors[cat]], s=60, alpha=0.7)
            if i < 10:  # Label only first 10 items
                ax2.annotate(name, (price, prot), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Price ($)')
        ax2.set_ylabel('Protein (g)')
        ax2.set_title('Price vs Protein', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=category_colors[cat], 
                                     markersize=8, label=cat) 
                          for cat in unique_categories[:8]]  # Limit legend items
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/price_vs_nutrition.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_expiration_timeline(self, items: List[Dict]):
        if not items:
            return
        
        # Extract expiration data
        expiry_data = []
        for item in items:
            days_to_expiry = item.get('days_to_expiry', 30)
            expiry_data.append({
                'name': item.get('name', 'Unknown')[:20],
                'days_to_expiry': days_to_expiry,
                'price': item.get('price', 0),
                'urgency': 'Critical' if days_to_expiry <= 1 
                          else 'High' if days_to_expiry <= 3
                          else 'Medium' if days_to_expiry <= 7
                          else 'Low'
            })
        
        # Sort by expiration date
        expiry_data.sort(key=lambda x: x['days_to_expiry'])
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(expiry_data) * 0.3)))
        
        # Color mapping for urgency
        urgency_colors = {'Critical': '#ff0000', 'High': '#ff8800', 
                         'Medium': '#ffbb00', 'Low': '#00bb00'}
        
        y_positions = range(len(expiry_data))
        colors = [urgency_colors[item['urgency']] for item in expiry_data]
        days = [item['days_to_expiry'] for item in expiry_data]
        names = [item['name'] for item in expiry_data]
        
        # Create horizontal bar chart
        bars = ax.barh(y_positions, days, color=colors, alpha=0.7)
        
        # Customize the chart
        ax.set_yticks(y_positions)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Days to Expiration')
        ax.set_title('Item Expiration Timeline', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for bar, days_val in zip(bars, days):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                   f'{days_val}d', ha='left', va='center', fontsize=9)
        
        # Add legend for urgency levels
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=urgency)
                          for urgency, color in urgency_colors.items()]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/expiration_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_algorithm_performance(self, performance: Dict):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Items selected by each algorithm
        algorithms = ['Greedy', 'Dynamic Programming', 'Heap Management']
        items_selected = [
            performance['greedy_algorithm']['items_selected'],
            performance['dynamic_programming']['items_selected'],
            performance['heap_management']['items_prioritized']
        ]
        
        bars1 = ax1.bar(algorithms, items_selected, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Items Selected/Processed by Algorithm', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Items')
        
        # Add value labels
        for bar, value in zip(bars1, items_selected):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value}', ha='center', va='bottom')
        
        # Time complexity visualization
        complexities = ['O(n log n)', 'O(nW)', 'O(n log n)']
        ax2.bar(algorithms, [1, 2, 1], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Algorithm Time Complexity (Relative)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Relative Complexity')
        ax2.set_ylim(0, 3)
        
        # Add complexity labels
        for i, (alg, comp) in enumerate(zip(algorithms, complexities)):
            ax2.text(i, 2.5, comp, ha='center', va='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Efficiency comparison (mock data based on algorithm characteristics)
        efficiency_scores = [85, 95, 80]  # Greedy, DP, Heap
        bars3 = ax3.bar(algorithms, efficiency_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_title('Algorithm Efficiency Score', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Efficiency Score (%)')
        ax3.set_ylim(0, 100)
        
        # Add efficiency labels
        for bar, score in zip(bars3, efficiency_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score}%', ha='center', va='bottom')
        
        # Algorithm strengths (radar-like visualization as bar chart)
        strengths = {
            'Speed': [95, 60, 90],
            'Optimality': [70, 100, 75],
            'Memory Usage': [90, 70, 85],
            'Practical Use': [85, 80, 95]
        }
        
        x = np.arange(len(algorithms))
        width = 0.2
        
        for i, (strength, values) in enumerate(strengths.items()):
            ax4.bar(x + i*width, values, width, label=strength, alpha=0.8)
        
        ax4.set_title('Algorithm Characteristics Comparison', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Score (0-100)')
        ax4.set_xlabel('Algorithms')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(['Greedy', 'DP', 'Heap'])
        ax4.legend()
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/algorithm_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_shopping_schedule(self, schedule: Dict):
        if not schedule:
            return
        
        days = list(schedule.keys())
        costs = [schedule[day]['cost'] for day in days]
        item_counts = [schedule[day]['item_count'] for day in days]
        urgent_counts = [schedule[day].get('urgent_items', 0) for day in days]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
        
        # Daily costs
        bars1 = ax1.bar(days, costs, color='skyblue', alpha=0.7)
        ax1.set_title('Daily Shopping Costs', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cost ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, cost in zip(bars1, costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${cost:.2f}', ha='center', va='bottom')
        
        # Daily item counts
        bars2 = ax2.bar(days, item_counts, color='lightgreen', alpha=0.7)
        ax2.set_title('Number of Items per Day', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Items')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars2, item_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count}', ha='center', va='bottom')
        
        # Urgent items per day
        bars3 = ax3.bar(days, urgent_counts, color='salmon', alpha=0.7)
        ax3.set_title('Urgent Items per Day (Expiring Soon)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Urgent Items')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, urgent in zip(bars3, urgent_counts):
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{urgent}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/shopping_schedule.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_macronutrient_balance(self, plan: Dict):
        items = plan['selected_items']
        if not items:
            return
        
        # Calculate total macronutrients
        total_protein = sum(item.get('protein', 0) for item in items)
        total_carbs = sum(item.get('carbohydrates', 0) for item in items)
        total_fat = sum(item.get('fat', 0) for item in items)
        
        # Convert to calories (protein: 4 cal/g, carbs: 4 cal/g, fat: 9 cal/g)
        protein_calories = total_protein * 4
        carb_calories = total_carbs * 4
        fat_calories = total_fat * 9
        
        total_macro_calories = protein_calories + carb_calories + fat_calories
        
        if total_macro_calories == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Macronutrient distribution by calories
        macro_calories = [protein_calories, carb_calories, fat_calories]
        macro_labels = ['Protein', 'Carbohydrates', 'Fat']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        wedges1, texts1, autotexts1 = ax1.pie(macro_calories, labels=macro_labels, colors=colors,
                                             autopct='%1.1f%%', startangle=90)
        ax1.set_title('Macronutrient Distribution by Calories', fontsize=14, fontweight='bold')
        
        # Macronutrient distribution by weight (grams)
        macro_grams = [total_protein, total_carbs, total_fat]
        wedges2, texts2, autotexts2 = ax2.pie(macro_grams, labels=macro_labels, colors=colors,
                                             autopct=lambda pct: f'{pct:.1f}%\n({pct/100*sum(macro_grams):.1f}g)',
                                             startangle=90)
        ax2.set_title('Macronutrient Distribution by Weight', fontsize=14, fontweight='bold')
        
        # Add nutritional recommendations text
        recommendations = []
        protein_pct = protein_calories / total_macro_calories * 100
        carb_pct = carb_calories / total_macro_calories * 100
        fat_pct = fat_calories / total_macro_calories * 100
        
        if protein_pct < 20:
            recommendations.append("Consider adding more protein-rich foods")
        if carb_pct < 40:
            recommendations.append("Consider adding more healthy carbohydrates")
        if fat_pct < 20:
            recommendations.append("Consider adding healthy fats")
        
        if recommendations:
            fig.text(0.5, 0.02, "Recommendations: " + "; ".join(recommendations),
                    ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/macronutrient_balance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self, plan: Dict, preferences: Dict):
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid of subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Budget utilization (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        total_budget = preferences['budget']
        used_budget = plan['total_cost']
        remaining = total_budget - used_budget
        
        ax1.pie([used_budget, remaining], labels=['Used', 'Remaining'], 
               autopct='%1.1f%%', colors=['#ff7f7f', '#7f7fff'])
        ax1.set_title('Budget Utilization')
        
        # 2. Category distribution (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        categories = {}
        for item in plan['selected_items']:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            ax2.bar(list(categories.keys())[:6], list(categories.values())[:6])
            ax2.set_title('Top Categories')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Expiration urgency (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        urgency_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for item in plan['selected_items']:
            days = item.get('days_to_expiry', 30)
            if days <= 1:
                urgency_counts['Critical'] += 1
            elif days <= 3:
                urgency_counts['High'] += 1
            elif days <= 7:
                urgency_counts['Medium'] += 1
            else:
                urgency_counts['Low'] += 1
        
        colors = ['#ff0000', '#ff8800', '#ffbb00', '#00bb00']
        ax3.bar(urgency_counts.keys(), urgency_counts.values(), color=colors)
        ax3.set_title('Expiration Urgency')
        
        # 4. Algorithm performance (top-far-right)
        ax4 = fig.add_subplot(gs[0, 3])
        perf = plan['algorithm_performance']
        alg_items = [perf['greedy_algorithm']['items_selected'],
                    perf['dynamic_programming']['items_selected'],
                    perf['heap_management']['items_prioritized']]
        ax4.bar(['Greedy', 'DP', 'Heap'], alg_items)
        ax4.set_title('Items by Algorithm')
        
        # 5. Price distribution (middle-left)
        ax5 = fig.add_subplot(gs[1, :2])
        prices = [item.get('price', 0) for item in plan['selected_items']]
        ax5.hist(prices, bins=15, alpha=0.7, color='skyblue')
        ax5.set_title('Price Distribution of Selected Items')
        ax5.set_xlabel('Price ($)')
        ax5.set_ylabel('Frequency')
        
        # 6. Nutritional content (middle-right)
        ax6 = fig.add_subplot(gs[1, 2:])
        nutrition = plan['nutritional_summary']
        nutrients = ['Calories', 'Protein (g)']
        values = [nutrition['daily_average_calories'], nutrition['daily_average_protein']]
        
        ax6.bar(nutrients, values, color=['orange', 'green'])
        ax6.set_title('Daily Average Nutrition')
        for i, v in enumerate(values):
            ax6.text(i, v + v*0.01, f'{v:.0f}', ha='center', va='bottom')
        
        # 7. Key metrics text summary (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create summary text
        summary_text = f"""
        SMART SHOPPING PLAN SUMMARY
        
        Total Items Selected: {len(plan['selected_items'])}
        Total Cost: ${plan['total_cost']:.2f} ({plan['budget_used']:.1f}% of budget)
        
        Daily Averages:
        • Calories: {nutrition['daily_average_calories']:.0f}
        • Protein: {nutrition['daily_average_protein']:.1f}g
        
        Urgent Items: {sum(1 for item in plan['selected_items'] if item.get('days_to_expiry', 30) <= 3)}
        Categories Covered: {len(set(item.get('category', 'unknown') for item in plan['selected_items']))}
        
        Algorithm Performance:
        • Greedy: {perf['greedy_algorithm']['items_selected']} items selected
        • Dynamic Programming: {perf['dynamic_programming']['items_selected']} items optimized
        • Heap Management: {perf['heap_management']['urgent_items']} urgent items prioritized
        """
        
        ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # Add title to entire dashboard
        fig.suptitle('Smart Shopping Planner - Comprehensive Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.savefig(f'{self.output_dir}/dashboard_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_cost_efficiency_analysis(self, items: List[Dict]):
        if not items:
            return
        
        # Calculate efficiency metrics
        efficiency_data = []
        for item in items:
            calories = item.get('calories', 0)
            protein = item.get('protein', 0)
            price = item.get('price', 0.01)
            
            if price > 0:
                calories_per_dollar = calories / price
                protein_per_dollar = protein / price
                
                efficiency_data.append({
                    'name': item.get('name', 'Unknown')[:15],
                    'category': item.get('category', 'unknown'),
                    'price': price,
                    'calories_per_dollar': calories_per_dollar,
                    'protein_per_dollar': protein_per_dollar,
                    'overall_efficiency': (calories_per_dollar + protein_per_dollar * 4) / 2
                })
        
        # Sort by overall efficiency
        efficiency_data.sort(key=lambda x: x['overall_efficiency'], reverse=True)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top 10 most efficient items by calories per dollar
        top_items = efficiency_data[:10]
        names = [item['name'] for item in top_items]
        calories_eff = [item['calories_per_dollar'] for item in top_items]
        
        bars1 = ax1.barh(range(len(names)), calories_eff, color='orange', alpha=0.7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names)
        ax1.set_xlabel('Calories per Dollar')
        ax1.set_title('Top 10: Calories per Dollar')
        ax1.invert_yaxis()
        
        # Top 10 most efficient items by protein per dollar
        protein_eff = [item['protein_per_dollar'] for item in top_items]
        bars2 = ax2.barh(range(len(names)), protein_eff, color='green', alpha=0.7)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Protein (g) per Dollar')
        ax2.set_title('Top 10: Protein per Dollar')
        ax2.invert_yaxis()
        
        # Efficiency by category
        category_efficiency = {}
        for item in efficiency_data:
            cat = item['category']
            if cat not in category_efficiency:
                category_efficiency[cat] = []
            category_efficiency[cat].append(item['overall_efficiency'])
        
        categories = list(category_efficiency.keys())
        avg_efficiency = [np.mean(category_efficiency[cat]) for cat in categories]
        
        bars3 = ax3.bar(categories, avg_efficiency, color=plt.cm.Set3(np.linspace(0, 1, len(categories))))
        ax3.set_ylabel('Average Efficiency Score')
        ax3.set_title('Average Efficiency by Category')
        ax3.tick_params(axis='x', rotation=45)
        
        # Price vs Efficiency scatter
        prices = [item['price'] for item in efficiency_data]
        overall_eff = [item['overall_efficiency'] for item in efficiency_data]
        categories_scatter = [item['category'] for item in efficiency_data]
        
        unique_cats = list(set(categories_scatter))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cats)))
        cat_colors = {cat: color for cat, color in zip(unique_cats, colors)}
        
        for i, (price, eff, cat) in enumerate(zip(prices, overall_eff, categories_scatter)):
            ax4.scatter(price, eff, c=[cat_colors[cat]], s=60, alpha=0.7)
        
        ax4.set_xlabel('Price ($)')
        ax4.set_ylabel('Overall Efficiency Score')
        ax4.set_title('Price vs Efficiency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cost_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_plan_summary_text(self, plan: Dict, preferences: Dict):
        """Save a text summary of the shopping plan."""
        summary_file = f"{self.output_dir}/../shopping_plan_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SMART SHOPPING PLANNER - DETAILED SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Budget Information
            f.write("BUDGET ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Budget: ${preferences['budget']:.2f}\n")
            f.write(f"Total Cost: ${plan['total_cost']:.2f}\n")
            f.write(f"Budget Used: {plan['budget_used']:.1f}%\n")
            f.write(f"Remaining: ${preferences['budget'] - plan['total_cost']:.2f}\n\n")
            
            # Nutritional Summary
            nutrition = plan['nutritional_summary']
            f.write("NUTRITIONAL SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Calories: {nutrition['total_calories']:.0f}\n")
            f.write(f"Total Protein: {nutrition['total_protein']:.1f}g\n")
            f.write(f"Daily Avg Calories: {nutrition['daily_average_calories']:.0f}\n")
            f.write(f"Daily Avg Protein: {nutrition['daily_average_protein']:.1f}g\n\n")
            
            # Selected Items
            f.write("SELECTED ITEMS\n")
            f.write("-" * 20 + "\n")
            for i, item in enumerate(plan['selected_items'], 1):
                f.write(f"{i:2d}. {item['name']:<25} ${item['price']:>6.2f} ")
                f.write(f"({item.get('category', 'unknown'):<12}) ")
                f.write(f"Expires: {item.get('days_to_expiry', 'N/A')} days\n")
            
            # Algorithm Performance
            f.write(f"\nALGORITHM PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            perf = plan['algorithm_performance']
            f.write(f"Greedy Algorithm: {perf['greedy_algorithm']['items_selected']} items\n")
            f.write(f"Dynamic Programming: {perf['dynamic_programming']['items_selected']} items\n")
            f.write(f"Heap Management: {perf['heap_management']['urgent_items']} urgent items\n")
            
            # Waste Prevention Tips
            f.write(f"\nWASTE PREVENTION TIPS\n")
            f.write("-" * 20 + "\n")
            for tip in plan['waste_reduction_tips']:
                f.write(f"• {tip}\n")
        
        print(f"Detailed summary saved to {summary_file}")
