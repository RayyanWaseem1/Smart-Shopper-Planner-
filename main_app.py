import pandas as pd
import numpy as np
import heapq
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Optional
import json

from greedy_algorithm import GreedyOptimizer
from dynamic_programming import DynamicProgrammingOptimizer
from heap_manager import ExpirationHeapManager

from data_processor import DataProcessor
from visualizer import ShoppingPlannerVisualizer

class SmartShoppingPlanner:
    def __init__(self, data_file: str = "data/food_nutrition_dataset.csv"):
        """Initialize the Smart Shopping Planner with data and algorithms."""
        self.data_processor = DataProcessor(data_file)
        self.greedy_optimizer = GreedyOptimizer()
        self.dp_optimizer = DynamicProgrammingOptimizer()
        self.heap_manager = ExpirationHeapManager()
        self.visualizer = ShoppingPlannerVisualizer()
        
        # Load and process the dataset
        self.df = self.data_processor.load_and_process_data()
        print(f"Loaded {len(self.df)} food items from dataset")
        
    def plan_shopping(self, user_preferences: Dict) -> Dict:
        """
        Main method to create an optimized shopping plan based on user preferences.
        
        Args:
            user_preferences: Dictionary containing:
                - budget: float
                - nutritional_goals: dict with min_calories, min_protein, etc.
                - dietary_restrictions: list of restrictions
                - shopping_list: list of preferred items
                - planning_days: int (number of days to plan for)
        
        Returns:
            Complete shopping plan with optimized selections
        """
        print("\n🛒 Creating your Smart Shopping Plan...")
        
        # Filter data based on dietary restrictions
        filtered_df = self.data_processor.filter_by_restrictions(
            self.df, user_preferences.get('dietary_restrictions', [])
        )
        
        results = {}
        
        # 1. GREEDY ALGORITHM: Get cost-effective items
        print("\n📊 Phase 1: Applying Greedy Algorithm for cost optimization...")
        greedy_selection = self.greedy_optimizer.optimize_by_cost_nutrition_ratio(
            filtered_df, 
            user_preferences['budget'],
            user_preferences.get('nutritional_goals', {})
        )
        results['greedy_selection'] = greedy_selection
        
        # 2. DYNAMIC PROGRAMMING: Solve the diet knapsack problem
        print("\n🎯 Phase 2: Applying Dynamic Programming for nutritional optimization...")
        dp_selection = self.dp_optimizer.solve_diet_knapsack(
            filtered_df,
            user_preferences['budget'],
            user_preferences.get('nutritional_goals', {})
        )
        results['dp_selection'] = dp_selection
        
        # 3. HEAP MANAGEMENT: Prioritize items by expiration
        print("\n⏰ Phase 3: Applying Heap Management for expiration optimization...")
        
        # Add expiration dates to selected items
        combined_selection = self._combine_selections(greedy_selection, dp_selection)
        items_with_expiration = self.data_processor.add_expiration_dates(combined_selection)
        
        expiration_priority = self.heap_manager.prioritize_by_expiration(items_with_expiration)
        results['expiration_priority'] = expiration_priority
        
        # 4. CREATE FINAL OPTIMIZED PLAN
        final_plan = self._create_final_plan(
            results, user_preferences, user_preferences.get('planning_days', 7)
        )
        
        return final_plan
    
    def _combine_selections(self, greedy_items: List, dp_items: List) -> List:
        """Combine and deduplicate items from different algorithms."""
        combined_dict = {}
        
        # Add greedy items
        for item in greedy_items:
            item_name = item['name']
            combined_dict[item_name] = item
            
        # Add DP items (may override greedy if better)
        for item in dp_items:
            item_name = item['name']
            if item_name in combined_dict:
                # Keep the one with better cost-nutrition ratio
                existing_ratio = combined_dict[item_name].get('cost_nutrition_ratio', float('inf'))
                new_ratio = item.get('cost_nutrition_ratio', float('inf'))
                if new_ratio < existing_ratio:
                    combined_dict[item_name] = item
            else:
                combined_dict[item_name] = item
                
        return list(combined_dict.values())
    
    def _create_final_plan(self, algorithm_results: Dict, preferences: Dict, planning_days: int) -> Dict:
        """Create the final optimized shopping plan."""
        
        # Get the prioritized items by expiration
        priority_items = algorithm_results['expiration_priority']
        
        # Calculate totals
        total_cost = sum(item['price'] for item in priority_items)
        total_calories = sum(item.get('calories', 0) for item in priority_items)
        total_protein = sum(item.get('protein', 0) for item in priority_items)
        
        # Create shopping schedule
        shopping_schedule = self._create_shopping_schedule(priority_items, planning_days)
        
        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(
            algorithm_results, preferences
        )
        
        final_plan = {
            'selected_items': priority_items,
            'total_cost': total_cost,
            'budget_used': (total_cost / preferences['budget']) * 100,
            'nutritional_summary': {
                'total_calories': total_calories,
                'total_protein': total_protein,
                'daily_average_calories': total_calories / planning_days,
                'daily_average_protein': total_protein / planning_days
            },
            'shopping_schedule': shopping_schedule,
            'algorithm_performance': performance_metrics,
            'waste_reduction_tips': self._generate_waste_reduction_tips(priority_items)
        }
        
        return final_plan
    
    def _create_shopping_schedule(self, items: List, days: int) -> Dict:
        """Create a day-by-day shopping schedule."""
        schedule = {}
        items_per_day = max(1, len(items) // days)
        
        for day in range(1, days + 1):
            start_idx = (day - 1) * items_per_day
            end_idx = start_idx + items_per_day if day < days else len(items)
            
            day_items = items[start_idx:end_idx]
            day_cost = sum(item['price'] for item in day_items)
            
            schedule[f'Day {day}'] = {
                'items': day_items,
                'cost': day_cost,
                'priority_level': 'High' if any(item.get('days_to_expiry', 10) <= 3 for item in day_items) else 'Medium'
            }
            
        return schedule
    
    def _calculate_performance_metrics(self, results: Dict, preferences: Dict) -> Dict:
        """Calculate performance metrics for each algorithm."""
        metrics = {
            'greedy_algorithm': {
                'items_selected': len(results['greedy_selection']),
                'avg_cost_nutrition_ratio': np.mean([
                    item.get('cost_nutrition_ratio', 0) for item in results['greedy_selection']
                ]),
                'execution_time': 'O(n log n)'
            },
            'dynamic_programming': {
                'items_selected': len(results['dp_selection']),
                'optimal_solution': True,
                'execution_time': 'O(nW) where W is budget'
            },
            'heap_management': {
                'items_prioritized': len(results['expiration_priority']),
                'urgent_items': len([
                    item for item in results['expiration_priority'] 
                    if item.get('days_to_expiry', 10) <= 3
                ]),
                'execution_time': 'O(n log n)'
            }
        }
        return metrics
    
    def _generate_waste_reduction_tips(self, items: List) -> List[str]:
        """Generate personalized tips to reduce food waste."""
        tips = []
        
        # Check for items expiring soon
        urgent_items = [item for item in items if item.get('days_to_expiry', 10) <= 3]
        if urgent_items:
            tips.append(f"🚨 Priority: Use {len(urgent_items)} items within 3 days to avoid waste")
            
        # Check for perishable items
        perishable = [item for item in items if item.get('category') in ['fruits', 'vegetables', 'dairy']]
        if perishable:
            tips.append(f"🥬 Store {len(perishable)} perishable items properly (refrigerate fruits/vegetables)")
            
        # General tips
        tips.extend([
            "📅 Plan meals around expiration dates",
            "🥡 Consider freezing items you won't use immediately",
            "📝 Update your shopping list based on what you actually consume"
        ])
        
        return tips
    
    def display_plan(self, plan: Dict):
        """Display the shopping plan in a user-friendly format."""
        print("\n" + "="*60)
        print("🎉 YOUR OPTIMIZED SMART SHOPPING PLAN")
        print("="*60)
        
        # Summary
        print(f"\n💰 Budget Analysis:")
        print(f"   Total Cost: ${plan['total_cost']:.2f}")
        print(f"   Budget Used: {plan['budget_used']:.1f}%")
        
        # Nutritional summary
        nutrition = plan['nutritional_summary']
        print(f"\n🥗 Nutritional Summary:")
        print(f"   Total Calories: {nutrition['total_calories']:.0f}")
        print(f"   Total Protein: {nutrition['total_protein']:.1f}g")
        print(f"   Daily Avg Calories: {nutrition['daily_average_calories']:.0f}")
        print(f"   Daily Avg Protein: {nutrition['daily_average_protein']:.1f}g")
        
        # Shopping schedule
        print(f"\n📅 Shopping Schedule:")
        for day, details in plan['shopping_schedule'].items():
            print(f"\n   {day} ({details['priority_level']} Priority) - ${details['cost']:.2f}:")
            for item in details['items'][:3]:  # Show first 3 items
                expiry_info = f" (expires in {item.get('days_to_expiry', 'N/A')} days)" if 'days_to_expiry' in item else ""
                print(f"     • {item['name']} - ${item['price']:.2f}{expiry_info}")
            if len(details['items']) > 3:
                print(f"     ... and {len(details['items']) - 3} more items")
        
        # Waste reduction tips
        print(f"\n♻️ Waste Reduction Tips:")
        for tip in plan['waste_reduction_tips']:
            print(f"   {tip}")
        
        # Algorithm performance
        print(f"\n⚡ Algorithm Performance:")
        perf = plan['algorithm_performance']
        print(f"   Greedy: {perf['greedy_algorithm']['items_selected']} items selected")
        print(f"   Dynamic Programming: {perf['dynamic_programming']['items_selected']} items optimized")
        print(f"   Heap Management: {perf['heap_management']['urgent_items']} urgent items prioritized")

def main():
    """Main function to run the Smart Shopping Planner."""
    print("🛒 Welcome to Smart Shopping Planner!")
    print("=" * 50)
    
    # Initialize the planner
    planner = SmartShoppingPlanner()
    
    # Example user preferences (you can modify these)
    user_preferences = {
        'budget': 150.0,
        'nutritional_goals': {
            'min_calories': 2000,
            'min_protein': 50,
            'min_carbs': 250,
            'min_fat': 65
        },
        'dietary_restrictions': ['vegetarian'],  # Options: ['vegetarian', 'vegan', 'gluten_free', 'low_sodium']
        'shopping_list': ['apples', 'chicken', 'rice', 'milk', 'bread'],
        'planning_days': 7
    }
    
    # Create the shopping plan
    plan = planner.plan_shopping(user_preferences)
    
    # Display the results
    planner.display_plan(plan)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    planner.visualizer.create_all_visualizations(plan, user_preferences)
    
    # Save the plan to JSON
    with open('output/shopping_plan.json', 'w') as f:
        json.dump(plan, f, indent=2, default=str)
    print("\n💾 Plan saved to 'output/shopping_plan.json'")

if __name__ == "__main__":
    main()
