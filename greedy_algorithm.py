import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class GreedyOptimizer:
    def __init__(self):
        self.name = "Greedy Cost-Nutrition Optimizer"
    
    def calculate_cost_nutrition_ratio(self, item: pd.Series) -> float:
        """
        Calculate cost-to-nutrition ratio for an item.
        Lower ratio means better value (more nutrition per dollar).
        
        Args:
            item: Pandas Series representing a food item
            
        Returns:
            Cost-to-nutrition ratio (lower is better)
        """
        # Nutritional value calculation (weighted sum of important nutrients)
        calories = max(item.get('calories', 0), 1)  # Avoid division by zero
        protein = max(item.get('protein', 0), 0.1)
        carbs = max(item.get('carbohydrates', 0), 0.1)
        fat = max(item.get('fat', 0), 0.1)
        
        # Weighted nutritional score (protein has higher weight for health)
        nutritional_score = (calories * 0.3) + (protein * 0.4) + (carbs * 0.2) + (fat * 0.1)
        
        # Price per unit of nutrition
        price = max(item.get('price', 0.01), 0.01)  # Avoid division by zero
        ratio = price / nutritional_score
        
        return ratio
    
    def optimize_by_cost_nutrition_ratio(self, df: pd.DataFrame, budget: float, 
                                       nutritional_goals: Dict = None) -> List[Dict]:
        """
        Greedy algorithm to select items with best cost-to-nutrition ratio within budget.
        
        Args:
            df: DataFrame containing food items
            budget: Maximum budget available
            nutritional_goals: Dictionary of nutritional requirements (optional)
            
        Returns:
            List of selected items with their details
        """
        if df.empty:
            return []
        
        # Calculate cost-nutrition ratio for all items
        df_copy = df.copy()
        df_copy['cost_nutrition_ratio'] = df_copy.apply(self.calculate_cost_nutrition_ratio, axis=1)
        
        # Sort by cost-nutrition ratio (ascending - lower is better)
        sorted_items = df_copy.sort_values('cost_nutrition_ratio').reset_index(drop=True)
        
        selected_items = []
        remaining_budget = budget
        nutritional_totals = {
            'calories': 0,
            'protein': 0,
            'carbohydrates': 0,
            'fat': 0
        }
        
        print(f"Starting greedy selection with budget: ${budget}")
        
        # Greedy selection: pick items with best ratio that fit in budget
        for idx, item in sorted_items.iterrows():
            item_price = item.get('price', 0)
            
            # Check if item fits in remaining budget
            if item_price <= remaining_budget:
                # Add item to selection
                selected_item = {
                    'name': item.get('name', f'Item_{idx}'),
                    'price': item_price,
                    'calories': item.get('calories', 0),
                    'protein': item.get('protein', 0),
                    'carbohydrates': item.get('carbohydrates', 0),
                    'fat': item.get('fat', 0),
                    'cost_nutrition_ratio': item['cost_nutrition_ratio'],
                    'category': item.get('category', 'unknown'),
                    'selection_method': 'greedy'
                }
                
                selected_items.append(selected_item)
                remaining_budget -= item_price
                
                # Update nutritional totals
                nutritional_totals['calories'] += item.get('calories', 0)
                nutritional_totals['protein'] += item.get('protein', 0)
                nutritional_totals['carbohydrates'] += item.get('carbohydrates', 0)
                nutritional_totals['fat'] += item.get('fat', 0)
                
                print(f"Selected: {selected_item['name']} - ${item_price:.2f} "
                      f"(ratio: {item['cost_nutrition_ratio']:.3f}, remaining budget: ${remaining_budget:.2f})")
                
                # Early termination if nutritional goals are met (optional optimization)
                if nutritional_goals and self._goals_satisfied(nutritional_totals, nutritional_goals):
                    print("Nutritional goals satisfied early - stopping greedy selection")
                    break
                
                # Stop if budget is exhausted
                if remaining_budget < 0.01:
                    break
        
        print(f"\nGreedy Algorithm Results:")
        print(f"Items selected: {len(selected_items)}")
        print(f"Total cost: ${budget - remaining_budget:.2f}")
        print(f"Budget utilization: {((budget - remaining_budget) / budget * 100):.1f}%")
        print(f"Average cost-nutrition ratio: {np.mean([item['cost_nutrition_ratio'] for item in selected_items]):.3f}")
        
        return selected_items
    
    def optimize_for_specific_nutrients(self, df: pd.DataFrame, budget: float, 
                                      target_nutrient: str, min_amount: float) -> List[Dict]:
        """
        Greedy algorithm optimized for a specific nutrient (e.g., protein, calories).
        
        Args:
            df: DataFrame containing food items
            budget: Maximum budget available
            target_nutrient: Name of the nutrient to optimize for
            min_amount: Minimum amount of the nutrient needed
            
        Returns:
            List of selected items
        """
        if df.empty or target_nutrient not in df.columns:
            return []
        
        # Calculate nutrient-to-cost ratio (higher is better for this case)
        df_copy = df.copy()
        df_copy['nutrient_cost_ratio'] = df_copy.apply(
            lambda row: max(row.get(target_nutrient, 0), 0.1) / max(row.get('price', 0.01), 0.01), 
            axis=1
        )
        
        # Sort by nutrient-cost ratio (descending - higher is better)
        sorted_items = df_copy.sort_values('nutrient_cost_ratio', ascending=False).reset_index(drop=True)
        
        selected_items = []
        remaining_budget = budget
        nutrient_total = 0
        
        print(f"Optimizing for {target_nutrient} (target: {min_amount}) with budget: ${budget}")
        
        for idx, item in sorted_items.iterrows():
            item_price = item.get('price', 0)
            item_nutrient = item.get(target_nutrient, 0)
            
            if item_price <= remaining_budget:
                selected_item = {
                    'name': item.get('name', f'Item_{idx}'),
                    'price': item_price,
                    target_nutrient: item_nutrient,
                    'nutrient_cost_ratio': item['nutrient_cost_ratio'],
                    'category': item.get('category', 'unknown'),
                    'selection_method': f'greedy_{target_nutrient}'
                }
                
                # Add all nutritional info
                for col in ['calories', 'protein', 'carbohydrates', 'fat']:
                    if col in item:
                        selected_item[col] = item[col]
                
                selected_items.append(selected_item)
                remaining_budget -= item_price
                nutrient_total += item_nutrient
                
                print(f"Selected: {selected_item['name']} - ${item_price:.2f} "
                      f"({target_nutrient}: {item_nutrient:.1f}, ratio: {item['nutrient_cost_ratio']:.3f})")
                
                # Check if target is reached
                if nutrient_total >= min_amount:
                    print(f"Target {target_nutrient} amount reached: {nutrient_total:.1f}")
                    break
                
                if remaining_budget < 0.01:
                    break
        
        return selected_items
    
    def optimize_by_preference_list(self, df: pd.DataFrame, budget: float, 
                                  preference_list: List[str]) -> List[Dict]:
        """
        Greedy algorithm that prioritizes items from user's preference list.
        
        Args:
            df: DataFrame containing food items
            budget: Maximum budget available
            preference_list: List of preferred item names
            
        Returns:
            List of selected items
        """
        if df.empty:
            return []
        
        selected_items = []
        remaining_budget = budget
        
        # First, try to get preferred items
        print(f"Prioritizing preferred items: {preference_list}")
        
        for preferred_item in preference_list:
            # Find items matching the preference (case-insensitive partial matching)
            matching_items = df[df['name'].str.contains(preferred_item, case=False, na=False)]
            
            if not matching_items.empty:
                # Get the cheapest matching item that fits budget
                affordable_matches = matching_items[matching_items['price'] <= remaining_budget]
                
                if not affordable_matches.empty:
                    best_match = affordable_matches.loc[affordable_matches['price'].idxmin()]
                    
                    selected_item = {
                        'name': best_match.get('name', preferred_item),
                        'price': best_match.get('price', 0),
                        'calories': best_match.get('calories', 0),
                        'protein': best_match.get('protein', 0),
                        'carbohydrates': best_match.get('carbohydrates', 0),
                        'fat': best_match.get('fat', 0),
                        'category': best_match.get('category', 'unknown'),
                        'selection_method': 'greedy_preference',
                        'preferred': True
                    }
                    
                    selected_items.append(selected_item)
                    remaining_budget -= selected_item['price']
                    
                    print(f"Found preferred: {selected_item['name']} - ${selected_item['price']:.2f}")
        
        # Fill remaining budget with best cost-nutrition ratio items
        if remaining_budget > 1.0:  # If significant budget remains
            remaining_df = df[~df['name'].isin([item['name'] for item in selected_items])]
            additional_items = self.optimize_by_cost_nutrition_ratio(remaining_df, remaining_budget)
            
            # Mark these as non-preferred
            for item in additional_items:
                item['preferred'] = False
                
            selected_items.extend(additional_items)
        
        return selected_items
    
    def _goals_satisfied(self, current_totals: Dict, goals: Dict) -> bool:
        """Check if nutritional goals are satisfied."""
        for nutrient, target in goals.items():
            if nutrient.startswith('min_'):
                nutrient_name = nutrient[4:]  # Remove 'min_' prefix
                if current_totals.get(nutrient_name, 0) < target:
                    return False
        return True
    
    def analyze_selection_efficiency(self, selected_items: List[Dict]) -> Dict:
        """
        Analyze the efficiency of the greedy selection.
        
        Args:
            selected_items: List of selected items from greedy algorithm
            
        Returns:
            Dictionary with efficiency metrics
        """
        if not selected_items:
            return {'error': 'No items selected'}
        
        total_cost = sum(item['price'] for item in selected_items)
        total_nutrition_score = sum(
            (item.get('calories', 0) * 0.3) + 
            (item.get('protein', 0) * 0.4) + 
            (item.get('carbohydrates', 0) * 0.2) + 
            (item.get('fat', 0) * 0.1) 
            for item in selected_items
        )
        
        avg_ratio = np.mean([item.get('cost_nutrition_ratio', 0) for item in selected_items])
        
        # Category distribution
        categories = {}
        for item in selected_items:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_items': len(selected_items),
            'total_cost': total_cost,
            'total_nutrition_score': total_nutrition_score,
            'average_cost_nutrition_ratio': avg_ratio,
            'cost_per_nutrition_unit': total_cost / max(total_nutrition_score, 1),
            'category_distribution': categories,
            'efficiency_grade': self._calculate_efficiency_grade(avg_ratio)
        }
    
    def _calculate_efficiency_grade(self, avg_ratio: float) -> str:
        """Calculate efficiency grade based on average cost-nutrition ratio."""
        if avg_ratio < 0.1:
            return 'A+ (Excellent)'
        elif avg_ratio < 0.2:
            return 'A (Very Good)'
        elif avg_ratio < 0.3:
            return 'B (Good)'
        elif avg_ratio < 0.5:
            return 'C (Average)'
        else:
            return 'D (Poor)'
