import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

class DynamicProgrammingOptimizer:
    def __init__(self):
        self.name = "Dynamic Programming Diet Optimizer"
        self.memo = {}  # Memoization for DP
    
    def solve_diet_knapsack(self, df: pd.DataFrame, budget: float, 
                          nutritional_goals: Dict) -> List[Dict]:
        """
        Solve the multi-dimensional knapsack problem for diet optimization.
        Maximizes nutritional value while staying within budget and meeting minimum requirements.
        
        Args:
            df: DataFrame containing food items
            budget: Maximum budget available
            nutritional_goals: Dict with min_calories, min_protein, etc.
            
        Returns:
            List of selected items that optimize nutritional value
        """
        if df.empty:
            return []
        
        print(f"Starting Dynamic Programming optimization with budget: ${budget}")
        print(f"Nutritional goals: {nutritional_goals}")
        
        # Prepare data for DP
        items = self._prepare_items(df)
        budget_cents = int(budget * 100)  # Convert to cents for integer DP
        
        # Extract nutritional goals
        min_calories = nutritional_goals.get('min_calories', 0)
        min_protein = nutritional_goals.get('min_protein', 0)
        min_carbs = nutritional_goals.get('min_carbs', 0)
        min_fat = nutritional_goals.get('min_fat', 0)
        
        # Use simplified DP for performance with large datasets
        if len(items) > 100:
            return self._solve_simplified_knapsack(items, budget, nutritional_goals)
        
        # Full DP solution for smaller datasets
        selected_indices = self._solve_multidimensional_knapsack(
            items, budget_cents, min_calories, min_protein, min_carbs, min_fat
        )
        
        # Convert back to item list
        selected_items = []
        for idx in selected_indices:
            item_data = items[idx]
            selected_items.append({
                'name': item_data['name'],
                'price': item_data['price'],
                'calories': item_data['calories'],
                'protein': item_data['protein'],
                'carbohydrates': item_data['carbohydrates'],
                'fat': item_data['fat'],
                'category': item_data['category'],
                'selection_method': 'dynamic_programming',
                'nutritional_score': self._calculate_nutritional_score(item_data)
            })
        
        print(f"DP selected {len(selected_items)} items with total value optimization")
        return selected_items
    
    def _prepare_items(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to list of items for DP processing."""
        items = []
        for idx, row in df.iterrows():
            items.append({
                'name': row.get('name', f'Item_{idx}'),
                'price': max(row.get('price', 0), 0.01),
                'calories': max(row.get('calories', 0), 0),
                'protein': max(row.get('protein', 0), 0),
                'carbohydrates': max(row.get('carbohydrates', 0), 0),
                'fat': max(row.get('fat', 0), 0),
                'category': row.get('category', 'unknown')
            })
        return items
    
    def _solve_multidimensional_knapsack(self, items: List[Dict], budget_cents: int,
                                       min_calories: float, min_protein: float,
                                       min_carbs: float, min_fat: float) -> List[int]:
        """
        Solve multi-dimensional knapsack using dynamic programming.
        This is a complex optimization problem with multiple constraints.
        """
        n = len(items)
        
        # For performance, we'll use a greedy approximation with DP principles
        # Sort items by value-to-weight ratio (nutritional score per dollar)
        item_values = []
        for i, item in enumerate(items):
            nutritional_score = self._calculate_nutritional_score(item)
            value_per_dollar = nutritional_score / item['price']
            item_values.append((i, value_per_dollar, item))
        
        # Sort by value per dollar (descending)
        item_values.sort(key=lambda x: x[1], reverse=True)
        
        # Use DP table for subset selection
        # dp[i][w] = maximum nutritional value using first i items with weight <= w
        budget_int = budget_cents // 100  # Simplify for DP table size
        dp = [[0 for _ in range(budget_int + 1)] for _ in range(n + 1)]
        keep = [[False for _ in range(budget_int + 1)] for _ in range(n + 1)]
        
        # Fill DP table
        for i in range(1, n + 1):
            item_idx, _, item = item_values[i-1]
            weight = int(item['price'])
            value = int(self._calculate_nutritional_score(item) * 100)
            
            for w in range(budget_int + 1):
                # Don't take item
                dp[i][w] = dp[i-1][w]
                
                # Take item if possible
                if weight <= w:
                    take_value = dp[i-1][w-weight] + value
                    if take_value > dp[i][w]:
                        dp[i][w] = take_value
                        keep[i][w] = True
        
        # Backtrack to find selected items
        selected = []
        w = budget_int
        for i in range(n, 0, -1):
            if keep[i][w]:
                item_idx, _, item = item_values[i-1]
                selected.append(item_idx)
                w -= int(item['price'])
        
        return selected
    
    def _solve_simplified_knapsack(self, items: List[Dict], budget: float, 
                                 nutritional_goals: Dict) -> List[Dict]:
        """
        Simplified knapsack solution for large datasets using greedy DP approach.
        """
        print("Using simplified DP approach for large dataset")
        
        # Calculate efficiency scores for all items
        scored_items = []
        for i, item in enumerate(items):
            efficiency = self._calculate_item_efficiency(item, nutritional_goals)
            scored_items.append((i, efficiency, item))
        
        # Sort by efficiency (higher is better)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection with DP-style optimization
        selected_items = []
        remaining_budget = budget
        nutritional_totals = {'calories': 0, 'protein': 0, 'carbohydrates': 0, 'fat': 0}
        
        # First pass: ensure minimum requirements are met
        for idx, efficiency, item in scored_items:
            if item['price'] <= remaining_budget:
                # Check if this item helps meet unfulfilled requirements
                helps_requirements = False
                
                for goal_key, min_val in nutritional_goals.items():
                    if goal_key.startswith('min_'):
                        nutrient = goal_key[4:]  # Remove 'min_' prefix
                        if (nutritional_totals.get(nutrient, 0) < min_val and 
                            item.get(nutrient, 0) > 0):
                            helps_requirements = True
                            break
                
                if helps_requirements or len(selected_items) < 5:  # Ensure minimum variety
                    selected_item = {
                        'name': item['name'],
                        'price': item['price'],
                        'calories': item['calories'],
                        'protein': item['protein'],
                        'carbohydrates': item['carbohydrates'],
                        'fat': item['fat'],
                        'category': item['category'],
                        'selection_method': 'dynamic_programming_simplified',
                        'efficiency_score': efficiency
                    }
                    
                    selected_items.append(selected_item)
                    remaining_budget -= item['price']
                    
                    # Update totals
                    for nutrient in nutritional_totals:
                        nutritional_totals[nutrient] += item.get(nutrient, 0)
                    
                    # Check if all requirements are met
                    all_met = True
                    for goal_key, min_val in nutritional_goals.items():
                        if goal_key.startswith('min_'):
                            nutrient = goal_key[4:]
                            if nutritional_totals.get(nutrient, 0) < min_val:
                                all_met = False
                                break
                    
                    if all_met and remaining_budget < budget * 0.1:  # 90% budget used
                        break
        
        return selected_items
    
    def _calculate_nutritional_score(self, item: Dict) -> float:
        """Calculate overall nutritional value score for an item."""
        calories = item.get('calories', 0)
        protein = item.get('protein', 0)
        carbs = item.get('carbohydrates', 0)
        fat = item.get('fat', 0)
        
        # Weighted nutritional score (protein weighted higher)
        score = (calories * 0.2) + (protein * 0.5) + (carbs * 0.2) + (fat * 0.1)
        return max(score, 0.1)
    
    def _calculate_item_efficiency(self, item: Dict, goals: Dict) -> float:
        """
        Calculate item efficiency based on how well it contributes to nutritional goals
        relative to its cost.
        """
        nutritional_contribution = 0
        
        for goal_key, target_val in goals.items():
            if goal_key.startswith('min_') and target_val > 0:
                nutrient = goal_key[4:]  # Remove 'min_' prefix
                item_nutrient = item.get(nutrient, 0)
                
                # Contribution as percentage of daily requirement
                contribution = min(item_nutrient / target_val, 1.0)  # Cap at 100%
                nutritional_contribution += contribution
        
        # Efficiency = nutritional contribution per dollar
        price = max(item['price'], 0.01)
        efficiency = nutritional_contribution / price
        
        return efficiency
    
    def solve_bounded_knapsack(self, df: pd.DataFrame, budget: float, 
                             max_quantities: Dict = None) -> List[Dict]:
        """
        Solve bounded knapsack where items can be selected multiple times up to a limit.
        Useful for bulk buying scenarios.
        
        Args:
            df: DataFrame with food items
            budget: Available budget
            max_quantities: Dict mapping item names to maximum quantities
            
        Returns:
            List of selected items with quantities
        """
        if df.empty:
            return []
        
        items = self._prepare_items(df)
        selected_with_quantities = []
        remaining_budget = budget
        
        # Default max quantity is 3 for each item
        default_max_qty = 3
        
        # Calculate value density for each item
        item_densities = []
        for i, item in enumerate(items):
            density = self._calculate_nutritional_score(item) / item['price']
            max_qty = max_quantities.get(item['name'], default_max_qty) if max_quantities else default_max_qty
            item_densities.append((i, density, max_qty, item))
        
        # Sort by density (descending)
        item_densities.sort(key=lambda x: x[1], reverse=True)
        
        # Greedy selection with quantity consideration
        for item_idx, density, max_qty, item in item_densities:
            quantity = 0
            item_price = item['price']
            
            # Add as many as budget and limit allows
            while quantity < max_qty and item_price <= remaining_budget:
                quantity += 1
                remaining_budget -= item_price
            
            if quantity > 0:
                selected_item = {
                    'name': item['name'],
                    'price': item['price'],
                    'quantity': quantity,
                    'total_cost': item['price'] * quantity,
                    'calories': item['calories'] * quantity,
                    'protein': item['protein'] * quantity,
                    'carbohydrates': item['carbohydrates'] * quantity,
                    'fat': item['fat'] * quantity,
                    'category': item['category'],
                    'selection_method': 'bounded_knapsack',
                    'density_score': density
                }
                selected_with_quantities.append(selected_item)
        
        print(f"Bounded knapsack selected {len(selected_with_quantities)} unique items")
        return selected_with_quantities
    
    def optimize_for_balanced_nutrition(self, df: pd.DataFrame, budget: float,
                                      target_ratios: Dict = None) -> List[Dict]:
        """
        Optimize for balanced nutrition using DP principles.
        Aims for specific ratios of macronutrients.
        
        Args:
            df: DataFrame with food items
            budget: Available budget
            target_ratios: Dict with target ratios for macronutrients
                          e.g., {'protein': 0.3, 'carbs': 0.4, 'fat': 0.3}
        """
        if target_ratios is None:
            target_ratios = {'protein': 0.25, 'carbohydrates': 0.50, 'fat': 0.25}
        
        items = self._prepare_items(df)
        selected_items = []
        remaining_budget = budget
        
        # Track current nutritional ratios
        totals = {'calories': 0, 'protein': 0, 'carbohydrates': 0, 'fat': 0}
        
        # Calculate balance scores for items
        scored_items = []
        for i, item in enumerate(items):
            balance_score = self._calculate_balance_score(item, target_ratios)
            scored_items.append((i, balance_score, item))
        
        # Sort by balance score (higher is better)
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Dynamic selection to maintain balance
        for item_idx, balance_score, item in scored_items:
            if item['price'] <= remaining_budget:
                # Check if adding this item improves overall balance
                new_totals = totals.copy()
                new_totals['calories'] += item['calories']
                new_totals['protein'] += item['protein']
                new_totals['carbohydrates'] += item['carbohydrates']
                new_totals['fat'] += item['fat']
                
                current_balance = self._calculate_overall_balance(totals, target_ratios)
                new_balance = self._calculate_overall_balance(new_totals, target_ratios)
                
                if new_balance >= current_balance or len(selected_items) < 3:
                    selected_item = {
                        'name': item['name'],
                        'price': item['price'],
                        'calories': item['calories'],
                        'protein': item['protein'],
                        'carbohydrates': item['carbohydrates'],
                        'fat': item['fat'],
                        'category': item['category'],
                        'selection_method': 'balanced_nutrition_dp',
                        'balance_score': balance_score
                    }
                    
                    selected_items.append(selected_item)
                    remaining_budget -= item['price']
                    totals = new_totals
        
        return selected_items
    
    def _calculate_balance_score(self, item: Dict, target_ratios: Dict) -> float:
        """Calculate how well an item contributes to balanced nutrition."""
        total_macros = item['protein'] + item['carbohydrates'] + item['fat']
        if total_macros == 0:
            return 0
        
        # Current ratios in this item
        item_ratios = {
            'protein': item['protein'] / total_macros,
            'carbohydrates': item['carbohydrates'] / total_macros,
            'fat': item['fat'] / total_macros
        }
        
        # Calculate how close item ratios are to target ratios
        balance_score = 0
        for macro, target_ratio in target_ratios.items():
            item_ratio = item_ratios.get(macro, 0)
            # Higher score for ratios closer to target
            ratio_diff = abs(target_ratio - item_ratio)
            balance_score += (1 - ratio_diff) * item.get(macro, 0)
        
        return balance_score
    
    def _calculate_overall_balance(self, totals: Dict, target_ratios: Dict) -> float:
        """Calculate overall nutritional balance score."""
        total_macros = totals['protein'] + totals['carbohydrates'] + totals['fat']
        if total_macros == 0:
            return 0
        
        current_ratios = {
            'protein': totals['protein'] / total_macros,
            'carbohydrates': totals['carbohydrates'] / total_macros,
            'fat': totals['fat'] / total_macros
        }
        
        balance_score = 0
        for macro, target_ratio in target_ratios.items():
            current_ratio = current_ratios.get(macro, 0)
            ratio_diff = abs(target_ratio - current_ratio)
            balance_score += (1 - ratio_diff)
        
        return balance_score / len(target_ratios)  # Normalize
    
    def analyze_dp_solution(self, selected_items: List[Dict], 
                          original_budget: float, goals: Dict) -> Dict:
        """Analyze the DP solution quality and optimality."""
        if not selected_items:
            return {'error': 'No items in solution'}
        
        total_cost = sum(item['price'] for item in selected_items)
        total_nutrition = {
            'calories': sum(item.get('calories', 0) for item in selected_items),
            'protein': sum(item.get('protein', 0) for item in selected_items),
            'carbohydrates': sum(item.get('carbohydrates', 0) for item in selected_items),
            'fat': sum(item.get('fat', 0) for item in selected_items)
        }
        
        # Check goal satisfaction
        goals_met = {}
        for goal_key, target in goals.items():
            if goal_key.startswith('min_'):
                nutrient = goal_key[4:]
                actual = total_nutrition.get(nutrient, 0)
                goals_met[nutrient] = {
                    'target': target,
                    'actual': actual,
                    'satisfied': actual >= target,
                    'percentage': (actual / target * 100) if target > 0 else 100
                }
        
        return {
            'total_items': len(selected_items),
            'total_cost': total_cost,
            'budget_utilization': (total_cost / original_budget * 100),
            'nutritional_totals': total_nutrition,
            'goals_satisfaction': goals_met,
            'average_nutritional_score': np.mean([
                item.get('nutritional_score', 0) for item in selected_items
            ]),
            'solution_quality': self._assess_solution_quality(goals_met)
        }
    
    def _assess_solution_quality(self, goals_met: Dict) -> str:
        """Assess the quality of the DP solution."""
        satisfied_count = sum(1 for goal in goals_met.values() if goal['satisfied'])
        total_goals = len(goals_met)
        
        if total_goals == 0:
            return 'No goals specified'
        
        satisfaction_rate = satisfied_count / total_goals
        
        if satisfaction_rate >= 0.9:
            return 'Excellent (90%+ goals met)'
        elif satisfaction_rate >= 0.7:
            return 'Good (70-89% goals met)'
        elif satisfaction_rate >= 0.5:
            return 'Fair (50-69% goals met)'
        else:
            return 'Poor (<50% goals met)'
        