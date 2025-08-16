import heapq
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

class ExpirationHeapManager:
    def __init__(self):
        self.name = "Expiration Priority Heap Manager"
        self.expiration_heap = []  # Min-heap for expiration dates
        self.priority_heap = []    # Min-heap for priority scores
        self.consumption_schedule = {}
    
    def prioritize_by_expiration(self, items: List[Dict]) -> List[Dict]:
        if not items:
            return []
        
        print(f"Prioritizing {len(items)} items by expiration date using heap")
        
        self.expiration_heap = []
        
        for i, item in enumerate(items):
            days_to_expiry = item.get('days_to_expiry', 30) 
            priority_score = self._calculate_priority_score(item)
            
            heap_entry = (days_to_expiry, priority_score, i, item)
            heapq.heappush(self.expiration_heap, heap_entry)
        
        prioritized_items = []
        temp_heap = self.expiration_heap.copy()
        
        while temp_heap:
            days_to_expiry, priority_score, item_index, item = heapq.heappop(temp_heap)
            
            # Add priority information to item
            prioritized_item = item.copy()
            prioritized_item.update({
                'priority_rank': len(prioritized_items) + 1,
                'days_to_expiry': days_to_expiry,
                'priority_score': priority_score,
                'urgency_level': self._get_urgency_level(days_to_expiry),
                'recommended_action': self._get_recommended_action(days_to_expiry),
                'selection_method': 'heap_prioritized'
            })
            
            prioritized_items.append(prioritized_item)
            
            # Log high-priority items
            if days_to_expiry <= 3:
                print(f"🚨 URGENT: {item['name']} expires in {days_to_expiry} days")
            elif days_to_expiry <= 7:
                print(f"⚠️  Soon: {item['name']} expires in {days_to_expiry} days")
        
        print(f"Heap prioritization complete. {len([i for i in prioritized_items if i['days_to_expiry'] <= 3])} urgent items identified.")
        
        return prioritized_items
    
    def create_consumption_schedule(self, prioritized_items: List[Dict], 
                                  planning_days: int = 7) -> Dict:
        if not prioritized_items:
            return {}
        
        print(f"Creating {planning_days}-day consumption schedule")
        
        #Create heaps for each day (to balance daily nutrition)
        daily_heaps = [[] for _ in range(planning_days)]
        schedule = {}
        
        #Distribute items across days based on expiration priority
        for item in prioritized_items:
            days_to_expiry = item['days_to_expiry']
            
            #Determine optimal day for consumption
            if days_to_expiry <= 1:
                target_day = 0 
            elif days_to_expiry <= planning_days:
                target_day = min(days_to_expiry - 1, planning_days - 1)
            else:
                #Distribute non-urgent items evenly
                target_day = len([i for i in prioritized_items if i['days_to_expiry'] > planning_days]) % planning_days
            
            #Add to target day's heap with nutritional value as priority
            nutritional_value = item.get('calories', 0) + (item.get('protein', 0) * 4)
            heapq.heappush(daily_heaps[target_day], (-nutritional_value, item))  # Negative for max-heap behavior
        
        for day in range(planning_days):
            day_items = []
            total_calories = 0
            total_cost = 0
            urgency_count = 0
            
            while daily_heaps[day]:
                neg_nutrition_value, item = heapq.heappop(daily_heaps[day])
                day_items.append(item)
                total_calories += item.get('calories', 0)
                total_cost += item.get('price', 0)
                
                if item['days_to_expiry'] <= 3:
                    urgency_count += 1
            
            day_key = f'Day {day + 1}'
            schedule[day_key] = {
                'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'items': day_items,
                'item_count': len(day_items),
                'total_calories': total_calories,
                'total_cost': total_cost,
                'urgent_items': urgency_count,
                'priority_level': 'HIGH' if urgency_count > 0 else 'MEDIUM' if len(day_items) > 3 else 'LOW',
                'recommendations': self._generate_daily_recommendations(day_items, urgency_count)
            }
        
        self.consumption_schedule = schedule
        return schedule
    
    def track_perishable_items(self, items: List[Dict]) -> Dict:
        print("Tracking perishable items with category-based heaps")
        
        # Category-based heaps
        category_heaps = {
            'highly_perishable': [],  # 1-3 days
            'moderately_perishable': [],  # 4-7 days
            'stable': [],  # 8+ days
            'frozen': []  # Special category for frozen items
        }
        
        #Categorize and add to appropriate heaps
        for item in items:
            days_to_expiry = item.get('days_to_expiry', 30)
            category = item.get('category', 'unknown').lower()
            
            #Determine perishability category
            if days_to_expiry <= 3 or category in ['dairy', 'meat', 'seafood', 'fresh_produce']:
                heap_category = 'highly_perishable'
            elif days_to_expiry <= 7 or category in ['bread', 'fruits', 'vegetables']:
                heap_category = 'moderately_perishable'
            elif category == 'frozen':
                heap_category = 'frozen'
            else:
                heap_category = 'stable'
            
            #Add to appropriate heap (priority by expiration date)
            priority = days_to_expiry
            heapq.heappush(category_heaps[heap_category], (priority, item))
        
        #Generate tracking summary
        tracking_summary = {}
        for category, heap in category_heaps.items():
            if heap:
                #Get items without modifying heap
                items_in_category = [item for _, item in sorted(heap)]
                
                tracking_summary[category] = {
                    'count': len(items_in_category),
                    'most_urgent': items_in_category[0] if items_in_category else None,
                    'average_days_to_expiry': np.mean([item.get('days_to_expiry', 30) for item in items_in_category]),
                    'total_value': sum(item.get('price', 0) for item in items_in_category),
                    'waste_risk': self._assess_waste_risk(items_in_category)
                }
        
        return tracking_summary
    
    def optimize_storage_priority(self, items: List[Dict]) -> List[Dict]:
        if not items:
            return []
        
        print("Optimizing storage priority using heap algorithm")
        
        storage_heap = []
        
        #Calculate storage priority for each item
        for item in items:
            storage_priority = self._calculate_storage_priority(item)
            
            #Add to heap (lower priority value = higher priority)
            heapq.heappush(storage_heap, (storage_priority, item))
        
        #Extract items in storage priority order
        prioritized_storage = []
        while storage_heap:
            priority, item = heapq.heappop(storage_heap)
            
            storage_item = item.copy()
            storage_item.update({
                'storage_priority': priority,
                'storage_recommendation': self._get_storage_recommendation(item),
                'handling_instructions': self._get_handling_instructions(item)
            })
            
            prioritized_storage.append(storage_item)
        
        return prioritized_storage
    
    def manage_inventory_rotation(self, current_inventory: List[Dict], 
                                new_items: List[Dict]) -> Dict:
        print("Managing inventory rotation with FIFO heap system")
        
        #Combined heap for all items (current + new)
        rotation_heap = []
        
        #Add current inventory with timestamps
        for item in current_inventory:
            # Use negative days to expiry for oldest-first priority
            days_to_expiry = item.get('days_to_expiry', 30)
            purchase_date = item.get('purchase_date', datetime.now() - timedelta(days=30))
            
            if isinstance(purchase_date, str):
                purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d')
            
            # Priority: older items first, then by expiration
            age_priority = (datetime.now() - purchase_date).days
            priority = (age_priority, days_to_expiry)
            
            heapq.heappush(rotation_heap, (priority, 'current', item))
        
        # Add new items
        for item in new_items:
            days_to_expiry = item.get('days_to_expiry', 30)
            age_priority = 0  # New items have age 0
            priority = (age_priority, days_to_expiry)
            
            heapq.heappush(rotation_heap, (priority, 'new', item))
        
        # Create rotation plan
        rotation_plan = {
            'use_first': [],  # Items to use immediately
            'use_soon': [],   # Items to use within a week
            'store_properly': [],  # New items to store
            'monitor': []     # Items to monitor closely
        }
        
        # Process heap to create rotation categories
        temp_heap = rotation_heap.copy()
        while temp_heap:
            (age, expiry_days), item_type, item = heapq.heappop(temp_heap)
            
            if expiry_days <= 2:
                rotation_plan['use_first'].append(item)
            elif expiry_days <= 7:
                rotation_plan['use_soon'].append(item)
            elif item_type == 'new':
                rotation_plan['store_properly'].append(item)
            else:
                rotation_plan['monitor'].append(item)
        
        return rotation_plan
    
    def _calculate_priority_score(self, item: Dict) -> float:
        days_to_expiry = item.get('days_to_expiry', 30)
        price = item.get('price', 0)
        nutritional_value = item.get('calories', 0) + (item.get('protein', 0) * 4)
        
        # Base priority is expiration urgency
        priority_score = days_to_expiry
        
        # Adjust for value (expensive items get slightly higher priority)
        if price > 10:
            priority_score -= 0.5
        elif price > 5:
            priority_score -= 0.2
        
        # Adjust for nutritional value (higher nutrition = higher priority)
        if nutritional_value > 500:
            priority_score -= 0.3
        elif nutritional_value > 200:
            priority_score -= 0.1
        
        # Category adjustments
        category = item.get('category', '').lower()
        if category in ['meat', 'dairy', 'seafood']:
            priority_score -= 1.0  # Higher priority for highly perishable
        elif category in ['fruits', 'vegetables']:
            priority_score -= 0.5
        
        return max(priority_score, 0.1)  # Ensure positive priority
    
    def _get_urgency_level(self, days_to_expiry: int) -> str:
        if days_to_expiry <= 1:
            return "CRITICAL"
        elif days_to_expiry <= 3:
            return "HIGH"
        elif days_to_expiry <= 7:
            return "MEDIUM"
        elif days_to_expiry <= 14:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _get_recommended_action(self, days_to_expiry: int) -> str:
        if days_to_expiry <= 1:
            return "Use immediately or freeze"
        elif days_to_expiry <= 3:
            return "Prioritize in next 2-3 meals"
        elif days_to_expiry <= 7:
            return "Plan to use this week"
        elif days_to_expiry <= 14:
            return "Monitor and plan usage"
        else:
            return "Store properly for future use"
    
    def _generate_daily_recommendations(self, day_items: List[Dict], urgent_count: int) -> List[str]:
        recommendations = []
        
        if urgent_count > 0:
            recommendations.append(f"🚨 {urgent_count} items need immediate attention")
            recommendations.append("Check refrigerator temperatures")
            recommendations.append("Consider meal prep to use multiple items")
        
        if len(day_items) > 5:
            recommendations.append("Heavy consumption day - plan larger meals")
            recommendations.append("Consider batch cooking or meal prep")
        
        # Category-specific recommendations
        categories = set(item.get('category', 'unknown') for item in day_items)
        if 'dairy' in categories:
            recommendations.append("🥛 Check dairy products for freshness")
        if 'meat' in categories or 'seafood' in categories:
            recommendations.append("🥩 Ensure proper refrigeration for proteins")
        if 'fruits' in categories or 'vegetables' in categories:
            recommendations.append("🥬 Store produce properly to extend life")
        
        return recommendations
    
    def _assess_waste_risk(self, items: List[Dict]) -> str:
        if not items:
            return "NO_RISK"
        
        urgent_items = len([item for item in items if item.get('days_to_expiry', 30) <= 3])
        total_items = len(items)
        
        risk_ratio = urgent_items / total_items
        
        if risk_ratio >= 0.5:
            return "HIGH"
        elif risk_ratio >= 0.25:
            return "MEDIUM"
        elif risk_ratio > 0:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _calculate_storage_priority(self, item: Dict) -> float:
        days_to_expiry = item.get('days_to_expiry', 30)
        category = item.get('category', 'unknown').lower()
        price = item.get('price', 0)
        
        # Base priority from expiration
        priority = days_to_expiry
        
        # Category adjustments for storage urgency
        if category in ['frozen']:
            priority += 10  # Lower priority - can be stored longer
        elif category in ['dairy', 'meat', 'seafood']:
            priority -= 2   # Higher priority - needs immediate proper storage
        elif category in ['fruits', 'vegetables']:
            priority -= 1   # Moderate priority
        
        # Price adjustment (expensive items get better storage priority)
        if price > 15:
            priority -= 1
        
        return max(priority, 0.1)
    
    def _get_storage_recommendation(self, item: Dict) -> str:
        category = item.get('category', 'unknown').lower()
        days_to_expiry = item.get('days_to_expiry', 30)
        
        if category == 'frozen':
            return "Keep frozen until ready to use"
        elif category in ['dairy', 'meat', 'seafood']:
            if days_to_expiry <= 3:
                return "Refrigerate immediately, use soon or freeze"
            else:
                return "Refrigerate at proper temperature (32-40°F)"
        elif category in ['fruits', 'vegetables']:
            return "Store in crisper drawer, some items at room temperature"
        elif category in ['bread', 'grains']:
            return "Store in cool, dry place or freeze for longer storage"
        else:
            return "Store in cool, dry pantry"
    
    def _get_handling_instructions(self, item: Dict) -> List[str]:
        category = item.get('category', 'unknown').lower()
        instructions = []
        
        if category in ['meat', 'seafood']:
            instructions.extend([
                "Keep refrigerated below 40°F",
                "Use separate cutting board",
                "Cook to safe internal temperature"
            ])
        elif category == 'dairy':
            instructions.extend([
                "Check expiration date regularly",
                "Keep refrigerated",
                "Use clean utensils when serving"
            ])
        elif category in ['fruits', 'vegetables']:
            instructions.extend([
                "Wash before eating",
                "Store appropriately (some need refrigeration)",
                "Check for spoilage regularly"
            ])
        elif category == 'frozen':
            instructions.extend([
                "Keep frozen until ready to use",
                "Thaw safely in refrigerator",
                "Do not refreeze after thawing"
            ])
        else:
            instructions.extend([
                "Store in cool, dry place",
                "Check for signs of spoilage",
                "Follow package instructions"
            ])
        
        return instructions
    
    def generate_waste_prevention_report(self, items: List[Dict]) -> Dict:
        if not items:
            return {'error': 'No items to analyze'}
        
        # Analyze items by urgency using heap
        urgency_heap = []
        for item in items:
            days_to_expiry = item.get('days_to_expiry', 30)
            heapq.heappush(urgency_heap, (days_to_expiry, item))
        
        # Extract items by urgency levels
        critical_items = []
        high_priority = []
        medium_priority = []
        low_priority = []
        
        temp_heap = urgency_heap.copy()
        while temp_heap:
            days, item = heapq.heappop(temp_heap)
            if days <= 1:
                critical_items.append(item)
            elif days <= 3:
                high_priority.append(item)
            elif days <= 7:
                medium_priority.append(item)
            else:
                low_priority.append(item)
        
        # Calculate potential waste value
        critical_value = sum(item.get('price', 0) for item in critical_items)
        high_value = sum(item.get('price', 0) for item in high_priority)
        
        # Generate actionable recommendations
        recommendations = []
        if critical_items:
            recommendations.append(f"🚨 URGENT: {len(critical_items)} items expire within 24 hours (${critical_value:.2f} value)")
            recommendations.append("Consider immediate meal prep or freezing")
        
        if high_priority:
            recommendations.append(f"⚠️ HIGH: {len(high_priority)} items expire within 3 days (${high_value:.2f} value)")
            recommendations.append("Plan meals around these items")
        
        if len(medium_priority) > 10:
            recommendations.append("📅 Consider batch cooking for medium-priority items")
        
        # Storage optimization suggestions
        storage_suggestions = []
        categories = {}
        for item in critical_items + high_priority:
            cat = item.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        for category, count in categories.items():
            if category in ['fruits', 'vegetables'] and count >= 3:
                storage_suggestions.append(f"Consider making smoothies or soups with {count} {category} items")
            elif category in ['meat', 'seafood'] and count >= 2:
                storage_suggestions.append(f"Freeze {count} {category} items if not using immediately")
        
        return {
            'total_items_analyzed': len(items),
            'urgency_breakdown': {
                'critical': len(critical_items),
                'high': len(high_priority),
                'medium': len(medium_priority),
                'low': len(low_priority)
            },
            'potential_waste_value': {
                'critical_risk': critical_value,
                'high_risk': high_value,
                'total_at_risk': critical_value + high_value
            },
            'immediate_actions': recommendations,
            'storage_optimizations': storage_suggestions,
            'waste_prevention_score': self._calculate_waste_prevention_score(
                len(critical_items), len(high_priority), len(items)
            )
        }
    
    def _calculate_waste_prevention_score(self, critical: int, high: int, total: int) -> Dict:
        if total == 0:
            return {'score': 100, 'grade': 'A+', 'description': 'No items to manage'}
        
        at_risk_percentage = ((critical * 2 + high) / total) * 100
        
        if at_risk_percentage <= 5:
            score = 95 + (5 - at_risk_percentage)
            grade = 'A+'
            description = 'Excellent waste prevention'
        elif at_risk_percentage <= 15:
            score = 80 + (15 - at_risk_percentage)
            grade = 'A'
            description = 'Good waste prevention'
        elif at_risk_percentage <= 30:
            score = 60 + (30 - at_risk_percentage) * (20/15)
            grade = 'B'
            description = 'Moderate waste risk'
        elif at_risk_percentage <= 50:
            score = 40 + (50 - at_risk_percentage) * (20/20)
            grade = 'C'
            description = 'High waste risk'
        else:
            score = max(20, 40 - (at_risk_percentage - 50))
            grade = 'D'
            description = 'Critical waste risk'
        
        return {
            'score': round(score, 1),
            'grade': grade,
            'description': description,
            'at_risk_percentage': round(at_risk_percentage, 1)
        }