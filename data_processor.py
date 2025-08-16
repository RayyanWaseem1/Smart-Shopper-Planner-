import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

class DataProcessor:
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.df = None
        
    def load_and_process_data(self) -> pd.DataFrame:
        """
        Load and preprocess the Kaggle nutritional values dataset.
        If the file doesn't exist, create a sample dataset.
        """
        try:
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded dataset from {self.data_file}")
            print(f"Original dataset shape: {self.df.shape}")
            print(f"Original columns: {list(self.df.columns)}")
            
        except FileNotFoundError:
            print(f"Dataset file {self.data_file} not found. Creating sample dataset...")
            self.df = self._create_sample_dataset()
            
        self.df = self._preprocess_kaggle_dataset(self.df)
        
        #Add synthetic pricing data if not present
        if 'price' not in self.df.columns:
            self.df = self._add_pricing_data(self.df)
            
        #Ensure required columns exist
        self.df = self._ensure_required_columns(self.df)
        
        print(f"Processed dataset: {len(self.df)} items with {len(self.df.columns)} features")
        return self.df
    
    def _preprocess_kaggle_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Preprocessing Kaggle nutritional dataset...")
        
        #Create a copy to avoid modifying original
        df_processed = df.copy()
        
        #Standardize column names to lowercase and replace spaces/special chars
        df_processed.columns = df_processed.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('%', 'percent').str.replace('-', '_')
        
        print(f"Standardized columns: {list(df_processed.columns)}")
        
        #Map common column variations to our standard names
        column_mapping = {}
        
        #Food name mapping
        name_candidates = ['name', 'food_name', 'description', 'food', 'item', 'product_name', 'food_item']
        for col in df_processed.columns:
            if any(candidate in col for candidate in name_candidates):
                column_mapping[col] = 'name'
                break
        
        #Category mapping
        category_candidates = ['category', 'food_group', 'group', 'type', 'class', 'food_category']
        for col in df_processed.columns:
            if any(candidate in col for candidate in category_candidates):
                column_mapping[col] = 'category'
                break
        
        #Calories mapping
        calorie_candidates = ['calories', 'energy', 'kcal', 'cal', 'energy_kcal']
        for col in df_processed.columns:
            if any(candidate in col for candidate in calorie_candidates):
                column_mapping[col] = 'calories'
                break
        
        #Protein mapping
        protein_candidates = ['protein', 'proteins', 'protein_g', 'protein_grams']
        for col in df_processed.columns:
            if any(candidate in col for candidate in protein_candidates):
                column_mapping[col] = 'protein'
                break
        
        #Carbohydrate mapping
        carb_candidates = ['carbohydrate', 'carbohydrates', 'carbs', 'carb', 'carbohydrate_g', 'carbs_g', 'total_carbohydrate']
        for col in df_processed.columns:
            if any(candidate in col for candidate in carb_candidates):
                column_mapping[col] = 'carbohydrates'
                break
        
        #Fat mapping
        fat_candidates = ['fat', 'fats', 'total_fat', 'fat_g', 'fats_g', 'lipid']
        for col in df_processed.columns:
            if any(candidate in col for candidate in fat_candidates):
                column_mapping[col] = 'fat'
                break
        
        #Apply column mapping
        df_processed = df_processed.rename(columns=column_mapping)
        print(f"Applied column mapping: {column_mapping}")
        print(f"Mapped columns: {list(df_processed.columns)}")
        
        #Handle missing values and data types
        numeric_columns = ['calories', 'protein', 'carbohydrates', 'fat']
        for col in numeric_columns:
            if col in df_processed.columns:
                # Convert to numeric, replacing non-numeric values with NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                # Fill missing values with 0 or reasonable defaults
                df_processed[col] = df_processed[col].fillna(0)
                # Ensure non-negative values
                df_processed[col] = df_processed[col].clip(lower=0)
        
        #Handle name column
        if 'name' not in df_processed.columns:
            # If no name column found, create one from index or first text column
            text_columns = df_processed.select_dtypes(include=['object']).columns
            if len(text_columns) > 0:
                df_processed['name'] = df_processed[text_columns[0]]
            else:
                df_processed['name'] = df_processed.index.map(lambda x: f'Food_Item_{x}')
        
        #Clean name column
        if 'name' in df_processed.columns:
            df_processed['name'] = df_processed['name'].astype(str).str.strip()
            # Remove very short or invalid names
            df_processed = df_processed[df_processed['name'].str.len() > 2].reset_index(drop=True)
        
        #Handle category column
        if 'category' not in df_processed.columns:
            df_processed['category'] = df_processed['name'].apply(self._infer_category_from_name)
        else:
            df_processed['category'] = df_processed['category'].astype(str).str.strip().str.lower()
            df_processed.loc[df_processed['category'].isin(['', 'nan', 'none', 'null']), 'category'] = 'unknown'
        
        #Remove duplicates and invalid entries
        df_processed = df_processed.drop_duplicates(subset=['name']).reset_index(drop=True)
        
        #Remove items with zero calories
        if 'calories' in df_processed.columns:
            df_processed = df_processed[df_processed['calories'] > 0].reset_index(drop=True)
        
        #Data validation and cleaning
        # emove outliers
        for col in numeric_columns:
            if col in df_processed.columns:
                q99 = df_processed[col].quantile(0.999)
                if q99 > 0:
                    df_processed = df_processed[df_processed[col] <= q99].reset_index(drop=True)
        
        print(f"Dataset after preprocessing: {len(df_processed)} items")
        return df_processed
    
    def _infer_category_from_name(self, name: str) -> str:
        name_lower = str(name).lower()
        
        #Define category keywords
        category_keywords = {
            'meat': ['beef', 'chicken', 'pork', 'lamb', 'turkey', 'ham', 'bacon', 'sausage', 'steak', 'ground beef'],
            'seafood': ['fish', 'salmon', 'tuna', 'cod', 'shrimp', 'crab', 'lobster', 'oyster', 'clam', 'sardine'],
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'cottage cheese', 'mozzarella', 'cheddar'],
            'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'blueberry', 'peach', 'pear', 'cherry', 'melon'],
            'vegetables': ['broccoli', 'carrot', 'spinach', 'tomato', 'potato', 'onion', 'pepper', 'lettuce', 'cucumber', 'celery'],
            'grains': ['rice', 'wheat', 'oat', 'barley', 'quinoa', 'bread', 'pasta', 'cereal', 'flour'],
            'nuts': ['almond', 'walnut', 'peanut', 'cashew', 'pecan', 'pistachio', 'hazelnut'],
            'oils': ['oil', 'olive oil', 'vegetable oil', 'coconut oil', 'butter'],
            'beverages': ['juice', 'soda', 'coffee', 'tea', 'wine', 'beer', 'water']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
        
        return 'unknown'
    
    def _create_sample_dataset(self) -> pd.DataFrame:
        
        # Sample food data with realistic nutritional information
        sample_foods = [
            # Proteins
            {'name': 'Chicken Breast', 'category': 'meat', 'calories': 165, 'protein': 31, 'carbohydrates': 0, 'fat': 3.6},
            {'name': 'Salmon Fillet', 'category': 'seafood', 'calories': 208, 'protein': 20, 'carbohydrates': 0, 'fat': 12},
            {'name': 'Ground Beef', 'category': 'meat', 'calories': 250, 'protein': 26, 'carbohydrates': 0, 'fat': 15},
            {'name': 'Eggs (dozen)', 'category': 'dairy', 'calories': 840, 'protein': 72, 'carbohydrates': 6, 'fat': 60},
            {'name': 'Greek Yogurt', 'category': 'dairy', 'calories': 100, 'protein': 17, 'carbohydrates': 9, 'fat': 0},
            {'name': 'Tofu', 'category': 'vegetarian', 'calories': 70, 'protein': 8, 'carbohydrates': 2, 'fat': 4},
            {'name': 'Black Beans', 'category': 'vegetarian', 'calories': 227, 'protein': 15, 'carbohydrates': 41, 'fat': 1},
            
            # Grains & Carbs
            {'name': 'Brown Rice', 'category': 'grains', 'calories': 216, 'protein': 5, 'carbohydrates': 45, 'fat': 1.8},
            {'name': 'Quinoa', 'category': 'grains', 'calories': 222, 'protein': 8, 'carbohydrates': 39, 'fat': 3.6},
            {'name': 'Whole Wheat Bread', 'category': 'bread', 'calories': 247, 'protein': 13, 'carbohydrates': 41, 'fat': 4},
            {'name': 'Oats', 'category': 'grains', 'calories': 389, 'protein': 17, 'carbohydrates': 66, 'fat': 7},
            {'name': 'Sweet Potato', 'category': 'vegetables', 'calories': 86, 'protein': 2, 'carbohydrates': 20, 'fat': 0.1},
            {'name': 'Pasta', 'category': 'grains', 'calories': 220, 'protein': 8, 'carbohydrates': 44, 'fat': 1.1},
            
            # Fruits
            {'name': 'Bananas', 'category': 'fruits', 'calories': 89, 'protein': 1.1, 'carbohydrates': 23, 'fat': 0.3},
            {'name': 'Apples', 'category': 'fruits', 'calories': 52, 'protein': 0.3, 'carbohydrates': 14, 'fat': 0.2},
            {'name': 'Oranges', 'category': 'fruits', 'calories': 47, 'protein': 0.9, 'carbohydrates': 12, 'fat': 0.1},
            {'name': 'Strawberries', 'category': 'fruits', 'calories': 32, 'protein': 0.7, 'carbohydrates': 8, 'fat': 0.3},
            {'name': 'Blueberries', 'category': 'fruits', 'calories': 57, 'protein': 0.7, 'carbohydrates': 14, 'fat': 0.3},
            {'name': 'Avocado', 'category': 'fruits', 'calories': 160, 'protein': 2, 'carbohydrates': 9, 'fat': 15},
            
            # Vegetables
            {'name': 'Broccoli', 'category': 'vegetables', 'calories': 34, 'protein': 3, 'carbohydrates': 7, 'fat': 0.4},
            {'name': 'Spinach', 'category': 'vegetables', 'calories': 23, 'protein': 3, 'carbohydrates': 4, 'fat': 0.4},
            {'name': 'Carrots', 'category': 'vegetables', 'calories': 41, 'protein': 1, 'carbohydrates': 10, 'fat': 0.2},
            {'name': 'Bell Peppers', 'category': 'vegetables', 'calories': 31, 'protein': 1, 'carbohydrates': 7, 'fat': 0.3},
            {'name': 'Tomatoes', 'category': 'vegetables', 'calories': 18, 'protein': 1, 'carbohydrates': 4, 'fat': 0.2},
            {'name': 'Onions', 'category': 'vegetables', 'calories': 40, 'protein': 1, 'carbohydrates': 9, 'fat': 0.1},
            {'name': 'Garlic', 'category': 'vegetables', 'calories': 149, 'protein': 6, 'carbohydrates': 33, 'fat': 0.5},
            
            # Dairy & Alternatives
            {'name': 'Milk (1 gallon)', 'category': 'dairy', 'calories': 2400, 'protein': 128, 'carbohydrates': 184, 'fat': 128},
            {'name': 'Cheese (cheddar)', 'category': 'dairy', 'calories': 402, 'protein': 25, 'carbohydrates': 1, 'fat': 33},
            {'name': 'Butter', 'category': 'dairy', 'calories': 717, 'protein': 1, 'carbohydrates': 0, 'fat': 81},
            {'name': 'Almond Milk', 'category': 'dairy_alternative', 'calories': 13, 'protein': 0.6, 'carbohydrates': 0.3, 'fat': 1.2},
            
            # Nuts & Seeds
            {'name': 'Almonds', 'category': 'nuts', 'calories': 579, 'protein': 21, 'carbohydrates': 22, 'fat': 50},
            {'name': 'Walnuts', 'category': 'nuts', 'calories': 654, 'protein': 15, 'carbohydrates': 14, 'fat': 65},
            {'name': 'Chia Seeds', 'category': 'seeds', 'calories': 486, 'protein': 17, 'carbohydrates': 42, 'fat': 31},
            {'name': 'Peanut Butter', 'category': 'nuts', 'calories': 588, 'protein': 25, 'carbohydrates': 20, 'fat': 50},
            
            # Frozen Items
            {'name': 'Frozen Berries', 'category': 'frozen', 'calories': 70, 'protein': 1, 'carbohydrates': 17, 'fat': 0.5},
            {'name': 'Frozen Vegetables', 'category': 'frozen', 'calories': 65, 'protein': 3, 'carbohydrates': 13, 'fat': 0.5},
            {'name': 'Frozen Fish', 'category': 'frozen', 'calories': 200, 'protein': 22, 'carbohydrates': 0, 'fat': 12},
            
            # Oils & Condiments
            {'name': 'Olive Oil', 'category': 'oils', 'calories': 884, 'protein': 0, 'carbohydrates': 0, 'fat': 100},
            {'name': 'Coconut Oil', 'category': 'oils', 'calories': 862, 'protein': 0, 'carbohydrates': 0, 'fat': 100},
            
            # Snacks
            {'name': 'Greek Crackers', 'category': 'snacks', 'calories': 120, 'protein': 3, 'carbohydrates': 20, 'fat': 3},
            {'name': 'Granola', 'category': 'snacks', 'calories': 471, 'protein': 11, 'carbohydrates': 61, 'fat': 22}
        ]
        
        return pd.DataFrame(sample_foods)
    
    def _add_pricing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        
        #Base prices by category (per 100g or typical serving)
        category_price_ranges = {
            'meat': (3.0, 8.0),
            'seafood': (4.0, 12.0),
            'dairy': (0.5, 4.0),
            'dairy_alternative': (1.0, 3.0),
            'fruits': (0.5, 3.0),
            'vegetables': (0.3, 2.5),
            'grains': (1.0, 3.0),
            'bread': (1.5, 4.0),
            'nuts': (4.0, 10.0),
            'seeds': (3.0, 8.0),
            'oils': (2.0, 6.0),
            'frozen': (2.0, 5.0),
            'snacks': (2.0, 6.0),
            'vegetarian': (1.5, 4.0),
            'beverages': (0.5, 3.0),
            'unknown': (1.0, 3.0)
        }
        
        prices = []
        for _, row in df.iterrows():
            category = row.get('category', 'unknown').lower()
            
            #Get base price range for category
            if category in category_price_ranges:
                min_price, max_price = category_price_ranges[category]
            else:
                min_price, max_price = category_price_ranges['unknown']
            
            #Adjust price based on protein content (higher protein = higher price)
            protein_factor = 1 + (row.get('protein', 0) / 50)  # Scale protein impact
            
            #Adjust price based on calories (energy density affects price)
            calorie_factor = 1 + (row.get('calories', 0) / 500)  # Scale calorie impact
            
            #Calculate final price with some randomness
            base_price = random.uniform(min_price, max_price)
            adjusted_price = base_price * protein_factor * calorie_factor * random.uniform(0.8, 1.2)
            
            #Round to realistic price (to nearest quarter)
            final_price = round(adjusted_price * 4) / 4
            
            prices.append(max(final_price, 0.25))  # Minimum price of $0.25
        
        df['price'] = prices
        return df
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {
            'name': 'Unknown Item',
            'category': 'unknown',
            'calories': 0,
            'protein': 0,
            'carbohydrates': 0,
            'fat': 0,
            'price': 1.0
        }
        
        for col, default_value in required_columns.items():
            if col not in df.columns:
                if col in ['calories', 'protein', 'carbohydrates', 'fat', 'price']:
                    df[col] = default_value
                else:
                    df[col] = default_value
        
        return df
    
    def filter_by_restrictions(self, df: pd.DataFrame, restrictions: List[str]) -> pd.DataFrame:
        if not restrictions:
            return df
        
        filtered_df = df.copy()
        
        print(f"Applying dietary restrictions: {restrictions}")
        
        for restriction in restrictions:
            restriction = restriction.lower().strip()
            
            if restriction == 'vegetarian':
                # Exclude meat and seafood
                filtered_df = filtered_df[
                    ~filtered_df['category'].str.lower().isin(['meat', 'seafood'])
                ].reset_index(drop=True)
                print(f"Vegetarian filter applied: {len(filtered_df)} items remaining")
                
            elif restriction == 'vegan':
                # Exclude all animal products
                excluded_categories = ['meat', 'seafood', 'dairy', 'eggs']
                filtered_df = filtered_df[
                    ~filtered_df['category'].str.lower().isin(excluded_categories)
                ].reset_index(drop=True)
                animal_keywords = ['cheese', 'milk', 'butter', 'yogurt', 'cream', 'egg', 'honey']
                for keyword in animal_keywords:
                    filtered_df = filtered_df[
                        ~filtered_df['name'].str.lower().str.contains(keyword, na=False)
                    ].reset_index(drop=True)
                print(f"Vegan filter applied: {len(filtered_df)} items remaining")
                
            elif restriction == 'gluten_free':
                # Exclude gluten-containing items
                gluten_keywords = ['bread', 'pasta', 'wheat', 'barley', 'rye', 'flour']
                for keyword in gluten_keywords:
                    filtered_df = filtered_df[
                        ~filtered_df['name'].str.lower().str.contains(keyword, na=False)
                    ].reset_index(drop=True)
                print(f"Gluten-free filter applied: {len(filtered_df)} items remaining")
                
            elif restriction == 'low_sodium':
                processed_keywords = ['canned', 'processed', 'deli', 'bacon', 'ham', 'sausage']
                for keyword in processed_keywords:
                    filtered_df = filtered_df[
                        ~filtered_df['name'].str.lower().str.contains(keyword, na=False)
                    ].reset_index(drop=True)
                print(f"Low-sodium filter applied: {len(filtered_df)} items remaining")
                
            elif restriction == 'keto' or restriction == 'low_carb':
                # Keep only items with low carbohydrates (< 10g per serving)
                filtered_df = filtered_df[
                    filtered_df['carbohydrates'] <= 10
                ].reset_index(drop=True)
                print(f"Keto/Low-carb filter applied: {len(filtered_df)} items remaining")
                
            elif restriction == 'dairy_free':
                # Exclude dairy products
                filtered_df = filtered_df[
                    filtered_df['category'].str.lower() != 'dairy'
                ].reset_index(drop=True)
                dairy_keywords = ['milk', 'cheese', 'butter', 'cream', 'yogurt']
                for keyword in dairy_keywords:
                    filtered_df = filtered_df[
                        ~filtered_df['name'].str.lower().str.contains(keyword, na=False)
                    ].reset_index(drop=True)
                print(f"Dairy-free filter applied: {len(filtered_df)} items remaining")
                
            elif restriction == 'high_protein':
                # Keep only items with high protein content (> 10g per serving)
                filtered_df = filtered_df[
                    filtered_df['protein'] >= 10
                ].reset_index(drop=True)
                print(f"High-protein filter applied: {len(filtered_df)} items remaining")
        
        print(f"Final filtered dataset: {len(filtered_df)} items")
        return filtered_df
    
    def add_expiration_dates(self, items: List[Dict]) -> List[Dict]:
        
        #Typical shelf life by category (in days)
        shelf_life_ranges = {
            'meat': (1, 5),
            'seafood': (1, 3),
            'dairy': (3, 14),
            'dairy_alternative': (5, 21),
            'fruits': (3, 10),
            'vegetables': (3, 14),
            'grains': (30, 365),
            'bread': (3, 7),
            'nuts': (30, 180),
            'seeds': (30, 120),
            'oils': (180, 730),
            'frozen': (30, 365),
            'snacks': (30, 180),
            'vegetarian': (7, 30),
            'beverages': (14, 365),
            'unknown': (7, 30)
        }
        
        items_with_expiry = []
        current_date = datetime.now()
        
        for item in items:
            item_copy = item.copy()
            category = item.get('category', 'unknown').lower()
            
            #Get shelf life range for category
            if category in shelf_life_ranges:
                min_days, max_days = shelf_life_ranges[category]
            else:
                min_days, max_days = shelf_life_ranges['unknown']
            
            purchase_age = random.randint(0, min_days)  # How many days ago was it purchased
            remaining_shelf_life = random.randint(min_days - purchase_age, max_days - purchase_age)
            remaining_shelf_life = max(remaining_shelf_life, 0)  # Ensure non-negative
            
            #Calculate expiration date
            expiration_date = current_date + timedelta(days=remaining_shelf_life)
            
            #Add expiration information to item
            item_copy.update({
                'expiration_date': expiration_date.strftime('%Y-%m-%d'),
                'days_to_expiry': remaining_shelf_life,
                'purchase_date': (current_date - timedelta(days=purchase_age)).strftime('%Y-%m-%d')
            })
            
            items_with_expiry.append(item_copy)
        
        print(f"Added expiration dates to {len(items_with_expiry)} items")
        return items_with_expiry
    
    def get_category_statistics(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}
        
        stats = {}
        
        stats['total_items'] = len(df)
        stats['categories'] = df['category'].nunique()
        
        #Category breakdown
        category_stats = {}
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            
            category_stats[category] = {
                'count': len(cat_data),
                'avg_price': cat_data['price'].mean() if 'price' in cat_data else 0,
                'avg_calories': cat_data['calories'].mean() if 'calories' in cat_data else 0,
                'avg_protein': cat_data['protein'].mean() if 'protein' in cat_data else 0,
                'price_range': {
                    'min': cat_data['price'].min() if 'price' in cat_data else 0,
                    'max': cat_data['price'].max() if 'price' in cat_data else 0
                }
            }
        
        stats['by_category'] = category_stats
        
        #Nutritional statistics
        if all(col in df.columns for col in ['calories', 'protein', 'carbohydrates', 'fat']):
            stats['nutrition_summary'] = {
                'avg_calories': df['calories'].mean(),
                'avg_protein': df['protein'].mean(),
                'avg_carbs': df['carbohydrates'].mean(),
                'avg_fat': df['fat'].mean(),
                'total_calories': df['calories'].sum(),
                'total_protein': df['protein'].sum()
            }
        
        #Price statistics
        if 'price' in df.columns:
            stats['price_summary'] = {
                'avg_price': df['price'].mean(),
                'median_price': df['price'].median(),
                'min_price': df['price'].min(),
                'max_price': df['price'].max(),
                'total_value': df['price'].sum()
            }
        
        return stats
    
    def search_items(self, df: pd.DataFrame, query: str, 
                    search_columns: List[str] = None) -> pd.DataFrame:
        if df.empty or not query:
            return df
        
        if search_columns is None:
            search_columns = ['name', 'category']
        
        query = query.lower()
        
        mask = pd.Series([False] * len(df))
        
        for col in search_columns:
            if col in df.columns:
                mask |= df[col].str.lower().str.contains(query, na=False)
        
        search_results = df[mask].reset_index(drop=True)
        print(f"Search for '{query}' found {len(search_results)} results")
        
        return search_results
    
    def get_items_by_budget_range(self, df: pd.DataFrame, min_price: float, 
                                 max_price: float) -> pd.DataFrame:
        if df.empty or 'price' not in df.columns:
            return df
        
        filtered_items = df[
            (df['price'] >= min_price) & (df['price'] <= max_price)
        ].reset_index(drop=True)
        
        print(f"Found {len(filtered_items)} items in price range ${min_price:.2f} - ${max_price:.2f}")
        
        return filtered_items
    
    def get_high_nutrition_items(self, df: pd.DataFrame, nutrition_thresholds: Dict = None) -> pd.DataFrame:
        if df.empty:
            return df
        
        if nutrition_thresholds is None:
            nutrition_thresholds = {
                'calories': 100,
                'protein': 5,
                'fat': 0  # No minimum fat requirement
            }
        
        mask = pd.Series([True] * len(df))
        
        for nutrient, threshold in nutrition_thresholds.items():
            if nutrient in df.columns:
                mask &= (df[nutrient] >= threshold)
        
        high_nutrition_items = df[mask].reset_index(drop=True)
        print(f"Found {len(high_nutrition_items)} items meeting high nutrition standards")
        
        return high_nutrition_items
    
    def balance_macronutrients(self, selected_items: List[Dict], 
                             target_ratios: Dict = None) -> Dict:
        if not selected_items:
            return {'error': 'No items provided'}
        
        if target_ratios is None:
            target_ratios = {
                'protein': 0.25,    #25% of calories from protein
                'carbohydrates': 0.50,  #50% of calories from carbs
                'fat': 0.25         #25% of calories from fat
            }
        
        #Calculate totals
        total_calories = sum(item.get('calories', 0) for item in selected_items)
        total_protein = sum(item.get('protein', 0) for item in selected_items)
        total_carbs = sum(item.get('carbohydrates', 0) for item in selected_items)
        total_fat = sum(item.get('fat', 0) for item in selected_items)
        
        if total_calories == 0:
            return {'error': 'No caloric content in selected items'}
        
        #Convert grams to calories (protein: 4 cal/g, carbs: 4 cal/g, fat: 9 cal/g)
        protein_calories = total_protein * 4
        carb_calories = total_carbs * 4
        fat_calories = total_fat * 9
        
        #Calculate actual ratios
        actual_ratios = {
            'protein': protein_calories / total_calories,
            'carbohydrates': carb_calories / total_calories,
            'fat': fat_calories / total_calories
        }
        
        #Calculate deviations from target
        deviations = {
            nutrient: abs(actual_ratios[nutrient] - target_ratios[nutrient])
            for nutrient in target_ratios
        }
        
        #Overall balance score (lower deviation = better balance)
        balance_score = 1 - (sum(deviations.values()) / len(deviations))
        balance_score = max(balance_score, 0)  # Ensure non-negative
        
        return {
            'totals': {
                'calories': total_calories,
                'protein': total_protein,
                'carbohydrates': total_carbs,
                'fat': total_fat
            },
            'actual_ratios': actual_ratios,
            'target_ratios': target_ratios,
            'deviations': deviations,
            'balance_score': balance_score,
            'balance_grade': self._get_balance_grade(balance_score),
            'recommendations': self._get_balance_recommendations(deviations, target_ratios)
        }
    
    def _get_balance_grade(self, balance_score: float) -> str:
        if balance_score >= 0.9:
            return 'A+'
        elif balance_score >= 0.8:
            return 'A'
        elif balance_score >= 0.7:
            return 'B'
        elif balance_score >= 0.6:
            return 'C'
        else:
            return 'D'
    
    def _get_balance_recommendations(self, deviations: Dict, targets: Dict) -> List[str]:
        recommendations = []
        
        #Find the most imbalanced nutrient
        max_deviation = max(deviations.values())
        most_imbalanced = [k for k, v in deviations.items() if v == max_deviation][0]
        
        if max_deviation > 0.1:  #Only recommend if significant imbalance
            if most_imbalanced == 'protein':
                recommendations.append("Consider adding more protein-rich foods (lean meats, legumes, dairy)")
            elif most_imbalanced == 'carbohydrates':
                recommendations.append("Consider adding more healthy carbs (fruits, vegetables, whole grains)")
            elif most_imbalanced == 'fat':
                recommendations.append("Consider adding healthy fats (nuts, avocados, olive oil)")
        
        #Check for overall balance
        if sum(deviations.values()) > 0.3:
            recommendations.append("Overall macronutrient balance could be improved")
            recommendations.append("Try to include foods from all major food groups")
        
        return recommendations
    
    def export_processed_data(self, df: pd.DataFrame, filename: str = "processed_food_data.csv"):
        try:
            df.to_csv(filename, index=False)
            print(f"Dataset exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting dataset: {e}")
            return False
    
    def create_meal_combinations(self, df: pd.DataFrame, target_calories: int = 500) -> List[Dict]:
        if df.empty:
            return []
        
        meal_combinations = []
        items = df.to_dict('records')
        
        for _ in range(10):
            combo = []
            combo_calories = 0
            combo_cost = 0
            
            available_items = items.copy()
            
            while combo_calories < target_calories * 0.8 and available_items:
                item = random.choice(available_items)
                available_items.remove(item)
                
                if combo_calories + item.get('calories', 0) <= target_calories * 1.2:
                    combo.append(item)
                    combo_calories += item.get('calories', 0)
                    combo_cost += item.get('price', 0)
            
            if combo:
                meal_combinations.append({
                    'items': combo,
                    'total_calories': combo_calories,
                    'total_cost': combo_cost,
                    'total_protein': sum(item.get('protein', 0) for item in combo),
                    'item_count': len(combo),
                    'cost_per_calorie': combo_cost / max(combo_calories, 1)
                })
        
        #Sort by cost per calorie
        meal_combinations.sort(key=lambda x: x['cost_per_calorie'])
        
        print(f"Generated {len(meal_combinations)} meal combinations")
        return meal_combinations[:5]
    
    def validate_kaggle_dataset(self, df: pd.DataFrame) -> Dict:
        validation_report = {
            'dataset_size': len(df),
            'column_count': len(df.columns),
            'columns_found': list(df.columns),
            'data_quality': {},
            'issues': [],
            'recommendations': []
        }
        
        required_cols = ['name', 'category', 'calories', 'protein', 'carbohydrates', 'fat']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation_report['issues'].append(f"Missing required columns: {missing_cols}")
        
        for col in ['calories', 'protein', 'carbohydrates', 'fat']:
            if col in df.columns:
                col_data = df[col]
                validation_report['data_quality'][col] = {
                    'missing_values': col_data.isnull().sum(),
                    'zero_values': (col_data == 0).sum(),
                    'negative_values': (col_data < 0).sum(),
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'range': [col_data.min(), col_data.max()]
                }
                
                if col_data.isnull().sum() > len(df) * 0.1:
                    validation_report['issues'].append(f"High missing values in {col}: {col_data.isnull().sum()}")
                
                if (col_data < 0).sum() > 0:
                    validation_report['issues'].append(f"Negative values found in {col}")
        
        if 'name' in df.columns:
            name_data = df['name']
            duplicate_names = name_data.duplicated().sum()
            if duplicate_names > 0:
                validation_report['issues'].append(f"Duplicate food names found: {duplicate_names}")
            
            short_names = (name_data.str.len() < 3).sum()
            if short_names > 0:
                validation_report['issues'].append(f"Very short food names found: {short_names}")
        
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            validation_report['category_distribution'] = category_counts.to_dict()
            
            if len(category_counts) > 50:
                validation_report['recommendations'].append("Consider consolidating categories (50+ unique categories found)")
            
            small_categories = category_counts[category_counts < 5].index.tolist()
            if small_categories:
                validation_report['recommendations'].append(f"Small categories found: {small_categories}")
        
        total_issues = len(validation_report['issues'])
        if total_issues == 0:
            validation_report['quality_score'] = 'Excellent'
        elif total_issues <= 2:
            validation_report['quality_score'] = 'Good'
        elif total_issues <= 5:
            validation_report['quality_score'] = 'Fair'
        else:
            validation_report['quality_score'] = 'Poor'
        
        return validation_report
    
    def preview_dataset_sample(self, df: pd.DataFrame, n: int = 10) -> None:
        print("\n" + "="*80)
        print("DATASET PREVIEW")
        print("="*80)
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\nFirst {n} rows:")
        print("-" * 80)
        
        for i in range(min(n, len(df))):
            row = df.iloc[i]
            print(f"\n{i+1:2d}. {row.get('name', 'Unknown')}")
            print(f"     Category: {row.get('category', 'N/A')}")
            print(f"     Nutrition: {row.get('calories', 0):.0f} cal, {row.get('protein', 0):.1f}g protein, {row.get('carbohydrates', 0):.1f}g carbs, {row.get('fat', 0):.1f}g fat")
            if 'price' in row:
                print(f"     Price: ${row.get('price', 0):.2f}")
        
        print("\n" + "="*80)
        
        if len(df) > 0:
            print("BASIC STATISTICS:")
            print("-" * 30)
            numeric_cols = ['calories', 'protein', 'carbohydrates', 'fat']
            for col in numeric_cols:
                if col in df.columns:
                    print(f"{col.capitalize():15}: {df[col].mean():6.1f} avg, {df[col].min():6.1f}-{df[col].max():6.1f} range")
            
            if 'category' in df.columns:
                print(f"\nCategories ({df['category'].nunique()}): {', '.join(df['category'].value_counts().head().index.tolist())}")
        
        print("="*80)
