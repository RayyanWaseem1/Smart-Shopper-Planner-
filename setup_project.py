import os
import requests
import pandas as pd

def create_directory_structure():
    """Create the necessary directory structure for the project."""
    directories = [
        'algorithms',
        'data',
        'output',
        'output/visualizations',
        'tests'
    ]
    
    print("Creating project directory structure...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create __init__.py files to make directories Python packages
    init_files = [
        'algorithms/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# This file makes the directory a Python package\n')
            print(f"✓ Created: {init_file}")

def download_sample_dataset():
    """Download or create a sample dataset for the project."""
    data_file = 'data/food_nutrition_dataset.csv'
    
    if os.path.exists(data_file):
        print(f"✓ Dataset already exists: {data_file}")
        return
    
    print("Creating sample dataset...")
    
    # Create comprehensive sample dataset
    sample_foods = [
        # Proteins
        {'name': 'Chicken Breast', 'category': 'meat', 'calories': 165, 'protein': 31, 'carbohydrates': 0, 'fat': 3.6},
        {'name': 'Salmon Fillet', 'category': 'seafood', 'calories': 208, 'protein': 20, 'carbohydrates': 0, 'fat': 12},
        {'name': 'Ground Beef (90% lean)', 'category': 'meat', 'calories': 176, 'protein': 26, 'carbohydrates': 0, 'fat': 8},
        {'name': 'Eggs (dozen)', 'category': 'dairy', 'calories': 840, 'protein': 72, 'carbohydrates': 6, 'fat': 60},
        {'name': 'Greek Yogurt', 'category': 'dairy', 'calories': 100, 'protein': 17, 'carbohydrates': 9, 'fat': 0},
        {'name': 'Tofu', 'category': 'vegetarian', 'calories': 70, 'protein': 8, 'carbohydrates': 2, 'fat': 4},
        {'name': 'Black Beans (canned)', 'category': 'vegetarian', 'calories': 227, 'protein': 15, 'carbohydrates': 41, 'fat': 1},
        {'name': 'Lentils (dry)', 'category': 'vegetarian', 'calories': 353, 'protein': 25, 'carbohydrates': 60, 'fat': 1.1},
        
        # Grains & Carbs
        {'name': 'Brown Rice', 'category': 'grains', 'calories': 216, 'protein': 5, 'carbohydrates': 45, 'fat': 1.8},
        {'name': 'Quinoa', 'category': 'grains', 'calories': 222, 'protein': 8, 'carbohydrates': 39, 'fat': 3.6},
        {'name': 'Whole Wheat Bread', 'category': 'bread', 'calories': 247, 'protein': 13, 'carbohydrates': 41, 'fat': 4},
        {'name': 'Oats (rolled)', 'category': 'grains', 'calories': 389, 'protein': 17, 'carbohydrates': 66, 'fat': 7},
        {'name': 'Sweet Potato', 'category': 'vegetables', 'calories': 86, 'protein': 2, 'carbohydrates': 20, 'fat': 0.1},
        {'name': 'Pasta (whole wheat)', 'category': 'grains', 'calories': 174, 'protein': 7, 'carbohydrates': 37, 'fat': 0.8},
        {'name': 'White Rice', 'category': 'grains', 'calories': 205, 'protein': 4, 'carbohydrates': 45, 'fat': 0.4},
        
        # Fruits
        {'name': 'Bananas', 'category': 'fruits', 'calories': 89, 'protein': 1.1, 'carbohydrates': 23, 'fat': 0.3},
        {'name': 'Apples', 'category': 'fruits', 'calories': 52, 'protein': 0.3, 'carbohydrates': 14, 'fat': 0.2},
        {'name': 'Oranges', 'category': 'fruits', 'calories': 47, 'protein': 0.9, 'carbohydrates': 12, 'fat': 0.1},
        {'name': 'Strawberries', 'category': 'fruits', 'calories': 32, 'protein': 0.7, 'carbohydrates': 8, 'fat': 0.3},
        {'name': 'Blueberries', 'category': 'fruits', 'calories': 57, 'protein': 0.7, 'carbohydrates': 14, 'fat': 0.3},
        {'name': 'Avocado', 'category': 'fruits', 'calories': 160, 'protein': 2, 'carbohydrates': 9, 'fat': 15},
        {'name': 'Grapes', 'category': 'fruits', 'calories': 62, 'protein': 0.6, 'carbohydrates': 16, 'fat': 0.2},
        {'name': 'Pineapple', 'category': 'fruits', 'calories': 50, 'protein': 0.5, 'carbohydrates': 13, 'fat': 0.1},
        
        # Vegetables
        {'name': 'Broccoli', 'category': 'vegetables', 'calories': 34, 'protein': 3, 'carbohydrates': 7, 'fat': 0.4},
        {'name': 'Spinach', 'category': 'vegetables', 'calories': 23, 'protein': 3, 'carbohydrates': 4, 'fat': 0.4},
        {'name': 'Carrots', 'category': 'vegetables', 'calories': 41, 'protein': 1, 'carbohydrates': 10, 'fat': 0.2},
        {'name': 'Bell Peppers', 'category': 'vegetables', 'calories': 31, 'protein': 1, 'carbohydrates': 7, 'fat': 0.3},
        {'name': 'Tomatoes', 'category': 'vegetables', 'calories': 18, 'protein': 1, 'carbohydrates': 4, 'fat': 0.2},
        {'name': 'Onions', 'category': 'vegetables', 'calories': 40, 'protein': 1, 'carbohydrates': 9, 'fat': 0.1},
        {'name': 'Garlic', 'category': 'vegetables', 'calories': 149, 'protein': 6, 'carbohydrates': 33, 'fat': 0.5},
        {'name': 'Lettuce', 'category': 'vegetables', 'calories': 15, 'protein': 1, 'carbohydrates': 3, 'fat': 0.2},
        {'name': 'Cucumber', 'category': 'vegetables', 'calories': 16, 'protein': 0.7, 'carbohydrates': 4, 'fat': 0.1},
        
        # Dairy & Alternatives
        {'name': 'Milk (1 gallon)', 'category': 'dairy', 'calories': 2400, 'protein': 128, 'carbohydrates': 184, 'fat': 128},
        {'name': 'Cheese (cheddar)', 'category': 'dairy', 'calories': 402, 'protein': 25, 'carbohydrates': 1, 'fat': 33},
        {'name': 'Butter', 'category': 'dairy', 'calories': 717, 'protein': 1, 'carbohydrates': 0, 'fat': 81},
        {'name': 'Almond Milk', 'category': 'dairy_alternative', 'calories': 13, 'protein': 0.6, 'carbohydrates': 0.3, 'fat': 1.2},
        {'name': 'Soy Milk', 'category': 'dairy_alternative', 'calories': 54, 'protein': 3.3, 'carbohydrates': 6, 'fat': 1.8},
        {'name': 'Cottage Cheese', 'category': 'dairy', 'calories': 98, 'protein': 11, 'carbohydrates': 4, 'fat': 4.3},
        
        # Nuts & Seeds
        {'name': 'Almonds', 'category': 'nuts', 'calories': 579, 'protein': 21, 'carbohydrates': 22, 'fat': 50},
        {'name': 'Walnuts', 'category': 'nuts', 'calories': 654, 'protein': 15, 'carbohydrates': 14, 'fat': 65},
        {'name': 'Chia Seeds', 'category': 'seeds', 'calories': 486, 'protein': 17, 'carbohydrates': 42, 'fat': 31},
        {'name': 'Peanut Butter', 'category': 'nuts', 'calories': 588, 'protein': 25, 'carbohydrates': 20, 'fat': 50},
        {'name': 'Sunflower Seeds', 'category': 'seeds', 'calories': 584, 'protein': 21, 'carbohydrates': 20, 'fat': 51},
        {'name': 'Cashews', 'category': 'nuts', 'calories': 553, 'protein': 18, 'carbohydrates': 30, 'fat': 44},
        
        # Frozen Items
        {'name': 'Frozen Berries', 'category': 'frozen', 'calories': 70, 'protein': 1, 'carbohydrates': 17, 'fat': 0.5},
        {'name': 'Frozen Vegetables Mix', 'category': 'frozen', 'calories': 65, 'protein': 3, 'carbohydrates': 13, 'fat': 0.5},
        {'name': 'Frozen Fish Fillets', 'category': 'frozen', 'calories': 200, 'protein': 22, 'carbohydrates': 0, 'fat': 12},
        {'name': 'Frozen Chicken Strips', 'category': 'frozen', 'calories': 180, 'protein': 15, 'carbohydrates': 12, 'fat': 8},
        
        # Oils & Condiments
        {'name': 'Olive Oil', 'category': 'oils', 'calories': 884, 'protein': 0, 'carbohydrates': 0, 'fat': 100},
        {'name': 'Coconut Oil', 'category': 'oils', 'calories': 862, 'protein': 0, 'carbohydrates': 0, 'fat': 100},
        {'name': 'Honey', 'category': 'sweeteners', 'calories': 304, 'protein': 0.3, 'carbohydrates': 82, 'fat': 0},
        {'name': 'Maple Syrup', 'category': 'sweeteners', 'calories': 260, 'protein': 0, 'carbohydrates': 67, 'fat': 0.06},
        
        # Snacks & Others
        {'name': 'Whole Grain Crackers', 'category': 'snacks', 'calories': 120, 'protein': 3, 'carbohydrates': 20, 'fat': 3},
        {'name': 'Granola', 'category': 'snacks', 'calories': 471, 'protein': 11, 'carbohydrates': 61, 'fat': 22},
        {'name': 'Dark Chocolate', 'category': 'snacks', 'calories': 546, 'protein': 5, 'carbohydrates': 61, 'fat': 31},
        {'name': 'Protein Bars', 'category': 'snacks', 'calories': 200, 'protein': 20, 'carbohydrates': 25, 'fat': 6},
        
        # Beverages
        {'name': 'Green Tea', 'category': 'beverages', 'calories': 2, 'protein': 0, 'carbohydrates': 0, 'fat': 0},
        {'name': 'Orange Juice', 'category': 'beverages', 'calories': 45, 'protein': 0.7, 'carbohydrates': 10, 'fat': 0.2},
        {'name': 'Coffee', 'category': 'beverages', 'calories': 2, 'protein': 0.3, 'carbohydrates': 0, 'fat': 0.02}
    ]
    
    # Convert to DataFrame and save
    df = pd.DataFrame(sample_foods)
    df.to_csv(data_file, index=False)
    print(f"✓ Created sample dataset: {data_file} ({len(sample_foods)} items)")

def create_algorithm_files():
    """Create empty algorithm files with basic structure."""
    algorithm_files = [
        ('algorithms/greedy_optimizer.py', 'Greedy Algorithm Implementation'),
        ('algorithms/dynamic_programming.py', 'Dynamic Programming Implementation'), 
        ('algorithms/heap_manager.py', 'Heap Management Implementation')
    ]
    
    for file_path, description in algorithm_files:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(f'"""\n{description}\n"""\n\n# Implementation will be added here\n')
            print(f"✓ Created: {file_path}")

def create_main_files():
    """Create main application files."""
    main_files = [
        ('data_processor.py', 'Data Processing Module'),
        ('visualizer.py', 'Visualization Module')
    ]
    
    for file_path, description in main_files:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(f'"""\n{description}\n"""\n\n# Implementation will be added here\n')
            print(f"✓ Created: {file_path}")

def create_readme():
    """Create a comprehensive README file."""
    readme_content = """# Smart Shopping Planner

A real-world application that utilizes data structures and algorithms to optimize grocery shopping under realistic constraints, including budget, dietary restrictions, expiration dates, and nutritional goals.

## Authors
- Ateeq Ur Rehman
- Rayyan Waseem

## Project Overview

This Smart Shopping Planner implements three core Data Structures and Algorithms concepts:

### 1. Greedy Algorithms
- Used for locally optimal decisions (e.g., selecting items based on cost-to-nutrition ratio)
- Builds basic planners for tight budgets by picking the next best item until constraints are met

### 2. Dynamic Programming
- Applied to solve the diet problem (variant of knapsack problem)
- Selects subset of food items to satisfy nutritional goals while staying within budget
- Optimizes for minimum calories, protein, carbohydrates, and fats

### 3. Heaps (Priority Queues)
- Manages and prioritizes perishable items based on expiration dates
- Recommends items that should be consumed or restocked first
- Minimizes food waste through intelligent scheduling

## Features

- **Budget Optimization**: Stay within your budget while maximizing nutritional value
- **Dietary Restrictions**: Support for vegetarian, vegan, gluten-free, and other dietary needs
- **Expiration Management**: Prioritize items by expiration date to minimize waste
- **Nutritional Goals**: Meet specific calorie, protein, and macronutrient targets
- **Visual Analytics**: Comprehensive charts and visualizations
- **Smart Scheduling**: Day-by-day consumption recommendations

## Installation

1. Clone or download this project
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script to create project structure:
```bash
python setup_project.py
```

## Usage

Run the main application:
```bash
python main_app.py
```

The program will:
1. Load the food dataset
2. Apply your dietary preferences and budget constraints
3. Run all three algorithms (Greedy, DP, Heap)
4. Generate an optimized shopping plan
5. Create visualizations and save results

## Project Structure

```
smart-shopping-planner/
├── algorithms/
│   ├── greedy_optimizer.py      # Greedy algorithm implementation
│   ├── dynamic_programming.py   # DP knapsack solver
│   └── heap_manager.py         # Heap-based expiration management
├── data/
│   └── food_nutrition_dataset.csv  # Food database
├── output/
│   ├── visualizations/         # Generated charts and graphs
│   └── shopping_plan.json     # Detailed shopping plan
├── main_app.py                # Main application
├── data_processor.py          # Data loading and preprocessing
├── visualizer.py             # Chart generation
└── requirements.txt          # Python dependencies
```

## Algorithm Details

### Greedy Algorithm
- **Time Complexity**: O(n log n)
- **Use Case**: Quick budget optimization with good results
- **Method**: Sorts items by cost-to-nutrition ratio and selects greedily

### Dynamic Programming
- **Time Complexity**: O(nW) where W is the budget
- **Use Case**: Optimal solution for nutritional requirements
- **Method**: Solves multi-dimensional knapsack problem

### Heap Management  
- **Time Complexity**: O(n log n)
- **Use Case**: Expiration date prioritization and waste reduction
- **Method**: Min-heap for expiration dates, max-heap for priority scores

## Sample Output

The program generates:
- **Shopping Plan**: Optimized list of items with costs and nutritional info
- **Budget Analysis**: Detailed breakdown of spending vs. budget
- **Nutritional Summary**: Daily averages and totals
- **Expiration Timeline**: Visual calendar of when items expire
- **Algorithm Performance**: Comparison of different approaches
- **Waste Reduction Tips**: Personalized recommendations

## Customization

Modify the user preferences in `main_app.py`:

```python
user_preferences = {
    'budget': 150.0,
    'nutritional_goals': {
        'min_calories': 2000,
        'min_protein': 50,
        'min_carbs': 250,
        'min_fat': 65
    },
    'dietary_restrictions': ['vegetarian'],
    'planning_days': 7
}
```

## Data Sources

- Food nutritional data from public datasets
- Pricing information simulated based on real market data
- Expiration dates calculated based on food category and storage conditions

## Contributing

This is a educational project demonstrating DSA concepts. Feel free to:
- Add new algorithms
- Enhance the dataset
- Improve visualizations
- Add new dietary restrictions
- Implement additional optimization strategies

## License

This project is for educational purposes. Feel free to use and modify for learning and academic projects.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    print("✓ Created: README.md")

def create_sample_config():
    """Create a sample configuration file."""
    config_content = """# Smart Shopping Planner Configuration

# Default user preferences (can be modified)
DEFAULT_BUDGET = 150.0
DEFAULT_PLANNING_DAYS = 7

# Nutritional targets
DEFAULT_NUTRITIONAL_GOALS = {
    'min_calories': 2000,
    'min_protein': 50,
    'min_carbohydrates': 250,
    'min_fat': 65
}

# Available dietary restrictions
AVAILABLE_RESTRICTIONS = [
    'vegetarian',
    'vegan', 
    'gluten_free',
    'dairy_free',
    'low_sodium',
    'keto',
    'high_protein'
]

# Algorithm weights (for customizing optimization)
GREEDY_WEIGHTS = {
    'calories': 0.3,
    'protein': 0.4,
    'carbohydrates': 0.2,
    'fat': 0.1
}

# Visualization settings
CHART_STYLE = 'seaborn'
OUTPUT_DPI = 300
FIGURE_SIZE = (12, 8)
"""
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("✓ Created: config.py")

def main():
    """Main setup function."""
    print("🛒 Smart Shopping Planner - Project Setup")
    print("=" * 50)
    
    try:
        # Create directory structure
        create_directory_structure()
        print()
        
        # Create sample dataset
        download_sample_dataset()
        print()
        
        # Create algorithm files
        create_algorithm_files()
        print()
        
        # Create main files
        create_main_files()
        print()
        
        # Create README
        create_readme()
        print()
        
        # Create configuration file
        create_sample_config()
        print()
        
        print("✅ Project setup completed successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy the algorithm implementations to their respective files")
        print("3. Copy the main application code to main_app.py")
        print("4. Copy the data processor code to data_processor.py") 
        print("5. Copy the visualizer code to visualizer.py")
        print("6. Run the application: python main_app.py")
        print("\n📊 Your Smart Shopping Planner is ready to use!")
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
