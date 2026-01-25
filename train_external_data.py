CONFIG = {
    # ============================================
    # CSV PATH - Use Kalimati dataset
    # ============================================
    'csv_path': 'kalimati_tarkari_dataset.csv',
    
    # ============================================
    # COLUMN MAPPINGS - Update based on STEP 2
    # ============================================
    'columns': {
        'date': 'Date',              # ← Column name for dates
        'commodity': 'Commodity',    # ← Column name for vegetable names
        'volume': 'Average',         # ← Kalimati usually doesn't have volume, use price
        'price': 'Average'           # ← Use 'Average', 'Maximum', or 'Minimum'
    },
    
    # ============================================
    # COMMODITIES - Common in Kalimati dataset
    # ============================================
    'commodities': [
        'Tomato',
        'Potato',
        'Onion',
        'Cauliflower',
        'Cabbage',
        'Beans',
        'Cucumber',
        'Carrot'
    ],
    
    # ============================================
    # TRAINING PARAMETERS
    # ============================================
    'sequence_length': 4,    # Use 4 weeks of history
    'epochs': 100,           # Training iterations
    'test_weeks': 8,         # Weeks to hold out for testing
    
    # ============================================
    # OUTPUT DIRECTORIES
    # ============================================
    'model_dir': 'models',
    'plot_dir': 'plots'
}
