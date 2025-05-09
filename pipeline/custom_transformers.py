from sklearn.base import BaseEstimator, TransformerMixin

from pipeline.util.peb_util import map_label_to_kwh


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops specified columns from a DataFrame."""

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x.drop(columns=self.columns_to_drop)
    
class NADropper(BaseEstimator, TransformerMixin):
    """Transformer that drops rows with NaN values in specified columns."""

    def __init__(self, subset=None):
        self.subset = subset

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x.dropna(subset=self.subset)
    
class NAReplacer(BaseEstimator, TransformerMixin):
    """Transformer that replaces specified values in a DataFrame."""

    def __init__(self, column, new_value):
        self.column = column
        self.new_value = new_value

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_copy = x.copy()
        
        if isinstance(self.column, list):
            # Handle list
            for col in self.column:
                x_copy = impute(x_copy, col, self.new_value)
        else:  
            x_copy = impute(x_copy, self.column, self.new_value)
        
        return x_copy

def impute(df, column, new_value):
    """Helper function to impute missing values in a DataFrame column."""
    
    new_val = new_value 
            
    if new_value == 'median':
        new_val = int(df[column].median())
    elif new_value == 'mode':
        new_val = df[column].mode()[0]

    df[column] = df[column].fillna(new_val)

    return df

class DuplicateDropper(BaseEstimator, TransformerMixin):
    """Transformer that drops duplicate rows from a DataFrame."""

    def __init__(self, subset=None):
        self.subset = subset
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return x.drop_duplicates(self.subset)
    
class EpcKwhCalculator(BaseEstimator, TransformerMixin):
    """Transformer that calculates epc_kwh from epcScore & province."""

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x_copy = x.copy()

        x_copy["epc_kwh"] = x_copy.apply(map_label_to_kwh, axis=1)

        return x_copy
        x_copy = x.copy()

        if self.columns is not None:
            for col in self.columns:
                order_by_price = x_copy.groupby(col)['price'].mean().round().sort_values(ascending=False)
                price_dict = order_by_price.to_dict()
                
                # Rescale to integers from 1 to len(unique values) - Thx to Copilot
                unique_values = sorted(set(price_dict.values()), reverse=True)
                scaled_dict = {cat: i+1 for i, cat in enumerate(unique_values)}
                
                # Map twice: first to price, then to scaled integer
                x_copy[col] = x_copy[col].map(price_dict).map(scaled_dict)
        
        return x_copy
    
class BooleanTransformer(BaseEstimator, TransformerMixin):
    """Transformer that converts boolean strings to integers (True=1, False=0)."""
    
    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x_copy = x.copy()
        
        bool_columns = x_copy.select_dtypes(include='bool').columns
        for column in bool_columns:
            x_copy[column] = x_copy[column].astype(int)
            
        return x_copy
    
class ToIntTransformer(BaseEstimator, TransformerMixin):
    """Transformer that converts columns to int, removing decimal parts."""

    def __init__(self, for_display=False):
        self.for_display = for_display

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x_copy = x.copy()

        for col in x_copy.columns:
            x_copy[col] = x_copy[col].astype(int)
        
        return x_copy