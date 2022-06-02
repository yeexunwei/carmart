import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



class DateTransformer(BaseEstimator, TransformerMixin):
    '''
    Convert date type data
    '''
    def __init__(self, reg_date=True, manufactured=True, lifespan=False, original_reg_date=False):
        self.reg_date = reg_date
        self.manufactured = manufactured
        self.lifespan = lifespan
        self.original_reg_date = original_reg_date
        self.columns = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X.loc[:, 'reg_date'] = pd.to_datetime(X.reg_date).dt.year
        if self.reg_date:
            X.loc[:, 'years_since_reg_date'] = pd.datetime.now().year - X['reg_date']
            X.drop('reg_date', axis=1)
        if self.manufactured:
            X.loc[X['manufactured'] > pd.datetime.now().year] = np.nan
            X.loc[:, 'years_since_manufactured'] = pd.datetime.now().year - X['manufactured']
        if self.lifespan:
            X.loc[:, 'lifespan'] = pd.to_datetime(X.lifespan).dt.year
            X.loc[:, 'years_to_lifespan'] = X['lifespan'] - pd.datetime.now().year
            X.loc[:, 'years_to_lifespan'] = X.loc[:, 'years_to_lifespan'].fillna(0)
        if self.original_reg_date:
             X.loc[:, 'original_reg_year'] = pd.to_datetime(X.original_reg_date).dt.year
        X = X.drop(['manufactured', 'reg_date', 'lifespan', 'original_reg_date'], axis=1)
        self.columns = list(X)
        return X
    
    def get_feature_names(self):
        return self.columns
    
    
class NumericalTransformer(BaseEstimator, TransformerMixin):
    '''
    Convert numerical data
    '''
    def __init__(self, dropna=True):
        self.dropna = dropna
        self.columns = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.dropna:
            X = X.drop(['indicative_price'], axis=1) # too many na
        self.columns = list(X)
        return X
    
    def get_feature_names(self):
        return self.columns
    

class OneHotCategoricalTransformer(BaseEstimator, TransformerMixin):
    '''
    Convert categorical data -> one-hot encoding 
    'title', 'make', 'model', 'type_of_vehicle'
    '''
    def __init__(self, make_from_title=True, make=True, model=True):
        self.make_from_title = make_from_title
        self.make = make
        self.model = model
        self.columns = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        '''Normalize make col using data from title'''
        X.loc[:, 'title'] = X['title'].str.replace(r"\(.*\)","").str.lower() # convert 'title' to lower
        X.loc[X['make'].isnull(), 'make'] = X['title'].str.split().str.get(0).str.lower() # fillna make with first word from title
        X = X.drop(['title'], axis=1) # drop 'title', data are available at 'make'
        
        X.loc[X['model'].isnull(), 'model'] = 'others'
        # if not self.make: # remove make, convert to 'luxury'
        #     # luxury_brands = ['aston martin', 'lagonda', 'duesenberg', 'cord', 'auburn', 'bmw', 'rolls-royce', 'imperial', 'chrysler', 'desoto', 'mercedes', 'mercedes-benz', 'maybach', 'hongqi', 'ferrari', 'lincoln', 'continental', 'zephyr', 'volvo cars', 'lynk & co', 'polestar', 'cadillac', 'lasalle', 'hispano-suiza', 'acura', 'greater eight', 'italia', 'hiphi', 'genesis motor', 'apollo', 'li auto', 'lucid motors', 'automobili pininfarina', 'ambassador', 'nash-healey', 'saab', 'aurus', 'nio', 'infiniti', 'ds', 'alfa romeo', 'maserati', 'packard', 'jaguar', 'land rover', 'daimler', 'lanchester', 'tesla', 'lexus', 'karma automotive', 'audi', 'bentley', 'bugatti', 'porsche', 'lamborghini', 'wey']
        #     # X['make'] = X['make'].fillna('budget')
        #     # X.loc[X['make'].isin(luxury_brands), 'brand_category'] = 'luxury'
        #     # X.loc[X['brand_category'] != 'luxury', 'brand_category'] = 'budget'
        #     X = X.drop(['make'], axis=1) # rm make col

        # if not self.model:
        #     X = X.drop(['model'], axis=1) # rm model col

        self.columns = list(X)
        return X
    
    def get_feature_names(self):
        return self.columns

    
class LabelCategoricalTransformer(BaseEstimator, TransformerMixin):
    '''
    Convert categorical data -> label encoding
    '''
    def __init__(self, category=True, fuel_type=True):
        self.category = category
        self.fuel_type = fuel_type
        self.columns = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.category:
            keywords = ['parf', 'premium ad', 'low mileage', 'imported used', 'coe', 'almost new', 'rare & exotic', 'hybrid', 'direct owner sale', 'sgcarmart warranty', 'vintage', 'sta evaluated', 'opc', 'consignment', 'electric']
            for keyword in keywords:
                X.loc[:, 'cat_' + keyword.replace(' ', '_')] = X['category'].str.contains(keyword) * 1
                
        if self.fuel_type:
            X['fuel_type'] = X['fuel_type'].fillna('diesel')
            X.loc[X['fuel_type'].str.contains('electric'), 'fuel_type'] = 1
            X.loc[X['fuel_type'] != 1, 'fuel_type'] = 0
        
        X.loc[X['transmission'] == 'manual', 'transmission'] = 1
        X.loc[X['transmission'] != 'manual', 'transmission'] = 0
        
        X = X.drop(['listing_id', 'eco_category', 'opc_scheme'], axis=1) # rm unique and redundant cols
        X = X.drop(['category'], axis=1) # convert to feature cols
        text_attribs = ['description', 'features', 'accessories']
        X = X.drop(text_attribs, axis=1) # remove text cols
        self.columns = list(X)
        return X
    
    def get_feature_names(self):
        return self.columns


class PreprocessedDataFrame():
    def __init__(self, X_train, y_train, target_encoding=False):
        self.X_train = X_train
        self.y_train = y_train
        self.target_encoding = target_encoding

        # categorize cols
        self.date_attribs = ['reg_date', 'manufactured', 'lifespan', 'original_reg_date']
        self.cat_attribs = ['listing_id', 'title', 'make', 'model', 'type_of_vehicle', 'category', 'transmission', 'fuel_type', 'eco_category', 'features', 'accessories', 'description', 'opc_scheme']
        self.oh_cat_attribs = ['title', 'make', 'model', 'type_of_vehicle']
        self.label_cat_attribs = list(set(self.cat_attribs) - set(self.oh_cat_attribs))
        self.num_attribs = list(set(self.X_train) - set(self.cat_attribs) - set(self.date_attribs))
        self.oh_cat_attribs.remove('type_of_vehicle')
        self.print_attribs_type()
        self.initialize_pipelines()

    def print_attribs_type(self):
        print('Date:', self.date_attribs)
        print('Num:', self.num_attribs)
        print('OneHot Cat:', self.oh_cat_attribs)
        print('Label Cat:', self.label_cat_attribs)
        print()

    def initialize_pipelines(self, make=True):
        '''Pipelines
        set feature to True if want to keep it
        '''
        date_pipeline = Pipeline([
            ('date_trans', DateTransformer(reg_date=True, manufactured=True, lifespan=False, original_reg_date=False)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('std_scaler', StandardScaler()),
        ])
        num_pipeline = Pipeline([
            ('num_trans', NumericalTransformer(dropna=True)),
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])
        if self.target_encoding:
            oh_cat_pipeline = Pipeline([
                ('cat_trans', OneHotCategoricalTransformer(make_from_title=True, make=True, model=True)),
            ])
        else:
            oh_cat_pipeline = Pipeline([
                ('cat_trans', OneHotCategoricalTransformer(make_from_title=True, make=True, model=True)),
                ('one_hot', OneHotEncoder(handle_unknown='ignore')),
            ])
        label_cat_pipeline = Pipeline([
            ('label_cat_trans', LabelCategoricalTransformer(category=True, fuel_type=True)),
        ])
        model_target_encoder = TargetEncoder()

        self.full_pipeline = ColumnTransformer([
            ('date', date_pipeline, self.date_attribs),
            ('oh_cat', oh_cat_pipeline, self.oh_cat_attribs),
            ('label_cat', label_cat_pipeline, self.label_cat_attribs),
            ('num', num_pipeline, self.num_attribs),
            ('encode_tov', OneHotEncoder(handle_unknown='ignore'), ['type_of_vehicle'])
        ])
        return self.full_pipeline
        # print(self.full_pipeline)
        # print()

    def check_feat_processing(self, check='oh_cat'): #num, oh_cat, label_cat, date
        '''Test one of the four preprocessing transformers'''
        if check == 'num':
            attribs_transformer = NumericalTransformer()
            transformed_df = attribs_transformer.transform(self.X_train[self.num_attribs].copy())
        elif check == 'oh_cat':
            attribs_transformer = OneHotCategoricalTransformer()
            transformed_df = attribs_transformer.transform(self.X_train[self.oh_cat_attribs].copy())
        elif check == 'label_cat':
            attribs_transformer = LabelCategoricalTransformer()
            transformed_df = attribs_transformer.transform(self.X_train[self.label_cat_attribs].copy())
        else:
            attribs_transformer = DateTransformer()
            transformed_df = attribs_transformer.transform(self.X_train[self.date_attribs].copy())
        return transformed_df

    def get_transformed_attribs(self):
        '''
        Get transformed attributes names
        '''
        def get_transformed_attribs_for(features_type):
            encoder = self.full_pipeline.named_transformers_[features_type]
            transformed_attribs = encoder[0].get_feature_names() #list(cat_encoder[1].categories_[0])
            return transformed_attribs

        # get transformed date cols
        date_transformed_attribs = get_transformed_attribs_for('date')

        # get transformed one hot categorical cols
        oh_cat_transformed_attribs = get_transformed_attribs_for('oh_cat')
        if not self.target_encoding:
            cat_encoder = self.full_pipeline.named_transformers_["oh_cat"]
            oh_cat_transformed_attribs = cat_encoder[1].get_feature_names(oh_cat_transformed_attribs) #list(cat_encoder[1].categories_[0])
            oh_cat_transformed_attribs = list(oh_cat_transformed_attribs)

        # get transformed label categorical cols
        label_cat_transformed_attribs = get_transformed_attribs_for('label_cat')

        # get transformed numerical cols
        num_transformed_attribs = get_transformed_attribs_for('num')

        # tov_transformed_attribs = get_transformed_attribs_for('encode_tov')
        tov_encoder = self.full_pipeline.named_transformers_["encode_tov"]
        tov_transformed_attribs = tov_encoder.get_feature_names(['tov']) #list(cat_encoder[1].categories_[0])
        tov_transformed_attribs = list(tov_transformed_attribs)
        
        self.transformed_attribs = date_transformed_attribs + oh_cat_transformed_attribs + label_cat_transformed_attribs + num_transformed_attribs + tov_transformed_attribs
        return self.transformed_attribs


    def build_dataframe(self):
        '''
        Fit and transform X_train
        '''
        self.X_train_transformed = self.full_pipeline.fit_transform(self.X_train)
        transformed_attribs = self.get_transformed_attribs()
        if not self.target_encoding:
            self.X_train_transformed = pd.DataFrame.sparse.from_spmatrix(self.X_train_transformed, columns=transformed_attribs)
        else:
            self.X_train_transformed = pd.DataFrame(self.X_train_transformed, columns=transformed_attribs)
        
        print('Input shape:', self.X_train.shape)
        print('Transformed shape:', self.X_train_transformed.shape)

        return self.X_train_transformed

    def transform_dataframe(self, X_test):
        '''
        Transform X_test
        '''
        X_test_prepared = self.full_pipeline.transform(X_test)
        transformed_attribs = self.get_transformed_attribs()
        if not self.target_encoding:
            X_test_prepared = pd.DataFrame.sparse.from_spmatrix(X_test_prepared, columns=transformed_attribs)
        else:
            X_test_prepared = pd.DataFrame(X_test_prepared, columns=transformed_attribs)
        
        print('Input shape:', X_test.shape)
        print('Transformed shape:', X_test_prepared.shape)

        return X_test_prepared
