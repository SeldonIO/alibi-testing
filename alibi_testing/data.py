import numpy as np
from sklearn.datasets import load_iris, load_boston
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from typing import Union, Tuple


############################################################################################
# The following is copy-pasted from the alibi repo to avoid a dependency on alibi itself
# TODO: figure out a cleaner way of doing this
class Bunch(dict):
    """
    Container object for internal datasets
    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


ADULT_URLS = ['https://storage.googleapis.com/seldon-datasets/adult/adult.data',
              'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
              'http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data']

MOVIESENTIMENT_URLS = ['https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz',
                       'http://www.cs.cornell.edu/People/pabo/movie-review-data/rt-polaritydata.tar.gz']


def fetch_movie_sentiment(return_X_y: bool = False, url_id: int = 0) -> Union[Bunch, Tuple[list, list]]:
    """
    The movie review dataset, equally split between negative and positive reviews.

    Parameters
    ----------
    return_X_y
        If true, return features X and labels y as Python lists, if False return a Bunch object
    url_id
        Index specifying which URL to use for downloading

    Returns
    -------
    Bunch
        Movie reviews and sentiment labels (0 means 'negative' and 1 means 'positive').
    (data, target)
        Tuple if ``return_X_y`` is true
    """
    url = MOVIESENTIMENT_URLS[url_id]
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise

    tar = tarfile.open(fileobj=BytesIO(resp.content), mode="r:gz")
    data = []
    labels = []
    for i, member in enumerate(tar.getnames()[1:]):
        f = tar.extractfile(member)
        for line in f.readlines():
            try:
                line.decode('utf8')
            except UnicodeDecodeError:
                continue
            data.append(line.decode('utf8').strip())
            labels.append(i)
    tar.close()
    if return_X_y:
        return data, labels

    target_names = ['negative', 'positive']
    return Bunch(data=data, target=labels, target_names=target_names)


def fetch_adult(features_drop: list = None, return_X_y: bool = False, url_id: int = 0) -> \
        Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    Downloads and pre-processes 'adult' dataset.
    More info: http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/

    Parameters
    ----------
    features_drop
        List of features to be dropped from dataset, by default drops ["fnlwgt", "Education-Num"]
    return_X_y
        If true, return features X and labels y as numpy arrays, if False return a Bunch object
    url_id
        Index specifying which URL to use for downloading

    Returns
    -------
    Bunch
        Dataset, labels, a list of features and a dictionary containing a list with the potential categories
        for each categorical feature where the key refers to the feature column.
    (data, target)
        Tuple if ``return_X_y`` is true
    """
    if features_drop is None:
        features_drop = ["fnlwgt", "Education-Num"]

    # download data
    dataset_url = ADULT_URLS[url_id]
    raw_features = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital Status',
                    'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
                    'Hours per week', 'Country', 'Target']
    try:
        resp = requests.get(dataset_url)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise

    raw_data = pd.read_csv(StringIO(resp.text), names=raw_features, delimiter=', ', engine='python').fillna('?')

    # get labels, features and drop unnecessary features
    labels = (raw_data['Target'] == '>50K').astype(int).values
    features_drop += ['Target']
    data = raw_data.drop(features_drop, axis=1)
    features = list(data.columns)

    # map categorical features
    education_map = {
        '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
        'Some-college': 'High School grad', 'Masters': 'Masters',
        'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates'
    }
    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar"
    }
    country_map = {
        'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
        'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
        'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
        'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
        'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
        'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
        'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
        'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
        'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
        'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
        'United-States': 'United-States', 'Vietnam': 'SE-Asia'
    }
    married_map = {
        'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
    }
    mapping = {'Education': education_map, 'Occupation': occupation_map, 'Country': country_map,
               'Marital Status': married_map}

    data_copy = data.copy()
    for f, f_map in mapping.items():
        data_tmp = data_copy[f].values
        for key, value in f_map.items():
            data_tmp[data_tmp == key] = value
        data[f] = data_tmp

    # get categorical features and apply labelencoding
    categorical_features = [f for f in features if data[f].dtype == 'O']
    category_map = {}
    for f in categorical_features:
        le = LabelEncoder()
        data_tmp = le.fit_transform(data[f].values)
        data[f] = data_tmp
        category_map[features.index(f)] = list(le.classes_)

    # only return data values
    data = data.values
    target_names = ['<=50K', '>50K']

    if return_X_y:
        return data, labels

    return Bunch(data=data, target=labels, feature_names=features, target_names=target_names, category_map=category_map)


############################################################################################

def get_iris_data(seed=42):
    """
    Load the Iris dataset.
    """
    data = load_iris()
    X, y = data.data, data.target
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # scale dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

    return {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'preprocessor': None,
        'metadata': {
            'feature_names': data.feature_names,
            'name': 'iris'
        }
    }


def get_boston_data(seed=42):
    """
    Load the Boston housing dataset.
    """
    dataset = load_boston()
    data = dataset.data
    labels = dataset.target
    feature_names = dataset.feature_names
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': None,
        'metadata': {
            'feature_names': feature_names,
            'name': 'boston'}
    }


def get_mnist_data():
    """
    Load the MNIST dataset.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    xmin, xmax = -.5, .5
    x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min())) * (xmax - xmin) + xmin
    x_test = ((x_test - x_test.min()) / (x_test.max() - x_test.min())) * (xmax - xmin) + xmin

    return {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'preprocessr': None,
        'metadata': {
            'name': 'mnist'
        }
    }


def get_movie_sentiment_data(seed=42):
    """
    Load and prepare movie sentiment dataset.
    """

    movies = fetch_movie_sentiment()
    data = movies.data
    labels = movies.target
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed)
    y_train = np.array(y_train)
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(x_train)

    return {
        'X_train': x_train,
        'y_train': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'preprocessor': vectorizer,
        'metadata': {'name': 'movie_sentiment'},
    }


def get_adult_data(seed=42, categorical_target=False):
    """
    Load and preprocess the Adult dataset.
    """

    # load raw data
    adult = fetch_adult()
    data = adult.data
    target = adult.target
    feature_names = adult.feature_names
    category_map = adult.category_map

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)

    # Create feature transformation pipeline
    ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
    ordinal_transformer = MinMaxScaler(feature_range=(-1.0, 1.0))

    categorical_features = list(category_map.keys())
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', ordinal_transformer, ordinal_features)
        ]
    )
    preprocessor.fit(X_train)

    if categorical_target:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    # calculate categorical variable columns with the modified feature columns
    cat_vars_ord = {}
    for ix, (key, val) in enumerate(category_map.items()):
        cat_vars_ord[ix] = len(val)
    ohe_keys = np.concatenate(([0], np.cumsum(list(cat_vars_ord.values()))[:-1]))
    cat_vars_ohe = {k: v for k, v in zip(ohe_keys, cat_vars_ord.values())}

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'metadata': {
            'feature_names': feature_names,
            'category_map': category_map,
            'cat_vars_ord': cat_vars_ord,
            'cat_vars_ohe': cat_vars_ohe,
            'name': 'adult'
        }
    }
