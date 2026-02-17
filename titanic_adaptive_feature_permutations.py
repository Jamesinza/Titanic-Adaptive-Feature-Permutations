import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


@tf.keras.utils.register_keras_serializable(package="Custom")
class LearnableFeaturePermute(tf.keras.layers.Layer):
    def __init__(self, num_features, num_iters=10, temperature=1.0):
        super().__init__()
        self.num_features = num_features
        self.num_iters = num_iters

        self.temperature = tf.Variable(
            temperature, trainable=False, dtype=tf.float32
        )
        self.logits = self.add_weight(
            shape=(num_features, num_features),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True
        )

    def _sinkhorn(self, log_alpha):
        # log_alpha: shape (F, F)
        logP = log_alpha
        for _ in range(self.num_iters):
            # normalize rows
            logP = logP - tf.reduce_logsumexp(logP, axis=1, keepdims=True)
            # normalize cols
            logP = logP - tf.reduce_logsumexp(logP, axis=0, keepdims=True)
        return tf.exp(logP)
        
    def call(self, x, training=None):
        if training:
            # Temperature annealing
            self.temperature.assign(tf.maximum(0.1, self.temperature * 0.999))
    
        P = self._sinkhorn(self.logits / self.temperature)
    
        # Entropy regularization
        entropy = -tf.reduce_sum(P * tf.math.log(P + 1e-8))
        self.add_loss(1e-3 * entropy)  # λ = 1e-3 is a good start
    
        out = tf.matmul(x, P)
        return out

        
def preprocess_titanic(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 80], labels=False)
    df['IsChild'] = (df['Age'] < 18).astype(int)
    
    embarked_map = {'C':0,'Q':1,'S':2}
    df['Embarked'] = df['Embarked'].map(embarked_map).fillna(2)
    
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles,'Rare')
    title_mapping = {'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4}
    df['Title'] = df['Title'].map(title_mapping)
        
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    deck_map = {letter:i for i, letter in enumerate('ABCDEFGU')}
    df['Deck'] = df['Deck'].map(deck_map)
        
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Sex_Pclass'] = df['Sex'] * df['Pclass']
    df['Title_Pclass'] = df['Title'] * df['Pclass']
    df['IsChild_Pclass'] = df['IsChild'] * df['Pclass']

    # Only the following columns will be used as input
    features = [
        'Pclass', 'Sex', 'Age', 'Fare',
        'FamilySize', 'IsAlone', 'FarePerPerson',
        'Title', 'IsChild', 'Embarked', 'AgeBin',
        'Sex_Pclass', 'Title_Pclass', 'Deck', 'IsChild_Pclass'
    ]
    # print(df[features])
    
    # Splitting inputs from target
    y = df['Survived'].values
    X = df[features].values
    X = np.nan_to_num(X, nan=0.0) # Filling few NaN's with 0.0

    # Data normalization will be built into model architecture
    scaler = tf.keras.layers.Normalization()
    scaler.adapt(X)
    return X, y, scaler
    

train_data = pd.read_csv('datasets/train.csv')
test_data = pd.read_csv('datasets/test.csv')

X, y, scaler = preprocess_titanic(train_data)
# X_test, y_test, _ = preprocess_titanic(test_data)

unique_classes     = np.unique(y)
class_weights      = compute_class_weight('balanced', classes=unique_classes, y=y)
class_weight_dicts = dict(enumerate(class_weights))
num_classes        = len(unique_classes)

# X = X.reshape(X.shape[0], 1, X.shape[-1]) # Reshape data for use with RNN
input_shape = (X.shape[-1],)

permute1 = LearnableFeaturePermute(X.shape[-1])
permute2 = LearnableFeaturePermute(X.shape[-1])

inputs = tf.keras.layers.Input(input_shape)
x = scaler(inputs)

x1 = permute1(x)
x1 = tf.keras.layers.Dense(128, activation='gelu', kernel_initializer='he_normal')(x1)
x1 = tf.keras.layers.Dropout(0.7)(x1)

x2 = permute2(x)
x2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x2)
x2 = tf.keras.layers.Dropout(0.7)(x2)

# Add all the branches, then create fusion weight
comb     = [x1,x2]
fusion   = tf.keras.layers.Concatenate()(comb)
weights  = tf.keras.layers.Dense(len(comb), activation='softmax')(fusion)

# Split weights and apply
w1 = tf.keras.layers.Lambda(lambda w: w[:, 0:1])(weights)
w2 = tf.keras.layers.Lambda(lambda w: w[:, 1:2])(weights)

x = tf.keras.layers.Add()([
    tf.keras.layers.Multiply()([x1, w1]),
    tf.keras.layers.Multiply()([x2, w2]),
])

outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True, verbose=1),
    # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=50),
    ]
history = model.fit(X, y, validation_split=0.1, epochs=10000, batch_size=8,
                    class_weight=class_weight_dicts, callbacks=callbacks, verbose=1)

