import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Carregar os dados
train = pd.read_csv('train_tit.csv')
test = pd.read_csv('test_tit.csv')

# Passo 1: Limpeza dos Dados
# Remover as colunas 'Name', 'Ticket' e 'Cabin' (essas colunas contêm texto irrelevante para análise)
train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Passo 2: Codificação das Variáveis Categóricas
# Codificar 'Sex' (masculino=0, feminino=1)
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])

# 'Embarked' é uma variável categórica com mais de 2 categorias. Usaremos One-Hot Encoding.
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

# Garantir que o conjunto de teste tenha as mesmas colunas após o One-Hot Encoding
for col in train.columns:
    if col not in test.columns:
        test[col] = 0  # Adicionar a coluna ao conjunto de teste, se faltar

for col in test.columns:
    if col not in train.columns:
        train[col] = 0  # Adicionar a coluna ao conjunto de treino, se faltar

# Passo 3: Tratar Valores Ausentes
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)

# Passo 4: Divisão entre variáveis independentes (X) e dependentes (y)
X = train.drop('Survived', axis=1)  # 'Survived' é a variável alvo
y = train['Survived']

# Dividir os dados em conjunto de treino e validação
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para adicionar porcentagens nos gráficos de barras
def add_percentage_text(ax, total):
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.text(p.get_x() + p.get_width() / 2., height + 10, f'{percentage:.1f}%', ha='center', fontsize=12)

# Passo 5: Verificar as colunas após One-Hot Encoding
print("\nColunas após One-Hot Encoding:")
print(train.columns)  # Verificar as colunas para garantir quais foram geradas

# Passo 6: Visualizações - Gráficos para todas as colunas

# 1. Gráficos para variáveis categóricas
categorical_columns = ['Sex', 'Pclass', 'Embarked_S', 'Embarked_Q']  # Variáveis categóricas

for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=col, data=train, palette='Set2')
    total = len(train)
    add_percentage_text(ax, total)
    plt.title(f'Distribuição de {col} com Porcentagens', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    plt.show()

# 2. Gráficos para variáveis numéricas
numerical_columns = ['Age', 'SibSp', 'Parch', 'Fare']  # Variáveis numéricas
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    if col == 'Fare':  # Fare tem uma distribuição com outliers, ajustamos para boxplot
        sns.boxplot(x=train[col], color='orange')
        plt.title(f'Distribuição de {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
    else:
        sns.histplot(train[col], bins=30, kde=True, color='teal')
        plt.title(f'Distribuição de {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    plt.show()

# 3. Gráficos de correlacionamento para variáveis numéricas
plt.figure(figsize=(10, 8))
correlation_matrix = train[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação entre Variáveis Numéricas', fontsize=16)
plt.show()
