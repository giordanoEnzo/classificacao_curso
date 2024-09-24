import pandas as pd
#%%
df = pd.read_csv("dados_estudantes.csv")
#%%
df.head()
#%%
df.info()
#%%
df['Target'].unique()
#%%
df.describe()
#%%
colunas_categoricas = ['Estado civil', 'Migração', 'Sexo', 'Estrangeiro', 'Necessidades educacionais especiais', 'Devedor',  'Taxas de matrícula em dia', 'Bolsista', 'Curso', 'Período', 'Qualificação prévia', 'Target']
#%%
df[colunas_categoricas].describe()
#%%
df['Estado civil'].value_counts(normalize=True)*100
#%%
import matplotlib.pyplot as plt
import seaborn as sns
#%%
sns.displot(df['Idade na matrícula'], bins=20)
plt.show()
#%%
color_dict = {'Desistente': '#e34c42', 'Graduado': '#4dc471', 'Matriculado': '#3b71db'}
sns.set_palette(list(color_dict.values()))
#%%
sns.displot(data=df, x='Idade na matrícula', hue='Target', kind='kde', fill=True)
plt.show()
#%%
df['Estrangeiro'].value_counts(normalize=True)*100
#%%
df['Sexo'].value_counts(normalize=True)*100
#%%
sns.countplot(x='Sexo', hue='Target', data=df)
plt.show()
#%%
sns.countplot(x='Devedor', hue='Target', data=df)
plt.show()
#%%
sns.countplot(x='Taxas de matrícula em dia', hue='Target', data=df)
plt.show()
#%%
sns.countplot(x='Bolsista', hue='Target', data=df)
plt.show()
#%%
import plotly.express as px
import plotly.io as pio
#%%
pio.renderers.default = 'browser' 
#%%
contagem = df.groupby(['Curso', 'Target']).size().reset_index(name='Contagem')
#%%
contagem['Porcentagem'] = contagem.groupby('Curso')['Contagem'].transform(lambda x: (x / x.sum()) * 100)
#%%
fig = px.bar(contagem, y='Curso', x='Porcentagem', color='Target', orientation='h', 
             color_discrete_map={'Desistente': '#e34c42', 'Graduado': '#4dc471', 'Matriculado': '#3b71db'})
fig.show()
#%%
sns.boxplot(x='Target', y='disciplinas 1º semestre (notas)', data=df)
plt.show()
#%%
sns.boxplot(x='Target', y='disciplinas 2º semestre (notas)', data=df)
plt.show()
#%%
df['Target'].value_counts(normalize=True)*100
#%%
from sklearn.preprocessing import OneHotEncoder
#%%
colunas_categoricas = ['Estado civil', 'Migração', 'Sexo', 'Estrangeiro', 'Necessidades educacionais especiais', 'Devedor',  'Taxas de matrícula em dia', 'Bolsista', 'Curso', 'Período', 'Qualificação prévia']

encoder = OneHotEncoder(drop='if_binary')

df_categorico = df[colunas_categoricas]

df_encoded = pd.DataFrame(encoder.fit_transform(df_categorico).toarray(),
                          columns=encoder.get_feature_names_out(colunas_categoricas))

df_final = pd.concat([df.drop(colunas_categoricas, axis=1), df_encoded], axis=1)
#%%
df_final
#%%
X = df_final.drop('Target', axis=1)
y = df_final['Target']
#%%
from sklearn.model_selection import train_test_split
#%%
X, X_teste, y, y_teste = train_test_split(X, y, test_size=0.15, stratify=y, random_state=0)
X_treino, X_val, y_treino, y_val = train_test_split(X, y, stratify=y, random_state=0)
#%%
from sklearn.ensemble import RandomForestClassifier
#%%
modelo_rf = RandomForestClassifier(random_state=0)
modelo_rf.fit(X_treino, y_treino)
y_pred = modelo_rf.predict(X_val)
print(f'Acurácia de treino: {modelo_rf.score(X_treino, y_treino)}')
print(f'Acurácia de validação: {modelo_rf.score(X_val, y_val)}')
#%%
#from sklearn.ensemble import GradientBoostingClassifier
#%%
#modelo_gb = GradientBoostingClassifier(random_state=0)
#modelo_gb.fit(X_treino, y_treino)
#y_pred = modelo_gb.predict(X_val)
#print(f'Acurácia de treino: {modelo_gb.score(X_treino, y_treino)}')
#print(f'Acurácia de validação: {modelo_gb.score(X_val, y_val)}')
#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#%%
matriz_confusao = confusion_matrix(y_val, y_pred)
#%%
visualizacao = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao, display_labels=modelo_rf.classes_)
visualizacao.plot(cmap='winter');
#%%
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, normalize='true', cmap='Blues');
#%%
from sklearn.metrics import classification_report
#%%
report = classification_report(y_val, y_pred)
print('Relatório de classificação:\n', report)
#%%
y_treino.value_counts()
#%%
from imblearn.over_sampling import SMOTE
#%%
oversample = SMOTE(random_state=0)
X_balanceado, y_balanceado = oversample.fit_resample(X_treino, y_treino)
#%%
y_balanceado.value_counts()
#%%
modelo = RandomForestClassifier(random_state=0, max_depth=10)
modelo.fit(X_balanceado, y_balanceado)
y_pred = modelo.predict(X_val)
#%%
print(classification_report(y_val, y_pred))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, normalize='true', cmap='Blues');
#%%
from imblearn.pipeline import Pipeline as imbpipeline
#%%
modelo = RandomForestClassifier(max_depth=10, random_state=0)
pipeline = imbpipeline([('oversample', SMOTE(random_state=0)), ('floresta', modelo)])
#%%
from sklearn.model_selection import cross_validate, StratifiedKFold
#%%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
cv_resultados = cross_validate(pipeline, X, y, cv=skf, scoring='recall_weighted')
cv_resultados['test_score']
#%%
media = cv_resultados['test_score'].mean()
desvio_padrao = cv_resultados['test_score'].std()
print(f'Intervalo de confiança: [{media - 2 * desvio_padrao}, {min(media + 2 * desvio_padrao, 1.0)}]')
#%%
oversample = SMOTE(random_state=0)
X_balanceado, y_balanceado = oversample.fit_resample(X, y)
modelo = RandomForestClassifier(random_state=0, max_depth=10)
modelo.fit(X_balanceado, y_balanceado)
y_pred = modelo.predict(X_teste)

print(classification_report(y_teste, y_pred))
ConfusionMatrixDisplay.from_predictions(y_teste, y_pred, normalize='true', cmap='Blues');
