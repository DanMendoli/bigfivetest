# Big Five Test

_Esta é uma aplicação web desenvolvida em Python e Django que fornece um teste de personalidade baseado no modelo Big Five. O objetivo deste projeto é ajudar os usuários a obter insights sobre seus traços de personalidade e melhorar o seu auto-conhecimento, bem como também auxiliar em teste de processos seletivos._

## Introdução

A aplicação consiste em uma página inicial que apresenta uma breve introdução sobre o teste de personalidade baseado no modelo Big Five. Em seguida, há cinco páginas com 10 questões cada, sendo que cada página contém perguntas referentes a um fator de personalidade. Após responder todas as perguntas, o usuário é direcionado para uma página com os resultados do teste, que apresenta uma análise dos traços de personalidade do usuário com base nas respostas dadas. Há na aplicação também uma página dedicada a explicar o que é o Modelo das Cinco Grandes Personalidade (Big Five Model).

## Dependências

Existe no repositório um arquivo chamado requirements.txt que contém todas as dependências necessárias para rodar a aplicação. Para instalar as dependências, basta executar o seguinte comando no terminal:

```
pip install -r requirements.txt
```

Ainda assim, caso seja necessário, as dependências são:

```
asgiref==3.7.2
contourpy==1.1.0
cycler==0.11.0
Django==4.2.5
fonttools==4.42.1
gunicorn==21.2.0
joblib==1.3.2
kiwisolver==1.4.5
matplotlib==3.7.3
numpy==1.25.2
packaging==23.1
pandas==2.1.0
Pillow==10.0.0
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2023.3.post1
scikit-learn==1.3.0
scipy==1.11.2
seaborn==0.12.2
six==1.16.0
sqlparse==0.4.4
threadpoolctl==3.2.0
tzdata==2023.3
yellowbrick==1.5
```

A versão da linguagem Python utilizada para o desenvolvimento do projeto foi a 3.11.4.

## Executando a aplicação localmente

Antes de qualquer coisa, é necessário realizar o download do arquivo csv utilizado para processo de clusterização dos dados. O arquivo pode ser encontrado [aqui](https://openpsychometrics.org/_rawdata/IPIP-FFM-data-8Nov2018.zip). Após o download, basta descompactar o .zip e colocar o arquivo chamado "_data-final.csv_" na pasta _data_, que pode ser acessar pelo caminho: `tcc/bigfivetest/data/`.

Após instalar as dependências em um ambiente virtual ou na instalação padrão do Python em sua máquina, basta executar seguir os seguintes passos, de acordo com o local onde foram instaladas as dependências:

### Ambiente virtual:

#### 1. Ative o ambiente virtual:

    1. No Windows: `<nome-do-ambiente-virutal>\Scripts\activate`
    2. No Linux: `source <nome-do-ambiente-virtual>/bin/activate`

#### 2. Navegue até a pasta do projeto

#### 3. Execute o comando `python manage.py runserver`

#### 4. Acesse o endereço `http://127.0.0.1:8000/` no navegador

#### 5. Pronto! A aplicação já está rodando localmente

### Instalação padrão do Python:

#### 1. Navegue até a pasta do projeto

#### 2. Execute o comando `python manage.py runserver`

#### 3. Acesse o endereço `http://127.0.0.1:8000/` no navegador

#### 4. Pronto! A aplicação já está rodando localmente

## Conceitos e tecnologias utilizadas

- Clusterização de dados
- Inteligência artificial
- Machine Learning
- Python
- Django
- HTML
- CSS
- Git
