from django.shortcuts import render

from django.shortcuts import render, redirect
from django.utils.safestring import mark_safe

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

from .forms import ExtroversionForm, NeuroticismForm, AgreeablenessForm, ConscientiousnessForm, OpennessForm

all_questions_responses = pd.DataFrame()


def index(request):  # pagina inicial
    return render(request, 'bigfivetest/index.html')


def extroversao(request):  # pagina de teste extroversao
    global all_questions_responses
    if request.method == 'POST':
        responses = []  # reseta a lista de respostas para cada usuário
        extroversion_form = ExtroversionForm(request.POST)

        if extroversion_form.is_valid():
            for i in range(1, 11):
                question = extroversion_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10'])

            # checa se o usuário já respondeu este formulário
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # atualiza as linhas existentes
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # adiciona novas colunas
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('neuroticismo')

    else:
        extroversion_form = ExtroversionForm()

    return render(request, 'bigfivetest/extroversao.html', {'extroversion_form': extroversion_form})


def neuroticismo(request):  # pagina de teste neuroticismo
    global all_questions_responses
    if request.method == 'POST':
        responses = []  # reseta a lista de respostas para cada usuário
        neuroticism_form = NeuroticismForm(request.POST)

        if neuroticism_form.is_valid():
            for i in range(1, 11):
                question = neuroticism_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10'])

            # checa se o usuário já respondeu este formulário
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # atualiza as linhas existentes
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # adiciona novas colunas
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('amabilidade')
    else:
        neuroticism_form = NeuroticismForm()

    return render(request, 'bigfivetest/neuroticismo.html', {'neuroticism_form': neuroticism_form})


def amabilidade(request):  # pagina de teste amabilidade
    global all_questions_responses
    if request.method == 'POST':
        responses = []  # reseta a lista de respostas para cada usuário
        agreeableness_form = AgreeablenessForm(request.POST)

        if agreeableness_form.is_valid():
            for i in range(1, 11):
                question = agreeableness_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10'])

            # checa se o usuário já respondeu este formulário
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # atualiza as linhas existentes
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # adiciona novas colunas
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('conscienciosidade')
    else:
        agreeableness_form = AgreeablenessForm()

    return render(request, 'bigfivetest/amabilidade.html', {'agreeableness_form': agreeableness_form})


def conscienciosidade(request):  # pagina de teste conscienciosidade
    global all_questions_responses
    if request.method == 'POST':
        responses = []  # reseta a lista de respostas para cada usuário
        conscientiousness_form = ConscientiousnessForm(request.POST)

        if conscientiousness_form.is_valid():
            for i in range(1, 11):
                question = conscientiousness_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10'])

            # checa se o usuário já respondeu este formulário
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # atualiza as linhas existentes
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # adiciona novas colunas
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('abertura')
    else:
        conscientiousness_form = ConscientiousnessForm()

    return render(request, 'bigfivetest/conscienciosidade.html', {'conscientiousness_form': conscientiousness_form})


def abertura(request):  # pagina de teste abertura
    global all_questions_responses
    if request.method == 'POST':
        responses = []  # reseta a lista de respostas para cada usuário
        openness_form = OpennessForm(request.POST)

        if openness_form.is_valid():
            for i in range(1, 11):
                question = openness_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10'])

            # checa se o usuário já respondeu este formulário
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # atualiza as linhas existentes
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # adiciona novas colunas
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('resultado')
    else:
        openness_form = OpennessForm()

    return render(request, 'bigfivetest/abertura.html', {'openness_form': openness_form})


def resultado(request):  # pagina de resultado
    data_raw = pd.read_csv('bigfivetest/data/data-final.csv', sep='\t')
    data = data_raw.copy()
    data.dropna(inplace=True)
    pd.options.display.max_columns = 150

    # exclui colunas não numéricas da imputação
    numeric_cols = data.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    data.drop(data.columns[50:107], axis=1, inplace=True)
    data.drop(data.columns[51:], axis=1, inplace=True)

    df_model = data.drop('country', axis=1)

    # accessa a variável global all_questions_responses
    global all_questions_responses

    # define 5 clusters e ajusta o modelo
    kmeans = KMeans(n_clusters=5)
    k_fit = kmeans.fit(df_model)

    # prevê os clusters
    pd.options.display.max_columns = 10
    predictions = k_fit.labels_
    df_model['Clusters'] = predictions
    pd.options.display.max_columns = 150

    # soma as respostas de cada fator de personalidade
    col_list = list(df_model)
    ext = col_list[0:10]
    est = col_list[10:20]
    agr = col_list[20:30]
    csn = col_list[30:40]
    opn = col_list[40:50]

    data_sums = pd.DataFrame()
    data_sums['extroversao'] = df_model[ext].sum(axis=1)/10
    data_sums['neuroticidade'] = df_model[est].sum(axis=1)/10
    data_sums['amabilidade'] = df_model[agr].sum(axis=1)/10
    data_sums['conscienciosidade'] = df_model[csn].sum(axis=1)/10
    data_sums['abertura'] = df_model[opn].sum(axis=1)/10
    data_sums['Grupos'] = predictions

    # visualiza a previzão dos clusters
    pca = PCA(n_components=2)
    pca.fit = pca.fit_transform(df_model)

    df_pca = pd.DataFrame(data=pca.fit, columns=['X', 'Y'])
    df_pca['Grupos'] = predictions
    df_pca['Grupos'] += 1

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='X', y='Y', hue='Grupos', data=df_pca, palette=[
                    '#FCD0A1', '#B1B695', '#A690A4', '#5E4B56', '#AFD2E9'])

    # salva a figura em um objeto BytesIO
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # incorpora a imagem na página HTML
    import base64
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    personality_cluster_img = f'<img src="data:image/png;base64,{img_data}" style="width: 100%;">'

    if not all_questions_responses.empty and not all_questions_responses.isnull().values.any():
        # Este código faz a predição do cluster do usuário com base nas respostas dadas pelo usuário em um questionário. Ele utiliza o modelo de KMeans previamente treinado para agrupar as respostas do usuário em um dos cinco clusters definidos. A variável all_questions_responses contém as respostas do usuário em formato de DataFrame, e a variável k_fit contém o modelo de KMeans treinado. A predição é armazenada na variável predict_user_cluster.
        predict_user_cluster = k_fit.predict(all_questions_responses)[0]

        cluster_counts = df_model['Clusters'].value_counts()
        user_cluster_count = cluster_counts[predict_user_cluster]

        user_col_list = list(all_questions_responses)
        user_ext = user_col_list[0:10]
        user_est = user_col_list[10:20]
        user_agr = user_col_list[20:30]
        user_csn = user_col_list[30:40]
        user_opn = user_col_list[40:50]

        all_questions_responses[user_ext] = all_questions_responses[user_ext].astype(
            int)
        all_questions_responses[user_est] = all_questions_responses[user_est].astype(
            int)
        all_questions_responses[user_agr] = all_questions_responses[user_agr].astype(
            int)
        all_questions_responses[user_csn] = all_questions_responses[user_csn].astype(
            int)
        all_questions_responses[user_opn] = all_questions_responses[user_opn].astype(
            int)

        user_data_sums = pd.DataFrame()
        user_data_sums['extroversao'] = all_questions_responses[user_ext].sum(
            axis=1)/10
        user_data_sums['neuroticidade'] = all_questions_responses[user_est].sum(
            axis=1)/10
        user_data_sums['amabilidade'] = all_questions_responses[user_agr].sum(
            axis=1)/10
        user_data_sums['conscienciosidade'] = all_questions_responses[user_csn].sum(
            axis=1)/10
        user_data_sums['abertura'] = all_questions_responses[user_opn].sum(
            axis=1)/10

        # calcula a média e o desvio padrão de cada fator de personalidade
        data_means = data_sums.mean()
        data_stds = data_sums.std()

        # calcula o z-score de cada fator de personalidade
        z_scores = {}
        for col in user_data_sums.columns:
            user_score = user_data_sums[col].values[0]
            data_mean = data_means[col]
            data_std = data_stds[col]
            z_score = (user_score - data_mean) / data_std
            z_scores[col] = round(z_score, 2)

        # cria exemplos de mensagens para cada fator de personalidade
        # mensagens para fatores de personalidade acima da média
        above_avg_examples = {
            'extroversao': ['Você tende a ser mais extrovertido(a) e sociável do que a maioria das pessoas. Você pode gostar de estar rodeado(a) de outras pessoas e de participar de atividades sociais.', 'Você pode ser mais propenso(a) a assumir riscos e a buscar novas experiências.'],
            'neuroticidade': ['Você tende a ser mais emocionalmente instável e ansioso(a) do que a maioria das pessoas. Você pode ter dificuldade em lidar com situações estressantes e ser mais propenso(a) a ter pensamentos negativos.', 'Você pode ser mais propenso(a) a evitar novas experiências e a se sentir desconfortável em situações desconhecidas.'],
            'amabilidade': ['Você tende a ser mais amigável e cooperativo(a) do que a maioria das pessoas. Você pode ser mais empático(a) e se preocupar mais com o bem-estar dos outros.', 'Você pode ser mais propenso(a) a evitar conflitos e a buscar harmonia em suas relações interpessoais.'],
            'conscienciosidade': ['Você tende a ser mais organizado(a) e responsável do que a maioria das pessoas. Você pode ser mais confiável e ter uma forte ética de trabalho.', 'Você pode ser mais propenso(a) a seguir regras e a se preocupar com a segurança e a estabilidade financeira.'],
            'abertura': ['Você tende a ser mais aberto(a) a novas ideias e experiências do que a maioria das pessoas. Você pode ser mais criativo(a) e ter uma mente mais flexível.', 'Você pode ser mais propenso(a) a questionar as normas e a buscar novas formas de pensar e agir.']
        }
        # mensagens para fatores de personalidade abaixo da média
        below_avg_examples = {
            'extroversao': ['Você tende a ser mais introvertido(a) e reservado(a) do que a maioria das pessoas. Você pode preferir atividades solitárias ou em pequenos grupos.', 'Você pode ser mais propenso(a) a evitar situações sociais e a se sentir desconfortável em grandes grupos.'],
            'neuroticidade': ['Você tende a ser mais emocionalmente estável e resiliente do que a maioria das pessoas. Você pode lidar melhor com situações estressantes e ter uma visão mais positiva da vida.', 'Você pode ser mais propenso(a) a buscar novas experiências e a se envolver em atividades criativas.'],
            'amabilidade': ['Você tende a ser mais desconfiado(a) e competitivo(a) do que a maioria das pessoas. Você pode ser mais crítico(a) e menos disposto(a) a cooperar com os outros.', 'Você pode ser mais propenso(a) a entrar em conflito com os outros e a se preocupar mais com seus próprios interesses do que com os dos outros.'],
            'conscienciosidade': ['Você tende a ser mais desorganizado(a) e impulsivo(a) do que a maioria das pessoas. Você pode ter dificuldade em cumprir prazos e em manter um ambiente de trabalho organizado.', 'Você pode ser mais propenso(a) a quebrar regras e a assumir riscos desnecessários.'],
            'abertura': ['Você tende a ser mais fechado(a) a novas ideias e experiências do que a maioria das pessoas. Você pode preferir seguir as normas e evitar situações desconhecidas.', 'Você pode ser mais propenso(a) a julgar as pessoas com base em estereótipos e a ter uma mente mais rígida.']
        }

        # gera mensagens indicando se o usuário está acima ou abaixo da média para cada fator de personalidade
        personality_factor_messages = {}
        for col in z_scores:
            if z_scores[col] > 0:
                personality_factor_messages[
                    col] = f"Você está {z_scores[col]} pontos acima da média mundial em {col}. {' '.join(above_avg_examples[col])}<br /><br />"
            elif z_scores[col] < 0:
                personality_factor_messages[col] = f"Você está {abs(z_scores[col])} pontos abaixo da média mundial em {col}. {' '.join(below_avg_examples[col])}<br /><br />"
            else:
                personality_factor_messages[col] = f"Sua pontuação em {col} é igual à média mundial. Isso significa que você tem uma tendência média em relação a esse fator de personalidade."

        personality_factor_messages_values = '\n'.join(
            personality_factor_messages.values())

        # cria um gráfico de barras com a pontuação do usuário em cada fator de personalidade
        plt.figure(figsize=(10, 5))
        plt.bar(user_data_sums.columns, user_data_sums.iloc[0])
        plt.xlabel('Fatores de Personalidade')
        plt.ylabel('Pontuação')
        plt.ylim(0, 5)

        # salva a figura em um objeto BytesIO
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # incorpora a imagem na página HTML
        import base64
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        test_result_chart = f'<img src="data:image/png;base64,{img_data}" style="width: 100%;">'

        # define as colunas de cada fator de personalidade
        ext = ['EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5',
               'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10']
        est = ['EST1', 'EST2', 'EST3', 'EST4', 'EST5',
               'EST6', 'EST7', 'EST8', 'EST9', 'EST10']
        agr = ['AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5',
               'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10']
        csn = ['CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5',
               'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10']
        opn = ['OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5',
               'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10']

        # define o cluster do usuário
        user_cluster = predict_user_cluster
        user_cluster_data = df_model[df_model['Clusters'] == user_cluster]
        user_cluster_means = {}

        # calcula a média de cada fator de personalidade para o cluster do usuário
        for personality in ['extroversao', 'neuroticidade', 'amabilidade', 'conscienciosidade', 'abertura']:
            if personality == 'extroversao':
                cols = ext
            elif personality == 'neuroticidade':
                cols = est
            elif personality == 'amabilidade':
                cols = agr
            elif personality == 'conscienciosidade':
                cols = csn
            elif personality == 'abertura':
                cols = opn
            user_cluster_means[personality] = user_cluster_data[cols].mean(
            ).mean()

        # calcula o percentil de cada fator de personalidade
        user_percentiles = {}
        for col in user_data_sums.columns:
            user_percentiles = {}
            for col in user_data_sums.columns:
                user_score = user_data_sums[col].values[0]
                cluster_mean = user_cluster_means[col]
                if user_score > cluster_mean:
                    percentile = ((user_score - cluster_mean) /
                                  cluster_mean) * 100
                elif user_score < cluster_mean:
                    percentile = ((cluster_mean - user_score) /
                                  cluster_mean) * -100
                else:
                    percentile = 0.0
                user_percentiles[col] = round(percentile, 2)

            # gera mensagens indicando se o usuário está acima ou abaixo da média para cada fator de personalidade
            more_less_extroversion = str(
                abs(user_percentiles['extroversao'])) + "% mais " if user_percentiles['extroversao'] > 0 else str(abs(user_percentiles['extroversao'])) + "% menos "
            more_less_neuroticism = str(
                abs(user_percentiles['neuroticidade'])) + "% mais " if 'neuroticidade' in user_percentiles and user_percentiles['neuroticidade'] > 0 else str(abs(user_percentiles['neuroticidade'])) + "% menos "
            more_less_agreeableness = str(
                abs(user_percentiles['amabilidade'])) + "% mais " if user_percentiles['amabilidade'] > 0 else str(abs(user_percentiles['amabilidade'])) + "% menos "
            more_less_conscientiousness = str(
                abs(user_percentiles['conscienciosidade'])) + "% mais " if user_percentiles['conscienciosidade'] > 0 else str(abs(user_percentiles['conscienciosidade'])) + "% menos "
            more_less_openness = str(
                abs(user_percentiles['abertura'])) + "% mais " if user_percentiles['abertura'] > 0 else str(abs(user_percentiles['abertura'])) + "% menos "

    else:
        print('Error: all_questions_responses is empty or has missing values')

    # retorna a página HTML com os resultados do teste
    return render(request, 'bigfivetest/resultado.html', {'test_result_chart': mark_safe(test_result_chart), 'predict_user_cluster': mark_safe(predict_user_cluster + 1), 'personality_cluster_img': mark_safe(personality_cluster_img), 'user_cluster_count': mark_safe(user_cluster_count), 'more_less_extroversion': mark_safe(more_less_extroversion), 'more_less_neuroticism': mark_safe(more_less_neuroticism), 'more_less_agreeableness': mark_safe(more_less_agreeableness), 'more_less_conscientiousness': mark_safe(more_less_conscientiousness), 'more_less_openness': mark_safe(more_less_openness), 'personality_factor_messages_values': mark_safe(personality_factor_messages_values)})


def oqueebigfive(request):  # pagina de o que é big five
    return render(request, 'bigfivetest/o-que-e-big-five.html')
