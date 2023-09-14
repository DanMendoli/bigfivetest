from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.utils.safestring import mark_safe

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


from .forms import ExtroversionForm, NeuroticismForm, AgreeablenessForm, ConscientiousnessForm, OpennessForm

all_questions_responses = pd.DataFrame()  # define the variable here


def index(request):
    return render(request, 'bigfivetest/index.html')


def extroversao(request):
    global all_questions_responses  # add this line to access the global variable
    if request.method == 'POST':
        responses = []  # reset the responses list for each user
        extroversion_form = ExtroversionForm(request.POST)

        if extroversion_form.is_valid():
            for i in range(1, 11):
                question = extroversion_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'EXT1', 'EXT2', 'EXT3', 'EXT4', 'EXT5', 'EXT6', 'EXT7', 'EXT8', 'EXT9', 'EXT10'])

            # check if user has already answered this form
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # update existing rows
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # add new columns
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('neuroticismo')

    else:
        extroversion_form = ExtroversionForm()

    return render(request, 'bigfivetest/extroversao.html', {'extroversion_form': extroversion_form})


def neuroticismo(request):
    global all_questions_responses  # add this line to access the global variable
    if request.method == 'POST':
        responses = []
        neuroticism_form = NeuroticismForm(request.POST)

        if neuroticism_form.is_valid():
            for i in range(1, 11):
                question = neuroticism_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'EST1', 'EST2', 'EST3', 'EST4', 'EST5', 'EST6', 'EST7', 'EST8', 'EST9', 'EST10'])

            # check if user has already answered this form
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # update existing rows
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # add new columns
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('agradabilidade')
    else:
        neuroticism_form = NeuroticismForm()

    return render(request, 'bigfivetest/neuroticismo.html', {'neuroticism_form': neuroticism_form})


def agradabilidade(request):
    global all_questions_responses  # add this line to access the global variable
    if request.method == 'POST':
        responses = []
        agreeableness_form = AgreeablenessForm(request.POST)

        if agreeableness_form.is_valid():
            for i in range(1, 11):
                question = agreeableness_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'AGR1', 'AGR2', 'AGR3', 'AGR4', 'AGR5', 'AGR6', 'AGR7', 'AGR8', 'AGR9', 'AGR10'])

            # check if user has already answered this form
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # update existing rows
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # add new columns
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('conscienciosidade')
    else:
        agreeableness_form = AgreeablenessForm()

    return render(request, 'bigfivetest/agradabilidade.html', {'agreeableness_form': agreeableness_form})


def conscienciosidade(request):
    global all_questions_responses  # add this line to access the global variable
    if request.method == 'POST':
        responses = []
        conscientiousness_form = ConscientiousnessForm(request.POST)

        if conscientiousness_form.is_valid():
            for i in range(1, 11):
                question = conscientiousness_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'CSN1', 'CSN2', 'CSN3', 'CSN4', 'CSN5', 'CSN6', 'CSN7', 'CSN8', 'CSN9', 'CSN10'])

            # check if user has already answered this form
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # update existing rows
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # add new columns
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('abertura')
    else:
        conscientiousness_form = ConscientiousnessForm()

    return render(request, 'bigfivetest/conscienciosidade.html', {'conscientiousness_form': conscientiousness_form})


def abertura(request):
    global all_questions_responses  # add this line to access the global variable
    if request.method == 'POST':
        responses = []
        openness_form = OpennessForm(request.POST)

        if openness_form.is_valid():
            for i in range(1, 11):
                question = openness_form.cleaned_data[f'question_{i}']
                responses.append(question)

            user_data = pd.DataFrame([responses], columns=[
                                     'OPN1', 'OPN2', 'OPN3', 'OPN4', 'OPN5', 'OPN6', 'OPN7', 'OPN8', 'OPN9', 'OPN10'])

            # check if user has already answered this form
            if len(all_questions_responses) > 0 and set(user_data.columns).issubset(set(all_questions_responses.columns)):
                # update existing rows
                all_questions_responses.loc[:, user_data.columns] = user_data
            else:
                # add new columns
                all_questions_responses = pd.concat(
                    [all_questions_responses, user_data], axis=1)

            return redirect('resultado')
    else:
        openness_form = OpennessForm()

    return render(request, 'bigfivetest/abertura.html', {'openness_form': openness_form})


def resultado(request):
    # data_raw = pd.read_csv('bigfivetest/data/data-final.csv', sep='\t')
    # data = data_raw.copy()
    # pd.options.display.max_columns = 150

    # from sklearn.impute import SimpleImputer

    # # exclude non-numeric columns from imputation
    # numeric_cols = data.select_dtypes(include='number').columns
    # imputer = SimpleImputer(strategy='mean')
    # data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    # data.drop(data.columns[50:107], axis=1, inplace=True)
    # data.drop(data.columns[51:], axis=1, inplace=True)

    # final_data_table_html = '<table style="width:100%; border-collapse: collapse; border: 1px solid black;">'

    # # add table header
    # final_data_table_html += '<thead><tr>'
    # for col in data.head().columns:
    #     final_data_table_html += f'<th style="border: 1px solid black; padding: 5px;">{col}</th>'
    # final_data_table_html += '</tr></thead>'

    # # add table body
    # final_data_table_html += '<tbody>'
    # for i in range(len(data.head())):
    #     final_data_table_html += '<tr>'
    #     for col in data.head().columns:
    #         final_data_table_html += f'<td style="border: 1px solid black; padding: 5px;">{data.head().iloc[i][col]}</td>'
    #     final_data_table_html += '</tr>'
    # final_data_table_html += '</tbody></table>'

    # --------------------------------------------------------------------------------------------------------------
    # add this line to access the global variable
    global all_questions_responses
    # add table header
    # user_data_table_html = '<table style="width:100%; border-collapse: collapse; border: 1px solid black;">'

    # user_data_table_html += '<thead><tr>'
    # for col in all_questions_responses.columns:
    #     user_data_table_html += f'<th style="border: 1px solid black; padding: 5px;">{col}</th>'
    # user_data_table_html += '</tr></thead>'

    # # add table body
    # user_data_table_html += '<tbody>'
    # for i in range(len(all_questions_responses)):
    #     user_data_table_html += '<tr>'
    #     for col in all_questions_responses.columns:
    #         user_data_table_html += f'<td style="border: 1px solid black; padding: 5px;">{all_questions_responses.iloc[i][col]}</td>'
    #     user_data_table_html += '</tr>'
    # user_data_table_html += '</tbody></table>'

    # df_model = data.drop('country', axis=1)

    # # Defining 5 clusters and fitting the model
    # kmeans = KMeans(n_clusters=5)
    # k_fit = kmeans.fit(df_model)

    # # Predicting the clusters
    # pd.options.display.max_columns = 10
    # predictions = k_fit.labels_
    # df_model['Clusters'] = predictions
    # # lembrar de apagar os prints
    # pd.options.display.max_columns = 150
    # print(df_model.groupby('Clusters').mean())
    # # lembrar de apagar os prints

    # Summing up the different questions groups
    # col_list = list(df_model)
    # ext = col_list[0:10]
    # est = col_list[10:20]
    # agr = col_list[20:30]
    # csn = col_list[30:40]
    # opn = col_list[40:50]

    # data_sums = pd.DataFrame()
    # data_sums['extroversao'] = df_model[ext].sum(axis=1)/10
    # data_sums['neuroticidade'] = df_model[est].sum(axis=1)/10
    # data_sums['agradabilidade'] = df_model[agr].sum(axis=1)/10
    # data_sums['conscienciosidade'] = df_model[csn].sum(axis=1)/10
    # data_sums['abertura'] = df_model[opn].sum(axis=1)/10
    # data_sums['clusters'] = predictions
    # print(data_sums.groupby('clusters').mean())

    # Visualizing the means for each cluster
    # data_clusters = data_sums.groupby('clusters').mean()
    # plt.figure(figsize=(22, 3))
    # for i in range(0, 5):
    #     plt.subplot(1, 5, i+1)
    #     plt.bar(data_clusters.columns, data_clusters.iloc[:, i], color='green', alpha=0.2)
    #     plt.plot(data_clusters.columns, data_clusters.iloc[:, i], color='red')
    #     plt.title('Cluster ' + str(i))
    #     plt.xticks(rotation=45)
    #     plt.ylim(0, 4)

    # Save the figure to a BytesIO object
    # import io
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)

    # # Embed the image in the HTML page
    # import base64
    # img_data = base64.b64encode(buf.read()).decode('utf-8')
    # img_tag = f'<img src="data:image/png;base64,{img_data}">'

    if not all_questions_responses.empty and not all_questions_responses.isnull().values.any():
        # predict_user_data = k_fit.predict(all_questions_responses)
        # print('User data cluster', predict_user_data)
        # print('User all questions type', type(all_questions_responses))

        user_col_list = list(all_questions_responses)
        user_ext = user_col_list[0:10]
        user_est = user_col_list[10:20]
        user_agr = user_col_list[20:30]
        user_csn = user_col_list[30:40]
        user_opn = user_col_list[40:50]

        # Assuming all_questions_responses is a pandas DataFrame
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
        user_data_sums['Extroversão'] = all_questions_responses[user_ext].sum(
            axis=1)/10
        user_data_sums['Neuroticidade'] = all_questions_responses[user_est].sum(
            axis=1)/10
        user_data_sums['Agradabilidade'] = all_questions_responses[user_agr].sum(
            axis=1)/10
        user_data_sums['Conscienciosidade'] = all_questions_responses[user_csn].sum(
            axis=1)/10
        user_data_sums['Abertura'] = all_questions_responses[user_opn].sum(
            axis=1)/10
        # print("Sum of user question groups:")
        # print(user_data_sums)

        # create table for results
        test_result_table_html = '<table style="width:100%; border-collapse: collapse; border: 1px solid black;">'
        # add table header
        test_result_table_html += '<thead><tr>'
        for col in user_data_sums.columns:
            test_result_table_html += f'<th style="border: 1px solid black; padding: 5px;">{col}</th>'
        test_result_table_html += '</tr></thead>'

        # add table body
        test_result_table_html += '<tbody>'
        for i in range(len(user_data_sums)):
            test_result_table_html += '<tr>'
            for col in user_data_sums.columns:
                test_result_table_html += f'<td style="border: 1px solid black; padding: 5px;">{user_data_sums.iloc[i][col]}</td>'
            test_result_table_html += '</tr>'
        test_result_table_html += '</tbody></table>'

    else:
        print('Error: all_questions_responses is empty or has missing values')

    return render(request, 'bigfivetest/resultado.html', {'test_result_table_html': mark_safe(test_result_table_html)})