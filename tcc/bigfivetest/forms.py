from django import forms

def create_personality_form(trait_name, questions):
    QUESTION_OPTIONS = (
        ('1', '1'),
        ('2', '2'),
        ('3', '3'),
        ('4', '4'),
        ('5', '5'),
    )
    fields = {}
    for i, question in enumerate(questions):
        field_name = f"question_{i+1}"
        fields[field_name] = forms.ChoiceField(label=question, choices=QUESTION_OPTIONS, widget=forms.RadioSelect(attrs={'class': 'app-form-question'}))
    return type(f"{trait_name}Form", (forms.Form,), fields)

ExtroversionForm = create_personality_form("Extroversion", [
    'Eu sou a alma da festa.',
    'Eu não falo muito.',
    'Eu me sinto confortável perto das pessoas.',
    'Eu me mantenho no fundo.',
    'Eu inicio conversas.',
    'Eu tenho pouco a dizer.',
    'Eu converso com muitas pessoas diferentes em festas.',
    'Eu não gosto de chamar a atenção para mim mesmo.',
    'Eu não me importo de ser o centro das atenções.',
    'Eu sou quieto(a) com estranhos.',
])

NeuroticismForm = create_personality_form("Neuroticism", [
    'Eu me estresso facilmente.',
    'Eu estou relaxado(a) na maior parte do tempo.',
    'Eu me preocupo com coisas.',
    'Raramente me sinto triste.',
    'Eu sou facilmente perturbado(a).',
    'Eu fico chateado(a) facilmente.',
    'Eu mudo muito de ânimo.',
    'Tenho frquentes oscilações de humor.',
    'Eu fico irritado(a) facilmente.',
    'Eu frequentemente me sinto triste.',
])

AgreeablenessForm = create_personality_form("Agreeableness", [
    'Eu sinto pouca preocupação pelos outros.',
    'Eu tenho interesse nas pessoas.',
    'Eu insulto as pessoas.',
    'Eu me solidarizo com os sentimentos dos outros.',
    'Não tenho interesse nos problemas das outras pessoas.',
    'Eu tenho um coração sensível.',
    'Realmente, não me interesso pelos outros.',
    'Eu dedico tempo aos outros.',
    'Eu sinto as emoções dos outros.',
    'Eu faço as pessoas se sentirem à vontade.',
])

ConscientiousnessForm = create_personality_form("Conscientiousness", [
    'Eu estou sempre preparado(a).',
    'Eu deixo minhas coisas espalhadas.',
    'Eu presto atenção aos detalhes.',
    'Eu bagunço as coisas.',
    'Eu concluo as tarefas imediatamente.',
    'Frequentemente esqueço de devolver as coisas para o seu devido lugar.',
    'Eu gosto de ordem.',
    'Eu evito minhas responsabilidades.',
    'Eu sigo uma agenda.',
    'Eu sou exigente no meu trabalho.',
])

OpennessForm = create_personality_form("Openness", [
    'Eu tenho um vocabulário rico.',
    'Eu tenho dificuldade em entender ideias abstratas.',
    'Eu tenho uma imaginação vívida.',
    'Eu não tenho interesse em ideias abstratas.',
    'Eu tenho excelentes ideias.',
    'Eu não tenho uma boa imaginação.',
    'Eu sou rápido(a) em entender as coisas.',
    'Eu uso palavras difíceis.',
    'Eu passo tempo refletindo sobre as coisas.',
    'Eu estou cheio(a) de ideias.',
])
