from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('resultado/', views.resultado, name='resultado'),
    path('extroversao/', views.extroversao, name='extroversao'),
    path('neuroticismo/', views.neuroticismo, name='neuroticismo'),
    path('amabilidade/', views.amabilidade, name='amabilidade'),
    path('conscienciosidade/', views.conscienciosidade, name='conscienciosidade'),
    path('abertura/', views.abertura, name='abertura'),
    path('o-que-e-big-five/', views.oqueebigfive, name='oqueebigfive')
]
