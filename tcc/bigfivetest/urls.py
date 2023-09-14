from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('resultado/', views.resultado, name='resultado'),
    path('extroversao/', views.extroversao, name='extroversao'),
    path('neuroticismo/', views.neuroticismo, name='neuroticismo'),
    path('agradabilidade/', views.agradabilidade, name='agradabilidade'),
    path('conscienciosidade/', views.conscienciosidade, name='conscienciosidade'),
    path('abertura/', views.abertura, name='abertura'),
]