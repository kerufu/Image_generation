from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('updateDB/', views.updateDB, name='updateDB'),
    path('show/<int:withPrediciton>/', views.show, name='show'),
    path('labelImage/<int:withPrediciton>/', views.labelImage, name='labelImage'),
    path('learnImage/', views.learnImage, name='learnImage'),
    path('generateImage/<int:isCAAE>/', views.generateImage, name='generateImage'),
    path('clearDataset/', views.clearDataset, name='clearDataset'),
]