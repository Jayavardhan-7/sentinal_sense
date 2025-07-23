from . import views
from django.urls import path, include

urlpatterns = [
    path("",views.base,name="base"),
    path("results/",views.results,name="results"),
    path("results/result",views.predict,name='predict'),
    
]
