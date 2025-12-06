from django.urls import path
from . import views

urlpatterns = [
    # MAIN HOME PAGE
    path("", views.home, name="home"),

    # OPTIMIZER HUB PAGE
    path("optimize/", views.optimizer_index, name="optimizer_index"),

    # CLASSIC OPTIMIZER
    path("optimize/classic/", views.classic_optimizer, name="classic_optimizer"),
    path("optimize/classic/run/", views.classic_optimizer_run, name="classic_optimizer_run"),

    # RISK PARITY
    path("optimize/risk-parity/", views.risk_parity_optimizer, name="risk_parity_optimizer"),

    # HIERARCHICAL RISK PARITY
    path("optimize/hrp/", views.hrp_optimizer, name="hrp_optimizer"),
]
