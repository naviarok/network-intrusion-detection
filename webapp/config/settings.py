"""Settings of the demo website. It has no database: it only runs the saved model."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Local demo only. Set DJANGO_SECRET_KEY and DJANGO_DEBUG=0 before deploying anywhere public.
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-key-not-for-production")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

INSTALLED_APPS = ["django.contrib.staticfiles", "detector"]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": ["django.template.context_processors.request"]},
}]
DATABASES = {}
STATIC_URL = "static/"
