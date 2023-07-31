from django.db import models

class pI(models.Model):
    path = models.CharField(max_length=255)

class isP(models.Model):
    path = models.CharField(max_length=255)
    label = models.IntegerField(max_length=1)

