from django.db import models

# Create your models here.


class UserInfor(models.Model):
    username = models.CharField(max_length=64)
    password = models.CharField(max_length=16, default='')
    sex = models.CharField(max_length=4)
    phone = models.CharField(max_length=20, default='')
    email = models.EmailField()


class BookInfor(models.Model):
    bookname = models.CharField(max_length=64)
    author = models.CharField(max_length=24, default='')
    ISBN = models.CharField(max_length=32)
    publisher = models.CharField(max_length=64)
    booknum = models.IntegerField()
    address = models.CharField(max_length=64)
