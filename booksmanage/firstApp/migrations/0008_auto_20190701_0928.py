# Generated by Django 2.2.2 on 2019-07-01 01:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('firstApp', '0007_auto_20190628_1628'),
    ]

    operations = [
        migrations.AddField(
            model_name='userinfor',
            name='password',
            field=models.CharField(default='', max_length=16),
        ),
        migrations.AlterField(
            model_name='userinfor',
            name='phone',
            field=models.CharField(default='', max_length=20),
        ),
    ]
