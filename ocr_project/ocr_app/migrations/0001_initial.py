# Generated by Django 5.2 on 2025-04-02 22:01

import ocr_app.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='OCRImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to=ocr_app.models.get_file_path)),
                ('processed_image', models.ImageField(blank=True, null=True, upload_to='processed/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('ocr_text', models.TextField(blank=True, null=True)),
                ('missing_letters', models.JSONField(blank=True, null=True)),
                ('confidence_data', models.JSONField(blank=True, null=True)),
            ],
        ),
    ]
