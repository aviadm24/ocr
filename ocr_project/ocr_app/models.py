from django.db import models
import uuid
import os


def get_file_path(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads/', filename)


class OCRImage(models.Model):
    image = models.ImageField(upload_to=get_file_path)
    processed_image = models.ImageField(upload_to='processed/', null=True, blank=True)
    preprocessed_image = models.ImageField(upload_to='preprocessed/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    ocr_text = models.TextField(blank=True, null=True)
    missing_letters = models.JSONField(blank=True, null=True)
    confidence_data = models.JSONField(blank=True, null=True)
    torah_specific_issues = models.JSONField(blank=True, null=True)
    is_hebrew = models.BooleanField(default=True)

    def __str__(self):
        return f"OCR Image {self.id} - {self.uploaded_at}"