from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from .models import OCRImage
from .services.ocr_service import OCRService


def index(request):
    """Render the main page"""
    return render(request, 'ocr_app/index.html')


@csrf_exempt
def process_image(request):
    """API endpoint to process the uploaded image"""
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']

        # Save the uploaded image
        ocr_image = OCRImage(image=image_file)
        ocr_image.save()

        # Process the image
        results = OCRService.process_image(ocr_image.image.path)

        # Update the model with results
        ocr_image.processed_image = results['processed_image_path']
        ocr_image.ocr_text = results['ocr_text']
        ocr_image.missing_letters = results['missing_letters']
        ocr_image.confidence_data = results['confidence_data']
        ocr_image.save()

        return JsonResponse({
            'status': 'success',
            'data': {
                'id': ocr_image.id,
                'ocr_text': ocr_image.ocr_text,
                'missing_letters': ocr_image.missing_letters,
                'confidence_data': ocr_image.confidence_data,
                'processed_image_url': ocr_image.processed_image.url,
                'original_image_url': ocr_image.image.url,
            }
        })

    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)