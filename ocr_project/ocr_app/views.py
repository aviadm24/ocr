from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import os
from .models import OCRImage
from .services.ocr_service import OCRService
from .services.hebrew_ocr_service import HebrewOCRService


def index(request):
    """Render the main page"""
    return render(request, 'ocr_app/index.html')


@csrf_exempt
def process_image(request):
    """API endpoint to process the uploaded image"""
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        is_hebrew = request.POST.get('is_hebrew', 'true').lower() == 'on'

        # Save the uploaded image
        ocr_image = OCRImage(image=image_file, is_hebrew=is_hebrew)
        ocr_image.save()

        # Process the image based on language selection
        if is_hebrew:
            # Preprocess for Torah scroll
            preprocessed_path = HebrewOCRService.preprocess_image_for_torah(ocr_image.image.path)

            # If a new preprocessed image was created
            if preprocessed_path != ocr_image.image.path:
                relative_path = os.path.relpath(preprocessed_path, settings.MEDIA_ROOT)
                ocr_image.preprocessed_image = relative_path
                ocr_image.save()

                # Use preprocessed image for OCR
                results = HebrewOCRService.process_torah_image(preprocessed_path)
            else:
                # Use original image if no preprocessing was needed
                results = HebrewOCRService.process_torah_image(ocr_image.image.path)

            # Update with Torah-specific issues
            ocr_image.torah_specific_issues = results.get('torah_specific_issues', [])
        else:
            # Use standard OCR service for non-Hebrew text
            results = OCRService.process_image(ocr_image.image.path)

        # Update the model with results
        ocr_image.processed_image = results['processed_image_path']
        ocr_image.ocr_text = results['ocr_text']
        ocr_image.missing_letters = results['missing_letters']
        ocr_image.confidence_data = results['confidence_data']
        ocr_image.save()

        response_data = {
            'status': 'success',
            'data': {
                'id': ocr_image.id,
                'ocr_text': ocr_image.ocr_text,
                'missing_letters': ocr_image.missing_letters,
                'confidence_data': ocr_image.confidence_data,
                'processed_image_url': ocr_image.processed_image.url,
                'original_image_url': ocr_image.image.url,
            }
        }

        # Add Hebrew-specific data if applicable
        if is_hebrew:
            response_data['data']['torah_specific_issues'] = ocr_image.torah_specific_issues
            if ocr_image.preprocessed_image:
                response_data['data']['preprocessed_image_url'] = ocr_image.preprocessed_image.url

        return JsonResponse(response_data)

    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)