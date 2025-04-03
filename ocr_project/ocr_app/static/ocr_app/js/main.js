document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    let confidenceChart = null;

    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        // Show loading indicator
        loadingIndicator.classList.remove('d-none');
        resultsContainer.classList.add('d-none');

        // Create form data
        const formData = new FormData(uploadForm);

        // Send request to server
        fetch('/api/process-image/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                displayResults(data.data);
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the image');
        })
        .finally(() => {
            loadingIndicator.classList.add('d-none');
        });
    });

    function displayResults(data) {
        // Display images
        document.getElementById('originalImage').src = data.original_image_url;
        document.getElementById('processedImage').src = data.processed_image_url;

        // Display extracted text
        document.getElementById('extractedText').textContent = data.ocr_text;

        // Display missing letters
        const missingLettersList = document.getElementById('missingLettersList');
        missingLettersList.innerHTML = '';

        if (data.missing_letters && data.missing_letters.length > 0) {
            const ul = document.createElement('ul');
            ul.className = 'list-group';

            data.missing_letters.forEach(item => {
                const li = document.createElement('li');
                li.className = 'list-group-item';

                let confidenceClass = 'low-confidence';
                if (item.confidence > 60) {
                    confidenceClass = 'medium-confidence';
                } else if (item.confidence > 30) {
                    confidenceClass = 'low-confidence';
                }

                li.innerHTML = `
                    <strong>Text:</strong> "${item.text}"
                    <span class="${confidenceClass}">
                        <strong>Confidence:</strong> ${item.confidence}%
                    </span>
                    <div><small>Position: (${item.position.join(', ')})</small></div>
                `;

                ul.appendChild(li);
            });

            missingLettersList.appendChild(ul);
        } else {
            missingLettersList.innerHTML = '<p class="text-success">No missing letters detected with high confidence.</p>';
        }

        // Create confidence chart
        createConfidenceChart(data.confidence_data);

        // Show results container
        resultsContainer.classList.remove('d-none');
    }

    function createConfidenceChart(confidenceData) {
        // Destroy previous chart if it exists
        if (confidenceChart) {
            confidenceChart.destroy();
        }

        // Filter out empty text entries and prepare data
        const filteredData = confidenceData.filter(item => item.text.trim() !== '');

        // Sort by confidence
        filteredData.sort((a, b) => a.confidence - b.confidence);

        // Limit to 30 items for readability if there are too many
        const chartData = filteredData.slice(0, Math.min(30, filteredData.length));

        // Prepare labels and data
        const labels = chartData.map(item => item.text || '?');
        const confidenceValues = chartData.map(item => item.confidence);
        const backgroundColors = confidenceValues.map(value => {
            if (value > 80) return 'rgba(75, 192, 192, 0.6)';  // Green
            if (value > 60) return 'rgba(255, 206, 86, 0.6)';  // Yellow
            return 'rgba(255, 99, 132, 0.6)';  // Red
        });

        // Create chart
        const ctx = document.getElementById('confidenceChart').getContext('2d');
        confidenceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'OCR Confidence (%)',
                    data: confidenceValues,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(color => color.replace('0.6', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Confidence (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Detected Text Characters/Words'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Character/Word Recognition Confidence'
                    },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                return `Text: "${tooltipItems[0].label}"`;
                            },
                            label: function(context) {
                                return `Confidence: ${context.raw}%`;
                            }
                        }
                    }
                }
            }
        });
    }
});