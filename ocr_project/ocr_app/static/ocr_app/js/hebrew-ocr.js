document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContainer = document.getElementById('resultsContainer');
    const hebrewCheckbox = document.getElementById('hebrewCheckbox');
    const torahIssuesSection = document.getElementById('torahIssuesSection');
    let confidenceChart = null;

    // Set the document direction based on Hebrew checkbox
    hebrewCheckbox.addEventListener('change', function() {
        document.querySelector('html').dir = this.checked ? 'rtl' : 'ltr';
        document.getElementById('extractedText').classList.toggle('text-hebrew', this.checked);
    });

    // Set initial direction
    document.querySelector('html').dir = hebrewCheckbox.checked ? 'rtl' : 'ltr';

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
        const extractedTextDiv = document.getElementById('extractedText');
        extractedTextDiv.textContent = data.ocr_text;
        extractedTextDiv.classList.toggle('text-hebrew', hebrewCheckbox.checked);

        // Display missing letters
        displayMissingLetters(data.missing_letters);

        // Display Torah-specific issues if available
        if (data.torah_specific_issues && data.torah_specific_issues.length > 0) {
            displayTorahIssues(data.torah_specific_issues);
            torahIssuesSection.classList.remove('d-none');
        } else {
            torahIssuesSection.classList.add('d-none');
        }

        // Create confidence chart
        createConfidenceChart(data.confidence_data);

        // Show results container
        resultsContainer.classList.remove('d-none');
    }

    function displayMissingLetters(missingLetters) {
        const missingLettersList = document.getElementById('missingLettersList');
        missingLettersList.innerHTML = '';

        if (missingLetters && missingLetters.length > 0) {
            const ul = document.createElement('ul');
            ul.className = 'list-group';

            missingLetters.forEach(item => {
                const li = document.createElement('li');
                li.className = 'list-group-item';

                let confidenceClass = 'low-confidence';
                if (item.confidence > 60) {
                    confidenceClass = 'medium-confidence';
                } else if (item.confidence > 30) {
                    confidenceClass = 'low-confidence';
                }

                li.innerHTML = `
                    <strong>Text:</strong> <span class="hebrew-letter">${item.text}</span>
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
    }

    function displayTorahIssues(torahIssues) {
        const torahIssuesList = document.getElementById('torahIssuesList');
        torahIssuesList.innerHTML = '';

        if (torahIssues && torahIssues.length > 0) {
            const ul = document.createElement('ul');
            ul.className = 'list-group';

            torahIssues.forEach(issue => {
                const li = document.createElement('li');
                li.className = 'list-group-item';

                let issueTypeText = '';
                switch(issue.type) {
                    case 'potentially_confused_letters':
                        issueTypeText = 'Potentially confused letters';
                        break;
                    case 'potentially_broken_letter':
                        issueTypeText = 'Potentially broken letter';
                        break;
                    default:
                        issueTypeText = 'Other issue';
                }

                li.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center">
                        <strong>${issueTypeText}:</strong>
                        <span class="hebrew-letter letter-issue">${issue.text}</span>
                    </div>
                    <div class="mt-2">${issue.description}</div>
                    <div class="mt-1"><small>Confidence: ${issue.confidence}%</small></div>
                `;

                ul.appendChild(li);
            });

            torahIssuesList.appendChild(ul);
        } else {
            torahIssuesList.innerHTML = '<p class="text-success">No Torah-specific issues detected.</p>';
        }
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

        // Limit to 20 items for readability of Hebrew characters
        const chartData = filteredData.slice(0, Math.min(20, filteredData.length));

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
                            text: 'Hebrew Characters'
                        },
                        ticks: {
                            font: {
                                family: "'SBL Hebrew', 'Times New Roman', serif",
                                size: 16
                            }
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Hebrew Character Recognition Confidence'
                    },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                return `Character: "${tooltipItems[0].label}"`;
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