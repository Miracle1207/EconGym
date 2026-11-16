function showAlert(type, message) {
    if (type === 'success') {
        alert(`Success: ${message}`);
    } else if (type === 'error') {
        alert(`Error: ${message}`);
    }
}

window.onload = function () {
    // Extract flash messages from hidden inputs
    var messages = document.getElementById('flash-messages');
    if (messages) {
        var flashMessages = JSON.parse(messages.value);
        flashMessages.forEach(function (msg) {
            showAlert(msg.type, msg.message);
        });
    }
}

function generateChart() {
    // Get all selected options in the multi_gov-select dropdown
    const selectedCharts = [];
    const selectElement = document.getElementById('charts');
    const selectedOptions = selectElement.selectedOptions;

    // Loop through selected options and push the values into an array
    for (let option of selectedOptions) {
        selectedCharts.push(option.value);
    }

    // Display the selected charts
    if (selectedCharts.length > 0) {
        document.getElementById('chart-container').innerHTML = 'Selected Charts: ' + selectedCharts.join(', ');
    } else {
        document.getElementById('chart-container').innerHTML = 'No charts selected';
    }
    console.log(selectedCharts)


    fetch('/generate_chart', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
            charts: selectedCharts.join(',')
        }),
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            const chartContainer = document.getElementById('chart-container');
            chartContainer.innerHTML = ''; // Clear previous charts

            if (data.chart_paths) {
                for (const [key, path] of Object.entries(data.chart_paths)) {
                    const height = (path.includes('Government Spending') ||
                        path.includes('GDP') || path.includes('Government Tax Income')) ? '1150px' : '650px';

                    const iframe = document.createElement('iframe');
                    iframe.src = path;
                    iframe.width = '100%';
                    iframe.height = height;
                    chartContainer.appendChild(iframe);
                }
            } else {
                alert('No data available for the selected value.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while generating the chart: ' + error.message);
        });
}
