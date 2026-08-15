// Initialize Chart
const ctx = document.getElementById('trendChart').getContext('2d');
const trendChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Temp History (°C)',
            data: [],
            borderColor: '#007bff',
            backgroundColor: 'rgba(0, 123, 255, 0.1)',
            fill: true,
            tension: 0.3
        }]
    },
    options: { responsive: true, maintainAspectRatio: false }
});

document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = document.getElementById('submitBtn');
    const errorMsg = document.getElementById('error-msg');
    
    // Prepare Data
    // Note: datetime-local returns "YYYY-MM-DDTHH:MM", we replace T with space
    const rawDate = document.getElementById('date_time').value;
    const formattedDate = rawDate.replace('T', ' ');

    const payload = {
        date_time: formattedDate,
        DewPointC: document.getElementById('DewPointC').value,
        humidity: document.getElementById('humidity').value,
        cloudcover: document.getElementById('cloudcover').value,
        uvIndex: document.getElementById('uvIndex').value,
        sunHour: document.getElementById('sunHour').value,
        precipMM: document.getElementById('precipMM').value,
        pressure: document.getElementById('pressure').value,
        windspeedKmph: document.getElementById('windspeedKmph').value,
        sunrise: document.getElementById('sunrise').value,
        sunset: document.getElementById('sunset').value
    };

    try {
        btn.textContent = "Processing...";
        btn.disabled = true;
        errorMsg.classList.add('hidden');

        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();

        if (res.ok) {
            // Update Number
            document.getElementById('resultValue').textContent = data.predicted_temperature;
            
            // Update Chart
            const timeLabel = formattedDate.substring(5, 16); // MM-DD HH:MM
            addDataToChart(trendChart, timeLabel, data.predicted_temperature);
        } else {
            throw new Error(data.error);
        }
    } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.remove('hidden');
    } finally {
        btn.textContent = "Predict Temperature";
        btn.disabled = false;
    }
});

function addDataToChart(chart, label, data) {
    if (chart.data.labels.length > 8) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    chart.data.labels.push(label);
    chart.data.datasets[0].data.push(data);
    chart.update();
}