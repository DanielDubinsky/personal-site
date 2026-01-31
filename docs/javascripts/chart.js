document.addEventListener("DOMContentLoaded", function () {
  var ctx = document.getElementById('yoloChart');
  if (ctx) {
      new Chart(ctx, {
          type: 'bar',
          data: {
              labels: ['ONNX (CPU)', 'Python (Hailo)', 'C++ (Hailo)'],
              datasets: [{
                  label: 'FPS (Higher is Better)',
                  data: [6.93, 36.22, 83.88],
                  backgroundColor: [
                      'rgba(255, 99, 132, 0.5)',
                      'rgba(54, 162, 235, 0.5)',
                      'rgba(75, 192, 192, 0.8)'
                  ],
                  borderColor: [
                      'rgba(255, 99, 132, 1)',
                      'rgba(54, 162, 235, 1)',
                      'rgba(75, 192, 192, 1)'
                  ],
                  borderWidth: 1
              }]
          },
          options: {
              responsive: true,
              scales: {
                  y: {
                      beginAtZero: true,
                      title: {
                          display: true,
                          text: 'Frames Per Second'
                      }
                  }
              },
              plugins: {
                  legend: {
                      display: false
                  },
                  title: {
                      display: true,
                      text: 'YOLO26n Performance Comparison (Raspberry Pi 5)'
                  }
              }
          }
      });
  }
});
