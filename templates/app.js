const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const uploadForm = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');

// Mostrar vista previa de la imagen al seleccionarla
imageInput.addEventListener('change', (e) => {
  const file = e.target.files[0];

  if (file) {
    const reader = new FileReader();

    reader.onload = function (event) {
      preview.src = event.target.result; 
      preview.style.display = 'block';  
    };

    reader.readAsDataURL(file); 
  } else {
    preview.style.display = 'none'; 
  }
});

// Manejar el envío del formulario
uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  if (!imageInput.files.length) {
    resultDiv.textContent = 'Por favor, selecciona una imagen.';
    return;
  }

  const formData = new FormData();
  formData.append('image', imageInput.files[0]);

  resultDiv.textContent = 'Procesando...';

  try {
    const response = await fetch('https://api.pythonFlask.com/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Error al comunicarse con la API');
    }

    const data = await response.json();
    resultDiv.textContent = `Número predicho: ${data.predicted_number}`;
  } catch (error) {
    resultDiv.textContent = `Error: ${error.message}`;
  }
});
