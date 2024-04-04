import requests

import requests

datos = {"tamano": 2000,
"habitaciones": 3}

respuesta = requests.post('http://localhost:5000/', json=datos)

print(respuesta.content)