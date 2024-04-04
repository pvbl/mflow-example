import requests


datos = {'X1': -0.035,
 'X2': -0.989,
 'X3': -1.698,
 'X4': 0.0753,
 'X5': -0.626}

respuesta = requests.post('http://localhost:5000/prediction', json=datos)

print(respuesta)