from Modelo import modelo, animal_misterioso

animal_previsto = modelo.predict([animal_misterioso])

if animal_previsto == 0:
    print('O animal é um pato')
elif animal_previsto == 2:
    print('O animal é um gato')
elif animal_previsto == 1:
    print('O animal é um ganso.')

print('Fim do programa.')