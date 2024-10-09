# Tipos de Machine Learning

![img](/assets/tipos.jpeg)

# Ideia central de "aprendizagem de máquina"

A ideia central de "aprendizagem de máquina" é a capacidade dos algoritmos poderem prever resultados ou realizar tarefas sem serem explicitamente programados para isso.

Embora o algoritmo em si seja explicitamente programado e definido pelos desenvolvedores, no código não tem um conjunto de regras fixas e especificas codificadas diretamente, o algoritmo é projetado para "aprender" com os dados.

O [algoritmo de aprendizado por reforço](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-por-reforco/resolvendo-problema-com-aprendizagem.py) por exemplo

![](/assets/taxi.gif)

Ele não é explicitamente programado com regras de caminhos especificos como if-else para buscar o passageiro e deixar no local de destino, nós utilizamos Q-Learning para **simular** o processo de aprendizagem, utilizando as recompensas para ajustar os parâmetros.

# Definição

Uma boa definição para Machine Learning é "um punhado de estatística, regressão linear e probabilidade aplicada"

Um bom exemplo disso é: 
- [aprendizado-supervisionado/classificacao normal/classifica-caes-e-porcos/poc-classificacao-sem-naive-bayes.py](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-supervisionado/classificacao%20normal/classifica-caes-e-porcos/poc-classificacao-sem-naive-bayes.py)
- [aprendizado-nao-supervisionado/recomendacao/filtragem-colaborativa/recomendacao-com-impl-do-algoritmo-knn.py](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-nao-supervisionado/recomendacao/filtragem-colaborativa/recomendacao-com-impl-do-algoritmo-knn.py)

Estas são implementações que não utilizam modelos, se você olhar essas implementações verá que parecem muito mais uma extração e análise de dados, pois não há nenhum processo de "aprendizagem" por meio dos dados, são implementações simples.

Mas por mais que não haja um processo de treinamento e aprendizado nessas implementações, elas ainda são consideradas machine learning, pois são capazes de prever resultados com base em dados, sem serem explicitamente programadas para isso, e essa é a ideia central!

Essas implementações podem não ser tão eficientes, mas são consideradas machine learning ainda assim.

Exemplo de implementações que utilizam modelos:
- [aprendizado-supervisionado/classificacao normal/classifica-caes-e-porcos/poc-classificacao.py](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-supervisionado/classificacao%20normal/classifica-caes-e-porcos/poc-classificacao.py)
- [aprendizado-nao-supervisionado/recomendacao/filtragem-colaborativa/recomendando-com-modelo-knn.py](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-nao-supervisionado/recomendacao/filtragem-colaborativa/recomendando-com-modelo-knn.py)

Vale lembrar que as implementações dos modelos não fogem muito dessas implementações mais simples, por mais que as implementações dos modelos sejam mais complexas, Machine Learning é "um punhado de estatística, regressão linear e probabilidade aplicada" que tem a capaciade de prever resultados ou realizar tarefas sem ser explicitamente programados para isso.

## Curiosidade: 

Antes de "Inteligência Artificial", esse campo de estudo tinha o nome de Estatística Multivariada, como esse nome não tinha muito apelo comercial, o termo foi trocado para "Inteligência Artificial", que soava mais atraente, evocava uma aura de ficção científica etc

## Deep Learning (sub-área da Machine Learning)
![img](/deep-learning/ai-deep-machine.png)

Tudo começou porque alguém se perguntou "E se a gente simulasse o cérebro humano?"

![img](/deep-learning/ideia.png)

E então criaram um modelo computacional inspirado no funcionamento do cérebro humano

O primeiro modelo neural, o neurônio de Mcculloch e Pitts, foi feito com portas lógicas: 

![img](/deep-learning/primeiro-modelo-neural.png)

E depois foram evoluindo para o que hoje chamamos de [Deep Learning](https://github.com/DeveloperArthur/machine-learning/blob/main/deep-learning)

![img](/deep-learning/timeline.png)

# Referencias

a maioria dos algoritmos deste projeto foram implementados e aprendidos estudando através destes materiais:

![img](/assets/classificacao.jpeg)
![img](/assets/recomendacao.png)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/zQUFxZsZODY/0.jpg)](https://www.youtube.com/watch?v=zQUFxZsZODY)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/uePeleYXD-0/0.jpg)](https://www.youtube.com/watch?v=uePeleYXD-0)
![img](/assets/deep-learning.png)