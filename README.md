# Tipos de Machine Learning

![img](/assets/tipos.jpeg)

# Definição

Uma boa definição para Machine Learning é "um punhado de estatística, regressão linear e probabilidade aplicada"

Um bom exemplo disso é: 
- [aprendizado-supervisionado/classificacao normal/classifica-caes-e-porcos/poc-classificacao-sem-naive-bayes.py](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-supervisionado/classificacao%20normal/classifica-caes-e-porcos/poc-classificacao-sem-naive-bayes.py)
- [aprendizado-nao-supervisionado/recomendacao/filtragem-colaborativa/recomendacao-com-impl-do-algoritmo-knn.py](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-nao-supervisionado/recomendacao/filtragem-colaborativa/recomendacao-com-impl-do-algoritmo-knn.py)

Se você olhar essas implementações verá que parecem muito mais uma extração e análise de dados, pois não há nenhum processo de "aprendizagem" por meio dos dados, são algoritmos simples, mas eles são machine learning.

O [algoritmo de aprendizado por reforço](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-por-reforco/resolvendo-problema-com-aprendizagem.py) por exemplo, eu fiz uma implementação usando Q-Learning, que aplica uma fórmula matemâtica em todas as células de uma tabela de 500x5 (estados x ações), dessa forma você tem uma tabela cheia de valores que representa o ambiente, e no algoritmo você coloca pro taxi se movimentar (com base nas células que estão ao redor dele) pra célula que tem o maior valor.

![](/assets/taxi.gif)

O taxi por si só não está traçando seu caminho sozinho, ele não aprendeu no sentido literal. Com base em uma fórmula matemâtica coloquei no algoritmo pro taxi se movimentar sempre para a célula de maior valor.

A ideia de "aprendizado de máquina" é referente aos modelos, essa aprendizagem ocorre durante seu treinamento, onde os parâmetros do algoritmo são ajustados com base nos dados de treino, os ajustes são feitos seguindo regras matemáticas e algoritmos específicos, e também pela capacidade de melhorar seu desempenho à medida que são expostos a mais dados de treinamento, quanto mais dados, mais "inteligentes", ou seja, maior será a precisão da resposta.

Um bom exemplo de implementações que utilizam modelos é:
- [aprendizado-supervisionado/classificacao normal/classifica-caes-e-porcos/poc-classificacao.py](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-supervisionado/classificacao%20normal/classifica-caes-e-porcos/poc-classificacao.py)
- [aprendizado-nao-supervisionado/recomendacao/filtragem-colaborativa/recomendando-com-modelo-knn.py](https://github.com/DeveloperArthur/machine-learning/blob/main/aprendizado-nao-supervisionado/recomendacao/filtragem-colaborativa/recomendando-com-modelo-knn.py)

Por mais que não haja um processo de treinamento e aprendizado nas implementações que não utilizam modelos, elas ainda são consideradas machine learning, pois machine learning nada mais é do que **fórmulas de estatística e probabilidade**, essas implementações podem não ser tão eficientes, mas são considerados algoritmos de machine learning ainda sim.

E vale lembrar que as implementações dos modelos não fogem muito dessas implementações mais simples, por mais que sejam implementações mais complexas, Machine Learning é "um punhado de estatística, regressão linear e probabilidade aplicada".

# Referencias

a maioria dos algoritmos deste projeto foram implementados e aprendidos estudando através destes materiais:

![img](/assets/classificacao.jpeg)
![img](/assets/recomendacao.png)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/zQUFxZsZODY/0.jpg)](https://www.youtube.com/watch?v=zQUFxZsZODY)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/uePeleYXD-0/0.jpg)](https://www.youtube.com/watch?v=uePeleYXD-0)
