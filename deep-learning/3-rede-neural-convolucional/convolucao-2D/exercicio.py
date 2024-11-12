'''
Ao convoluir um kernel com uma imagem, o resultado será uma nova
imagem com o mapa de ativação destacando padrões específicos do
kernel utilizado. No contexto de classificação de imagens, redes
convolucionais aprendem o conjunto de kernels que melhor auxiliam
na distinção entre as classes. Um exemplo disso seria planejar
kernels com padrões exclusivos de uma determinada classe, tal que
quando esses kernels sejam ativados, já sabemos a que categoria a
imagem de entrada se refere.

De acordo com a figura a seguir, considere que gostaríamos de
distinguir entre as imagens 1 e 2, desenhos minimalistas de uma
casa e uma estrela, respectivamente. Qual dos filtros seria mais
adequado para realizar essa tarefa?

resultado.png

Resposta:
Filtro A!!!
A imagem da casa possui bordas verticais que casam perfeitamente
com o padrão do filtro A, enquanto a estrela possui bordas
horizontais e diagonais.

Filtro B não, ambas as imagens possuem padrões de borda diagonal,
que seriam identificados por esse filtro.

Filtro C não,aAmbas as imagens possuem padrões de borda horizontal,
que seriam identificados por esse filtro.
'''