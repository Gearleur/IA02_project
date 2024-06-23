# Le projet Gopher and Dodo ou le plus gros flop de l'Histoire

## Introduction

Pour commencer ce titre un peu aggicheur je vais vous raconter un projet d'IA02 qui m'a pris plsu d'une nuit sans dormir pendant plusieur semaines et qui fut un échec total.

Le projet Gopher and Dodo est un projet pour l'UV d'IA02 dans laquelle il fallait mettre en place une IA pour joeur au jeux Dodo et Gopher que vous trouverez dans le dossier Règles.
Ce sont des jeux extremment simple et qui sont très facile à mettre en place. (Ce qui est en partie la cause mon echec).

## Modélisation des jeux

Je vais passer le fait d'expliqer les règles du jeu mais je vais m'attarder sur deux modélisation des jeux. Dans chaque dossier de jeu vous avez game et game_2. La première modélisation est la plus complexe car elle devait s'adapter pour pouvoir implementer AlphaZero. Elle prend en compte un systeme de tableau pour gerer les états et un systeme qui permet d'encoder cette état pour pouvoir le passer dans un réseau de neurone. Les parties qui diffère d'une modélisation de bases sont : get_encoded_state et next_state_encoded. La fonction get_encoded_state permet en ayant un etat de jeu donné, permet d'avoir trois matrice de jeu avec : les coups adverse, les coups jouable et enfin les coups joué par le joueur.next_state_encoded quand a elle permet de passer d'un état de jeu à un autre en donnant un coup à partir d'une matrice 1D de l'ensemble des coups jouable.
Par la suite cette modélisation nous simplifira l'utilisation d'un model ResNet pour AlphaZero.

Les eeuxième modélisaiton sont beaucoup plus classqiue avec l'utilisation de dictioanire pour gerer les états et les coups. Cela permet de simplifier l'implémentation de l'IA comme MinMax et simplifie grandement les calculs et la gestion des états.

## Première exploration

apres avoir modélisé les jeux assez rapidement et fait une premeire version de l'IA, j'ai constaté que rapidement j'arrivasi a faire une IA de MinMax. l'IA n'était pas très performante mais elle arrivait a jouer correctement. ce qui me satisfia.

Ainsi, pour aller plus loin, j'ai décidé de m'atteler à une des dernière découverte pour les jeux par google, l'implémentation de AlphaZero. J'ai donc commencé à lire les articles de google et à essayer de comprendre comment cela fonctionnait.

## AlphaZero

Pour commencer, laissez-moi vous expliquer comment AlphaZero fusionne l'intuition et la raison pour créer une intelligence artificielle exceptionnelle. Il y a deux modes de pensée dans le raisonnement humain : un mode rapide basé sur l'intuition et un mode lent guidé par des règles explicites. 

Dans AlphaZero, le mode rapide est représenté par un réseau de neurones qui prend un état de jeu et produit une politique (une distribution de probabilité sur les actions) et une valeur (un score indiquant la qualité de cet état pour le joueur actuel). 
Le mode lent, quant à lui, est incarné par une recherche d'arbre de Monte Carlo (MCTS). Imaginez que nous réfléchissons à la prochaine action à prendre dans un jeu d'information parfaite comme le jeu de Réaction en chaîne. 

Nous pourrions avoir une intuition sur les meilleures actions à prendre. Cette intuition initiale peut être exprimée sous forme de distribution de probabilité sur les actions, attribuant une probabilité plus élevée aux bonnes actions et plus faible aux mauvaises. Cette distribution est notre "politique" pour cet état donné. Pour améliorer cette politique initiale, nous pouvons envisager les mouvements futurs possibles, en utilisant notre intuition pour évaluer les états intermédiaires et éviter de passer trop de temps sur des nœuds à faible valeur. 
Après cette recherche d'arbre, nous aurons une meilleure idée des actions à entreprendre, obtenant ainsi une politique améliorée. Ce processus est appelé "amplification" et il est réalisé par MCTS dans AlphaZero. Ensuite, nous utilisons cette politique améliorée pour optimiser notre réseau de neurones, en minimisant la perte d'entropie croisée entre la politique améliorée et la politique initiale, ainsi qu'une autre perte entre les prédictions de valeur du réseau de neurones et la valeur réelle obtenue à la fin d'une partie. En combinant ces deux processus, AlphaZero parvient à développer des agents experts capables de jouer à des jeux de manière très efficace.

## MCTS

Pour comprendre en détail toutes les étapes de la recherche d'arbre de Monte Carlo (MCTS), nous devons commencer par une vue d'ensemble. Dans MCTS appliqué aux jeux, nous effectuons des simulations répétées du jeu à partir d'un état de plateau donné. Dans la MCTS traditionnelle, ces simulations sont menées jusqu'à la fin du jeu. Cependant, l'implémentation de MCTS dans AlphaZero est différente de la méthode traditionnelle car AlphaZero utilise également un réseau de neurones entraîné pour fournir des politiques et des valeurs pour un état de plateau donné.

Les entrées de l'algorithme de recherche dans AlphaZero sont un état de plateau (noté σ) et le nombre d'itérations (également appelé le nombre de simulations) pour lesquelles nous souhaitons exécuter MCTS. Dans notre cas, la sortie de cet algorithme de recherche serait la politique à partir de laquelle nous sélectionnerions une action à jouer pour l'état σ.

L'arbre est construit de manière itérative. Chaque nœud de l'arbre contient un état de plateau et des informations sur les actions valides possibles dans cet état. En utilisant cette structure, AlphaZero peut améliorer continuellement ses décisions en combinant la recherche approfondie de MCTS avec les prédictions fournies par le réseau de neurones, ce qui conduit à une politique de jeu optimisée pour chaque situation rencontrée.

![State](img/mcts.png)*

### Sélection

La première étape de MCTS est la sélection. On choisit les meilleures arêtes à partir du nœud racine jusqu'à atteindre un nœud terminal ou un nœud non exploré. Les "meilleures arêtes" sont déterminées par un équilibre entre exploration et exploitation, guidé par le réseau de neurones. L'exploration consiste à découvrir de nouvelles informations en visitant de nouveaux nœuds, tandis que l'exploitation utilise les informations existantes pour choisir les nœuds prometteurs.

En pratique, cette phase de sélection suit les arêtes avec les scores les plus élevés, équilibrant les gains attendus et le potentiel de découverte. Cela garantit que l'algorithme explore suffisamment tout en exploitant les actions bénéfiques, maximisant ainsi les chances de trouver une stratégie gagnante.


## Comprendre la règle PUCT

AlphaZero utilise une règle appelée PUCT (Predictor Upper Confidence bounds applied to Trees) pour trouver un équilibre. Cette règle a été conçue de manière empirique, inspirée par les travaux de Rosin dans un cadre de bandits avec prédicteurs. Un article récent de DeepMind discute de quelques alternatives à la PUCT.

La règle PUCT a été développée pour gérer les compromis entre exploration et exploitation dans les arbres de recherche. Elle utilise des prédictions pour guider la recherche, permettant à AlphaZero de naviguer efficacement dans l'espace de jeu. 

Si vous voulez plus d'information sur la règle PUCT, je vous invite à lire suivant article : [PUCT](https://medium.com/@bentou.pub/alphazero-from-scratch-in-pytorch-for-the-game-of-chain-reaction-part-2-b2e7edda14fb)

mais voici une petite image de la formule expliquant la règle PUCT et l'exploration

![PUCT](img/puct.png)*


![Exploration](img/exploration_puct.png)*

Pour bien comprendre comment fonctionne la règle PUCT d'AlphaZero, prenons un exemple concret. Disons que notre réseau neuronal, après avoir été entraîné, nous dit avec une probabilité de 0,3 qu'il faut jouer une action particulière, appelons-la "a". On intègre cette probabilité de 0,3 dans la partie exploration de notre règle PUCT.

Imaginons maintenant que l'état "s" appartient au nœud parent et que l'état obtenu en prenant l'action "a" sur "s" appartient au nœud enfant. Si on visite un nœud particulier trop souvent dans notre recherche MCTS, pour éviter cela et encourager l'exploration d'autres nœuds, on inclut le nombre de visites du nœud enfant dans le dénominateur, et on le normalise en utilisant le nombre total de visites du nœud parent.

Pourquoi prend-on la racine carrée du nombre de visites du nœud parent ? Cette règle PUCT a été conçue de manière empirique, et c'est ce qui a donné les meilleurs résultats parmi toutes les options testées par les chercheurs. En gros, on peut voir ça comme une manière de normaliser le terme child.N + 1 dans le dénominateur.

Il y a un hyperparamètre appelé c_puct que l'on voit dans la figure ci-dessus. Cette constante équilibre les termes d'exploitation et d'exploration. Une valeur typique pour cet hyperparamètre est de 2.

Maintenant qu'on a une idée de comment obtenir PUCT(s, a), revenons à l'étape de sélection dans MCTS.

L'étape de sélection est comme montré dans le bloc ci-dessous : en partant de la racine, on cherche de manière répétée le nœud enfant ayant la valeur PUCT maximale jusqu'à ce qu'on atteigne un nœud dont l'état est encore None (pas encore exploré/initialisé).

### Expansion et evaluation

Une fois la sélection faite, la prochaine étape est d'étendre et d'évaluer ce nœud (dont l'état est encore None). L'extension signifie qu'on initialise l'état du nœud sélectionné selon les règles du jeu. Si le nœud est terminal, on laisse l'état à None et on marque le nœud comme terminal avec l'information du gagnant.

Toutes les nouvelles arêtes du nœud sélectionné sont aussi initialisées. Par exemple, après l'extension, l'arbre ressemblera à la figure ci-dessous.

Ensuite, on évalue le nœud étendu. Par évaluation, on cherche la récompense attendue pour le joueur à ce nœud. Le MCTS traditionnel effectue des simulations depuis le nœud étendu pour trouver la valeur à la fin du jeu, souvent de manière aléatoire.

Le MCTS d'AlphaZero est différent. Ici, on utilise la valeur de sortie du réseau neuronal pour déterminer la valeur du nœud étendu.

C'est un peu comme évaluer une position aux échecs : on calcule quelques coups dans notre tête et on utilise notre intuition pour estimer la qualité de la position résultante, sans faire de simulations jusqu'à la fin du jeu avec des actions aléatoires.

### Backup

Après avoir évalué le nœud étendu, il faut mettre à jour les valeurs Q (réalisées par les valeurs de récompense totales et les comptes de visites totales) pour tous les nœuds, depuis la racine jusqu'au nœud étendu. C'est ce qu'on appelle l'étape de backup dans le MCTS.


## Réseau de neurones

### Schéma global du réseau

```plaintext
Input
  |
Start Block: Conv2d -> BatchNorm2d -> ReLU
  |
Backbone: [ResBlock] x num_resBlocks
  |
+-------------------+
|                   |
Policy Head         Value Head
|                   |
Conv2d -> BatchNorm -> Conv2d -> BatchNorm
ReLU                ReLU
Flatten             Flatten
Linear              Linear
```


### Diagramme du réseau

```plaintext
Input
  |
  V
-------------------------
| Start Block           |
| Conv2d (3 -> num_hidden) |
| BatchNorm2d           |
| ReLU                  |
-------------------------
  |
  V
-------------------------
| Backbone (num_resBlocks)|
| [ResBlock] x num_resBlocks|
-------------------------
  |
  V
-------------------------------------
| Policy Head                      | Value Head
| Conv2d (num_hidden -> 32)        | Conv2d (num_hidden -> 3)
| BatchNorm2d                      | BatchNorm2d
| ReLU                             | ReLU
| Flatten                          | Flatten
| Linear -> game.action_size       | Linear -> 1
|                                  | Tanh
-------------------------------------
```


## Améliorations faites

La première amélioration que j'ia pu faire pour l'entrainnement est de paralleliser l'entrainnement. En effet, l'entrainnement d'AlphaZero est très long et peut prendre plusieurs jours. Ainsi, j'ai décidé de paralléliser l'entrainnement en utilisant plusieurs processus pour entrainer le réseau de neurone. On retrouve ca dans AlphaZeroParallele et MCTSParallele. Cela permet de gagner un temps considérable. Il a fallut egalement traiter des listes de listes d'états encodés ce qui ne fut pas une mince affaire au début.

## Résultats

Sur mon ordinateur je n'ai pu entrainner mon model uniquement sur 40 parties ce qui est clairement insuffisant pour avoir un model performant. Dans le fichier visualisation, vous pourrez voir les résultats de l'entrainnement des model si vous décidé d'ssayer de les entrainer. 
Dans visualisation, vous pourrez voir les résultats de l'entrainnement des model si vous décidé d'essayer de les entrainer. 

Je pense qu'il faut compter environ 3000 parties pour avoir un model performant. Donc il faut clairement améliorer le code pour l'optimiser en particulier pour la mdoélisation du jeu.


## Amélioration possible
Je n'ai utiliser qu'au début un tableau pour voir les pions placé. Mais prendre un state muni d'une grid et d'un dictionnaire n'est clairmeent pas une mauvaise idée et donc il faudrait revoir la modélisation du jeu pour l'optimiser.

Une autre idée que je n'ai pas eu le temsp de faire est de considérer uniquement un état de jeu et par une rotation de 60 degrés, on retrouve facilement les autres états de jeu. Cela permettrait de réduire le nombre d'état de jeu et donc de réduire le temps d'entrainnement. Une des Amélioration les plus importante que je n'ai malheureusement pas eu le temps de faire. (c'est, j'imagine,  ceux sur quoi vous nous attendiez evidement...)

Enfin, comme vous avez à disposition des IA des jeux plutot très performante, une bonne idée serait de commencer l'entrainnement du model avec ces IA pour qu'elle atteigne plus rapidement un niveau de jeu correcte. Je pense que l'entrainnement de l'IA sera extremement plus rapide au début.


## Conclusion

Ce projet clairement j'ai adoré le faire, j'ai appris enormement de chose. Mon seul regret est de ne pas avoir pu le finir correctement. Il y a énormement de chose que j'aurai voulu faire et c'etait surement trop ambitieux pour notre groupe. Je ne me suis peut être pas assez rapproché de professeur pour m'aider au bon déroulement de ce projet. Mais je suis très content de ce que j'ai pu faire et j'espère que vous avez apprécié ce projet autant que moi.


## utilisations

### Attention Important 
avant d'installer les requierements, il faut installer pytorch et cuda en fonction de votre machine. Pour cela, je vous invite à vous rendre sur le site de pytorch et de suivre les instructions pour installer pytorch en fonction de votre machine. voir [Pytorch](https://pytorch.org/get-started/locally/)

### Installation des requierements
```bash
pip install -r requirements.txt
```

### Lancer l'entrainnement
```bash
python train_mcts.py
```

- "num_searches": nombre de recherche pour MCTS
- "C": constante pour la règle PUCT
- "num_iterations": nombre d'itération pour l'entrainnement
- "num_selfPlay_iterations": nombre de partie pour l'entrainnement
- "num_parallel_games": nombre de partie en parallèle
- "num_epochs": nombre d'epoch pour l'entrainnement
- "batch_size": taille du batch : c'est le nombre de partie que l'on va jouer avant de faire un backpropagation
- "temperature": temperature pour la distribution de probabilité
- "dirichlet_epsilon": epsilon pour la distribution de dirichlet
- "dirichlet_alpha": alpha pour la distribution de dirichlet

Si vous voulez load un model pour continuer l'entrainnement, vous pouvez ajouter décommenter la ligne 17 et 18 et ajouter le chemin du model à load et de l'optimisateur.

Enfin num_resBlocks et num_hidden sont les hyperparamètres du réseau de neurone. Ils servent à définir le nombre de block de résiduel et le nombre de neurone dans le réseau de neurone. Comme les jeux sont plutot simple, il n'est pas nécessaire d'avoir un réseau de neurone très complexe.

### Lancer la visualisation
```bash
python visualisation.py
```

### Lancer un jeu
```bash
python main.py
```


