# Arthur Docquois et Aurélien Gosse 🤙

# Séance 1 : 3 Mars

## Première utilisation

Premier objectif : noter les performances du système de base avec différents nombres de particules pour avoir un point de comparaison.

| Nombre particule | State update per second |
|------------------|-------------------------|
| 500 | 750 |
| 1000 | 180 |
| 1500 | 85 |
| 2000 | 48 |
| 3000 | 22 |
| 5000 | 8 |
| 10000 (Objectif) | 2 |

On observe une baisse drastique des performances avec l'augmentation du nombre de particules.

## Stratégie qu'on souhaite adopter

Nous souhaitons faire tourner le programme sur un CPU en optimisant un maximum son utilisation.

Pistes :

* OpenMP pour faire du TLP
* OpenMPI pour faire du NLP
* XSIMD pour faire du DLP

## Débuts avec OpenMP

#### Threading naïf

Première approche : utilisation de pragma omp for naïve

Voici le code réalisé :

```
#pragma omp parallel for
    for (int i = 0; i < n_particles; i ++)
    {
        for (int j = 0; j <n_particles; j++)
        {
             ...
		}
	}
#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
            ...
	}
```

La commande pour le lancer :

```
OMP_NUM_THREADS=XX ./build/bin/galax -c CPU_FAST
```

L'ordinateur sur lequel on fait tourner le code possède 12 threads (résultat après commande htop salle D03-007)

| Nombre de particule, nombre de thread | FPS | Gain par rapport à la base (%) |
|---------------------------------------|-----|--------------------------------|
| 2000, 4 | 125 | 160% |
| 2000, 8 | 180 | 275% |
| 2000, 10 | 200 | 316% |
| 2000, 12 | 140 | 191% |

Il faut noter que ces tests sont effectués alors que d'autres applications tournent sur l'ordinateur. Ce qui peut expliquer le fait que l'on ralentisse plus quand on a 12 threads.

On augmente le nombre de particules :

| Nombre de particule, nombre de thread | FPS | Gain par rapport à la base (%) |
|---------------------------------------|-----|--------------------------------|
| 5000, 4 | 25 | 212% |
| 5000, 8 | 30 | 275% |
| 5000, 10 | 37 | 362% |
| 5000, 12 | 38 | 375% |
| 10000, 4 | 6 | 200% |
| 10000, 8 | 8 | 300% |
| 10000, 10 | 9\.5 | 375% |
| 10000, 12 | 11 | 450% |

On observe une accélération significative, qui augmente légerement avec le nombre de threads utilisés.

#### Exploration du code en recherche d'optimisation

Première idée, on remarque qu'on utilise 3 fois :

```
acceleration = diff * dij * initsate.masses
```

On fait :

```
float dijTimeMass = 0.0 
...
for {
   for {
      ...
      dijTimeMass = dij * initstate.masses[j]
   }
}
```

On observe également que le calcul de l'attraction de i par j est l'inverse de celle de j par i. Une optimisation est donc :

```
    for (int i = 0; i < n_particles; i ++)
    {
        for (int j = 0; j <i; j++)
        {
             ...
             accelerationx[i] += diffx * dij * inistate.masses[j];
             ...
             accelerationx[j]  -= diffx * dij * inistate.masses[i];
		}
	}
```

::: success
On a divisé par 2 le nombre de tour de boucle par rapport à l'algorithme de base.

:::

## Prochaine séance

Continuer l'exploration du code et des optimisations, regarder autour du calcul de l'inverse de la racine de dij.

Réfléchir à un début de vectorisation : comment présenter les données pour qu'elle soit proche en mémoire ?

# Séance 2 : 9 Mars

Tests divers qui ne sont pas liés à la parallélisation du code.

#### Utilisation de la fonction std::pow

```
// Calcul de base 
dij = 10/ (dij*dij*dij);
// Test optimisation 
dij = 10 / std::pow(dij,3);
```

::: warn
Cela ne permet pas d'optimiser les fps, on en perd. De plus il y a une erreur d'approximation importante qui apparaît.

:::

Cette fonction réalise en effet une approximation pour le calcul des puissances. De plus voici une étude sur le temps de calcul de pow : <https://baptiste-wicht.com/posts/2017/09/cpp11-performance-tip-when-to-use-std-pow.html>, on peut observer qu'il augmente fortement pour les puissances supérieures à 3.

#### Écriture d'une fonction fast_inv_sqrt

Test d'une fonction écrite à la main pour le calcul rapide de l'inverse de la racine :

```
float inverse_rsqrt( float number )
{
 const float threehalfs = 1.5F;
 float x2 = number * 0.5F;
 float y = number;
 long i = * ( long * ) &y; 
 i = 0x5f3759df - ( i >> 1 );
 y = * ( float * ) &i;
 y = y * ( threehalfs - ( x2 * y * y ) );
 return y;
}
```

::: warn
Cette fonction permet une petite accélération mais réalise une approximation. Il est préférable d'utiliser des instructions déjà présentes comme celles proposées par intel.

:::

# Séance 3 : 10 Mars

::: info
Nous avons aujourd'hui commencé par implémenter un inverse de racine carrée inverse avec les instructions intel AVX.

:::

#### Utilisation des instructions AVX intel

```
// Passage en vecteur AVX
__m256 x_vec = _mm256_set1_ps(dij); 

// Calcul de l'inverse de la racine carrée
__m256 inv_sqrt_x_vec = _mm256_rsqrt_ps(x_vec); 

// Multiplication
inv_sqrt_x_vec = _mm256_mul_ps(inv_sqrt_x_vec,_mm256_mul_ps(inv_sqrt_x_vec,inv_sqrt_x_vec));

//reconversion en float
float inv_sqrt_x = _mm256_cvtss_f32(inv_sqrt_x_vec);
					
dij = 10.0 * inv_sqrt_x;
```

On remarque que cette amélioration permet bien de réduire le nombre de cycles d'executions du calul :

![Comparaison](.attachments.3521666/GALAX_ISR%20%282%29.png)

Avec nos deux améliorations, voici un tableau récapitulatif du gain de performance total, sur les ordinateurs de la salle K2-123 cette fois ci :

| Nombre de points | Sans amélioration (fps) | Avec amélioration (fps) | gain de performances (%) |
|------------------|-------------------------|-------------------------|--------------------------|
| 2000 | 27 | 60 | 112% |
| 4000 | 8 | 16 | 100% |
| 6000 | 3\.7 | 7\.5 | 103% |

Nous avons réussi à doubler nos performances de calcul de base seulement en optimisant le code de base.

::: error
A noter que l'utilisation de ces fonctions ne donne pas la même approximation sur le résultat que celle obtenue avec le calcul précédent, on a donc une légère dérivation de la solution par rapport à la solution de base.

:::

Nous avons ensuite ajouté ces améliorations dans le programme parallélisé à l'aide de openMP. Voici les nouveaux résultats obtenus (On repart des valeurs de base vu le changement de PC)

| Nombre de points, threads | code de base (fps) | Code parallélisé (fps) | gain de performances (%) |
|---------------------------|--------------------|------------------------|--------------------------|
| 2000, 2 | 27 | 60 | 112% |
| 2000, 4 | 27 | 64 | 137% |
| 2000,6 | 27 | 65 | 140% |
| 4000,2 | 8 | 16 | 100% |
| 4000,4 | 8 | 20 | 150% |
| 4000,6 | 8 | 22 | 175% |
| 6000,2 | 3\.7 | 8 | 116% |
| 6000,4 | 3\.7 | 10\.5 | 184% |
| 6000,6 | 3\.7 | 11\.7 | 216% |

Comparé au premier ordinateur qui était équipé de 12 threads, cette ordinateur n'a que 4 threads, ce qui laisse moins de threads libres pour faire les caculs. On se retrouve donc avec des gains de performances beaucoup plus faible qu'auparavent. Cependant, on peur observer que plus le nombre de threads utilisé est grand, plus le gain en performances est grand. Il faut quand même noter qu'avec des valeurs de fps faibles, les valeurs en pourcentage de gain augmentent assez vite entre deux valeurs.

#### Utilisation de XSIMD

Une idée d'optimisation est l'utilisation de la bibliothèque xsimd pour réaliser des opérations sur des vecteurs.

L'idée est de récupérer les positions sous en vecteur en les loadant dans des batch.

```
const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);

b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

std::vector<float> once(b_type::size,1.0);
b_type once_v = b_type::load_unaligned(&once[0]);
```

Ensuite on parcourt toutes les particules une à une et on réalise les calculs avec ces vecteurs.

```
for(int j=0; j<n_particles; j ++){
			b_type diffx = particles.x[j] - rposx_i;
			b_type diffy = particles.y[j] - rposy_i;
			b_type diffz = particles.z[j] - rposz_i;	

			b_type dij = diffx * diffx + diffy * diffy + diffz * diffz;	

			dij = xs::rsqrt(xs::max(once_v,dij)); 
			dij = 10.0 * dij * dij * dij;

			raccx_i = raccx_i + diffx * dij * initstate.masses[j];
			raccy_i = raccy_i + diffy * dij * initstate.masses[j];
			raccz_i = raccz_i + diffy * dij * initstate.masses[j];

			raccx_i.store_unaligned(&accelerationsx[i]);
			raccy_i.store_unaligned(&accelerationsy[i]);
			raccz_i.store_unaligned(&accelerationsz[i]);
}
```

À la fin on stocke les résultats contenus dans les batchs dans les vecteurs.

| Nombre de points | code de base (fps) | code xsimd (fps) | gain de performances (%) |
|------------------|--------------------|------------------|--------------------------|
| 2000 | 27 | 100 | 270% |
| 4000 | 8 | 70 | 775% |
| 6000 | 3\.7 | 30 | 711% |
| 10000 | 1\.38 | 12 | 770% |

On remarque une bien plus grande augmentation en performances qu'avec l'utilisation de OpenMP. On remarque aussi que cette augmentation en performances est plus grande à partir d"un certain nombre de points où l'utilisation de xsimd a vraiment plus d'interet.

::: error
Cependant, nous avons modifié le calcul de l'inverse de la racine carrée dans la boucle, ce qui modifie un peu le résultat. Nous obtenons donc une petite accumulation d'erreur comme lorsque nous avons modifié cette même opération avec les instructions intel AVX.

:::

L'idée est maintenant d'optimiser cette idée, plusieurs pistes s'offrent à nous :

* Utiliser des vecteurs SIMD sur la seconde boucle
* Voir si on ne peut pas faire des batchs avec les particules j
* Voir le chargement des batchs : unaligned VS aligned
* Essayer d'utiliser les threads

::: warn
Vérifier qu'on a bien parcouru toutes les particules et qu'il n'en reste pas

:::

#### XSIMD et threads

Nous avons maintenant combiné notre implémentation XSIMD avec de simples praga omp for pour paralléliser le code sur des threads.

```
#pragma omp parallel for simd
    for (int i = 0; i < n_particles; i += b_type::size)
    {  ...

#pragma omp parallel for
	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
...
```

On obtient les performances suivantes (avec la fenêtre graphique active) :

| Nombre de particules, nombre de threads | code de base (fps) | code parallélisé (fps) | gain en performances (%) |
|-----------------------------------------|--------------------|------------------------|--------------------------|
| 2000,2 | 27 | 500 | 1750% |
| 2000,4 | 27 | 200 | 640% |
| 2000,6 | 27 | 200 | 640% |
| 6000,2 | 3\.7 | 65 | 1650% |
| 6000,4 | 3\.7 | 80 | 2060% |
| 6000,6 | 3\.7 | 75 | 1930% |
| 10000,2 | 1\.38 | 24 | 1640% |
| 10000,4 | 1\.38 | 35 | 2440% |
| 10000,6 | 1\.38 | 35 | 2440% |

::: warn
On observe une grande augmentation de performances pour le 2000 particules, 2 threads que l'on ne retrouve pas pour le 4 et le 6 threads. On peut l'expliquer par le fait que on perd trop de puissance à répartir et controler la charge plutot qu'à faire les calculs, étant donné qu'il y n'y en a pas assez. Ainsi avec plus de points, le phénomène disparait et il est plus rentable d'utiliser plus de threads.

:::

On voit très peu de performances supplémentaires entre 4 et 6 threads étant donné que les processeur n'est doté que de 4 threads.

::: success
On obtient un gain de performance de l'ordre de x2 à x3 par rapport au xsimd sans threads.

:::

::: info
Sans fenêtre graphique, on monte à 47 fps avec la simulation à 10 000 points, 4 threads.

:::

#### Objectifs de la séance prochaine :

* Commencer à toucher à OpenMPI ?
* Commencer à regarder éventuellement les optimisations mémoires sur le cache
* Des pistes sur d'autres types d'algorithmes ?

# Séance 4 : 17 Mars

## Objectifs

* Réparer la simulation : attention au i != j
* Limiter le nombre d'itérations et les accès mémoires grâce à des permutations
* Trouver une référence du maximum de performance atteignable en théorie
* Regarder si besoin de parallélisme sur les autres boucles que la boucle principale

## Performance maximale théorique

Avec f la fréquence, P le parallélisme (xsimd), T le nombre de threads

Pour trouver f :

```
watch -n 0.5 proc/cpuinfo | grep MHz
```

f = 3.3 GHz

On a P = 8 et T=4

Donc en théorie on peut réaliser :

105\.6 G opération par seconde.

Il faut maintenant regarder le nombre d'opération à effectuer dans la boucle principale (on regarde le throughput) pour l'architecture skylake :

* L'addition (utilisé 8 fois) : 0.5 cycle
* La multiplication (utilisé 12 fois)  : 0.5 cycle
* Le max (utilisé une fois) : 0.5 cycle
* rsqrt (utilisé une fois) : 1 cycle
* Store unaligned (utilisé 3 fois) : 1 cycle

Ce qui fait 18 cycles pour une itération de la seconde boucle, donc il faut encore multiplier par le nombre de particules pour avoir le nombre de cycles de la boucle intérieure.

Puis pour le calcul d'une frame il faut encore multiplier par le nombre de particules divisées par 8 en ajoutant les opérations de load :

* Load unaligned (utilisé 7 fois) : 0.5 cycles

Donc pour 10 000 particules pour faire une frame il faut environ : 1.8e9 cycles

::: success
Donc en théorie le maximum atteignable est d'environ : 58 fps avec notre implémentation. Sachant qu'elle peut être encore optimisée.

:::

## Réparation de la divergence

La divergence vient du fait que l'on continue de prendre en compte les cas où on calcule la distance d'une particule avec elle même. On va progressivement induire une erreur à mesure que l'on repasse sur les mêmes particules. Pour cela avec xsimd, on va s'occuper du cas où notre particule j se trouve dans le vecteur de 8 particules actuel. On utilise un masque de bits afin de supprimer l’interaction entre la particule et elle même, de cette manière 

```
if (!((j < i) | (j>i+7))) //Si j se trouve entre i et i+8
{
	b_type mask = b_type::load_unaligned(&masks[j-i][0]);
	dij = dij * mask;
}
```

On obtient bien au final une réduction de l'erreur. 

## Limitation des accès mémoires

L'idée est de passer le parcours dans la seconde boucle en chargeant les particules j dans des vecteurs et de faire des rotations dans ce vecteur pour ne faire qu'un seul accès mémoire.

Il y a besoin de mettre à jour la librairie xsimd. (git checkout master, git pull, re-compiler)

### Objectifs pour la prochaine séance

Terminer le code pour la version xsimd améliorée

Piste : utilisation de make_sized_batch

# Séance 5 : 24 Mars

## Amélioration du code xsimd

Nous avons implémenté l'amélioration des accès mémoires, en utilisant des rotations :

  
```
    struct Rot{
	    static constexpr unsigned get(unsigned i, unsigned n){
		    return (i+n-1)%n;
		}
	};
	...
	auto constexpr mask = xs::make_batch_constant<xs::batch<uint32_t,xs::avx2>, Rot>();
	for(int i ...){
		 ...
		 for(int j ...){
			 ...
			 for(int k = 0; i<7; k++){
				calcul;
				rposx_j = xs::swizzle(rposx_j,mask);
				...
			}
		...
		}
	}
```
	 
Résultats :

| Nombre de particules, nombre de threads | code de base (fps) | code parallélisé (fps) | gain en performances (%) |
|-----------------------------------------|--------------------|------------------------|--------------------------|
| 2000,2 | 27 | 950 | 3400% |
| 2000,4 | 27 | 1800 | 6500% |
| 2000,6 | 27 | 1000 | 3600% |
| 6000,2 | 3\.7 | 93 | 2400% |
| 6000,4 | 3\.7 | 170 | 4500% |
| 6000,6 | 3\.7 | 130 | 3400% |
| 10000,2 | 1\.38 | 40 | 2800% |
| 10000,4 | 1\.38 | 76 | 5400% |
| 10000,6 | 1\.38 | 60 | 4250% |



# Boite à idée

Changer les calculs pour des calculs plus rapide, exemple : \*2 -> shift, calcul de l'inverse de la racine (fast inverse square root)

::: success
Attraction i -> j = - attraction j -> i

:::

Deux particules très éloignées ont peu d'influence l'une sur l'autre mais le calcul sqrt est quand même fait

Coupler le SIMD et le TLP avec l'instruction pragma omp for simd

Regardez !  la galaxie s'enfuie

## Ressources

<https://xsimd.readthedocs.io/en/latest/vectorized_code.html>

<https://github.com/Ohrelianne/galax_eleves>

<https://godbolt.org/>

<https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>

<https://github.com/xtensor-stack/xsimd>
