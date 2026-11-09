# Évaluation de modèles de Computer Vision — Détection, Segmentation d'instances et Segmentation sémantique

> **Notebook :** `Evaluation_CV_3_Datasets_70_20_10.ipynb` (Google Colab, GPU T4 recommandé)
> **Objectif :** découvrir et comprendre les **méthodes d'évaluation** des modèles de vision par ordinateur, en les appliquant à trois tâches différentes, chacune avec son dataset, son modèle et ses métriques.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Prérequis et installation](#2-prérequis-et-installation)
3. [Méthodologie d'évaluation commune](#3-méthodologie-dévaluation-commune)
4. [Tâche 1 — Détection d'objets (Brain Tumor)](#4-tâche-1--détection-dobjets-brain-tumor)
5. [Tâche 2 — Segmentation d'instances (Apple)](#5-tâche-2--segmentation-dinstances-apple)
6. [Tâche 3 — Segmentation sémantique (ADE20K, classe « mur »)](#6-tâche-3--segmentation-sémantique-ade20k-classe--mur-)
7. [Glossaire des métriques](#7-glossaire-des-métriques)
8. [Résultats obtenus](#8-résultats-obtenus)
9. [Analyse et interprétation](#9-analyse-et-interprétation)
10. [Points d'attention et limites](#10-points-dattention-et-limites)
11. [Structure des fichiers générés](#11-structure-des-fichiers-générés)
12. [Pistes d'amélioration](#12-pistes-damélioration)
13. [Reproductibilité](#13-reproductibilité)

---

## 1. Vue d'ensemble

Le notebook compare **trois familles de tâches** de vision par ordinateur, qui se distinguent par la *granularité* de la prédiction :

| Tâche | Ce que le modèle prédit | Dataset | Modèle | Métriques sur le test |
|---|---|---|---|---|
| **Object Detection** | Une boîte englobante (*bounding box*) + une classe par objet | Brain Tumor (Ultralytics) | YOLO11n | Precision, Recall, mAP@50, mAP@50-95 |
| **Instance Segmentation** | Un masque pixel par objet, chaque instance distinguée | Apple (Roboflow, COCO-segmentation) | YOLO11n-seg | Mask Precision, Mask Recall, Mask mAP@50, Mask mAP@50-95 |
| **Semantic Segmentation** | Une classe par pixel (sans distinguer les instances) | ADE20K via `merve/scene_parse_150` (classe « mur ») | U-Net compact (PyTorch) + YOLO11n-seg en comparaison | Pixel Accuracy, IoU, mIoU, Dice |

```
Détection        →  « Il y a un objet dans cette boîte »
Instances        →  « Voici le contour exact de CHAQUE objet »
Sémantique       →  « Chaque pixel appartient à la classe X ou non »
```

> ⚠️ **Les métriques de tâches différentes ne sont pas comparables entre elles** : les datasets, les classes et les critères d'évaluation diffèrent.

---

## 2. Prérequis et installation

### Environnement

- Google Colab (*Exécution → Modifier le type d'exécution → GPU T4*)
- Connexion Internet (téléchargement des datasets et des poids pré-entraînés)
- Python 3.10+

### Dépendances

```bash
pip install ultralytics torchvision pyyaml scikit-learn tqdm matplotlib pillow
pip install roboflow                      # dataset Apple
pip install -U "datasets" "pyarrow>=15.0.0"   # dataset ADE20K (Hugging Face)
```

Le notebook utilise aussi `opencv-python` (`cv2`), `pandas`, `numpy` et `torch` (déjà présents sur Colab).

### Hyperparamètres globaux

| Paramètre | Valeur | Rôle |
|---|---|---|
| `SEED` | 42 | Graine aléatoire (reproductibilité) |
| `EPOCHS_DET` | 12 | Époques d'entraînement — détection |
| `EPOCHS_INST` | 12 | Époques d'entraînement — instances |
| `EPOCHS_SEM` | 8 | Époques d'entraînement — sémantique |
| `IMGSZ_YOLO` | 416 | Taille d'image des modèles YOLO |
| `batch` | 8 (YOLO) / 16 (U-Net) | Taille de lot |

> Les nombres d'époques sont volontairement **faibles** pour que le notebook reste exécutable rapidement. Ils peuvent être augmentés pour de meilleurs résultats.

---

## 3. Méthodologie d'évaluation commune

### 3.1 Découpage 70 / 20 / 10

Pour chaque dataset, les données sont réparties en trois ensembles disjoints :

| Ensemble | Part | Utilisation |
|---|---|---|
| **Train** | 70 % | Apprentissage des poids du modèle |
| **Validation** | 20 % | Suivi de l'entraînement, **sélection du meilleur modèle** (`best.pt`) |
| **Test** | 10 % | **Évaluation finale, une seule fois**, après l'entraînement |

**Pourquoi trois ensembles ?** Si l'on choisissait le meilleur modèle avec le jeu de test, celui-ci deviendrait indirectement un jeu d'entraînement et la performance finale serait **surestimée**. Le jeu de test reste donc « vierge » jusqu'à la fin.

### 3.2 Fonction de découpage

```python
def split_70_20_10(items, seed=SEED):
    items = sorted(items, key=str)
    train, remaining = train_test_split(items, test_size=0.30, random_state=seed)
    val, test = train_test_split(remaining, test_size=1/3, random_state=seed)
    # vérification qu'aucun élément n'est partagé entre les ensembles
    return {'train': train, 'val': val, 'test': test}
```

- Premier découpage : 70 % / 30 %.
- Les 30 % restants sont découpés en 2/3 (20 % du total) et 1/3 (10 % du total).
- `sorted(...)` + `random_state` garantissent un découpage **déterministe**.
- Une assertion vérifie l'**absence de fuite de données** (*data leakage*) entre les ensembles.

### 3.3 Principe général : entraîner → sélectionner → tester

```
Train (70 %) ──► apprentissage
Val   (20 %) ──► choix du meilleur checkpoint (best.pt / meilleur mIoU)
Test  (10 %) ──► métriques finales rapportées
```

---

## 4. Tâche 1 — Détection d'objets (Brain Tumor)

### Dataset
- **Brain Tumor** (jeu officiel Ultralytics, `brain-tumor.yaml`), léger (~4 Mo), téléchargé automatiquement.
- Images médicales annotées en *bounding boxes*, **2 classes** : `negative` et `positive`.
- Les images d'origine sont **réunies puis redécoupées** en 70/20/10 (au lieu de réutiliser le découpage d'origine).

| Ensemble | Images |
|---|---|
| Train | 770 (69,9 %) |
| Validation | 220 (20,0 %) |
| Test | 111 (10,1 %) |

### Pipeline
1. `check_det_dataset('brain-tumor.yaml', autodownload=True)` télécharge le dataset.
2. `discover_pairs()` apparie chaque image à son fichier d'annotation `.txt` (format YOLO).
3. `repartition_yolo()` recrée l'arborescence `images/{train,val,test}` et `labels/{train,val,test}`, avec des noms de fichiers uniques.
4. Génération d'un `data.yaml`.
5. Entraînement de **YOLO11n** (modèle *nano*, le plus léger de la famille) à partir de poids pré-entraînés (`yolo11n.pt`).
6. Chargement de `best.pt` et évaluation avec `model.val(split='test')`.
7. Visualisation de 3 prédictions (seuil de confiance `conf=0.25`).

### Métriques
Extraites de `det_test.box` : `mp` (Precision moyenne), `mr` (Recall moyen), `map50`, `map` (= mAP@50-95).

> ⚠️ **Avertissement :** les images médicales servent uniquement à illustrer l'évaluation informatique. Ce modèle **n'est pas un outil de diagnostic**.

---

## 5. Tâche 2 — Segmentation d'instances (Apple)

### Dataset
- Projet Roboflow **`appledetect`** (version 5), exporté au format **COCO-segmentation**.
- **1 classe** : `apple`.
- Chaque pomme est annotée par un **polygone** individuel.

### Conversion COCO → YOLO
YOLO attend un fichier `.txt` par image, avec une ligne par instance :

```
<class_id> x1 y1 x2 y2 ... xn yn      (coordonnées normalisées entre 0 et 1)
```

La fonction `convert_coco_to_yolo()` :
1. lit le fichier `_annotations.coco.json` ;
2. indexe images et annotations ;
3. copie chaque image ;
4. normalise les coordonnées des polygones (division par largeur / hauteur) ;
5. écrit les fichiers d'annotation YOLO.

### Découpage
Ce dataset **conserve le découpage fourni par Roboflow** (`train/`, `valid/`, `test/`) ; il n'est pas redécoupé en 70/20/10. Le jeu de test contient **97 images** (dont 10 sans pomme, soit 123 instances).

### Entraînement et évaluation
- Modèle : **YOLO11n-seg** (pré-entraîné), 12 époques, `imgsz=416`.
- Les métriques proviennent de `inst_test.seg` — c'est la **Mask mAP** (et non la mAP des boîtes) qui est la métrique principale, car elle mesure la qualité des **masques**.

---

## 6. Tâche 3 — Segmentation sémantique (ADE20K, classe « mur »)

### Dataset
- `merve/scene_parse_150` (ADE20K, Hugging Face), **500 premiers exemples** du split `train`.
- Tâche **binaire** : pixel `mur` (valeur `1` dans l'annotation ADE20K) **vs** tout le reste.

| Ensemble | Images |
|---|---|
| Train | 350 |
| Validation | 100 |
| Test | 50 |

### Prétraitement (`SemanticDataset`)
- Redimensionnement à **128 × 128**.
- Image : interpolation bilinéaire, normalisation `/255`.
- Masque : interpolation **nearest** (indispensable pour ne pas créer de valeurs de classe intermédiaires), puis binarisation `mask == 1`.

### Modèle A — U-Net compact (PyTorch)

Architecture d'encodeur-décodeur avec connexions de saut (*skip connections*) :

```
Entrée (3×128×128)
  e1: ConvBlock 3→16   ──────────────────────┐
  e2: ConvBlock 16→32  ───────────────┐      │
  e3: ConvBlock 32→64  ────────┐      │      │
  center: ConvBlock 64→128     │      │      │
  d3: upsample + concat(e3) ◄──┘      │      │
  d2: upsample + concat(e2) ◄─────────┘      │
  d1: upsample + concat(e1) ◄────────────────┘
  out: Conv 1×1 → 2 classes
```

Chaque `ConvBlock` = `Conv3×3 → BatchNorm → ReLU` répété deux fois.

- **Perte :** `CrossEntropyLoss(ignore_index=255)`
- **Optimiseur :** Adam, `lr=1e-3`
- **Sélection :** le checkpoint avec le meilleur **mIoU de validation** est sauvegardé (`unet_best.pth`) puis rechargé pour le test.

### Évaluation par matrice de confusion pixel

```python
def pixel_confusion(model, loader):   # matrice 2×2 cumulée sur tout le jeu
def semantic_metrics(cm):             # calcule les métriques depuis la matrice
```

La matrice de confusion est **cumulée sur l'ensemble du jeu de test** avant de calculer les métriques (et non moyennée image par image), ce qui est la pratique standard pour le mIoU.

### Modèle B — YOLO11n-seg (comparaison)
Le notebook exporte aussi ce dataset au format YOLO-segmentation (polygones extraits via `cv2.findContours`) et entraîne un YOLO11n-seg sur 8 époques. C'est une approche **de segmentation d'instances appliquée à une tâche sémantique**, utile pour illustrer que le choix du modèle doit correspondre à la nature de la tâche.

---

## 7. Glossaire des métriques

### 7.1 Notions de base

| Terme | Définition |
|---|---|
| **TP** (vrai positif) | Prédiction correcte d'un objet existant |
| **FP** (faux positif) | Prédiction d'un objet qui n'existe pas |
| **FN** (faux négatif) | Objet existant manqué par le modèle |
| **IoU** (*Intersection over Union*) | Taux de recouvrement entre une prédiction et la vérité terrain |

```
            Aire(Prédiction ∩ Vérité terrain)
   IoU  =  ───────────────────────────────────
            Aire(Prédiction ∪ Vérité terrain)
```

L'IoU vaut 0 (aucun recouvrement) à 1 (recouvrement parfait).

### 7.2 Détection et segmentation d'instances

| Métrique | Formule / sens | Lecture |
|---|---|---|
| **Precision** | `TP / (TP + FP)` | Parmi les détections, combien sont justes ? (évite les fausses alarmes) |
| **Recall** | `TP / (TP + FN)` | Parmi les objets réels, combien sont trouvés ? (évite les oublis) |
| **AP** (*Average Precision*) | Aire sous la courbe Precision-Recall d'une classe | Résume le compromis precision/recall |
| **mAP@50** | Moyenne des AP sur les classes, avec un seuil IoU = 0,50 | Critère « tolérant » |
| **mAP@50-95** | Moyenne des mAP sur les seuils IoU 0,50 → 0,95 (pas de 0,05) | Critère **strict**, métrique de référence COCO |
| **Mask mAP** | Idem, mais l'IoU est calculé sur les **masques** | Qualité des contours pixel par pixel |

> Pour les modèles YOLO, la valeur `mp`/`mr` correspond à la moyenne de la Precision/Recall sur toutes les classes.

### 7.3 Segmentation sémantique

| Métrique | Formule | Lecture |
|---|---|---|
| **Pixel Accuracy** | `pixels corrects / pixels totaux` | Simple, mais **trompeuse si les classes sont déséquilibrées** |
| **IoU par classe** | `TP / (TP + FP + FN)` | Recouvrement de la classe |
| **mIoU** | Moyenne des IoU de chaque classe | Métrique de référence en segmentation sémantique |
| **Dice (F1 pixel)** | `2·TP / (2·TP + FP + FN)` | Proche de l'IoU, plus indulgent ; `Dice = 2·IoU / (1 + IoU)` |

---

## 8. Résultats obtenus

Résultats obtenus **uniquement sur les jeux de test**. Ils sont également exportés dans `cv_metrics_test.csv`.

### 8.1 Détection d'objets — Brain Tumor (111 images de test)

| Métrique | Valeur |
|---|---|
| Precision | 0,4999 |
| Recall | 0,8563 |
| mAP@50 | 0,5785 |
| mAP@50-95 | 0,4294 |

Détail par classe :

| Classe | Images | Instances | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| negative | 54 | 56 | 0,495 | 0,893 | 0,540 | 0,409 |
| positive | 57 | 61 | 0,505 | 0,820 | 0,617 | 0,449 |

### 8.2 Segmentation d'instances — Apple (97 images de test, 123 instances)

| Métrique | Valeur |
|---|---|
| Mask Precision | 0,9792 |
| Mask Recall | 0,8780 |
| Mask mAP@50 | 0,9596 |
| Mask mAP@50-95 | 0,8795 |

### 8.3 Segmentation sémantique — mur vs reste (50 images de test)

**U-Net compact (PyTorch) :**

| Métrique | Valeur |
|---|---|
| Pixel Accuracy | 0,8461 |
| IoU arrière-plan | 0,8386 |
| IoU « classe cible » | 0,2333 |
| **mIoU** | **0,5360** |
| Dice « classe cible » | 0,3784 |

**YOLO11n-seg (comparaison) :**

| Métrique | Valeur |
|---|---|
| Mask Precision | 0,0745 |
| Mask Recall | 0,0536 |
| Mask mAP@50 | 0,0134 |
| Mask mAP@50-95 | 0,0027 |

Évolution de l'entraînement du U-Net (mIoU de validation) :

| Époque | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Loss | 0,575 | 0,468 | 0,419 | 0,386 | 0,370 | 0,361 | 0,349 | 0,344 |
| mIoU val | 0,429 | 0,476 | 0,429 | 0,429 | 0,425 | 0,459 | 0,429 | 0,551 |

---

## 9. Analyse et interprétation

### Détection (Brain Tumor) — performance moyenne
- Le **Recall élevé (0,86)** indique que le modèle trouve la plupart des objets.
- La **Precision faible (0,50)** montre beaucoup de fausses détections : à `conf` bas, le modèle « sur-détecte ».
- L'écart entre mAP@50 (0,58) et mAP@50-95 (0,43) indique que les boîtes sont assez bien localisées mais pas parfaitement ajustées.
- Causes probables : entraînement court (12 époques), image réduite à 416 px, modèle *nano*, tâche difficile (images médicales).

### Instances (Apple) — très bonne performance
- Mask mAP@50 de **0,96** et mAP@50-95 de **0,88** : masques précis et bien ajustés.
- Precision (0,98) > Recall (0,88) : le modèle est prudent ; il manque quelques pommes mais se trompe rarement.
- La tâche est visuellement simple (une seule classe, objet bien contrasté).

### Sémantique — résultat contrasté
- **Pixel Accuracy de 0,85 trompeuse :** la classe « arrière-plan » domine. Un modèle qui prédirait *toujours* « arrière-plan » obtiendrait déjà un score élevé.
- L'**IoU de la classe cible (0,23)** et le **Dice (0,38)** révèlent la vraie difficulté : le modèle segmente mal la classe d'intérêt.
- Le **mIoU (0,54)** est tiré vers le haut par l'arrière-plan.
- La validation est **instable** (mIoU oscille entre 0,42 et 0,55) : signe d'un entraînement trop court et d'un jeu de validation réduit (100 images).
- **YOLO11n-seg échoue** (mAP@50 ≈ 0,01) : il est conçu pour des *instances* compactes, pas pour de grandes régions comme des murs. Le U-Net, conçu pour la prédiction dense par pixel, est bien plus adapté.

### Enseignements clés
1. **Une seule métrique ne suffit pas** : la Pixel Accuracy cache le déséquilibre, d'où l'intérêt de l'IoU, du mIoU et du Dice.
2. **Precision et Recall se lisent ensemble** : un Recall élevé avec une Precision faible (détection) et l'inverse (instances) racontent des histoires différentes.
3. **mAP@50 vs mAP@50-95** : le second mesure la qualité de localisation.
4. **Le modèle doit correspondre à la tâche** (cf. YOLO-seg vs U-Net en sémantique).
5. **Le jeu de test ne sert qu'une fois**, pour rapporter la performance finale.

---

## 10. Points d'attention et limites

### ⚠️ Incohérences à corriger dans le notebook

1. **Textes Markdown obsolètes.**
   - La section 2 est intitulée « *Package-Seg* » et décrit des colis, alors que le code utilise le dataset **Apple**.
   - La section 3 est intitulée « *Oxford-IIIT Pet* » (chats/chiens, trimap, contours `label=3`), alors que le code utilise **ADE20K / classe « mur »**.
2. **Étiquettes trompeuses dans le code du U-Net.** Les noms `'IoU animal'`, `'Dice animal'` et `display_labels=['Background','Animal']` désignent en réalité la classe **mur**. Pour cohérence, les renommer (`'IoU wall'`, etc.).
3. **Valeur `255` jamais utilisée.** `ignore_index=255` et `valid = yb != 255` sont hérités de la version Oxford-Pet ; avec ADE20K binaire, aucun pixel n'est ignoré. Sans conséquence, mais inutile.
4. **Découpage Apple non conforme à 70/20/10.** Le découpage Roboflow d'origine est conservé (contrairement à l'annonce en en-tête du notebook).
5. **Tableau de synthèse.** Le dataset est nommé `semantic_yolo_dataset` et seul le U-Net (`sem_scores`) est repris ; les scores YOLO de la tâche sémantique (`sem_yolo_scores`) ne figurent pas dans la synthèse.

### 🔐 Sécurité
**Une clé API Roboflow est écrite en clair dans le notebook.** Il faut la **révoquer et la régénérer** dans votre compte Roboflow avant tout partage (GitHub, Drive public, etc.), puis la lire depuis les secrets Colab :

```python
from google.colab import userdata
rf = Roboflow(api_key=userdata.get('ROBOFLOW_API_KEY'))
```

### Limites méthodologiques
- **Petits jeux de test** (50 à 111 images) : les métriques ont une forte variance ; un petit écart n'est pas significatif.
- **Entraînements courts** (8 à 12 époques) : les résultats sont indicatifs, non optimaux.
- **Sous-échantillon ADE20K** : seulement 500 images, redimensionnées à 128×128 (perte de détails).
- **Une seule graine** : aucune mesure de variabilité entre exécutions.
- **Découpage aléatoire simple** (pas de stratification) : possible déséquilibre de classes entre ensembles.
- **Brain Tumor :** si plusieurs images proviennent du même patient, le découpage aléatoire peut introduire une **fuite de données** entre train et test.

---

## 11. Structure des fichiers générés

```
/content/cv_evaluation/
├── brain_tumor_702010/          # Dataset détection redécoupé 70/20/10
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── apple_yolo_dataset/          # Dataset Apple converti en format YOLO-seg
├── semantic_yolo_dataset/       # Dataset sémantique au format YOLO-seg
├── runs/
│   ├── detection/weights/best.pt
│   ├── instances/weights/best.pt
│   └── semantic_yolo/weights/best.pt
├── unet_best.pth                # Meilleur U-Net (mIoU val)
└── cv_metrics_test.csv          # Synthèse des métriques de test
```

Chaque dossier `runs/*` contient aussi les courbes (`results.png`), les matrices de confusion, les courbes PR/F1 et des exemples de prédictions (`plots=True`).

---

## 12. Pistes d'amélioration

- Augmenter les époques (50–100) et utiliser un `early stopping`.
- Passer à `yolo11s` / `yolo11m` et à `imgsz=640`.
- Ajuster le seuil de confiance (`conf`) en s'appuyant sur la courbe Precision-Recall pour améliorer la Precision en détection.
- Utiliser des **fonctions de perte adaptées au déséquilibre** (Dice loss, Focal loss) pour la segmentation sémantique.
- Employer un encodeur pré-entraîné (ex. U-Net avec ResNet, DeepLabV3, SegFormer) et augmenter la résolution / le nombre d'images ADE20K.
- Appliquer de la **data augmentation** (flip, rotation, variations de luminosité).
- Répéter les expériences sur plusieurs graines ou utiliser une **validation croisée** ; fournir des intervalles de confiance.
- Faire un **découpage par patient** (Brain Tumor) pour éviter les fuites de données.
- Ajouter une analyse d'erreurs qualitative (faux positifs / faux négatifs visualisés).

---

## 13. Reproductibilité

- Graine fixée : `SEED = 42` (Python, NumPy, PyTorch, scikit-learn, Hugging Face `train_test_split`, Ultralytics).
- Découpages déterministes (`sorted` + `random_state`).
- Les résultats peuvent varier légèrement selon le GPU, les versions de bibliothèques et le non-déterminisme de certaines opérations CUDA.

---

## Pour aller plus loin

- Documentation Ultralytics : <https://docs.ultralytics.com>
- Métriques de détection COCO : <https://cocodataset.org/#detection-eval>
- U-Net (Ronneberger et al., 2015) : *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- ADE20K / Scene Parsing : <https://groups.csail.mit.edu/vision/datasets/ADE20K/>
