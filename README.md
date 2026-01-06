
### Installation
```bash
# Cloner le repository
git clone https://github.com/username/malaria-detection-cnn
cd malaria-detection-cnn

# Construction en mode release
cargo build --release

# Préparation des données
mkdir -p data/{Parasitized,Uninfected}
# Placer les images dans les dossiers respectifs
```

### Structure des Données
```
data/
├── Parasitized/          # 13,779 images infectées
│   ├── cell_1.png
│   ├── cell_2.png
│   └── ...
└── Uninfected/           # 13,779 images saines
    ├── cell_1.png  
    ├── cell_2.png
    └── ...
```

### Lancement de l'Entraînement
```bash
# Mode équilibré (recommandé)
cargo run --release

# Mode debug (développement)
cargo run

# Tests unitaires
cargo test

# Benchmark
cargo bench
```

## 📁 Structure du Projet

```
Burn_malaria_model_2/
├── Cargo.toml                 # Configuration Rust
├── Cargo.lock                 # Verrouillage des dépendances
├── src/
│   ├── main.rs                # Point d'entrée principal
│   ├── config/
│   │   └── model_config.rs    # Configuration hyperparamètres
│   ├── model/
│   │   └── malaria_cnn.rs     # Architecture CNN
│   ├── data/
│   │   └── dataset.rs         # Dataset et batcher
│   └── training/
│       └── trainer.rs         # Logique d'entraînement
├── data/                      # Dataset (à créer)
│   ├── Parasitized/
│   └── Uninfected/
└── malaria-model-balanced/    # Modèles sauvegardés (auto-généré)
```

## 🎓 Apprentissage et Découvertes

### ✅ Succès Techniques
1. **Performance Rust** : 50-100x plus rapide que Python équivalent
2. **Optimisation Mémoire** : Gestion efficace des 27,558 images
3. **Convergence Stable** : BatchNorm et learning rate adaptatif
4. **Qualité Préservée** : 90% de la précision originale avec 98% de temps en moins

### 🚧 Défis Rencontrés
1. **Temps d'Entraînement Initial** : 4 jours estimés → optimisation nécessaire
2. **Gestion Mémoire** : Cache vs performance → compromis trouvé
3. **Compilation Rust** : Courbe d'apprentissage du borrow checker
4. **Data Loading** : Parallélisation et optimisation I/O

### 🔧 Solutions Implémentées
1. **Réduction Dimensions** : 128×128 → 80×80 (qualité préservée)
2. **Architecture Léger** : Réduction paramètres 70%
3. **Cache Intelligent** : Préchargement partiel et parallélisation
4. **Batch Processing** : Augmentation batch size pour optimisation CPU

## 🔄 Évolution du Projet

### Phase 1: Prototype Initial
- ✅ Architecture CNN de base
- ✅ Pipeline de données fonctionnel
- ✅ Entraînement basique opérationnel

### Phase 2: Optimisation Performance  
- ✅ Réduction temps entraînement (4 jours → 4 heures)
- ✅ Optimisation mémoire et calcul
- ✅ Implémentation métriques avancées

### Phase 3: Industrialisation
- ✅ Code modulaire et maintenable
- ✅ Configuration externalisée
- ✅ Sauvegarde/chargement modèles

## 🔮 Roadmap et Améliorations Futures

### 🎯 Court Terme (1-2 mois)
- [ ] **Data Augmentation** avancée (rotation, flip, contraste)
- [ ] **Cross-Validation** k-fold pour robustesse
- [ ] **Visualisation** des features maps et attention
- [ ] **API REST** pour inference en production

### 🚀 Moyen Terme (3-6 mois)  
- [ ] **Transfer Learning** avec modèles pré-entraînés
- [ ] **Segmentation** des parasites dans les cellules
- [ ] **Multi-Class Classification** (espèces de Plasmodium)
- [ ] **Déploiement Mobile** avec ONNX/TFLite

### 🔬 Long Terme (6+ mois)
- [ ] **Federated Learning** pour confidentialité des données
- [ ] **Active Learning** pour annotation automatique
- [ ] **Integration LIS/HIS** systèmes hospitaliers
- [ ] **Validation Clinique** multi-centres

## 🏥 Impact Médical et Sociétal

### Bénéfices Directs
- **Diagnostic Accéléré** : Minutes → secondes
- **Accessibilité** : Zones rurales et ressources limitées
- **Standardisation** : Réduction variabilité inter-opérateur
- **Coût Réduit** : Automatisation des analyses de routine

### Applications Potentielles
1. **Télémédecine** : Diagnostic à distance
2. **Screening de Masse** : Campagnes de santé publique  
3. **Recherche** : Analyse de grands datasets épidémiologiques
4. **Éducation** : Outil d'apprentissage pour techniciens

## 🤝 Contribution

### Guide de Contribution
1. **Fork** le repository
2. **Feature Branch** : `git checkout -b feature/amazing-feature`
3. **Commit** : `git commit -m 'Add amazing feature'`
4. **Push** : `git push origin feature/amazing-feature`
5. **Pull Request**

### Standards de Code
- **Rustfmt** pour le formatage
- **Clippy** pour les lintings
- **Tests Unitaires** pour chaque module
- **Documentation** exhaustive

### Développement Local
```bash
# Installation environnement
rustup component add clippy rustfmt

# Vérification code
cargo clippy -- -D warnings
cargo fmt --check

# Tests
cargo test
cargo test -- --nocapture  # Avec output
```

## 📄 Licence

Ce projet est distribué sous licence **MIT** - voir le fichier [LICENSE](LICENSE) pour plus de détails.

### Citation Académique
Si vous utilisez ce code dans un contexte de recherche, merci de citer :
```
@software{malaria_detection_2024,
  author = {FOSSOUO WATO MARTIAL},
  title = {Malaria Detection CNN with Burn Framework},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rustnew/Malaria_model_2}}
}
```

## 🙏 Remerciements

- **Équipe Burn** pour le framework exceptionnel
- **Communauté Rust** pour le support et les ressources
- **NIH** pour le dataset de frottis sanguins publics
- **Contributeurs** qui améliorent continuellement le projet


### Lancer l'API d'inférence (Rust)
```bash
# À la racine du projet
MODEL_PATH=./malaria-model.bin cargo run --bin server
# L'API écoute par défaut sur http://localhost:8080
```

Endpoints:
- `GET /health` → renvoie `ok`
- `POST /predict` (multipart/form-data, champ `image`) → renvoie `{ class, probabilities }`

### Lancer l'interface Inference UI (Vite + React)
```bash
cd inference-ui
# Optionnel: créer un fichier .env.local pour configurer l'URL de l'API
echo "VITE_API_BASE=http://localhost:8080" > .env.local

npm install
npm run dev   # ouvre http://localhost:5173
```

Dans l'UI, rendez-vous sur la page « Analyze » (menu en haut) pour:
- téléverser une image de frottis sanguin (drag & drop ou sélection de fichier)
- envoyer la requête à l'API `/predict`
- visualiser la classe prédite (Parasitized / Uninfected) et les probabilités

Note CORS: le serveur autorise les origines en développement (Any). Pour la production, restreindre l'origine côté serveur si nécessaire.
