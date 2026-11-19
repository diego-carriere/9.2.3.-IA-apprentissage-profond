# Rapport — Phase 0 (CPU, sans GPU)

## 1. Résumé
Tout ce qui ne demande pas de GPU est construit et testé (89 tests verts) : règles, masque de tokens, génération contrainte, métriques, best-of-n, évaluation, jeu d'évaluation figé, corpus source, enseignant du mode A et code SFT.
La garantie dure tient : 100 % de sorties sans « e » avec le masque (smoke Qwen3-0.6B).
Le pipeline du mode A produit 86 % de paires utilisables, sans mot inventé, pour environ 0,3 appel Claude Code par source.
Reste à faire sur le PC GPU : étape 1 (baseline Qwen3-4B), puis la génération des données à l'échelle et le SFT.

## 2. Ce qui a été fait / mesuré

**Fait**
- `rules.py` : 97 caractères Unicode interdits (via NFKD : é, ẽ, ｅ, 𝐞…).
- `token_mask.py` : règle exacte aux frontières UTF-8 ; seuls EOS et fin de tour sont autorisés parmi les tokens spéciaux ; cache disque.
- `generate.py` (LogitsProcessor, few-shot optionnel), `metrics.py` (avec `oov_rate`), `rerank.py`, `scripts/eval.py`.
- Jeu d'évaluation figé : 200 phrases + 50 paragraphes + 30 pièges, issus de 19 livres Gutenberg du domaine public. Hash `f45b381a316d8715`.
- Corpus source : 8 693 textes (10 livres Gutenberg hors éval + Wikipédia FR), découpés en 7 831 / 431 / 431.
- `data_build.py` : enseignant `claude -p` (abonnement), avec cache, plafond d'appels, réparation (mots avec « e » et mots inexistants) et filtres.
- `train_sft.py` : prompt identique à l'inférence, loss uniquement sur la réponse, loss masquée en option. **Non exécuté** (GPU requis).

**Mesuré**

| Mesure | Valeur |
|---|---|
| Tokens Qwen3 interdits | 48 351 / 151 669 (31,9 %), identiques pour 0.6B et 4B-Instruct-2507 |
| Test de propriété sur le masque | 0 violation : toutes les paires frontière (environ 200 k) + 100 k paires aléatoires ; une règle cassée exprès est bien détectée |
| Smoke éval, Qwen3-0.6B CPU (5 textes) | `lipogram_rate` = 1,000 (n = 1 et 2) ; environ 50 s par texte pour 2 candidats |
| Enseignant, 50 sources (v2, avec lexique) | 43 gardées ; 19 sans « e » au 1er essai ; 30 réparations réussies sur 31 ; rejets : 6 ratios de longueur, 1 `has_e` |
| Enseignant, 50 sources (v1, sans lexique) | 46 gardées, mais 21 avec des mots inventés et 34 commençant par une minuscule |
| Coût enseignant | 14 appels pour 50 sources ; environ 1 700 tokens de réflexion par source ; aucun euro facturé (quota de l'abonnement) |

## 3. Écarts par rapport au prompt
- Masque : règle exacte par coupure au lieu de « bannir tout token incomplet ». La règle large rendait ù, û, ï, ÿ impossibles à produire.
- Tokens spéciaux : seuls EOS et fin de tour sont autorisés, et non tous.
- `repetition_penalty` et `no_repeat_ngram_size` sont désactivés par défaut : HF les applique aussi au prompt, ce qui empêcherait de recopier les passages sans « e » de la source.
- Best-of-n évalué sur des préfixes de 16 candidats, au lieu de régénérer pour chaque n.
- Jeu d'évaluation issu de Gutenberg uniquement : registre XIXe siècle ou plus ancien.
- Enseignant via la CLI Claude Code (abonnement) plutôt qu'un client API.
- Ajout d'un contrôle lexique, absent du prompt mais indispensable (cf. décision 18).
- SFT sans few-shot (le prompt reste identique entre entraînement et inférence), pour tenir dans `max_seq_length` = 1024.

## 4. Problèmes ouverts / risques
- `sem_sim` (e5) est tassée (de 0,88 à 0,95 sur les données, environ 0,91 sur du charabia) : elle doit être recalibrée avant l'étape 1.
- `fluency` est biaisée contre les mots rares ; seule la note humaine fait foi.
- Le quota de l'abonnement limite la génération : environ 850 appels pour environ 2 500 paires, sur plusieurs sessions.
- Non testé sans GPU : le chargement 4-bit (bitsandbytes), Unsloth sur Blackwell, `train()`.
- Angle mort récurrent de l'enseignant (« mental ») ; contresens ponctuels (noms propres remplacés).
- RAM de ce PC : swap saturé pendant les essais avec le 0.6B.

## 5. Recommandation
1. Recalibrer `sem_sim`, puis lancer l'étape 1 sur le PC GPU (`configs/baseline.yaml`) et produire le rapport du POINT D'ARRÊT 1.
2. En parallèle (sans GPU), générer les données du mode A par sessions jusqu'à environ 2 500 paires, en relisant un échantillon toutes les 500.

## 6. Commandes
```bash
source ../.venv/bin/activate
pip install -r requirements.txt
pytest -q                                              # 89 tests
python scripts/mask_report.py                          # docs/reports/token_mask.md
python scripts/eval.py --config configs/smoke_cpu.yaml # smoke CPU
python scripts/build_eval.py                           # régénère data/eval (hash f45b381a316d8715)
python scripts/build_data.py corpus                    # data/processed/sources.jsonl
python scripts/build_data.py pairs --limit 50          # enseignant (quota Claude Code, reprise via cache)
# PC GPU
python scripts/eval.py --config configs/baseline.yaml
python scripts/train_sft.py --config configs/sft.yaml
```
