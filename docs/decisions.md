# Décisions

Chaque entrée : décision, raison, date. Les plus récentes en bas.

## 2026-11-19 — Phase 0 (développement sur CPU, sans GPU)

1. **Développement sur CPU avec Qwen3-0.6B, cible Qwen3-4B-Instruct-2507 sur GPU.**
   Les deux modèles ont le même tokenizer, donc le même masque (vérifié dans `docs/reports/token_mask.md`). Tout le pipeline est testé ici ; sur GPU, seul le nom du modèle change.

2. **Environnement CPU en Python 3.14** (venv `../.venv`, `requirements.txt`). transformers 5.23 et sentence-transformers 6.3 s'y installent. L'environnement GPU (Python 3.12, CUDA ≥ 12.8, Unsloth, bitsandbytes) sera figé à part dans `requirements-gpu.txt`, sur le PC GPU.

3. **Définition du « e » interdit** (`rules.py`) : un caractère est interdit s'il appartient à `FORBIDDEN_CHARS` ou si sa décomposition NFKD en contient un. Cela couvre `e` + accent combinant, ẽ, ę, ainsi que ｅ (pleine chasse) et 𝐞 (mathématique). Au total, 97 caractères Unicode sont interdits. Les positions se calculent sur le texte NFC.

4. **Masque de tokens : règle exacte aux frontières UTF-8 au lieu de « bannir tout token incomplet ».** La règle large du prompt (puis une première version plus fine, qui bannissait tout octet de continuation pouvant terminer *un* caractère interdit) rendait « ù », « û », « ï », « ÿ », « Ù », « Û » et « Ÿ » impossibles à produire : Qwen les tokenise en `Ã` + `¹`, et `¹` termine aussi ẹ (`E1 BA B9`). La règle retenue bannit un seul côté de chaque coupure possible :
   - lead `C3`/`C5` (2 octets, partagés avec les lettres françaises) : on bannit le token qui *commence* par l'octet de continuation interdit (`A9` pour é) ;
   - tout autre lead : on bannit le token qui *finit* par un préfixe incomplet du caractère.
   Test exhaustif sur toutes les paires (fin incomplète × début continuation), environ 200 k pour Qwen, plus 100 k paires aléatoires. Un test de mutation vérifie que le test détecte bien une règle cassée. Résultat : 31,9 % du vocabulaire Qwen interdit.

5. **Tokens spéciaux : seuls EOS et fin de tour (`<|im_end|>`, `<|endoftext|>`) sont autorisés**, et non tous les tokens spéciaux comme le demande le prompt : `<think>`, les appels d'outils, etc. feraient dérailler la génération.

6. **`repetition_penalty` = 1.0 et `no_repeat_ngram_size` = 0 par défaut.** Dans HF, ces deux processeurs voient aussi le prompt : ils pénaliseraient la recopie des mots sans « e » de la source, donc la fidélité. Pour limiter les boucles, on plafonne `max_new_tokens` à 2 × longueur de la source + 32, puis on s'appuie sur la détection de dégénérescence et le best-of-n.

7. **Best-of-n évalué par préfixes** : on génère 16 candidats par texte une seule fois ; le best-of-n pour n se calcule sur les n premiers. Les candidats étant i.i.d., l'estimation est sans biais et coûte 4 fois moins cher que 4 passes.

8. **Embedding pour `sem_sim` : `intfloat/multilingual-e5-small`** (118 M paramètres, environ 470 Mo, tourne sur CPU dans la RAM disponible, multilingue). On utilise le préfixe `query: ` des deux côtés, comme recommandé pour une tâche symétrique. Limite connue : biais lexical. Un lipogramme change forcément de mots, ce qui fait baisser `sem_sim`. On compare donc des variantes entre elles, sans viser un seuil absolu.

9. **`len_ratio` en caractères.** Les lipogrammes emploient des mots de longueur différente ; les caractères sont moins biaisés que les mots.

10. **`fluency` = log-probabilité moyenne par token sous le modèle de base non masqué.** Biais structurel : un lipogramme emploie des mots rares. On n'interprète que des écarts entre variantes ; la notation humaine fait foi. Pour le reranking, on utilise `exp(fluency)`, la probabilité moyenne géométrique par token, déjà dans [0, 1], sans constante arbitraire.

11. **Score composite du reranking** : 0,5 × `sem_sim` + 0,3 × `exp(fluency)` + 0,2 × score de longueur (1 dans [0,7 ; 1,3], décroissance linéaire jusqu'à 0 à 0,5 au-delà). Poids de départ, à revoir après la notation humaine de l'étape 1.

12. **Inférence du 4B en 4-bit (nf4, bitsandbytes).** En bf16, le 4B pèse environ 8 Go et ne tient pas en VRAM. `fluency` est donc mesurée sous le modèle quantifié (impact à vérifier sur un échantillon).

13. **Approximation de la fréquence des tokens dans le rapport du masque : ordre des fusions BPE.** Elle est biaisée vers l'anglais. Elle sera remplacée par des fréquences sur le corpus français une fois celui-ci construit.

## En attente

- **Enseignant du mode A** : backend `claude -p` (abonnement Claude Code, pas de clé API). Avant toute génération, il faut vérifier que les conditions d'utilisation autorisent l'usage des sorties pour entraîner un modèle. Plan B : un enseignant open-weight via OpenRouter, ou le mode B sur GPU.
- **Hash du jeu d'évaluation** : à consigner une fois `data/eval/` figé.
